
from typing import List, Union, Optional
import copy
from collections import defaultdict
import torch
import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin
from dataclasses import dataclass, field

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import process_image, collate_fn
import re
# using factory to initialize the list

from vagen.env.register import REGISTERED_ENVS
from vagen.env.base import EnvFeedback, EnvConfig

# TODO an extra class for recorder
@dataclass
class QwenVLRolloutConifg:
    window_size=5
    max_prompt_length=1024
    max_turns=5
    n_gpu_per_node=1 # used for multigpu batch balancing
    # use factory to initialize the list
    sptk_for_loss_mask: List[str] = field(default_factory=lambda: ['<|box_start|>', '<|box_end|>'])
    
class QwenVLRolloutManger():
    def __init__(self,
                 actor_rollout_wg,
                 tokenizer: PreTrainedTokenizer,
                 config: QwenVLRolloutConifg,
                 processor: Optional[ProcessorMixin] = None,
                 verbose: bool = False,
                 truncation='error',
                 ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.verbose = verbose
        self.truncation = truncation
        self.recorder= None # defaultdict(list)
        self.envs = None # dict
        self.env_states = None # dict
        self.batch_idx_to_env_id = None # dict
    
        
    def reset(self, env_configs: List[EnvConfig]):
        """
        Reset environments based on provided configurations, reusing environments when possible.
        - For env with same config and env_name, reuse the same environment (reset)
        - For env with different config or env_name, close the old environment and create a new one
        - Reset the recorder
        
        Args:
            env_configs: List of environment configurations containing env_name, config, and seed
        
        Returns:
            Initial observations and info from all environments
        """
        # Step 1: Sort environments into buckets by env_name and config
        # Try to reuse environemnts with the same config and env_name
        
        env_buckets = defaultdict(set)
        new_envs = {}
        
        if self.envs is None:
            self.envs = {}
            
        for env_id, env in self.envs.items():
            env_name = env.name_repr()
            env_config = env.config_repr(env.config)
            bucket_key = f"{env_name}:{env_config}"
            env_buckets[bucket_key].add(env_id)
        
        for i, cfg in enumerate(env_configs):
            env_id = i
            env_name = cfg.env_name
            env_config = cfg.env_config
            seed = cfg.seed
            
            # Create bucket key
            config_key = REGISTERED_ENVS[env_name].config_repr(env_config)
            bucket_key = f"{env_name}:{config_key}"
            
            # Check if we have an available environment with the same config
            if bucket_key in env_buckets and env_buckets[bucket_key]:
                old_env_id = env_buckets[bucket_key].pop()
                new_envs[env_id] = {
                    "env":self.envs[old_env_id],
                    "seed":seed,
                    "config":env_config,
                }
            else:
                # don't initialize the environment here, close unused environments first
                new_envs[env_id] = {
                    "env_class":REGISTERED_ENVS[env_name],
                    "seed":seed,
                    "config":env_config,
                }
        
        # Close unused environments
        for bucket_key, env_ids in env_buckets.items():
            for env_id in env_ids:
                self.envs[env_id].close()
                del self.envs[env_id]

        
        # Step 2: Reset environments and collect observations/info
        
        if self.recorder is not None:
            del self.recorder
        self.recorder = defaultdict(list)
        initial_obs = {}
        initial_info = {}
        for k, v in new_envs.items():
            if "env" in v:
                self.envs[k] = v["env"]
            else:
                assert "env_class" in v
                self.envs[k] = v["env_class"](**v["config"])
            env_feedback = self.envs[k].reset(v["seed"])
            initial_obs[env_id] = env_feedback.env_observation
            initial_info[env_id] = env_feedback.info
            self.record(env_id, env_feedback)
        
        self.env_states = {env_id: {'step': 0, 'done': False} for env_id in self.envs}
        
        return initial_obs, initial_info
    
    
    def record(self, env_id, env_feedback: EnvFeedback):
        """
        Record each step's obs, info, done, reward,
        Please include "llm_raw_response" in info # it will be decoded by rollout manager and pass to env, then should pass back
        """
        # Create a record entry for this step
        record_entry = {
            'env_id': env_id,
            'done': env_feedback.done,
            'reward': env_feedback.reward,
            'info': copy.deepcopy(env_feedback.info) if env_feedback.info is not None else None
        }
        
        # Process observations if provided
        env_observation = env_feedback.env_observation
        if env_observation is not None:
            template = env_observation.observation_template
            mm_observation = env_observation.multi_modal_observation
            
            # Get all placeholders from mm_observation keys
            placeholders = list(mm_observation.keys())
            position_for_each_placeholder = defaultdict(list)
            placeholder_counter = 0

            if placeholders and self.processor is not None:
                record_entry["image_data"] = []
            
            # for each placeholder, find all occurance in template (find its position)
            for placeholder in placeholders:
                position_for_each_placeholder[placeholder] = [m.start() for m in re.finditer(placeholder, template)]
                placeholder_counter += len(position_for_each_placeholder[placeholder])
            

            while placeholder_counter > 0:
                # choose the first placeholder in all positions
                first_index, first_index_placeholder = None, None
                for placeholder, positions in position_for_each_placeholder.items():
                    if positions:
                        if first_index is None:
                            first_index = positions[0]
                            first_index_placeholder = placeholder
                        elif positions[0] < first_index:
                            first_index = positions[0]
                            first_index_placeholder = placeholder
                # replace the first placeholder with <image>
                template = template[:first_index] + '<image>' + template[first_index+len(first_index_placeholder):]
                placeholder_counter -= 1
                position_for_each_placeholder[first_index_placeholder].pop(0)
                if "image_data" in record_entry:
                    record_entry["image_data"].append(process_image(mm_observation[first_index_placeholder]))
            
            record_entry["text_template"] = template
            
        # if env_observation is not None:
        #     template = env_observation.observation_template
        #     mm_observation = env_observation.multi_modal_observation

        #     mllm_keys = re.findall(r'<image([^>]+)>', template)
        #     # Process multimodal inputs if present in observation
        #     if mllm_keys and self.processor is not None:
        #         record_entry["image_data"] = [process_image(mm_observation[key]) for key in mllm_keys]
        #         #record_entry["image_inputs"] = self.processor.image_processor(record_entry["image_data"], return_tensors='pt')
        #         record_entry["text_template"] = re.sub(r'<image([^>]+)>', '<image>', template)

        self.recorder[env_id].append(record_entry)

    def __getitem__(
            self, 
            recording, 
            step, 
            window_size,
            compute_loss_mask=False,
            final=False,
        ):
        """
        Given a recording, generate the input for MLLM
        
        Args:
            recording: List of dictionaries containing recorded environment interactions
            step: Current step to generate input for
            window_size: Number of past steps to include in the context
        
        Returns:
            Dictionary containing properly formatted inputs for the MLLM
        """
        assert step >= 0
        
        start_step = max(0, step - window_size)
        end_step = step
        history = recording[start_step: end_step + 1]
        chat = []
        
  
        env_id = history[0]['env_id']
        chat.append({"role": "system", "content": self.envs[env_id].get_task_instruction()})
        
   
        if history:
            first_record = history[0]
            if 'text_template' in first_record:
                chat.append({"role": "user", "content": first_record['text_template']})
    
        for i, record in enumerate(history[1:], 1):
            chat.append({"role": "user", "content": record['text_template']})
            
    
            if i < len(history) - 1 or final:
                assert 'llm_raw_response' in record['info']
                llm_raw_response = record['info']['llm_raw_response']
                # filtering llm generated <image> tokens
                llm_raw_response = re.sub(r'<image>', '', llm_raw_response)
                if compute_loss_mask:
                    # filtering special tokens for llm_raw_response, then adding them to the beginning and end of the response
                    sptk_b = self.config.sptk_for_loss_mask[0]
                    sptk_e = self.config.sptk_for_loss_mask[1]
                    llm_raw_response = re.sub(sptk_e, '', llm_raw_response)
                    llm_raw_response = re.sub(sptk_b, '', llm_raw_response)
                    llm_raw_response = sptk_b + llm_raw_response + sptk_e
                chat.append({"role": "assistant", "content": record['info']['llm_raw_response']})
        chat.append({"role": "assistant", "content": "<think>"})
        
        # TODO for vllm no embedding
        image_data=[image for image in record["image_data"] for record in history if 'multi_modal_inputs' in record]
        image_inputs = self.processor.image_processor(image_data, return_tensors='pt')
        has_images = len(image_data) > 0
        
        
        # modified from verl.utils.dataset.rl_dataset.py
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        row_dict = {}
        if has_images:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_data'] = {'image': image_data}
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if compute_loss_mask:
            input_ids,attention_mask,loss_mask=self.compute_loss_mask(input_ids,attention_mask)
            
        if self.image_key in row_dict:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['loss_mask']=loss_mask[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
       
    def gen_batch(self, step, window_size):
        
        batch=[]
        self.batch_idx_to_env_id = {}
        batch_idx = 0
        for env_id in self.envs.keys():
            if self.env_states[env_id]['done']:
                continue
            batch.append(self.__getitem__(self.recorder[env_id], step, window_size))
            self.batch_idx_to_env_id[batch_idx] = env_id
            batch_idx += 1
        if len(batch)%self.config.n_gpu_per_node!=0:
            # Pad the batch to make it divisible by n_gpu_per_node
            while len(batch)%self.config.n_gpu_per_node!=0:
                batch.append(batch[-1])
        return collate_fn(batch)
    
    def compute_loss_mask(self,input_ids,attention_mask):
        # There will be different stratgy to handel special tokens in the list
        # 1. remove them, in this case we need to fill the hole by adding pad in the right and shift the sequence left
        # 2. keep them, attention mask will be 0 for them
        # 3. Replace them with pad token
        
        # let's use 2nd for now
        
        sptk_b= self.tokenizer.convert_tokens_to_ids(self.config.sptk_for_loss_mask[0])
        sptk_e= self.tokenizer.convert_tokens_to_ids(self.config.sptk_for_loss_mask[1])
        loss_mask = torch.ones_like(input_ids) # 0 for no loss, 1 for loss
        sptk_s_indices = (input_ids[0] == sptk_b).nonzero().flatten()
        sptk_e_indices = (input_ids[0] == sptk_e).nonzero().flatten()
        attention_mask[0][sptk_s_indices] = 0
        attention_mask[0][sptk_e_indices] = 0
        for s,e in zip(sptk_s_indices,sptk_e_indices):
            loss_mask[0][s+1:e]=1
        return input_ids,attention_mask,loss_mask
    
    def rollout_loop(self):
        """
        Step the environment and record the results
        
        Returns:
            Dictionary containing the results of the step
        """
        for step in self.config.max_turns-1:
            input_batch = self.gen_batch(step, self.config.window_size)
            output_batch=self.actor_rollout_wg(input_batch)
            responses_str = self.tokenizer.batch_decode(
                output_batch.batch['responses'], 
                skip_special_tokens=True
            )
            
            for batch_idx, env_id in self.batch_idx_to_env_id.items(): # TODO whether multiple actions in one rollout are considered here
                env_feedback = self.envs[env_id].step(responses_str[batch_idx])
                self.env_states[env_id]['step'] += 1
                self.env_states[env_id]['done'] = env_feedback.done
                self.record(env_id, env_feedback)
        
        
        
    def get_final_trajectory(self) -> DataProto:
        """
        Get the final trajectory of all environments
        
        Returns:
            Dictionary containing the final trajectory of all environments
        """
        batch_list = []
        for env_id in self.envs.keys():
            row_dict = self.__getitem__(self.recorder[env_id], self.env_states[env_id]['step'], self.config.window_size,loss_mask=True,final=True)
            row_dict['reward_model'] = {"style": "given", "ground_truth": {"reward": self.envs[env_id].get_traj_reward()}}
            batch_list.append(row_dict)
        batch_dict = collate_fn(batch_list)
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        return batch