# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict,List,Any
from copy import deepcopy
from collections import defaultdict
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from vagen.rollout.qwen_rollout.rollout_manager import QwenVLRolloutManager
from vagen.rollout.qwen_rollout.rollout_manager_service import QwenVLRolloutManagerService
WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    MASKED_GAE = 'masked_gae'
    BI_LEVEL_GAE = 'bi_level_gae'
    TURN_WISE_GAE = 'turn_wise_gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'
    MULTI_TURN_GRPO = 'multi_turn_grpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,high_level_gamma=1.0):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                    values=values,
                                                                    eos_mask=response_mask,
                                                                    gamma=gamma,
                                                                    lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.MASKED_GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        gae_mask = data.batch['gae_mask'][:, -response_length:]
        advantages, returns =core_algos.compute_gae_advantage_return_with_loss_mask(token_level_rewards=token_level_rewards,
                                                                values=values,
                                                                loss_mask=gae_mask,
                                                                gamma=gamma,
                                                                lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.BI_LEVEL_GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        #assert "multi_turn_token_level_rewards" in data.batch.keys()
        # assert "loss_mask" in data.batch.keys()
        # loss_mask = data.batch['loss_mask'][:, -response_length:]
        if "loss_mask" in data.batch.keys():
            loss_mask = data.batch['loss_mask'][:, -response_length:]
        else:
            loss_mask=data.batch['attention_mask'][:, -response_length:]
        advantages, returns = core_algos.compute_bi_level_gae_advantage_return(token_level_rewards=data.batch['token_level_rewards'],
                                                                        values=values,
                                                                        loss_mask=loss_mask,
                                                                        gamma=gamma,
                                                                        lam=lam,
                                                                        high_level_gamma=high_level_gamma,
                                                                        reward_mask=data.batch['end_of_response_position_mask'][:, -response_length:])
        
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.TURN_WISE_GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        #assert "multi_turn_token_level_rewards" in data.batch.keys()
        # assert "loss_mask" in data.batch.keys()
        # loss_mask = data.batch['loss_mask'][:, -response_length:]
        if "loss_mask" in data.batch.keys():
            loss_mask = data.batch['loss_mask'][:, -response_length:]
        else:
            loss_mask=data.batch['attention_mask'][:, -response_length:]
        advantages, returns = core_algos.compute_turn_wise_gae_advantage_return(token_level_rewards=data.batch['token_level_rewards'],
                                                                        values=values,
                                                                        loss_mask=loss_mask,
                                                                        reward_mask=data.batch['end_of_response_position_mask'][:, -response_length:],
                                                                        lam=lam,
                                                                        high_level_gamma=high_level_gamma,
                                                                        )
        
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        
        
        
        if "loss_mask" in data.batch.keys():
            loss_mask = data.batch['loss_mask'][:, -response_length:]
            
            # valid_token_level_rewards_positions = token_level_rewards[0].nonzero(as_tuple=True)[0]
            # valid_loss_positions = loss_mask[0].nonzero(as_tuple=True)[0]
            # print(f"[DEBUG]valid_token_level_rewards_positions={valid_token_level_rewards_positions}")
            # print(f"[DEBUG]valid_loss_positions={valid_loss_positions}")
            # seems here only need to replace eos_mask with loss_mask
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=loss_mask,
                                                                        index=index)
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                            eos_mask=response_mask,
                                                                            index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    # elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
    #     token_level_rewards = data.batch['token_level_rewards']
    #     responses = data.batch['responses']
    #     response_length = responses.size(-1)
    #     attention_mask = data.batch['attention_mask']
    #     response_mask = attention_mask[:, -response_length:]
    #     advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
    #         token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
    #     data.batch['advantages'] = advantages
    #     data.batch['returns'] = returns
    # elif adv_estimator == AdvantageEstimator.REMAX:
    #     token_level_rewards = data.batch['token_level_rewards']
    #     index = data.non_tensor_batch['uid']
    #     responses = data.batch['responses']
    #     response_length = responses.size(-1)
    #     attention_mask = data.batch['attention_mask']
    #     response_mask = attention_mask[:, -response_length:]

    #     reward_baselines = data.batch['reward_baselines']

    #     advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
    #                                                                      reward_baselines=reward_baselines,
    #                                                                      eos_mask=response_mask)

    #     data.batch['advantages'] = advantages
    #     data.batch['returns'] = returns
    # elif adv_estimator == AdvantageEstimator.RLOO:
    #     token_level_rewards = data.batch['token_level_rewards']
    #     index = data.non_tensor_batch['uid']
    #     responses = data.batch['responses']
    #     response_length = responses.size(-1)
    #     attention_mask = data.batch['attention_mask']
    #     response_mask = attention_mask[:, -response_length:]
    #     advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
    #                                                                     eos_mask=response_mask,
    #                                                                     index=index)
    #     data.batch['advantages'] = advantages
    #     data.batch['returns'] = returns
    
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    if "loss_mask" in batch.batch.keys():
        # end_of_response_position_mask=batch.batch["end_of_response_position_mask"]
        response_length = batch.batch['loss_mask'].sum(-1).float()
        prompt_length = (batch.batch['attention_mask'].sum(-1)-batch.batch['loss_mask'].sum(-1)).float()
        response_part_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['loss_mask'][:, -response_part_length:]
    else:
        response_length = batch.batch['responses'].shape[-1]

        prompt_mask = batch.batch['attention_mask'][:, :-response_length]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        prompt_length = prompt_mask.sum(-1).float()
        response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator in [AdvantageEstimator.GAE, AdvantageEstimator.BI_LEVEL_GAE,
                                                   AdvantageEstimator.MASKED_GAE,AdvantageEstimator.TURN_WISE_GAE]:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.MULTI_TURN_GRPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
        self.test_rollout_config=None
        self.test_rollout_manager=None
        

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus
            if config.algorithm.adv_estimator == AdvantageEstimator.TURN_WISE_GAE:
                assert config.critic.get('use_reward_mask', False), \
                    "TURN_WISE_GAE needs reward mask"

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_batch_size is None:
            val_batch_size=len(self.val_dataset)
        else:
            val_batch_size=self.config.data.val_batch_size
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=val_batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        # assert len(
        #     self.val_dataloader
        # ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves." # for agent training we still use val batch size

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, log_rst: List[Dict[str, Any]]):
        """Log a table of validation samples with multiple images per sample to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print('WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np
        inputs=[]
        outputs=[]
        scores=[]
        images=[]
        question_ids=[]

        for item in log_rst:
            inputs.append(item['config_id'])
            outputs.append(item['output_str'])
            scores.append(item['metrics']['score'])
            images.append(item['image_data'])
            if 'question_id' in item:
                question_ids.append(item['question_id'])
        
        
        
        # Handle the case where images might not be provided
        if images is None:
            samples = list(zip(inputs, outputs, scores)) if len(question_ids)==0 else list(zip(inputs, outputs, scores, question_ids))
            has_images = False
            max_images_per_sample = 0
        else:
            # Here, images is expected to be a list of lists, where each inner list contains images for one sample
            samples = list(zip(inputs, outputs, scores, images)) if len(question_ids)==0 else list(zip(inputs, outputs, scores, question_ids, images))
            has_images = True
            # Find maximum number of images in any sample
            max_images_per_sample = max(len(img_list) if isinstance(img_list, (list, tuple)) else 1 for img_list in images)

        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState()
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        if has_images:
            columns = ["step"]
            for i in range(len(samples)):
                if len(question_ids) == 0:
                    columns.extend([f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"])
                else:
                    columns.extend([f"input_{i+1}", f"output_{i+1}", f"score_{i+1}", f"question_id_{i+1}"])
                columns.extend([f"image_{i+1}_{j+1}" for j in range(max_images_per_sample)])
        else:
            if len(question_ids) == 0:
                columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])
            else:
                columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}", f"question_id_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        
        for sample in samples:
            if has_images:
                if len(question_ids)==0:
                    input_text, output_text, score, sample_images = sample
                    row_data.extend([input_text, output_text, score])
                else:
                    input_text, output_text, score, question_id, sample_images = sample
                    row_data.extend([input_text, output_text, score, question_id])
                
                # Handle if sample_images is a single image or list of images
                if not isinstance(sample_images, (list, tuple)):
                    sample_images = [sample_images]
                    
                # Convert each image to wandb.Image
                wandb_images = []
                for img in sample_images:
                    if not isinstance(img, wandb.Image):
                        img = wandb.Image(img)
                    wandb_images.append(img)
                    
                # Pad with None if there are fewer images than max_images_per_sample
                wandb_images.extend([None] * (max_images_per_sample - len(wandb_images)))
                row_data.extend(wandb_images)
            else:
                row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table
    
    def _validate(self, map_to_id=False):
        print(f"[DEBUG] validation at global step {self.global_steps} begins")
        # Lists to collect samples for the table
    
        if self.test_rollout_manager==None:
            if self.config.rollout_manager.get("use_service",False):
                self.test_rollout_manager =QwenVLRolloutManagerService(
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=self.config.rollout_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    split="val",
                )
            else:
                self.test_rollout_manager =QwenVLRolloutManager(
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=self.config.rollout_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor, 
                )
        
        validation_rst=[]
        env_id_2_question_id = {} # for CrossViewEnv
        
        
        for batch_dict in self.val_dataloader:
            
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            # pop these keys so it will not cause error when rollout
            if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            env_configs = [
                    batch.non_tensor_batch['extra_info'][i]
                    for i in range(len(batch))
                ]
            
            initial_obs, initial_info = self.test_rollout_manager.reset(env_configs)
            self.test_rollout_manager.rollout_loop()
            micro_validation_rst = self.test_rollout_manager.recording_to_log() # data source == inputs in our current setting, outputs=whole trjecotry
            
            if map_to_id:
                for env_id, info in initial_info.items():
                    env_id_2_question_id[env_id] = info["question_id"]
                for item in micro_validation_rst:
                    env_id = item.get("env_id")
                    if env_id in env_id_2_question_id:
                        item["question_id"] = env_id_2_question_id[env_id]

            validation_rst.extend(micro_validation_rst)
        
        self._maybe_log_val_generations_to_wandb(validation_rst)
        metric_dict = self.log_rst_to_metrics_dict(validation_rst,mode='val')
        print(f"[DEBUG] validation at global step {self.global_steps} ends")
        return metric_dict

    def log_rst_to_metrics_dict(self,rst,mode='train'):
        metric_dict = {}
        
        
        metrics_by_config_id = defaultdict(dict)  # a dict of dict of list
        
        for item in rst:
            for k,v in item["metrics"].items():
                if k not in metrics_by_config_id[item["config_id"]]:
                    metrics_by_config_id[item["config_id"]][k] = []
                metrics_by_config_id[item["config_id"]][k].append(v)
        
        
        for config_id, metrics in metrics_by_config_id.items():
            for k,v in metrics.items():
                metric_dict[f'{mode}/{k}/{config_id}'] = np.mean(v)
        
        return metric_dict
            

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _process_in_mini_batches(self,batch, rollout_manager, mini_batch_size):
        """
        Process the batch in mini-batches.
        
        Args:
            batch: DataProto containing the data
            rollout_manager: Manager for rollout operations
            mini_batch_size: Size of each mini-batch to process
        
        Returns:
            Tuple of (final_combined_batch_output, combined_rst)
        """
        batch_size = len(batch)
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size  # Ceiling division
        
        all_final_gen_batch_outputs = []
        all_rst = []
        
        for i in range(num_mini_batches):
            start_idx = i * mini_batch_size
            end_idx = min((i + 1) * mini_batch_size, batch_size)
            actual_mini_batch_size = end_idx - start_idx
            print(f"Processing mini-batch {i+1}/{num_mini_batches}, size: {actual_mini_batch_size}")
            
            # Extract env_configs for this mini-batch
            mini_batch_env_configs = [
                batch.non_tensor_batch['extra_info'][j]
                for j in range(start_idx, end_idx)
            ]
            
            # Reset and process this mini-batch
            rollout_manager.reset(mini_batch_env_configs)
            rollout_manager.rollout_loop()
            mini_batch_output = rollout_manager.generate_batch_for_update()
            mini_batch_rst = rollout_manager.recording_to_log()
            
            # Store results
            all_final_gen_batch_outputs.append(mini_batch_output)
            all_rst.extend(mini_batch_rst)  # Extend the list since rst is a list
        
       
            combined_output = DataProto.concat(all_final_gen_batch_outputs)
       
        
        return combined_output, all_rst
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate(map_to_id=self.config.trainer.get('val_only', False))
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            score_key = next((k for k in val_metrics.keys() if 'score' in k and k.startswith('val/')), 'score')
            best_score = val_metrics[score_key]
            print(f'best_score: {best_score}')
            if self.config.trainer.get('val_only', False):
                return
        else:
            best_score = float('-inf')

        # we start from step 1
        self.global_steps += 1

        if self.config.rollout_manager.get("use_service",False):
            rollout_manager = QwenVLRolloutManagerService(
                actor_rollout_wg=self.actor_rollout_wg,
                config=self.config.rollout_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                split="train",
            )
        else:
            rollout_manager = QwenVLRolloutManager(
                actor_rollout_wg=self.actor_rollout_wg,
                config=self.config.rollout_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'global_steps: {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # pop these keys so it will not cause error when rollout
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                
                # We control vanilla-grpo sampling param here (start from init state s0, sample n_trajectory)
                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],dtype=object)
                batch = batch.repeat(repeat_times=self.config.rollout_manager.n_trajectory, interleave=True)
                
                    
                

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                       
                        mini_batch_size=self.config.rollout_manager.get('mini_batch_size',len(batch))
                        final_gen_batch_output, rst=self._process_in_mini_batches(batch, rollout_manager, mini_batch_size) 
                        train_metrics=self.log_rst_to_metrics_dict(rst=rst,mode='train')
                        metrics.update(train_metrics)
                    print(f"[DEBUG] step {self.global_steps} rollout ends")
                    batch = batch.union(final_gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        
                        if self.config.rollout_manager.use_multi_turn_reward:
                        #VAGEN: TODO: use a new reward_fn to combine the results from reward model and rule-based multi-turn token reward.
                            response_len=batch.batch['responses'].shape[1]
                            batch.batch['token_level_scores'] = batch.batch['multi_turn_token_level_rewards'][:,-response_len:]
                        else:
                            # we combine with rule-based rm
                            reward_tensor = self.reward_fn(batch)
                            batch.batch['token_level_scores'] = reward_tensor 
                

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  high_level_gamma=self.config.algorithm.high_level_gamma,)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                        # save the best model
                        print(f'val_metrics: {val_metrics}')
                        score_key = next((k for k in val_metrics.keys() if 'score' in k and k.startswith('val/')), 'score')
                        if val_metrics[score_key] > best_score:
                            best_score = val_metrics[score_key]
                            print(f'best_score: {best_score} at global_steps: {self.global_steps}')
                            if self.config.trainer.save_freq > 0:
                                with _timer('save_checkpoint', timing_raw):
                                    self._save_checkpoint()

                    # if self.config.trainer.save_freq > 0 and \
                    #         self.global_steps % self.config.trainer.save_freq == 0:
                    #     with _timer('save_checkpoint', timing_raw):
                    #         self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    # if self.config.trainer.save_freq > 0 and \
                    #         (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                    #     with _timer('save_checkpoint', timing_raw):
                    #         self._save_checkpoint()
                    return
