# vagen/mllm_agent/inference_rollout/inference_rollout_service.py

import os
import time
import json
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import PIL

from vagen.rollout.base_rollout import BaseRollout
from vagen.server.client import BatchEnvClient
from vagen.env import REGISTERED_ENV

class InferenceRolloutService(BaseRollout):
    """
    Implementation of BaseRollout for inference.
    Handles batch-wise environment interaction and model inference
    without training-specific components.
    """
    
    def __init__(self,
                 config: Dict,
                 model_interface,
                 base_url: str = "http://localhost:5000",
                 timeout: int = 600,
                 max_workers: int = 10,
                 split: str = "test",
                 debug: bool = False):
        """
        Initialize the inference rollout service.
        
        Args:
            config: Configuration dictionary
            model_interface: Model interface for generating responses
            base_url: URL of the environment service
            timeout: Timeout for HTTP requests in seconds
            max_workers: Maximum number of worker threads
            split: Data split name (test/val)
            debug: Enable debug logging
        """
        self.config = config
        self.model_interface = model_interface
        self.base_url = base_url
        self.timeout = timeout
        self.max_workers = max_workers
        self.split = split
        self.debug = debug
        
        # Debug saving setup
        self.debug_save_dir = config.get("debug_save_dir", None)
        
        # Initialize environment client
        self.env_client = BatchEnvClient(
            base_url=base_url,
            timeout=timeout,
            max_workers=max_workers
        )
        
        # Environment tracking
        self.envs = {}  # Maps env_id to config instances
        self.env_states = {}  # Maps env_id to environment state
        self.recordings = {}  # Maps env_id to recorded trajectory
        self.system_prompts = {}  # Maps env_id to system prompt
        
        # Max number of steps from config
        self.max_steps = config.get("max_steps", 10)
        
        # Show progress bar
        self.show_progress = config.get("show_progress", True)
        
        # Debug messages
        if self.debug:
            print(f"Initialized InferenceRolloutService with model: {model_interface.get_model_info()['name']}")
    
    def reset(self, env_configs: List[Dict]) -> Dict[str, Tuple[Dict, Dict]]:
        """
        Reset environments based on provided configurations.
        
        Args:
            env_configs: List of environment configurations
            
        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        # Clean up existing environments if any
        if self.envs:
            self.close()
        
        # Reset tracking structures
        self.envs = {}
        self.env_states = {}
        self.recordings = {}
        self.system_prompts = {}
        
        # Prepare environment configurations
        ids2configs = {}
        ids2seeds = {}
        
        for i, cfg in enumerate(env_configs):
            env_id = f"{self.split}_{i}"
            ids2configs[env_id] = cfg
            ids2seeds[env_id] = cfg.get("seed", 42)
            
            # Store configuration for reference
            self.envs[env_id] = REGISTERED_ENV[cfg["env_name"]]["config_cls"](**cfg["env_config"])
        
        if self.debug:
            print(f"Creating {len(env_configs)} environments...")
        
        # Create and reset environments
        self.env_client.create_environments_batch(ids2configs)
        reset_results = self.env_client.reset_batch(ids2seeds)
        
        # Get system prompts
        self.system_prompts = self.env_client.get_system_prompts_batch(list(self.envs.keys()))
        
        # Initialize recordings and state tracking
        for env_id, (obs, info) in reset_results.items():
            # Initialize recording with system prompt and first observation
            self.recordings[env_id] = [
                {"role": "system", "content": self.system_prompts[env_id]},
                {"role": "user", "content": obs["obs_str"]}
            ]
            
            # Track multi-modal data if present
            if "multi_modal_data" in obs:
                # Store multimodal data with the message
                self.recordings[env_id][-1]["multi_modal_data"] = obs["multi_modal_data"]
            
            # Initialize environment state
            self.env_states[env_id] = {
                "step": 0,
                "done": False,
                "last_obs": obs,
                "last_info": info,
                "rewards": [],
                "metrics": {
                    "turn_metrics": defaultdict(list),
                    "traj_metrics": defaultdict(list)  # Changed to defaultdict to accumulate all metrics
                }
            }
        
        if self.debug:
            print(f"Reset {len(reset_results)} environments")
        
        return reset_results
    
    def run(self, max_steps: int = None) -> None:
        """
        Run inference on all environments until completion or max steps.
        
        Args:
            max_steps: Maximum number of steps to run for each environment,
                      overrides config value if provided
        """
        if max_steps is None:
            max_steps = self.max_steps
        
        # Track active environments
        active_envs = set(self.envs.keys())
        
        # Progress bar
        progress_iter = range(max_steps)
        if self.show_progress:
            progress_iter = tqdm(progress_iter, desc="Inference steps")
        
        # Main inference loop
        for step in progress_iter:
            if not active_envs:
                if self.debug:
                    print(f"All environments completed after {step} steps")
                break
            
            # Collect prompts for active environments
            env_messages = {}
            
            for env_id in active_envs:
                # Get conversation history for this environment
                env_messages[env_id] = self.recordings[env_id]

            print(f"[DEBUG] env_messages: {env_messages}")
            
            # Generate responses using model interface
            start_time = time.time()
            responses = self._generate_batch_responses(env_messages)
            gen_time = time.time() - start_time
            
            # Step environments with responses
            next_active_envs = set()
            
            # Group responses for batch step
            ids2actions = {env_id: response for env_id, response in responses.items()}
            
            # Step environments using service
            step_results = self.env_client.step_batch(ids2actions)
            
            for env_id, (obs, reward, done, info) in step_results.items():
                # Update state
                self.env_states[env_id]["step"] += 1
                self.env_states[env_id]["done"] = done
                self.env_states[env_id]["last_obs"] = obs
                self.env_states[env_id]["last_info"] = info
                self.env_states[env_id]["rewards"].append(reward)
                
                # Store llm_raw_response in info
                info["llm_raw_response"] = responses[env_id]
                
                # Update metrics - properly handle all metrics from the environment
                if "metrics" in info:
                    # Update trajectory metrics
                    for k, v in info["metrics"].get("traj_metrics", {}).items():
                        if isinstance(v, list):
                            self.env_states[env_id]["metrics"]["traj_metrics"][k].extend(v)
                        else:
                            self.env_states[env_id]["metrics"]["traj_metrics"][k] = v
                    
                    # Update turn metrics (accumulate lists)
                    for k, v in info["metrics"].get("turn_metrics", {}).items():
                        if isinstance(v, list):
                            self.env_states[env_id]["metrics"]["turn_metrics"][k].extend(v)
                        else:
                            self.env_states[env_id]["metrics"]["turn_metrics"][k].append(v)
                
                # Add assistant response to recording
                self.recordings[env_id].append({
                    "role": "assistant",
                    "content": responses[env_id]
                })
                
                # Add user observation to recording if not done
                if not done:
                    user_message = {"role": "user", "content": obs["obs_str"]}
                    
                    # Track multi-modal data if present
                    if "multi_modal_data" in obs:
                        user_message["multi_modal_data"] = obs["multi_modal_data"]
                    
                    self.recordings[env_id].append(user_message)
                    next_active_envs.add(env_id)
            
            # Update active environments for next iteration
            active_envs = next_active_envs
            
            if self.debug or (step % 5 == 0 and self.show_progress):
                # Print progress stats every 5 steps
                completion_rate = (len(self.envs) - len(active_envs)) / len(self.envs) * 100
                avg_steps = sum(self.env_states[env_id]["step"] for env_id in self.env_states) / len(self.env_states)
                print(f"Step {step+1}: {completion_rate:.1f}% environments completed, {len(active_envs)} active, avg steps: {avg_steps:.1f}, gen time: {gen_time:.3f}s")
    
    def _generate_batch_responses(self, env_messages: Dict[str, List[Dict]]) -> Dict[str, str]:
        """
        Generate responses for multiple environments.
        
        This method now properly aligns with the model interface that expects
        message lists directly, and handles multimodal data according to the
        training rollout format.
        
        Args:
            env_messages: Dictionary mapping environment IDs to conversation histories
            
        Returns:
            Dictionary mapping environment IDs to generated responses
        """
        responses = {}
        env_ids = list(env_messages.keys())
        prompts = []
        
        # Collect all messages for batch generation
        for env_id in env_ids:
            messages = env_messages[env_id]
            prompts.append(messages)  # Model interface expects message lists directly
        
        # Generate responses for all prompts
        # The model interface will handle multimodal data extraction internally
        batch_results = self.model_interface.generate(prompts)
        
        # Extract responses
        for i, env_id in enumerate(env_ids):
            responses[env_id] = batch_results[i]["text"]
        
        return responses
    
    def recording_to_log(self) -> List[Dict]:
        """
        Format and return results in a format compatible with logging.
        
        Returns:
            List of dictionaries with:
            - env_id: Environment ID
            - config_id: Configuration ID
            - output_str: Formatted output string
            - image_data: List of images (if applicable)
            - metrics: Dictionary of metrics for this environment
        """
        results = []
        
        # Recursive function to convert all NumPy types to native Python types
        def convert_numpy_types(obj):
            import numpy as np
            
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(i) for i in obj)
            else:
                return obj
        
        # Get final rewards for all environments (no longer used)
        # reward_results = self.env_client.compute_reward_batch(list(self.envs.keys()))
        
        for env_id in self.envs:
            # Get environment configuration ID
            config_id = self.envs[env_id].config_id()
            
            # Get step count
            step_count = self.env_states[env_id]["step"]
            
            # Extract images from all messages
            image_data = []
            
            for message in self.recordings[env_id]:
                if "multi_modal_data" in message:
                    for key, values in message["multi_modal_data"].items():
                        if key == "<image>" or "image" in key.lower():
                            for value in values:
                                # Handle different image formats
                                if isinstance(value, PIL.Image.Image):
                                    image_data.append(value)
                                # Handle serialized images from the service
                                elif isinstance(value, dict) and "__pil_image__" in value:
                                    from vagen.server.serial import deserialize_pil_image
                                    image_data.append(deserialize_pil_image(value))
            
            # Format the conversation
            output_str = ""
            for msg in self.recordings[env_id]:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    output_str += f"System: {content}\n\n"
                elif role == "user":
                    output_str += f"User: {content}\n\n"
                elif role == "assistant":
                    output_str += f"Assistant: {content}\n\n"
            
            # Get completion status
            done = self.env_states[env_id]["done"]
            
            # ======= Key Modifications =======
            # Accumulate rewards from each step
            accumulated_rewards = sum(self.env_states[env_id]["rewards"])
            
            # Collect grounding and worldmodeling rewards
            all_turn_metrics = self.env_states[env_id]["metrics"].get("turn_metrics", {})
            grounding_rewards = all_turn_metrics.get("grounding_reward", [])
            worldmodeling_rewards = all_turn_metrics.get("worldmodeling_reward", [])
            
            # Calculate total rewards
            total_grounding_reward = sum(grounding_rewards) if isinstance(grounding_rewards, list) else grounding_rewards
            total_worldmodeling_reward = sum(worldmodeling_rewards) if isinstance(worldmodeling_rewards, list) else worldmodeling_rewards
            
            # Total score = step rewards + grounding rewards + worldmodeling rewards
            total_score = accumulated_rewards + total_grounding_reward + total_worldmodeling_reward
            
            print(f"[SCORE DEBUG] env_id={env_id}, steps={accumulated_rewards}, grounding={total_grounding_reward}, worldmodeling={total_worldmodeling_reward}, total={total_score}")
            # ======= End of Modifications =======
            
            # Collect metrics
            metrics = {
                "score": convert_numpy_types(total_score),  # Use our calculated total score
                "done": 1.0 if done else 0.0,
                "step": convert_numpy_types(step_count),
            }
            
            # Add turn metrics (handle both averaged and list data)
            turn_metrics = self.env_states[env_id]["metrics"]["turn_metrics"]
            for k, v in turn_metrics.items():
                if isinstance(v, list) and v:
                    # Average list values
                    metrics[f"avg_{k}"] = convert_numpy_types(sum(v) / len(v))
                    # Also keep the raw list
                    metrics[f"all_{k}"] = convert_numpy_types(v)
                else:
                    # Direct value
                    metrics[k] = convert_numpy_types(v)
            
            # Add trajectory metrics (keep all metrics)
            traj_metrics = self.env_states[env_id]["metrics"]["traj_metrics"]
            for k, v in traj_metrics.items():
                if isinstance(v, list) and v:
                    # For list values, take the last one (most recent state)
                    metrics[k] = convert_numpy_types(v[-1] if v else 0)
                    # Also keep the full history
                    metrics[f"history_{k}"] = convert_numpy_types(v)
                else:
                    metrics[k] = convert_numpy_types(v)
            
            # Add to results
            results.append({
                "env_id": env_id,
                "config_id": config_id,
                "output_str": output_str,
                "image_data": image_data,
                "metrics": convert_numpy_types(metrics),
            })
        
        # Save debug information if enabled
        if self.debug_save_dir:
            self._save_debug_info(results)
        
        return results
    
    def close(self) -> None:
        """
        Close all environments and clean up resources.
        """
        # Close environments through service
        if self.envs:
            self.env_client.close_batch(list(self.envs.keys()))
        
        # Clear tracking structures
        self.envs = {}
        self.env_states = {}
        self.recordings = {}
        self.system_prompts = {}
    
    def _save_debug_info(self, results: List[Dict]) -> None:
        """Save detailed debug information for each environment."""
        import os
        
        # Create debug directory
        os.makedirs(self.debug_save_dir, exist_ok=True)
        print(f"Saving debug information to {self.debug_save_dir}")
        
        for result in results:
            env_id = result["env_id"]
            
            # Clean recordings - remove image objects for JSON serialization
            clean_recordings = []
            for msg in self.recordings.get(env_id, []):
                clean_msg = msg.copy()
                if "multi_modal_data" in clean_msg:
                    # Replace image objects with placeholders
                    clean_multi_modal = {}
                    for key, values in clean_msg["multi_modal_data"].items():
                        if key == "<image>":
                            clean_multi_modal[key] = [f"<CURRENT_IMAGE_SAVED_SEPARATELY_{i}>" for i in range(len(values))]
                        elif key == "<target_image>":
                            clean_multi_modal[key] = [f"<TARGET_IMAGE_SAVED_SEPARATELY_{i}>" for i in range(len(values))]
                        elif "image" in key.lower():
                            clean_multi_modal[key] = [f"<IMAGE_SAVED_SEPARATELY_{i}>" for i in range(len(values))]
                        else:
                            clean_multi_modal[key] = values
                    clean_msg["multi_modal_data"] = clean_multi_modal
                clean_recordings.append(clean_msg)
            
            # Clean env_states - remove image objects
            env_states = self.env_states.get(env_id, {})
            clean_env_states = {}
            for key, value in env_states.items():
                if key == "last_obs" and isinstance(value, dict):
                    clean_obs = {}
                    for obs_key, obs_value in value.items():
                        if obs_key == "multi_modal_data" and isinstance(obs_value, dict):
                            clean_multi_modal = {}
                            for mm_key, mm_values in obs_value.items():
                                if mm_key == "<image>":
                                    clean_multi_modal[mm_key] = [f"<CURRENT_IMAGE_IN_ENV_STATES_{i}>" for i in range(len(mm_values)) if hasattr(mm_values, '__len__')]
                                elif mm_key == "<target_image>":
                                    clean_multi_modal[mm_key] = [f"<TARGET_IMAGE_IN_ENV_STATES_{i}>" for i in range(len(mm_values)) if hasattr(mm_values, '__len__')]
                                elif "image" in mm_key.lower():
                                    clean_multi_modal[mm_key] = [f"<IMAGE_IN_ENV_STATES_{i}>" for i in range(len(mm_values)) if hasattr(mm_values, '__len__')]
                                else:
                                    clean_multi_modal[mm_key] = mm_values
                            clean_obs[obs_key] = clean_multi_modal
                        else:
                            clean_obs[obs_key] = obs_value
                    clean_env_states[key] = clean_obs
                else:
                    clean_env_states[key] = value
            
            # Create debug data structure (without image objects)
            debug_data = {
                "env_id": env_id,
                "config_id": result["config_id"],
                "recordings": clean_recordings,
                "env_states": clean_env_states,
                "final_metrics": result["metrics"],
                "output_str": result["output_str"]
            }

            print(f"[DEBUG] debug_data: {debug_data}")

            
            # Save JSON file
            json_path = os.path.join(self.debug_save_dir, f"{env_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
            
            # Save images if present - with better naming to distinguish current vs target
            if result["image_data"]:
                print(f"[DEBUG] len(result['image_data']): {len(result['image_data'])}")
                image_dir = os.path.join(self.debug_save_dir, f"{env_id}_images")
                os.makedirs(image_dir, exist_ok=True)
                
                # Track image indices for current and target images
                current_img_idx = 0
                target_img_idx = 0
                
                # Go through recordings to identify image types
                for msg in self.recordings.get(env_id, []):
                    print(f"[DEBUG] msg: {msg}")
                    if "multi_modal_data" in msg:
                        for key, values in msg["multi_modal_data"].items():
                            if key == "<image>":
                                # Current object images
                                for value in values:
                                    if isinstance(value, PIL.Image.Image) and current_img_idx < len(result["image_data"]):
                                        img_path = os.path.join(image_dir, f"current_step_{current_img_idx}.png")
                                        result["image_data"][2 * current_img_idx].save(img_path)
                                        current_img_idx += 1
                            elif key == "<target_image>":
                                # Target object images
                                for value in values:
                                    if isinstance(value, PIL.Image.Image) and (current_img_idx + target_img_idx) < len(result["image_data"]):
                                        img_path = os.path.join(image_dir, f"target_step_{target_img_idx}.png")
                                        print(f"[DEBUG] len(result['image_data']): {len(result['image_data'])}, current_img_idx: {current_img_idx}, target_img_idx: {target_img_idx}")
                                        result["image_data"][2 * (current_img_idx-1) + 1].save(img_path)
                                        target_img_idx += 1
                
                # Save any remaining images with generic names (fallback)
                total_saved = current_img_idx + target_img_idx
                for i in range(total_saved, len(result["image_data"])):
                    img = result["image_data"][i]
                    if isinstance(img, PIL.Image.Image):
                        img_path = os.path.join(image_dir, f"unknown_step_{i}.png")
                        img.save(img_path)
        
        print(f"Debug information saved for {len(results)} environments")
        
        if self.debug:
            print("Closed all environments and cleaned up resources")
