from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base.base_service import BaseService
from vagen.env.rearrangement.env import RearrangementEnv
from vagen.env.rearrangement.env_config import RearrangementEnvConfig
from vagen.server.serial import serialize_observation
from .service_config import RearrangementServiceConfig
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper

class RearrangementService(BaseService):
    """
    Service class for Rearrangement environments based on AI2-THOR.
    Implements batch operations with parallel processing for efficiency.
    """
    
    def __init__(self, config: RearrangementServiceConfig):
        """
        Initialize the RearrangementService.
        
        Args:
            config: Service configuration including max workers and devices
        """
        self.max_workers = config.max_workers
        self.device_status = {device_id: set() for device_id in config.devices}
        self.environments = {}
        self.env_configs = {}
        self.config = config
        print(f"[DEBUG] {self.config}")
    
    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        """
        Create multiple Rearrangement environments in parallel.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
                Each config should contain:
                - env_name: Should be "rearrangement"
                - env_config: Rearrangement specific configuration
        """
        # Define worker function
        def create_single_env(env_id, config):
            # Verify environment type
            env_name = config.get('env_name', 'rearrangement')
            if env_name != 'rearrangement':
                return env_id, None, f"Expected environment type 'rearrangement', got '{env_name}'"
            
            env_config_dict = config['env_config']
            env_config = RearrangementEnvConfig(**env_config_dict)
            env = RearrangementEnv(env_config)
            
            return env_id, env, None
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_env_id = {
                executor.submit(create_single_env, env_id, config): env_id 
                for env_id, config in ids2configs.items()
            }
            
            for future in as_completed(future_to_env_id):
                env_id = future_to_env_id[future]
                try:
                    env_id, env, error = future.result()
                    if error:
                        print(f"Failed to create environment {env_id}: {error}")
                    else:
                        self.environments[env_id] = env
                        self.env_configs[env_id] = ids2configs[env_id]['env_config']
                        print(f"Successfully created environment {env_id}")
                except Exception as exc:
                    print(f"Environment {env_id} generated an exception: {exc}")

    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Reset multiple environments in parallel.

        Args:
            ids2seeds: A dictionary where each key is an environment ID and the corresponding
                      value is the seed for that environment

        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, info)
        """
        results = {}

        def reset_single_env(env_id, seed):
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
            return env_id, (serialized_observation, info), None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_env_id = {
                executor.submit(reset_single_env, env_id, seed): env_id
                for env_id, seed in ids2seeds.items()
            }

            for future in as_completed(future_to_env_id):
                env_id = future_to_env_id[future]
                try:
                    env_id, result, error = future.result()
                    if error:
                        results[env_id] = ({}, {"error": error})
                    else:
                        results[env_id] = result
                except Exception as exc:
                    results[env_id] = ({}, {"error": str(exc)})

        return results

    def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Execute actions in multiple environments in parallel.

        Args:
            ids2actions: A dictionary where each key is an environment ID and the corresponding
                        value is the action to execute in that environment

        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, reward, done, info)
        """
        def step_single_env(env_id, action):
            if env_id not in self.environments:
                return env_id, None, f"Environment {env_id} not found"

            try:
                result = self.environments[env_id].step(action)
                # Extract components from result
                observation = result.get('observation', {})
                reward = result.get('reward', 0.0)
                done = result.get('done', False)
                info = {k: v for k, v in result.items() if k not in ['observation', 'reward', 'done']}

                # Serialize observation for transmission
                serialized_observation = serialize_observation(observation)
                return env_id, (serialized_observation, reward, done, info), None
            except Exception as e:
                return env_id, None, str(e)

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_env_id = {
                executor.submit(step_single_env, env_id, action): env_id
                for env_id, action in ids2actions.items()
            }

            for future in as_completed(future_to_env_id):
                env_id = future_to_env_id[future]
                try:
                    env_id, result, error = future.result()
                    if error:
                        results[env_id] = ({}, 0.0, True, {"error": error})
                    else:
                        results[env_id] = result
                except Exception as exc:
                    results[env_id] = ({}, 0.0, True, {"error": str(exc)})

        return results

    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        """
        Get system prompts for multiple environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to system prompts
        """
        results = {}
        for env_id in env_ids:
            if env_id in self.environments:
                try:
                    results[env_id] = self.environments[env_id].get_system_prompt()
                except Exception as e:
                    results[env_id] = f"Error getting system prompt: {str(e)}"
            else:
                results[env_id] = f"Environment {env_id} not found"
        
        return results

    def get_action_spaces_batch(self, env_ids: List[str]) -> Dict[str, List[str]]:
        """
        Get action spaces for multiple environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to action space descriptions
        """
        results = {}
        for env_id in env_ids:
            if env_id in self.environments:
                try:
                    results[env_id] = self.environments[env_id].get_action_space()
                except Exception as e:
                    results[env_id] = [f"Error getting action space: {str(e)}"]
            else:
                results[env_id] = [f"Environment {env_id} not found"]
        
        return results

    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple Rearrangement environments and clean up resources in parallel.

        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all environments
        if env_ids is None:
            env_ids = list(self.environments.keys())

        # Define worker function
        def close_single_env(env_id):
            env = self.environments[env_id]
            env.close()
            return None


        # Use ThreadPoolExecutor for parallel closing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all closing tasks
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]

            # Wait for all tasks to complete
            for future in as_completed(futures):
                error = future.result()
                if error:
                    print(f"Error closing environment: {error}")

        # Remove closed environments from dictionaries
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)

    def get_environment_info(self, env_id: str) -> Dict[str, Any]:
        """
        Get information about a specific environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Dictionary containing environment information
        """
        if env_id not in self.environments:
            return {'error': f'Environment {env_id} not found'}
        
        env = self.environments[env_id]
        return {
            'phase': env.current_phase,
            'step_count': env.step_count,
            'max_steps': env.max_steps,
            'valid_actions': env.get_valid_actions(),
            'config': self.env_configs.get(env_id, {})
        }

    def list_environments(self) -> List[str]:
        """
        List all active environment IDs.

        Returns:
            List of environment IDs
        """
        return list(self.environments.keys())

    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        """
        Compute the total reward for multiple Rearrangement environments in parallel.

        Args:
            env_ids: A list of environment IDs

        Returns:
            A dictionary mapping each environment ID to its computed total reward
        """
        results = {}
        for env_id in env_ids:
            if env_id in self.environments:
                try:
                    # For rearrangement, reward is based on success rate
                    env = self.environments[env_id]
                    if hasattr(env, '_calculate_success_rate'):
                        success_rate = env._calculate_success_rate()
                        results[env_id] = success_rate * 10.0  # Scale reward
                    else:
                        results[env_id] = 0.0
                except Exception as e:
                    print(f"Error computing reward for environment {env_id}: {e}")
                    results[env_id] = 0.0
            else:
                results[env_id] = 0.0

        return results
