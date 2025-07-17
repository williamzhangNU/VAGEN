from typing import Dict, List, Tuple, Optional, Any
from vagen.env.base.base_service import BaseService
from vagen.server.serial import serialize_observation

from .env import SpatialGym
from .env_config import SpatialGymConfig
from .service_config import SpatialGymServiceConfig

class SpatialGymService(BaseService):
    
    def __init__(self, config: SpatialGymServiceConfig):
        self.environments = {}
        self.env_configs = {}
        self.config = config
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        for env_id, config in ids2configs.items():
            env_config_dict = config.get('env_config', {})
            env_config = SpatialGymConfig(**env_config_dict)
            env = SpatialGym(env_config)
            self.environments[env_id] = env
            self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        results = {}
        
        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, info)
        
        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        results = {}
        
        for env_id, action in ids2actions.items():
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, reward, done, info)
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        results = {}
        
        for env_id in env_ids:
            env = self.environments[env_id]
            # SpatialGym doesn't have compute_reward method, use current reward
            results[env_id] = 0.0
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        results = {}
        
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.system_prompt()
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        for env_id in env_ids:
            if env_id in self.environments:
                env = self.environments[env_id]
                if hasattr(env, 'close'):
                    env.close()
        
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None) 