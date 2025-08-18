from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
from typing import Optional, List
import random

# Temporary base config to avoid import issues
@dataclass
class BaseEnvConfig(ABC):
    format_reward: float = 0.5
    image_placeholder: str = "<image>"
    special_token_list: Optional[List[str]] = field(default_factory=lambda: ["<think>", "</think>", "<answer>", "</answer>"])
    action_sep: str = ","

    @abstractmethod
    def config_id(self) -> str:
        pass

    def get(self, key, default=None):
        return getattr(self, key, default)

    def generate_seeds(self, size, seed=0, n_candidate: int = 20000) -> list:
        random.seed(seed)
        seeds = random.sample(range(0, n_candidate+size), size)
        return seeds

@dataclass
class RearrangementEnvConfig(BaseEnvConfig):
    env_name: str = "rearrangement"
    resolution: int = 255
    eval_set: str = 'base'
    down_sample_ratio: float = 1.0
    fov: int = 100
    multiview: bool = False
    render_mode: str= 'vision'
    max_actions_per_step: int = 5
    max_action_penalty: float = -0.1
    format_reward: float = 0.5
    gpu_device: int = 0
    prompt_format: str = "free_think" 
    # for rearrangement phases
    success_threshold: float = 0.5
    step_length: float = 0.5

    # state reward configs (optional)
    max_objects_in_state: int = 10
    use_state_reward: bool = False
    grounding_reward_weight: float = 0.5
    worldmodeling_reward_weight: float = 0.5

    def config_id(self) -> str:
        id_fields = ["eval_set","render_mode", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"RearrangementEnvConfig({id_str})"

