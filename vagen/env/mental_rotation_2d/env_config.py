from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields
from typing import Optional, List, Union

@dataclass
class MentalRotation2DEnvConfig(BaseEnvConfig):
    env_name: str = "mental_rotation_2d"
    
    # Image placeholders
    image_placeholder: str = "<image>"
    
    # Rotation settings
    rotation_direction: str = "clockwise"  # "clockwise" or "counterclockwise"
    rotation_granularity: int = 15  # degrees (e.g., 15, 30, 45, 90)
    
    # Task settings
    task_name: str = "mental_rotation_2d_d30"  # Name of the JSON file in dataset folder
    
    # Environment behavior
    max_actions_per_step: int = 1  # Usually 1 for rotation tasks
    max_steps: int = 10  # Maximum steps before episode ends
    
    # Rendering settings - only vision mode supported
    image_size: tuple = (224, 224)  # Size for rendered images
    
    # Prompt format
    prompt_format: str = "free_think"  # Format for LLM prompts
    
    # Reward settings
    success_reward: float = 10.0  # Reward for reaching target orientation
    step_penalty: float = -0.1  # Small penalty per step to encourage efficiency
    angle_tolerance: float = 5.0  # Tolerance in degrees for considering success
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        id_fields = [
            "rotation_direction", "rotation_granularity", "task_name", 
            "max_actions_per_step", "max_steps", "prompt_format"
        ]
        id_str = ",".join([
            f"{field.name}={getattr(self, field.name)}" 
            for field in fields(self) 
            if field.name in id_fields
        ])
        return f"MentalRotation2DEnvConfig({id_str})"

if __name__ == "__main__":
    config = MentalRotation2DEnvConfig()
    print(config.config_id())
    print(config)
