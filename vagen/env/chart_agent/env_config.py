from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields, field
from typing import Optional, List, Union

@dataclass
class ChartAgentEnvConfig(BaseEnvConfig):
    split: str = "train" # train, test
    image_path: str = "ChartAgent/images"
    image_size: tuple = (300, 300)
    return_cropped: bool = False  # Whether to return cropped regions
    highlight_boxes: bool = True  # Whether to highlight bounding box areas
    type: str = "chart_agent"  # Environment type identifier
    def config_id(self) -> str:
        return f"ChartAgentEnv"
    def generate_seeds(self, size, seed=0, n_candidate = 20000):
        return [i for i in range(size)]
            
            
            
if __name__ == "__main__":
    config = ChartAgentEnvConfig()
    print(config.config_id())