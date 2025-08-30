from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class MentalRotation2DServiceConfig(BaseServiceConfig):
    service_name: str = "mental_rotation_2d"
    
    # Service-specific configurations can be added here
    # For now, inherit all from BaseServiceConfig
    
    def config_id(self) -> str:
        """Generate unique identifier for this service configuration."""
        return f"MentalRotation2DServiceConfig(service_name={self.service_name})"

if __name__ == "__main__":
    config = MentalRotation2DServiceConfig()
    print(config.config_id())
    print(config)
