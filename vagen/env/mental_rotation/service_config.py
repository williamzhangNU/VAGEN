from dataclasses import dataclass
from vagen.env.base.base_service_config import BaseServiceConfig
 
@dataclass
class MentalRotationServiceConfig(BaseServiceConfig):
    use_state_reward: bool = False
    max_workers: int = 4
    timeout: int = 120