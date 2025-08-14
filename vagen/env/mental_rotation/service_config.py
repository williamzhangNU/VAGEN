from dataclasses import dataclass, field
from vagen.env.base.base_service_config import BaseServiceConfig
 
@dataclass
class MentalRotationServiceConfig(BaseServiceConfig):
    devices: list = field(default_factory=lambda: [0])  # GPU indices, empty list [] means CPU
    use_state_reward: bool = False
    max_workers: int = 4
    timeout: int = 120