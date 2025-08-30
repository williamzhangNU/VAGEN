from vagen.env.base.base_service import BaseService
from .env import MentalRotation2DEnv
from .env_config import MentalRotation2DEnvConfig
from .service_config import MentalRotation2DServiceConfig

class MentalRotation2DService(BaseService):
    """
    Service wrapper for Mental Rotation 2D environment.
    
    This class handles the parallel backend functionality and integrates
    with the VAGEN server/client API system.
    """
    
    def __init__(self, service_config: MentalRotation2DServiceConfig, env_config: MentalRotation2DEnvConfig):
        """
        Initialize the Mental Rotation 2D service.
        
        Args:
            service_config: Configuration for the service layer
            env_config: Configuration for the environment
        """
        super().__init__(service_config, env_config)
        self.env_config = env_config
        self.service_config = service_config
    
    def create_env(self) -> MentalRotation2DEnv:
        """
        Create a new Mental Rotation 2D environment instance.
        
        Returns:
            New MentalRotation2DEnv instance
        """
        return MentalRotation2DEnv(self.env_config)

if __name__ == "__main__":
    # Test service creation
    from .env_config import MentalRotation2DEnvConfig
    from .service_config import MentalRotation2DServiceConfig
    
    env_config = MentalRotation2DEnvConfig()
    service_config = MentalRotation2DServiceConfig()
    
    service = MentalRotation2DService(service_config, env_config)
    print("Service created successfully!")
    print(f"Service config: {service_config.config_id()}")
    print(f"Env config: {env_config.config_id()}")
    
    # Test environment creation through service
    env = service.create_env()
    print("Environment created through service successfully!")
