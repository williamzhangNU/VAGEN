# vagen/mllm_agent/model_interface/openai/model_config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from vagen.inference.model_interface.base_model_config import BaseModelConfig

@dataclass
class ZhipuModelConfig(BaseModelConfig):
    """Configuration for Zhipu API model interface."""
    
    # Zhipu specific parameters
    api_key: Optional[str] = None  # If None, will use environment variable
    
    # Model parameters
    model_name: str = "glm-4.5v"
    max_retries: int = 3
    timeout: int = 180
    
    # Generation parameters (inherited from base)
    # max_tokens, temperature already defined in base
    max_tokens: Optional[int] = None # 
    
    # Provider identifier
    provider: str = "zhipu"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        return f"ZhipuModelConfig({self.model_name},max_tokens={self.max_tokens},temp={self.temperature})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about Zhipu provider."""
        return {
            "description": "Zhipu API for GLM models",
            "supports_multimodal": True,
            "supported_models": [
                "glm-4.5v"
            ],
            "default_model": "glm-4.5v"
        }