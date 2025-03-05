from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from copy import deepcopy
from transformers import AutoTokenizer
import torch
from PIL import Image
import numpy as np
from dataclasses import dataclass, field


# ===============================
# Prompt Template
# ===============================

@dataclass
class PromptTemplate:
    """Dataclass for managing environment-specific prompts."""
    system_prompt: str = ""
    instruction_prompt: str = ""
    init_observation_template: str = ""
    valid_action_template: str = ""
    invalid_action_template: str = ""

    def get_task_instruction(self) -> str:
        """Get the system and instruction prompts"""
        return self.system_prompt + '\n' +self.instruction_prompt

@dataclass
class EnvFeedback:
    """Dataclass for managing environment feedback."""
    observation_template: str = ""
    multi_modal_observation: Dict = field(default_factory=dict)
    action_str: str = ""
    step_reward: float = 0.0
    done: bool = False
    env_finished_before: bool = False
    info: Dict = field(default_factory=dict)





# ===============================
# Base Env
# ===============================
class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    The class needs to be consistent with gymnasium interface.
    Should inplement: step, reset, 

    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment
            
        Returns:
            rendered environment
        """

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """

    @abstractmethod
    def success(self) -> bool:
        """Check if the current environment is successful."""

    @abstractmethod
    def finished(self) -> bool:
        """Check if the current environment is finished."""

    @abstractmethod
    def render(self, mode: str = 'text') -> Any:
        """Render the environment."""

    @abstractmethod
    def copy(self) -> 'BaseEnv':
        """Create a deep copy of the environment."""

    @abstractmethod
    def close(self):
        """Close the environment."""




# ===============================
# Base Game
# ===============================
class BaseGame(ABC):
    """
    Abstract base class for all games. Main function: step() and reset()
    - step() Workflow:
        1. Raw input --> Preprocess --> [Action];
        2. [Action] --> Env.step --> Feedback;
        3. Feedback --> Postprocess --> Observation;
    - reset() function: reset environment; return initial observation (instruction)
    """

    PROMPT_TEMPLATE = PromptTemplate()

    def __init__(self, **env_config):
        self.env_config = env_config

    @classmethod
    def get_task_instruction(cls) -> str:
        """Get the initial instruction for the environment."""
        return cls.PROMPT_TEMPLATE.get_task_instruction()

    @staticmethod
    def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
        """Convert a numpy array to a PIL RGBA image."""
        if numpy_array.shape[-1] == 3:
            # Convert RGB to RGBA by adding an alpha channel
            height, width, _ = numpy_array.shape
            rgba_array = np.zeros((height, width, 4), dtype=numpy_array.dtype)
            rgba_array[:, :, 0:3] = numpy_array
            rgba_array[:, :, 3] = 255  # Set alpha channel to fully opaque
            return Image.fromarray(rgba_array, mode='RGBA')
        elif numpy_array.shape[-1] == 4:
            # Already has 4 channels, assume it's RGBA
            return Image.fromarray(numpy_array, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB) or 4 (RGBA).")

    @abstractmethod
    def _preprocess(self, text: str) -> Dict:
        """Preprocess the raw text from LLM to action space."""

    @abstractmethod
    def _postprocess(self, step_result: Dict) -> EnvFeedback:
        """Postprocess the observation from environment to feedback for LLM."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> EnvFeedback:
        """Reset the environment."""

    @abstractmethod
    def step(self, action: List) -> List[EnvFeedback]:
        """Execute one step in the environment."""

    @abstractmethod
    def finished(self) -> bool:
        """Check if the current environment is finished."""
    
    @abstractmethod
    def success(self) -> bool:
        """Check if the current environment is successful."""

    @abstractmethod
    def close(self):
        """Close the environment."""

    @classmethod
    def name_repr(cls) -> str:
        """Get the name of the environment."""
        return cls.__name__
    
    @classmethod
    @abstractmethod
    def config_repr(cls, config: Dict) -> str:
        """Get the config of the environment."""
    




# class BaseEnv(ABC):
#     """
#     Abstract base class for all environments.
#     The class needs to be consistent with gymnasium interface.
#     Should inplement: step, reset, 

#     """
#     INVALID_ACTION = 0
#     PENALTY_FOR_INVALID = -1
#     PROMPT = QwenPromptTemplate()

#     @staticmethod
#     def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
#         """Convert a numpy array to a PIL RGBA image."""
#         return Image.fromarray(numpy_array, mode='RGBA')

#     @classmethod
#     def get_init_instruction(cls) -> str:
#         """Get the initial instruction for the environment."""
#         return cls.PROMPT.get_init_instruction()
        
#     @classmethod
#     @abstractmethod
#     def preprocess(cls, text: str) -> Dict:
#         """Preprocess the raw text from LLM."""

#     @classmethod
#     @abstractmethod
#     def postprocess(cls, text: str) -> str:
#         """Postprocess the feedback from environment for LLM."""

#     @abstractmethod
#     def reset(self, seed: Optional[int] = None) -> Any:
#         """
#         Reset the environment.
#         NOTE: the environment should be same for the same seed
#         Args:
#             seed: Seed for the environment
            
#         Returns:
#             rendered environment
#         """
#         pass

#     @abstractmethod
#     def step(self, action) -> Tuple[Any, float, bool, Dict]:
#         """
#         Execute one step in the environment.
#         NOTE should also handle predefined invalid action (0)
#         Args:
#             action: Action to take, must be in action space, or default invalid action
            
#         Returns:
#             observation (rendered environment), reward, done, info
#         """
#         pass

#     @abstractmethod
#     def success(self) -> bool:
#         """Check if the current environment is successful."""
#         pass

#     @abstractmethod
#     def finished(self) -> bool:
#         """Check if the current environment is finished."""
#         pass

#     @abstractmethod
#     def render(self, mode: str = 'tiny_rgb_array') -> Any:
#         """Render the environment."""
#         pass

#     @abstractmethod
#     def copy(self) -> 'BaseEnv':
#         """Create a deep copy of the environment."""
#         pass
