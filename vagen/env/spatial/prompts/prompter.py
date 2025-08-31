import numpy as np
from typing import Optional
from vagen.env.spatial.Base.tos_base import ActionSequence, EvaluationManager, CognitiveMapManager
from vagen.env.spatial.Base.tos_base import Room, Agent
from vagen.env.spatial.Base.tos_base.utils.room_utils import get_room_description
from vagen.env.spatial.Base.tos_base.managers.cognitive_map_manager import COGMAP_EXP_REQUIRED_INSTRUCTION, COGMAP_EVAL_REQUIRED_INSTRUCTION
from .prompts import *

class Prompter:
    """A class to generate prompts for the SpatialGym environment."""

    ACTIVE_INSTRUCTION = ACTIVE_INSTRUCTION
    PASSIVE_INSTRUCTION = PASSIVE_INSTRUCTION
    EVALUATION_INSTRUCTION = EVALUATION_INSTRUCTION
    SHORT_EXPLORATION_PROMPT = SHORT_EXPLORATION_PROMPT
    SHORT_EVALUATION_PROMPT = SHORT_EVALUATION_PROMPT
    COGMAP_EXP_REQUIRED_INSTRUCTION = COGMAP_EXP_REQUIRED_INSTRUCTION
    COGMAP_EVAL_REQUIRED_INSTRUCTION = COGMAP_EVAL_REQUIRED_INSTRUCTION

    # Add FORMAT_PROMPT for backward compatibility
    FORMAT_PROMPT = "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text."

    # Add image prompt constants
    TOPDOWN_PROMPT = "\n\nTopdown view: {placeholder}\n{object_info}"
    # OBLIQUE_PROMPT = "\n\nOblique view: {placeholder}\n{object_info}"

    def __init__(self, config, image_handler, np_random: np.random.RandomState):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random

    def _get_topdown_prompt(self, prompt_template: str, room) -> str:
        """Generate topdown view prompt with object information."""
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        for idx, obj in enumerate(room.objects):
            obj_info += f"\nObject {idx + 1}: {obj.name}"
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def _get_oblique_prompt(self, prompt_template: str, room) -> str:
        """Generate oblique view prompt with object information."""
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        for idx, obj in enumerate(room.objects):
            obj_info += f"\nObject {idx + 1}: {obj.name}"
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def get_initial_observation_prompt(
            self,
            room: Room,
            agent: Agent,
            eval_manager: Optional[EvaluationManager] = None,
            cogmap_manager: Optional[CognitiveMapManager] = None,
            exp_history = None,
            **kwargs
        ) -> dict:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        room_desc = get_room_description(room, agent, with_topdown=self.config.prompt_config['topdown'])

        result = {}

        # Add topdown image descriptions if enabled
        if self.config.prompt_config['topdown']:
            room_desc += self._get_topdown_prompt(self.TOPDOWN_PROMPT, room)

        cogmap_instruction = cogmap_manager.get_cognitive_map_instruction() if cogmap_manager else ""

        # Prepare image list in the EXACT order of placeholders
        exp_imgs = []
        if (exp_history and isinstance(exp_history, dict)
            and 'multi_modal_data' in exp_history
            and self.config.image_placeholder in exp_history['multi_modal_data']):
            exp_imgs = list(exp_history['multi_modal_data'][self.config.image_placeholder])

        images = []  # start fresh, append strictly by placeholder order

        # Add instruction image first (corresponds to {instruction_example})
        if self.image_handler:
            try:
                images.append(self.image_handler.get_image('instruction'))
            except Exception:
                pass

        # Text placeholder for {instruction_example}
        instruction_example_str = self.config.image_placeholder

        if self.config.exp_type == 'active':
            exp_instructions = ActionSequence.get_usage_instructions() + f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
            active_instruction = self.ACTIVE_INSTRUCTION
            obs_str = active_instruction.format(
                room_info=room_desc,
                exp_instructions=exp_instructions,
                cogmap_instruction=cogmap_instruction,
                instruction_example=instruction_example_str
            )

            if self.config.prompt_config.get('cogmap', False):
                obs_str += '\n' + self.COGMAP_EXP_REQUIRED_INSTRUCTION

            # Add topdown image if enabled (after instruction image)
            if self.config.prompt_config['topdown'] and self.image_handler:
                images.append(self.image_handler.get_image('topdown'))

            result['multi_modal_data'] = {self.config.image_placeholder: images}

        else:
            # PASSIVE mode
            include_history = not (self.config.prompt_config['topdown'])

            if include_history:
                exp_history_str = f"## Exploration History\n{exp_history.get('obs_str', '')}" \
                                if isinstance(exp_history, dict) else ""
            else:
                exp_history_str = ""

            obs_str = self.PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_history=exp_history_str,
                cogmap_instruction=cogmap_instruction,
                instruction_example=instruction_example_str
            )

            if self.config.prompt_config.get('cogmap', False):
                obs_str += '\n' + self.COGMAP_EVAL_REQUIRED_INSTRUCTION

            if eval_manager is not None:
                obs_str += f"\n{self.get_evaluation_prompt(eval_manager)}"

            # Add images in placeholder order
            if self.image_handler and self.config.prompt_config['topdown']:
                images.append(self.image_handler.get_image('topdown'))
            
            # Add exploration history images if included
            if include_history and exp_imgs:
                images.extend(exp_imgs)

            result['multi_modal_data'] = {self.config.image_placeholder: images}

        result['obs_str'] = obs_str + "\n" + self.FORMAT_PROMPT
        return result
        
            

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return self.EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
