import numpy as np
from vagen.env.spatial.Base.tos_base import EvaluationManager, CognitiveMapManager, ActionSequence, Room
from vagen.env.spatial.Base.tos_base.managers.cognitive_map_manager import get_cognitive_map_instruction, COGMAP_REQUIRED_INSTRUCTION
from vagen.env.spatial.utils.generate_history import AutoExplore
from .prompts import *

class Prompter:
    """A class to generate prompts for the SpatialGym environment."""
    ACTIVE_INSTRUCTION = ACTIVE_INSTRUCTION
    ACTIVE_INSTRUCTION_SHORTER = ACTIVE_INSTRUCTION_SHORTER
    EVALUATION_INSTRUCTION = EVALUATION_INSTRUCTION
    FORMAT_PROMPT = FORMAT_PROMPT
    TOPDOWN_PROMPT = TOPDOWN_PROMPT
    OBLIQUE_PROMPT = OBLIQUE_PROMPT
    COGMAP_REQUIRED_INSTRUCTION = COGMAP_REQUIRED_INSTRUCTION

    def __init__(self, config, image_handler, np_random):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random

    def _get_topdown_prompt(self, prompt_template: str, room: Room) -> str:
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        for idx, obj in enumerate(room.objects):
            obj_info += f"\nObject {idx + 1}: {obj.name}"
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def get_initial_observation_prompt(
            self, room: Room,
            eval_manager: EvaluationManager = None,
            cogmap_manager: CognitiveMapManager = None,
            **kwargs,
        ) -> dict:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        room_desc = room.get_room_description()
        if self.config.prompt_with_topdown:
            room_desc += self._get_topdown_prompt(TOPDOWN_PROMPT, room)
        if self.config.prompt_with_oblique:
            room_desc += self._get_topdown_prompt(OBLIQUE_PROMPT, room)
            
        if self.config.exp_type == 'active':
            exp_instructions = ""
            if cogmap_manager:
                cogmap_instruction = cogmap_manager.get_cognitive_map_instruction()
                exp_instructions += f"\n{cogmap_instruction}"
            exp_instructions += f"## Action Instructions\n{ActionSequence.get_usage_instructions()}\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
            obs_str = self.ACTIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_instructions=exp_instructions
            )
            result = {'obs_str': obs_str + "\n" + self.FORMAT_PROMPT}
            if self.config.prompt_with_topdown:
                result['multi_modal_data'] = {self.config.image_placeholder: [self.image_handler.get_image('topdown')]}
            return result
        
        else:
            exp_history = ""
            images = []
            if not self.config.prompt_with_topdown and not self.config.prompt_with_oblique:
                exp_history_obs = AutoExplore(room, self.np_random, self.image_handler).gen_exp_history()
                exp_history = f"## Exploration History\n{exp_history_obs['obs_str']}"
                images.extend(exp_history_obs['multi_modal_data'][self.config.image_placeholder])
            elif self.config.prompt_with_topdown:
                images.append(self.image_handler.get_image('topdown'))
            elif self.config.prompt_with_oblique:
                images.append(self.image_handler.get_image('oblique'))
                
            obs_str = self._PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_history=exp_history
            )
            if eval_manager:
                obs_str += f"\n{self.get_evaluation_prompt(eval_manager)}"

            result = {'obs_str': obs_str + "\n" + self.FORMAT_PROMPT}
            if images:
                result['multi_modal_data'] = {self.config.image_placeholder: images}
            return result
        
            

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return self.EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
