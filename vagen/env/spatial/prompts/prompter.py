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

    def __init__(self, config, image_handler, np_random: np.random.RandomState, enable_think: bool = True):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random
        # Set output format based on parsing setting (enable_think)
        self.FORMAT_PROMPT = (
            "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text."
            if enable_think else
            "Always output: <answer> [your answer] </answer> with no extra text."
        )
        
    def _get_object_labels(self) -> dict:
        """Get object labels from meta data."""
        object_labels = {}
        if hasattr(self.image_handler, 'json_data') and 'objects' in self.image_handler.json_data:
            for obj in self.image_handler.json_data['objects']:
                if 'label' in obj:
                    object_labels[obj['name']] = obj['label']
        return object_labels

    def _get_topdown_prompt(self, prompt_template: str, room) -> str:
        """Generate topdown view prompt with object information."""
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        
        # Get object labels from meta data if available
        if hasattr(self.image_handler, 'json_data') and 'room_object_assignments' in self.image_handler.json_data:
            object_labels = {}
            for room_assignments in self.image_handler.json_data['room_object_assignments'].values():
                for assignment in room_assignments:
                    object_labels[assignment['name']] = assignment['label']
            
            for obj in room.objects:
                label = object_labels.get(obj.name, "?")
                obj_info += f"\nObject {label}: {obj.name}"
        else:
            # Fallback to simple numbering
            for idx, obj in enumerate(room.objects):
                obj_info += f"\nObject {idx + 1}: {obj.name}"
                
        obj_info += "\nNote: All objects in the orientation instruction image are facing towardsthe camera and the labels match the objects listed below."
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def _get_oblique_prompt(self, prompt_template: str, room) -> str:
        """Generate oblique view prompt with object information."""
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        
        # Get object labels from meta data if available
        if hasattr(self.image_handler, 'json_data') and 'room_object_assignments' in self.image_handler.json_data:
            object_labels = {}
            for room_assignments in self.image_handler.json_data['room_object_assignments'].values():
                for assignment in room_assignments:
                    object_labels[assignment['name']] = assignment['label']
            
            for obj in room.objects:
                label = object_labels.get(obj.name, "?")
                obj_info += f"\nObject {label}: {obj.name}"
        else:
            # Fallback to simple numbering
            for idx, obj in enumerate(room.objects):
                obj_info += f"\nObject {idx + 1}: {obj.name}"
                
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def get_initial_observation_prompt(
            self,
            room: Room,
            agent: Agent,
            eval_manager: Optional[EvaluationManager] = None,
            exp_history = None
        ) -> dict:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        room_desc = get_room_description(room, agent, with_topdown=self.config.prompt_config['topdown'])
        
        # Replace object list with numbered version using meta data labels
        objects = [o for o in room.all_objects if not hasattr(o, 'room_id') or not isinstance(getattr(o, 'room_id', None), list)]
        if objects:
            object_labels = self._get_object_labels()
            numbered_objects = [f"{object_labels.get(obj.name, '?')}. {obj.name}" for obj in objects]
            if numbered_objects:
                import re
                room_desc = re.sub(r"Objects: ([^\n]+)", f"Objects: {', '.join(numbered_objects)}", room_desc)

        result = {}

        # Add topdown image descriptions if enabled
        if self.config.prompt_config['topdown']:
            room_desc += self._get_topdown_prompt(self.TOPDOWN_PROMPT, room)

        images = [self.image_handler.get_image('instruction')]
        
        # Add orientation instruction image if available
        try:
            orientation_img = self.image_handler.get_image('orientation_instruction')
            images.append(orientation_img)
        except KeyError:
            # orientation_instruction.png not found, continue without it
            pass

        if self.config.exp_type == 'active':
            exp_instructions = ActionSequence.get_usage_instructions() + f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
            active_instruction = self.ACTIVE_INSTRUCTION
            
            # Add orientation instruction placeholder if we have the image
            instruction_placeholders = self.config.image_placeholder
            if len(images) > 1:  # We have both instruction and orientation_instruction
                instruction_placeholders += f"\n\nOrientation Instructions: {self.config.image_placeholder}\nNote: All objects in the orientation instruction image are facing towards the camera and the labels match the objects listed below."
            
            obs_str = active_instruction.format(
                room_info=room_desc,
                exp_instructions=exp_instructions,
                instruction_example=instruction_placeholders
            )

            # Add topdown image if enabled (after instruction image)
            if self.config.prompt_config['topdown']:
                images.append(self.image_handler.get_image('topdown'))

            result['multi_modal_data'] = {self.config.image_placeholder: images}

        else:
            if self.config.prompt_config['topdown']:
                exp_history_str = ""
                images.append(self.image_handler.get_image('topdown'))
            else:
                exp_history_str = f"## Exploration History\n{exp_history['obs_str']}"
                images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])

            # Add orientation instruction placeholder if we have the image
            instruction_placeholders = self.config.image_placeholder
            # Count images that aren't from exp_history
            base_image_count = 1  # instruction image
            if len(images) > 1 and not self.config.prompt_config['topdown']:
                # We have orientation_instruction (not counting exp_history images)
                instruction_placeholders += f"\n\nOrientation Instructions: {self.config.image_placeholder}\nNote: All objects in the orientation instruction image are facing towards the camera and the labels match the objects listed below."
            
            obs_str = self.PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_history=exp_history_str,
                instruction_example=instruction_placeholders
            )
            
            obs_str += f"\n{self.get_evaluation_prompt(eval_manager)}"

            result['multi_modal_data'] = {self.config.image_placeholder: images}

        result['obs_str'] = obs_str + "\n" + self.FORMAT_PROMPT
        return result
        
            

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return self.EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
