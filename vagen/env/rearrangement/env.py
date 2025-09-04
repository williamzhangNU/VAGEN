from vagen.env.base.base_env import BaseEnv
import ai2thor.controller
import numpy as np
import json
import re
from ai2thor.platform import CloudRendering
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import RearrangementEnvConfig
from .prompt import FORMAT_PROMPT, INIT_PROMPT, STEP_PROMPT


class RearrangementEnv(BaseEnv):
    ValidEvalSets = ['base']


    def __init__(self, config: RearrangementEnvConfig):
        """Initialize the Rearrangement environment.

        Args:
            config: Configuration for the environment including resolution, FOV,
                   eval set, render mode, etc.
        """
        super().__init__()
        self.config = config
        self.controller = None
        self.current_task = None
        self.step_count = 0
        self.max_steps = 10
        self.success_threshold = config.success_threshold

        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        # Initialize AI2-THOR controller
        self._init_controller()

        # Load dataset
        self._load_dataset()

    def _init_controller(self):
        """Initialize the AI2-THOR controller."""
        controller_kwargs = {
            'agentMode': 'default',
            'visibilityDistance': 10,
            'scene': 'FloorPlan1',
            'gridSize': self.config.step_length,
            'snapToGrid': True,
            'rotateStepDegrees': 90,
            'renderDepthImage': False,
            'renderInstanceSegmentation': True,
            'width': self.config.width,
            'height': self.config.height,
            'fieldOfView': self.config.fov,
            'platform': CloudRendering
        }

        self.controller = ai2thor.controller.Controller(**controller_kwargs)

    def _load_dataset(self):
        """Load the rearrangement dataset (list of tasks)."""
        import os
        dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', f'{self.config.eval_set}.json')
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)  # list of task dicts
        print(f"Loaded {len(self.dataset)} rearrangement tasks from {dataset_path}")

    def reset(self, seed=None):
        """Reset the environment for the base dataset task.

        This env shows two images to LLM at the beginning: the target object's
        before (initial) and after (moved) positions. Only 4 movement actions are allowed.
        """
        # Pick task index
        idx = seed if seed is not None else 0
        if not isinstance(self.dataset, list):
            # Backward compatibility in case dataset was loaded differently
            tasks = self.dataset.get('tasks', [])
        else:
            tasks = self.dataset
        if len(tasks) == 0:
            raise RuntimeError("No tasks found in dataset")
        if idx >= len(tasks):
            idx = idx % len(tasks)

        self.current_task = tasks[idx]
        self.step_count = 0
        self.target_objects = {}
        self.starting_objects = {}

        # Reset scene and teleport agent
        scene = self.current_task['scene']
        self.controller.reset(scene=scene)

        agent_view = self.current_task.get('agent_view', {})
        agent_pos = agent_view.get('position')
        agent_rot_vec = agent_view.get('rotation')
        self.controller.step(
            action="Teleport",
            position=agent_pos,
            rotation=agent_rot_vec,
            horizon=0
        )

        before_frame = self.controller.last_event.frame.copy()

        # Get original rotation of target object from metadata
        self.target_object_id = self.current_task['target_object_id']
        self.original_position = self.current_task['original_position']
        self.final_position = self.current_task['final_position']
        self.original_rotation = None
        for obj in self.controller.last_event.metadata.get('objects', []):
            if obj.get('objectId') == self.target_object_id:
                self.original_rotation = obj.get('rotation', {"x": 0.0, "y": 0.0, "z": 0.0})
                break
        assert self.original_rotation is not None

        self.controller.step(
            action="PlaceObjectAtPoint",
            objectId=self.target_object_id,
            position=self.final_position,
            rotation=self.original_rotation,
        )
        assert self.controller.last_event.metadata.get('lastActionSuccess', False)
        after_frame = self.controller.last_event.frame.copy()

        # Revert object back to original position for the interactive phase

        self.controller.step(
            action="PlaceObjectAtPoint",
            objectId=self.target_object_id,
            position=self.original_position,
            rotation=self.original_rotation,
        )
        assert self.controller.last_event.metadata.get('lastActionSuccess', False)
        # Track object position state
        self.object_position = dict(self.original_position)
        self.target_position = dict(self.final_position)

        # Prepare choice list for LLM: candidate object types (from entire dataset)
        self.candidate_types = list(set({t.get('target_object_type') for t in tasks if t.get('target_object_type')}))
        self.selected_type = None

        # Store frames for potential re-prompting
        self._reset_frames = (before_frame, after_frame)

        # Build initial prompt (single phase): ask to Select(obj)
        image_placeholder = self.config.get("image_placeholder", "<image>")
        obs_str = INIT_PROMPT.format(
            image_placeholder=image_placeholder,
            candidate_types=", ".join(self.candidate_types),
            step=self.max_steps,
        ) + f"\n{FORMAT_PROMPT}"
        multi_modal = {image_placeholder: [convert_numpy_to_PIL(before_frame), convert_numpy_to_PIL(after_frame)]}
        return {"obs_str": obs_str, "multi_modal_data": multi_modal}, {}


    def _get_observation(self):
        """Get current observation from the environment."""
        event = self.controller.last_event

        # Always return RGB image (CloudRendering default)
        rgb_image = event.frame
        return convert_numpy_to_PIL(rgb_image)

    def _render(self, init_obs=True):
        """Render observation with images and task description.

        - On reset (init_obs=True): provide two images [before, after] in the same
          placeholder and a concise instruction.
        - On step: provide current camera view only and the same instruction.
        """

        return {}

    def step(self, llm_raw_response: str):
        """Single-phase step: parse <think>, <answer>, then execute Actions via _execute_action.
        Expected answer format: Actions: [<action_1>, ..., <action_n>]
        Supported actions: Select(ObjectType), Move(ahead|back|left|right, meters), Term()
        """
        import re
        self.step_count += 1

        # 1) Parse <think> and <answer>
        ans_m = re.search(r"<answer>\s*(.*?)\s*</answer>", llm_raw_response, flags=re.IGNORECASE | re.DOTALL)
        answer_text = ans_m.group(1).strip() if ans_m else llm_raw_response.strip()

        # 2) Delegate full answer_text to executor for parsing and execution
        return self._execute_action(answer_text)

    def _execute_action(self, answer_text: str):
        """Parse and execute the full action sequence from answer_text.
        Returns (obs, reward, done, info)
        """
        import re, math
        img_ph = self.config.get("image_placeholder", "<image>")
        reward = 0.0
        done = False
        obs_str = ''
        # Extract ordered actions
        pattern_select = re.compile(r"Select\s*\(\s*([^)]+)\s*\)", re.IGNORECASE)
        pattern_move = re.compile(r"Move\s*\(\s*(ahead|back|left|right)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", re.IGNORECASE)
        pattern_term = re.compile(r"Term\s*\(\s*\)", re.IGNORECASE)
        matches = []
        for m in pattern_select.finditer(answer_text):
            matches.append((m.start(), 'select', m.group(1)))
        for m in pattern_move.finditer(answer_text):
            matches.append((m.start(), 'move', (m.group(1), m.group(2))))
        for m in pattern_term.finditer(answer_text):
            matches.append((m.start(), 'term', None))
        matches.sort(key=lambda x: x[0])

        # Execute sequentially
        for _, kind, payload in matches:
            if kind == 'select':
                chosen = str(payload).strip()
                correct = str(self.current_task.get('target_object_type', '')).strip()
                if chosen.lower() != correct.lower():
                    return {"obs_str": 'you choose the wrong object'}, reward, True, {}
                self.selected_type = chosen
                obs_str += f"Select({chosen}): success\n"

            elif kind == 'move':
                direction, meters_str = payload
                meters = float(meters_str)
                if not getattr(self, 'selected_type', None):
                    obs_str += "You must select an object before moving.\n"
                    break
                # Agent-relative axes
                yaw_deg = float(self.controller.last_event.metadata["agent"]["rotation"]["y"])
                yaw = math.radians(yaw_deg)
                fwd = {"x": math.sin(yaw), "z": math.cos(yaw)}
                right = {"x": math.cos(yaw), "z": -math.sin(yaw)}
                dir_lc = direction.lower()
                if dir_lc == 'ahead':
                    dx_unit, dz_unit = fwd['x'], fwd['z']
                elif dir_lc == 'back':
                    dx_unit, dz_unit = -fwd['x'], -fwd['z']
                elif dir_lc == 'right':
                    dx_unit, dz_unit = right['x'], right['z']
                elif dir_lc == 'left': 
                    dx_unit, dz_unit = -right['x'], -right['z']
                else:
                    obs_str += f"Invalid direction: {direction}. Valid options: ahead|back|left|right.\n"
                    break
                self.object_position['x'] = float(self.object_position['x']) + dx_unit * meters
                self.object_position['z'] = float(self.object_position['z']) + dz_unit * meters
                ev = self.controller.step(
                    action="PlaceObjectAtPoint",
                    objectId=self.target_object_id,
                    position=self.object_position,
                    rotation=self.original_rotation,
                )
                if ev.metadata['lastActionSuccess']:
                    obs_str += f"Move({direction},{meters}): 'success'\n"
                else:
                    obs_str += f"Move({direction},{meters}): 'fail because it may collide with other objects or out of bound'\n"
                    print(f"Move({direction},{meters}) failed: ",ev.metadata['errorMessage'])
                    break

            else:  # term
                done = True
                break

        # Distance to target (for info)
        dx = float(self.object_position['x']) - float(self.target_position['x'])
        dz = float(self.object_position['z']) - float(self.target_position['z'])
        dist = float((dx**2 + dz**2) ** 0.5)
        done = done or (dist <= self.success_threshold) or (self.step_count >= self.max_steps)

        # Compose observation
        obs_str += STEP_PROMPT.format(image_placeholder=img_ph) + f"\n{FORMAT_PROMPT}"
        obs = {"obs_str": obs_str, "multi_modal_data": {img_ph: [convert_numpy_to_PIL(self.controller.last_event.frame)]}}
        return obs, reward, done, {}

    def system_prompt(self) -> str:
        return "You are an AI assistant that answers visual questions based on images."

    def close(self):
        """Close the environment."""
        if self.controller:
            self.controller.stop()

    def get_env_state(self):
        """
        Get the current state of the rearrangement environment focusing on visible objects.

        Returns:
            Dict: Contains current phase, target objects, visible objects,
                and task completion information
        """
        event = self.controller.last_event

        # Get visible objects
        visible_objects = []
        for obj in event.metadata['objects']:
            if obj['visible']:
                visible_objects.append({
                    'name': obj['name'],
                    'objectType': obj['objectType'],
                    'position': obj['position'],
                    'rotation': obj['rotation'],
                    'pickupable': obj.get('pickupable', False),
                    'openable': obj.get('openable', False),
                    'isOpen': obj.get('isOpen', False)
                })

        # Get success metrics
        success_rate, avg_distance = self.measure_success()

        # Get agent information
        agent_metadata = event.metadata["agent"]
        agent_position = agent_metadata["position"]
        agent_rotation = agent_metadata["rotation"]["y"]

        return {
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'success_rate': success_rate,
            'average_distance': avg_distance,
            'agent_position': agent_position,
            'agent_rotation': agent_rotation,
            'visible_objects': visible_objects,
            'target_objects': self.target_objects,
            'starting_objects': self.starting_objects,
            'instruction': self.current_task.get('instruction', '') if self.current_task else '',
            'task_id': getattr(self, 'current_task_id', 0)
        }

    def get_action_space(self):
        """Deprecated: no discrete action space (LLM outputs free-form action sequence)."""
        return []

    def get_valid_actions(self):
        """Deprecated helper: use prompt to guide action formats."""
        return []
