from vagen.env.base.base_env import BaseEnv
import ai2thor.controller
import numpy as np
import json
from ai2thor.platform import CloudRendering
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import RearrangementEnvConfig
from .prompt import WALKTHROUGH_SYSTEM_PROMPT, UNSHUFFLE_SYSTEM_PROMPT, init_observation_template, action_template

class RearrangementEnv(BaseEnv):
    """Rearrangement environment with two-phase workflow: walkthrough and unshuffle."""   

    ValidEvalSets = ['base']

    # Available actions for rearrangement
    ACTION_LOOKUP = {
        "moveahead": 1,
        "moveback": 2,
        "moveright": 3,
        "moveleft": 4,
        "rotateright": 5,
        "rotateleft": 6,
        "lookup": 7,
        "lookdown": 8,
        "pickup": 9,
        "putdown": 10,
        "open": 11,
        "close": 12,
        "done": 13
    }

    # Action descriptions
    DISCRETE_SKILLSET = [
        "Move forward by 0.5 meter",
        "Move backward by 0.5 meter", 
        "Move rightward by 0.5 meter",
        "Move leftward by 0.5 meter",
        "Rotate to the right by 90 degrees",
        "Rotate to the left by 90 degrees",
        "Tilt the camera upward by 30 degrees",
        "Tilt the camera downward by 30 degrees",
        "Pick up an object",
        "Put down an object",
        "Open an object",
        "Close an object",
        "Finish the current phase"
    ]

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
        self.current_phase = "walkthrough"  # "walkthrough" or "unshuffle"
        self.walkthrough_memory = []  # Store observations from walkthrough phase
        self.target_objects = {}  # Store target object states
        self.starting_objects = {}  # Store starting object states
        self.step_count = 0
        self.max_steps = 500
        self.success_threshold = config.success_threshold
        
        # Initialize AI2-THOR controller
        self._init_controller()
        
        # Load dataset
        self._load_dataset()

    def _init_controller(self):
        """Initialize the AI2-THOR controller."""
        controller_kwargs = {
            'agentMode': 'default',
            'visibilityDistance': 1.5,
            'scene': 'FloorPlan1',
            'gridSize': self.config.step_length,
            'snapToGrid': True,
            'rotateStepDegrees': 90,
            'renderDepthImage': False,
            'renderInstanceSegmentation': True,
            'width': self.config.resolution,
            'height': self.config.resolution,
            'fieldOfView': self.config.fov,
            'platform': CloudRendering
        }
            
        self.controller = ai2thor.controller.Controller(**controller_kwargs)

    def _load_dataset(self):
        """Load the rearrangement dataset."""
        import os
        dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', f'{self.config.eval_set}.json')
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset['tasks'])} rearrangement tasks")

    def reset(self, seed=None):
        """Reset the environment to a specific task.

        Args:
            seed: Random seed for reproducibility (used as task_id)

        Returns:
            Tuple of (observation, info)
        """
        # Use seed as task_id, default to 0
        task_id = seed if seed is not None else 0
        if task_id >= len(self.dataset['tasks']):
            task_id = 0
            
        self.current_task = self.dataset['tasks'][task_id]
        self.current_phase = "walkthrough"
        self.walkthrough_memory = []
        self.step_count = 0
        
        # Reset scene
        scene = self.current_task['scene']
        self.controller.reset(scene=scene)
        
        # Set agent position
        agent_pose = self.current_task['agentPose']
        self.controller.step(
            action="Teleport",
            position=agent_pose['position'],
            rotation=dict(x=0, y=agent_pose['rotation'], z=0),
            horizon=agent_pose['horizon']
        )
        
        # Set up target state (walkthrough phase)
        self._setup_target_state()
        
        return self._render(init_obs=True), {}

    def _setup_target_state(self):
        """Set up the target state for walkthrough phase."""
        # Store target object positions
        self.target_objects = {}
        
        # Set objects to target positions (this is what agent should observe)
        if 'rearrangement_data' in self.current_task:
            starting_poses = self.current_task['rearrangement_data']['starting_poses']
            for pose in starting_poses:
                obj_name = pose['objectName']
                # Set object to target position for walkthrough
                self.controller.step(
                    action="SetObjectPoses",
                    objectPoses=[{
                        'objectName': obj_name,
                        'position': pose['position'],
                        'rotation': pose['rotation']
                    }]
                )
                
                # Store target state
                self.target_objects[obj_name] = {
                    'position': pose['position'],
                    'rotation': pose['rotation']
                }

    def _setup_starting_state(self):
        """Set up the starting state for unshuffle phase."""
        # Randomly shuffle objects to create the starting state
        if 'rearrangement_data' in self.current_task:
            starting_poses = self.current_task['rearrangement_data']['starting_poses']

            # Create shuffled positions (simple random displacement)
            for pose in starting_poses:
                obj_name = pose['objectName']
                original_pos = pose['position']

                # Add random displacement
                shuffled_pos = {
                    'x': original_pos['x'] + np.random.uniform(-1.0, 1.0),
                    'y': original_pos['y'],
                    'z': original_pos['z'] + np.random.uniform(-1.0, 1.0)
                }

                # Set object to shuffled position
                self.controller.step(
                    action="SetObjectPoses",
                    objectPoses=[{
                        'objectName': obj_name,
                        'position': shuffled_pos,
                        'rotation': pose['rotation']
                    }]
                )

                # Store starting state
                self.starting_objects[obj_name] = {
                    'position': shuffled_pos,
                    'rotation': pose['rotation']
                }

    def _get_observation(self):
        """Get current observation from the environment."""
        event = self.controller.last_event

        # Always return RGB image (CloudRendering default)
        rgb_image = event.frame
        return convert_numpy_to_PIL(rgb_image)

    def _render(self, init_obs=True):
        """Render the environment observation.

        This method creates the observation dict with image and prompt information,
        formatted based on whether this is the initial observation or a subsequent one.

        Args:
            init_obs: Whether this is the initial observation

        Returns:
            Observation dict
        """
        img_placeholder = self.config.get("image_placeholder", "<image>")

        # Get the RGB frame from the environment
        frame = self.controller.last_event.frame

        # Convert to PIL image for multimodal inputs
        multi_modal_data = {
            img_placeholder: [convert_numpy_to_PIL(frame)]
        }

        # Format the observation string based on current phase
        if init_obs:
            instruction = self.current_task['instruction'] if self.current_task else "Complete the rearrangement task"
            obs_str = init_observation_template(
                observation=img_placeholder,
                instruction=instruction,
                phase=self.current_phase
            )
        else:
            obs_str = action_template(
                observation=img_placeholder,
                instruction=self.current_task.get('instruction', '') if self.current_task else '',
                phase=self.current_phase
            )

        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data
        }

    def step(self, action_str: str):
        """Execute an action in the environment.

        Args:
            action_str: String representation of the action

        Returns:
            Dictionary containing observation, reward, done, info
        """
        self.step_count += 1

        # Parse action
        actions = self._parse_action(action_str)

        # Check for phase completion
        if "done" in actions or "WALKTHROUGH_DONE" in action_str or "UNSHUFFLE_DONE" in action_str:
            if self.current_phase == "walkthrough":
                return self._complete_walkthrough(action_str)
            elif self.current_phase == "unshuffle":
                return self._complete_unshuffle()

        # Execute actions
        reward = 0.0
        env_feedback = ""

        for action in actions:
            if action in self.ACTION_LOOKUP:
                success = self._execute_action(action)
                if success:
                    env_feedback += f"{action} executed successfully. "
                    reward += 0.1
                else:
                    env_feedback += f"{action} failed. "
                    reward -= 0.1
            else:
                env_feedback += f"Invalid action: {action}. "
                reward -= 0.2

        # Check if done
        done = self.step_count >= self.max_steps

        # Create info dict
        info = {
            'phase': self.current_phase,
            'instruction': self.current_task['instruction'] if self.current_task else '',
            'env_feedback': env_feedback,
            'last_action_success': True  # Simplified for now
        }

        return self._render(init_obs=False), reward, done, info

    def _parse_action(self, action_str: str):
        """Parse action string into list of actions."""
        # Use existing parse function
        if hasattr(PARSE_FUNC_MAP, 'get'):
            parse_func = PARSE_FUNC_MAP.get('rearrangement', PARSE_FUNC_MAP.get('default'))
            if parse_func:
                return parse_func(action_str)

        # Fallback parsing
        actions = []
        for action in action_str.lower().split(','):
            action = action.strip()
            if action in self.ACTION_LOOKUP:
                actions.append(action)
        return actions

    def _execute_action(self, action: str):
        """Execute a single action."""
        try:
            if action == "moveahead":
                event = self.controller.step(action="MoveAhead")
            elif action == "moveback":
                event = self.controller.step(action="MoveBack")
            elif action == "moveright":
                event = self.controller.step(action="MoveRight")
            elif action == "moveleft":
                event = self.controller.step(action="MoveLeft")
            elif action == "rotateright":
                event = self.controller.step(action="RotateRight")
            elif action == "rotateleft":
                event = self.controller.step(action="RotateLeft")
            elif action == "lookup":
                event = self.controller.step(action="LookUp")
            elif action == "lookdown":
                event = self.controller.step(action="LookDown")
            elif action == "pickup":
                # Find nearest pickupable object
                event = self._pickup_nearest_object()
            elif action == "putdown":
                event = self.controller.step(action="PutObject")
            elif action == "open":
                event = self._open_nearest_object()
            elif action == "close":
                event = self._close_nearest_object()
            else:
                return False

            return event.metadata['lastActionSuccess']
        except Exception as e:
            print(f"Action execution failed: {e}")
            return False

    def _pickup_nearest_object(self):
        """Pick up the nearest pickupable object."""
        event = self.controller.last_event
        pickupable_objects = [obj for obj in event.metadata['objects']
                            if obj['pickupable'] and obj['visible']]

        if pickupable_objects:
            # Find nearest object
            agent_pos = event.metadata['agent']['position']
            nearest_obj = min(pickupable_objects,
                            key=lambda obj: np.linalg.norm([
                                obj['position']['x'] - agent_pos['x'],
                                obj['position']['z'] - agent_pos['z']
                            ]))

            return self.controller.step(action="PickupObject", objectId=nearest_obj['objectId'])
        else:
            return self.controller.step(action="PickupObject")

    def _open_nearest_object(self):
        """Open the nearest openable object."""
        event = self.controller.last_event
        openable_objects = [obj for obj in event.metadata['objects']
                          if obj['openable'] and obj['visible'] and not obj['isOpen']]

        if openable_objects:
            agent_pos = event.metadata['agent']['position']
            nearest_obj = min(openable_objects,
                            key=lambda obj: np.linalg.norm([
                                obj['position']['x'] - agent_pos['x'],
                                obj['position']['z'] - agent_pos['z']
                            ]))

            return self.controller.step(action="OpenObject", objectId=nearest_obj['objectId'])
        else:
            return self.controller.step(action="OpenObject")

    def _close_nearest_object(self):
        """Close the nearest closeable object."""
        event = self.controller.last_event
        closeable_objects = [obj for obj in event.metadata['objects']
                           if obj['openable'] and obj['visible'] and obj['isOpen']]

        if closeable_objects:
            agent_pos = event.metadata['agent']['position']
            nearest_obj = min(closeable_objects,
                            key=lambda obj: np.linalg.norm([
                                obj['position']['x'] - agent_pos['x'],
                                obj['position']['z'] - agent_pos['z']
                            ]))

            return self.controller.step(action="CloseObject", objectId=nearest_obj['objectId'])
        else:
            return self.controller.step(action="CloseObject")

    def _complete_walkthrough(self, action_str: str):
        """Complete the walkthrough phase and transition to unshuffle."""
        # Extract memory from action string if provided
        if "WALKTHROUGH_DONE" in action_str:
            # Try to extract JSON memory
            try:
                import re
                json_match = re.search(r'\[.*\]', action_str, re.DOTALL)
                if json_match:
                    memory_json = json_match.group()
                    self.walkthrough_memory = json.loads(memory_json)
            except:
                pass

        # Transition to unshuffle phase
        self.current_phase = "unshuffle"
        self.step_count = 0

        # Set up starting state (shuffled objects)
        self._setup_starting_state()

        # Get new observation
        observation = self._get_observation()

        # Create unshuffle prompt
        instruction = "根据之前的备忘录，将发生变化的物体恢复至目标状态。"
        prompt = init_observation_template(
            observation=observation,
            instruction=instruction,
            phase=self.current_phase
        )

        return {
            'observation': observation,
            'prompt': prompt,
            'phase': self.current_phase,
            'instruction': instruction,
            'done': False,
            'reward': 1.0,  # Reward for completing walkthrough
            'env_feedback': "Walkthrough phase completed. Starting unshuffle phase."
        }

    def _complete_unshuffle(self):
        """Complete the unshuffle phase."""
        # Calculate success based on object positions
        success_rate = self._calculate_success_rate()

        reward = success_rate * 10.0  # Scale reward
        done = True

        return {
            'observation': self._get_observation(),
            'prompt': f"Unshuffle phase completed. Success rate: {success_rate:.2f}",
            'phase': self.current_phase,
            'instruction': "Task completed.",
            'done': done,
            'reward': reward,
            'env_feedback': f"Task completed with {success_rate:.2f} success rate."
        }

    def _calculate_success_rate(self):
        """Calculate success rate based on object positions."""
        if not self.target_objects:
            return 0.0

        event = self.controller.last_event
        current_objects = {obj['objectId']: obj for obj in event.metadata['objects']}

        total_objects = len(self.target_objects)
        successful_objects = 0

        for obj_name, target_state in self.target_objects.items():
            if obj_name in current_objects:
                current_obj = current_objects[obj_name]
                current_pos = current_obj['position']
                target_pos = target_state['position']

                # Calculate distance
                distance = np.sqrt(
                    (current_pos['x'] - target_pos['x'])**2 +
                    (current_pos['z'] - target_pos['z'])**2
                )

                if distance < self.success_threshold:
                    successful_objects += 1

        return successful_objects / total_objects if total_objects > 0 else 0.0

    def measure_success(self):
        """Check if the rearrangement task has been completed successfully.

        Returns:
            success: Float indicating success rate (0.0 to 1.0)
            distance: Average distance of objects from target positions
        """
        success_rate = self._calculate_success_rate()

        # Calculate average distance for additional info
        if not self.target_objects:
            return success_rate, 0.0

        event = self.controller.last_event
        total_distance = 0.0
        object_count = 0

        for obj_name, target_info in self.target_objects.items():
            for obj in event.metadata['objects']:
                if obj['name'] == obj_name:
                    current_pos = obj['position']
                    target_pos = target_info['position']
                    distance = ((current_pos['x'] - target_pos['x']) ** 2 +
                              (current_pos['z'] - target_pos['z']) ** 2) ** 0.5
                    total_distance += distance
                    object_count += 1
                    break

        avg_distance = total_distance / object_count if object_count > 0 else 0.0
        return success_rate, avg_distance

    def compute_reward(self) -> float:
        """
        Compute final reward for the rearrangement task.

        Returns:
            Final reward based on success rate and completion status
        """
        success_rate = self._calculate_success_rate()

        # Base reward from success rate
        base_reward = success_rate * 10.0

        # Bonus for completing the task
        if self.current_phase == "unshuffle" and success_rate > 0.8:
            base_reward += 5.0

        return base_reward

    def system_prompt(self) -> str:
        """Get the system prompt for the current phase."""
        if self.current_phase == "walkthrough":
            return WALKTHROUGH_SYSTEM_PROMPT
        else:
            return UNSHUFFLE_SYSTEM_PROMPT

    def get_system_prompt(self):
        """Get the system prompt for the current phase (legacy method)."""
        return self.system_prompt()

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
            'current_phase': self.current_phase,
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
        """Get the action space description."""
        return self.DISCRETE_SKILLSET

    def get_valid_actions(self):
        """Get list of valid action names."""
        return list(self.ACTION_LOOKUP.keys())
