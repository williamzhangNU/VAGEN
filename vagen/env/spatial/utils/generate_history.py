"""
Generate exploration history using DFS
"""
import numpy as np
import os
import json
from collections import Counter
from PIL import Image
from typing import List, Tuple, Dict

from vagen.env.spatial.Base.tos_base import (
    Room,
    DirectionalGraph,
    DirPair,
    Dir,
    DirectionSystem,
    BaseAction,
    MoveAction,
    RotateAction,
    ObserveAction,
    TermAction,
    ExplorationManager,
    Object,
    Agent,
)

class AutoExplore:
    """
    Automatically explore the environment
    TODO: use BFS to get shortest path
    """
    
    def __init__(self, room: Room, np_random: np.random.Generator, json_data: dict = None, image_dir: str = None, image_size: tuple = (224, 224)):
        self.room = room.copy()
        self.np_random = np_random
        self.exp_manager = ExplorationManager(self.room)
        self.image_placeholder = "<image>"
        
        # Initialize image handling if data provided
        self.image_map = {}
        self.name_to_cam = {}
        if json_data and image_dir:
            self._init_images(json_data, image_dir, image_size)

    def _init_images(self, json_data: dict, image_dir: str, image_size: tuple):
        """Initialize image mapping similar to SpatialGym."""
        # Map object names to camera IDs
        self.name_to_cam = {'central': 'central'}
        for obj in json_data.get('objects', []):
            obj_pos = (obj['pos']['x'], obj['pos']['z'])
            obj_name = obj['model']
            for cam in json_data.get('cameras', []):
                if cam['id'] == 'central':
                    continue
                cam_pos = (cam['position']['x'], cam['position']['z'])
                if obj_pos == cam_pos:
                    self.name_to_cam[obj_name] = cam['id']
                    break
        
        # Load all images
        for entry in json_data.get('images', []):
            key = (entry['cam_id'], entry['direction'])
            filename = entry['file']
            path = os.path.join(image_dir, filename)
            if os.path.exists(path):
                self.image_map[key] = Image.open(path).resize(image_size, Image.LANCZOS)

    def _get_current_position_direction(self) -> Tuple[str, str]:
        """Get current agent position and direction."""
        agent = self.exp_manager.exploration_room.agent
        
        # Find position: which object is at same location as agent
        position_name = 'central'  # default
        for obj in self.exp_manager.exploration_room.objects:
            if np.allclose(obj.pos, agent.pos):
                position_name = obj.name
                break
        
        # Get direction from agent_anchor
        agent_anchor_ori = tuple(self.exp_manager.agent_anchor.ori)
        direction_name = {(0, 1): 'north', (1, 0): 'west', (0, -1): 'south', (-1, 0): 'east'}[agent_anchor_ori]
        
        return position_name, direction_name

    def _generate_history_passive(self) -> Tuple[List[str], List[List], List[Image.Image]]:
        """
        Generate exploration history of egocentric exploration using ExplorationManager
        NOTE oracle generation
        
        Returns:
            observe_result: list of observation messages
            actions: list of Action instances in chronological order
            images: list of images for each observe action
        """
        assert self.room.agent is not None, "Agent is not in the room"

        observe_result, actions, actions_in_a_turn, images = [], [], [], []
        agent_idx = self.exp_manager._get_index(self.room.agent.name)

        while True:
            unknown_pairs = self.exp_manager.get_unknown_pairs()
            if not unknown_pairs:
                # no unknown pairs --> terminate
                actions.append([TermAction()])
                break

            # Get unknowns involving agent
            agent_unknown_pairs = [(pair[1], pair[0]) if pair[0] == agent_idx else pair 
                                 for pair in unknown_pairs if agent_idx in pair]
            
            if not agent_unknown_pairs:
                # Move to best position (next object)
                counts = Counter()
                for i, j in unknown_pairs:
                    counts[i] += 1
                    counts[j] += 1
                next_obj_idx = max(counts, key=counts.get)
                obj_name = self.exp_manager.objects[next_obj_idx].name
                
                # Turn to face target before moving
                rotation = self._find_rotation_to_see_object(next_obj_idx)
                if rotation != 0:
                    action = RotateAction(rotation)
                    actions_in_a_turn.append(action)
                    self.exp_manager.execute_action(action)
                
                # Move to target object
                action = MoveAction(obj_name)
                actions_in_a_turn.append(action)
                self.exp_manager.execute_action(action)
                continue
            
            # Check if there are already visible unknowns in current direction
            current_visible_count = sum(1 for target_idx, _ in agent_unknown_pairs 
                                      if self._would_be_visible_after_rotation(target_idx, 0))
            
            # turn to best direction only if current direction has no visible unknowns
            if current_visible_count == 0:
                best_direction = self._find_best_direction(agent_unknown_pairs)
                if best_direction != 0:
                    action = RotateAction(best_direction)
                    actions_in_a_turn.append(action)
                    self.exp_manager.execute_action(action)
            
            # Perform observation
            action = ObserveAction()
            actions_in_a_turn.append(action)
            success, message, data = self.exp_manager._execute_and_update(action)
            
            # for visual, use image placeholder as observation
            observe_result.append(f"You observe: {self.image_placeholder}")
            
            # Get current image if available
            if self.image_map:
                position, direction = self._get_current_position_direction()
                cam_id = self.name_to_cam.get(position, 'central')
                if (cam_id, direction) in self.image_map:
                    images.append(self.image_map[(cam_id, direction)])
                else:
                    images.append(None)
            else:
                images.append(None)
            
            # Observation marks end of turn
            actions.append(actions_in_a_turn)
            actions_in_a_turn = []

        return observe_result, actions, images

    def _would_be_visible_after_rotation(self, target_idx: int, rotation: int) -> bool:
        """Check visibility after rotation"""
        agent = self.exp_manager.exploration_room.agent
        target = self.exp_manager.objects[target_idx]
        if target.name == agent.name:
            return True
        
        rotations = {
            0: np.array([[1, 0], [0, 1]]),
            90: np.array([[0, -1], [1, 0]]),
            270: np.array([[0, 1], [-1, 0]]),
            180: np.array([[-1, 0], [0, -1]]),
        }
        rotated_ori = agent.ori @ rotations[rotation]
        temp_agent = Object(name='temp', pos=agent.pos, ori=rotated_ori)  
        return BaseAction._is_visible(temp_agent, target)
    
    def _find_best_direction(self, agent_unknown_pairs: List[Tuple[int, int]]) -> int:
        """Find direction with most visible unknowns"""
        best_count, best_direction = 0, 0
        
        for rotation in [0, 90, 180, 270]:
            count = sum(1 for target_idx, _ in agent_unknown_pairs 
                       if self._would_be_visible_after_rotation(target_idx, rotation))
            if count > best_count:
                best_count, best_direction = count, rotation
        
        return best_direction

    def _find_rotation_to_see_object(self, target_idx: int) -> int:
        """Find rotation needed to see specific object"""
        for rotation in [0, 90, 180, 270]:
            if self._would_be_visible_after_rotation(target_idx, rotation):
                return rotation
        return 0
    
    def _format_history_to_obs(self, observe_result: List[str], actions: List[List[BaseAction]], images: List[Image.Image]) -> Dict:
        """Convert history and actions to obs format with multi_modal_data."""
        turn_strings = []
        observe_idx = 0
        valid_images = []
        
        for turn_num, turn_actions in enumerate(actions, 1):
            action_strings = []
            
            for action in turn_actions:
                if isinstance(action, ObserveAction) and observe_idx < len(observe_result):
                    action_strings.append(observe_result[observe_idx])
                    # Collect valid images
                    if observe_idx < len(images) and images[observe_idx] is not None:
                        valid_images.append(images[observe_idx])
                    observe_idx += 1
                else:
                    action_strings.append(action.success_message())
            
            turn_strings.append(f"{turn_num}. {' '.join(action_strings)}")
        
        obs_str = "\n".join(turn_strings)
        
        # Create observation dict
        obs = {'obs_str': obs_str}
        if valid_images:
            obs['multi_modal_data'] = {self.image_placeholder: valid_images}
        
        return obs
    
    def gen_exp_history(self) -> Dict:
        """Generate exploration history in obs format."""
        observe_result, actions, images = self._generate_history_passive()
        return self._format_history_to_obs(observe_result, actions, images)

if __name__ == "__main__":
    import re
    from vagen.env.spatial.Base.tos_base.utils.room_utils import initialize_room_from_json
    from gymnasium.utils import seeding
    import json
    import os

    # Load data
    json_path = os.path.join(os.path.dirname(__file__), "../output/run_00/meta_data.json")
    image_dir = os.path.dirname(json_path)
    json_data = json.load(open(json_path, 'r'))

    rng1 = seeding.np_random(2)[0]
    room = initialize_room_from_json(json_data)
    print(room)
    room.plot(render_mode='img')

    explorer = AutoExplore(room, rng1, json_data, image_dir)
    exploration_obs = explorer.gen_exp_history()
    
    print("Observation string:")
    print(exploration_obs['obs_str'])
    
    if 'multi_modal_data' in exploration_obs:
        images = exploration_obs['multi_modal_data']['<image>']
        print(f"\nImages collected: {len(images)}")
        
        # Save images to local directory
        output_dir = os.path.join(os.path.dirname(__file__), "saved_images")
        os.makedirs(output_dir, exist_ok=True)
        
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"exploration_image_{i:03d}.png")
            image.save(image_path)
            print(f"Saved image {i+1} to {image_path}")
    else:
        print("\nNo images collected")
