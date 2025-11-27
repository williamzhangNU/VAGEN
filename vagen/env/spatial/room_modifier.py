import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from vagen.env.spatial.Base.tos_base import Room, Object

from dataclasses import dataclass

@dataclass
class ChangedObject:
    name: str
    pos: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None # (prev, curr)
    ori: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None # (prev, curr)

    def to_dict(self):
        res = {'name': self.name}
        if self.pos:
            res['pos'] = {'prev': self.pos[0], 'curr': self.pos[1]}
        if self.ori:
            res['ori'] = {'prev': self.ori[0], 'curr': self.ori[1]}
        return res
    
    def merge(self, other: 'ChangedObject'):
        if self.name != other.name:
            raise ValueError(f"Cannot merge changes for different objects: {self.name} vs {other.name}")
        if other.pos:
            self.pos = other.pos
        if other.ori:
            self.ori = other.ori

    @classmethod
    def parse(cls, text: str) -> 'ChangedObject':
        """
        Parse string like 'apple: position', 'chair orientation', 'table moved'.
        Tries to be robust to formatting differences.
        """
        import re
        text = text.strip().strip('"\'').lower()
        
        # Define keywords
        pos_keywords = ['position', 'location', 'moved', 'pos']
        ori_keywords = ['orientation', 'rotation', 'rotated', 'ori', 'facing']
        
        # Try to split by colon first, then by space
        parts = []
        if ':' in text:
            parts = text.split(':', 1)
        else:
            # Split by last space to separate name from type
            # This assumes name doesn't contain the type keyword at the end
            # A better approach might be to find the keyword and split there
            found_keyword = False
            for kw in pos_keywords + ori_keywords:
                if text.endswith(kw):
                    name = text[:-len(kw)].strip()
                    type_str = kw
                    parts = [name, type_str]
                    found_keyword = True
                    break
            if not found_keyword:
                # Fallback: split by space
                parts = text.rsplit(' ', 1)
                
        if len(parts) != 2:
            raise ValueError(f"Cannot parse: {text}")
            
        name = parts[0].strip().strip('"\'')
        change_type = parts[1].strip().strip('"\'')
        
        is_pos = any(k in change_type for k in pos_keywords)
        is_ori = any(k in change_type for k in ori_keywords)
        
        if not (is_pos or is_ori):
             raise ValueError(f"Unknown change type: {change_type}")

        return cls(name=name, pos=True if is_pos else None, ori=True if is_ori else None)

class RoomModifier:
    """Base class for room modifications."""
    def modify(self, room: Room) -> Tuple[Room, List[ChangedObject]]:
        raise NotImplementedError

class ObjectModifier(RoomModifier):
    """Modifies objects in the room by moving or rotating them."""
    
    def __init__(self, seed: int, n_changes: int = 1, modification_type: Optional[str] = None):
        """
        Args:
            seed: Random seed for modification
            n_changes: Number of objects to modify
            modification_type: 'move', 'rotate', or None (randomly choose)
        """
        self.np_random = np.random.default_rng(seed)
        self.n_changes = n_changes
        self.modification_type = modification_type

    def modify(self, room: Room) -> Tuple[Room, List[ChangedObject]]:
        """
        Randomly selects n objects and either moves them to a new valid position or rotates them.
        Returns the modified room and a list of changes.
        """
        modified_room = room.copy()
        changes_map: Dict[str, ChangedObject] = {}
        
        # Get all movable objects (exclude walls, doors, etc. if any)
        candidates = [obj for obj in modified_room.all_objects if obj.name != 'agent']
        
        if not candidates:
            return modified_room, []

        # Filter candidates for rotation if forced to rotate
        if self.modification_type == 'rotate':
            candidates = [obj for obj in candidates if obj.has_orientation]
            if not candidates:
                 return modified_room, []

        # Select n unique objects to modify
        n = min(self.n_changes, len(candidates))
        target_objs = self.np_random.choice(candidates, size=n, replace=False)
        
        for target_obj in target_objs:
            # Decide whether to move or rotate
            mod_type = self.modification_type or 'move'
            
            # If random (None), pick one.
            if self.modification_type is None:
                mod_type = self.np_random.choice(['move', 'rotate'])

            change_obj = None
            if mod_type == 'move':
                change_obj = self._move_object(modified_room, target_obj)
            elif mod_type == 'rotate':
                if target_obj.has_orientation:
                    change_obj = self._rotate_object(target_obj)
                else:
                    # Fallback to move if selected object cannot rotate
                    change_obj = self._move_object(modified_room, target_obj)
            
            if change_obj:
                if change_obj.name in changes_map:
                    changes_map[change_obj.name].merge(change_obj)
                else:
                    changes_map[change_obj.name] = change_obj
            
        return modified_room, list(changes_map.values())

    def _move_object(self, room: Room, obj: Object) -> Optional[ChangedObject]:
        prev_pos = tuple(float(x) for x in obj.pos)
        assert hasattr(room, 'mask') and room.mask is not None, "Room must have a mask"
        mask = room.mask
        valid_indices = np.argwhere((mask >= 1) & (mask < 100))
        occupied = {tuple(o.pos) for o in room.all_objects} # Includes obj itself, will filter later if needed, but simpler to just check collision
        
        # Find available positions
        # Optimization: Filter valid_indices directly
        available_pos = [tuple(pos) for pos in valid_indices if tuple(pos) not in occupied]
        
        if available_pos:
            new_pos = available_pos[self.np_random.choice(len(available_pos))]
            obj.pos = np.array(new_pos)
            # Update room_id based on new position's mask value
            obj.room_id = int(mask[new_pos[0], new_pos[1]])
            return ChangedObject(
                name=obj.name,
                pos=(prev_pos, tuple(float(x) for x in obj.pos))
            )
        else:
            # Fallback to rotate if possible
            if obj.has_orientation:
                return self._rotate_object(obj)
        return None

    def _rotate_object(self, obj: Object) -> Optional[ChangedObject]:
        rotations = [
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, -1]),
            np.array([-1, 0])
        ]
        current_ori_tuple = tuple(float(x) for x in obj.ori)
        possible_oris = [r for r in rotations if tuple(r) != tuple(obj.ori)]
        if possible_oris:
            new_ori = possible_oris[self.np_random.choice(len(possible_oris))]
            obj.ori = new_ori
            return ChangedObject(
                name=obj.name,
                ori=(current_ori_tuple, tuple(float(x) for x in obj.ori))
            )
        return None

if __name__ == "__main__":
    # Simple test case
    print("Running RoomModifier tests...")
    obj1 = Object(name="obj1", pos=np.array([1, 1]), has_orientation=True)
    obj2 = Object(name="obj2", pos=np.array([2, 2]), has_orientation=False)
    
    # Mock Room
    class MockRoom(Room):
        def __init__(self, objects, name="mock_room", mask=None, gates=None):
            self.name = name
            self.objects = objects
            self.all_objects = objects
            self.mask = mask if mask is not None else np.zeros((5, 5))
            if mask is None:
                self.mask[1:4, 1:4] = 1 # valid room area
            self.gates = gates or []
            
    room = MockRoom([obj1, obj2])
    
    modifier = ObjectModifier(seed=42, n_changes=2)
    mod_room, changes = modifier.modify(room)
    
    print(f"Changes: {changes}")
    assert len(changes) > 0
    print("Test passed!")
