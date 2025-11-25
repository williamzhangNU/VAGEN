import numpy as np
from typing import Tuple, Optional, List
from vagen.env.spatial.Base.tos_base import Room, Object

class RoomModifier:
    """Base class for room modifications."""
    def modify(self, room: Room, np_random: np.random.Generator) -> Tuple[Room, str]:
        raise NotImplementedError

class SingleObjectModifier(RoomModifier):
    """Modifies a single object in the room by moving or rotating it."""
    
    def __init__(self, modification_type: Optional[str] = None):
        """
        Args:
            modification_type: 'move', 'rotate', or None (randomly choose)
        """
        self.modification_type = modification_type

    def modify(self, room: Room, np_random: np.random.Generator) -> Tuple[Room, str]:
        """
        Randomly selects an object and either moves it to a new valid position or rotates it.
        Returns the modified room and the name of the modified object.
        """
        modified_room = room.copy()
        
        # Get all movable objects (exclude walls, doors, etc. if any)
        candidates = [obj for obj in modified_room.all_objects if obj.name != 'agent']
        
        if not candidates:
            return modified_room, ""

        # Filter candidates for rotation if forced to rotate
        if self.modification_type == 'rotate':
            candidates = [obj for obj in candidates if obj.has_orientation]
            assert candidates, "No rotatable objects found"

        target_obj = np_random.choice(candidates)
        
        # Decide whether to move or rotate
        mod_type = self.modification_type or 'move'
        
        if mod_type == 'move':
            self._move_object(modified_room, target_obj, np_random)
        elif mod_type == 'rotate':
            if target_obj.has_orientation:
                self._rotate_object(target_obj, np_random)
            else:
                # Fallback to move if selected object cannot rotate (shouldn't happen with logic above but safe guard)
                self._move_object(modified_room, target_obj, np_random)
            
        return modified_room, target_obj.name

    def _move_object(self, room: Room, obj: Object, np_random: np.random.Generator):
        if hasattr(room, 'mask') and room.mask is not None:
            mask = room.mask
            valid_indices = np.argwhere((mask >= 1) & (mask < 100))
            occupied_positions = {tuple(o.pos) for o in room.all_objects if o.name != obj.name}
            occupied_positions.add(tuple(obj.pos)) # exclude current
            
            available_pos = []
            for idx in valid_indices:
                pos = tuple(idx)
                if pos not in occupied_positions:
                    available_pos.append(pos)
            
            if available_pos:
                new_pos = available_pos[np_random.choice(len(available_pos))]
                obj.pos = np.array(new_pos)
            else:
                # Fallback to rotate if possible
                if obj.has_orientation:
                    self._rotate_object(obj, np_random)
        else:
             if obj.has_orientation:
                self._rotate_object(obj, np_random)

    def _rotate_object(self, obj: Object, np_random: np.random.Generator):
        rotations = [
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, -1]),
            np.array([-1, 0])
        ]
        current_ori_tuple = tuple(obj.ori)
        possible_oris = [r for r in rotations if tuple(r) != current_ori_tuple]
        if possible_oris:
            new_ori = possible_oris[np_random.choice(len(possible_oris))]
            obj.ori = new_ori

if __name__ == "__main__":
    # Simple test case
    print("Running RoomModifier tests...")
    obj1 = Object(name="obj1", pos=np.array([1, 1]), has_orientation=True)
    obj2 = Object(name="obj2", pos=np.array([2, 2]), has_orientation=False)
    
    # Mock Room
    class MockRoom(Room):
        def __init__(self, objects):
            self.objects = objects
            self.mask = np.zeros((5, 5))
            self.mask[1:4, 1:4] = 1 # valid room area
            self.gates = []
            
    room = MockRoom([obj1, obj2])
    rng = np.random.default_rng(42)
    
    modifier = SingleObjectModifier()
    mod_room, target = modifier.modify(room, rng)
    
    print(f"Modified object: {target}")
    orig_obj = next(o for o in room.all_objects if o.name == target)
    mod_obj = next(o for o in mod_room.all_objects if o.name == target)
    
    print(f"Original pos: {orig_obj.pos}, ori: {orig_obj.ori}")
    print(f"Modified pos: {mod_obj.pos}, ori: {mod_obj.ori}")
    
    assert not (np.array_equal(orig_obj.pos, mod_obj.pos) and np.array_equal(orig_obj.ori, mod_obj.ori)), "Object should be modified"
    print("Test passed!")
