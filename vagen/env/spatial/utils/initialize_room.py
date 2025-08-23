import numpy as np
from typing import Dict, Any, Tuple
from vagen.env.spatial.Base.tos_base.utils.room_utils import Room, Object, Agent, RoomGenerator

# for initializing visual-based
def initialize_room_from_json(json_data: Dict[str, Any], mask: np.ndarray) ->  Tuple[Room, Agent]:
    """
    Initialize a Room from your metadata JSON, which now has:
      - objects: list of {oid, model, pos:{x,y,z}, rot:{x,y,z}, size:[w,h]}
      - cameras: list of {id, label, position:{x,y,z}, rotation:{y}}
      - room_size, screen_size, etc.
    """
    # Rotation to orientation vector mapping
    rotation_map = {0: np.array([0, 1]), 90: np.array([1, 0]), 180: np.array([0, -1]), 270: np.array([-1, 0])}
    offset = np.array((8,7))
    # 1) Parse all objects
    objects = []
    for obj in json_data['objects']:
        if 'door' not in obj['name']:
            objects.append(Object(
                name=obj['name'],
                pos=np.array([obj["pos"]["x"], obj["pos"]["z"]]),
                ori=rotation_map.get(obj["rot"]["y"]) if obj["attributes"]["has_orientation"] else np.array([1, 0]),
                has_orientation=obj["attributes"]["has_orientation"]
            ))
        
    # 2) Room size metadata
    room_name = json_data.get("name", "room_from_json")

    # 3) Build and return
    agent_pos = [camera['position'] for camera in json_data['cameras'] if camera['id'] == 'agent'][0]
    agent_pos = np.array([agent_pos["x"], agent_pos["z"]])
    for obj in objects:
        obj.pos +=offset
    agent_pos +=offset
    gates = RoomGenerator._gen_gates_from_mask(mask)

    # Update gate room_ids to match the connected rooms from JSON data
    door_objects = [obj for obj in json_data['objects'] if 'door' in obj['name']]
    for gate in gates:
        for door in door_objects:
            if set(gate.room_id) == set(door["attributes"]['connected_rooms']):
                gate.name = door['name']

    return Room(objects=objects, mask=mask, name=room_name, gates=gates), Agent(pos=agent_pos,room_id=1,init_room_id=1)
