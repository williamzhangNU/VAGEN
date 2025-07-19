import numpy as np
from typing import Dict, Any
from vagen.env.spatial.Base.tos_base.utils.room_utils import Room, Object, Agent

# for initializing visual-based
def initialize_room_from_json(json_data: Dict[str, Any]) -> Room:
    """
    Initialize a Room from your metadata JSON, which now has:
      - objects: list of {oid, model, pos:{x,y,z}, rot:{x,y,z}, size:[w,h]}
      - cameras: list of {id, label, position:{x,y,z}, rotation:{y}}
      - room_size, screen_size, etc.
    """
    # Rotation to orientation vector mapping
    rotation_map = {0: np.array([0, 1]), 90: np.array([1, 0]), 180: np.array([0, -1]), 270: np.array([-1, 0])}
    
    # 1) Parse all objects
    objects = []
    for obj in json_data['objects']:
        objects.append(Object(
            name=obj['name'],
            pos=np.array([obj["pos"]["x"], obj["pos"]["z"]]),
            ori=rotation_map.get(obj["rot"]["y"]),
            has_orientation=True
        ))
    # 2) Room size metadata
    room_name = json_data.get("name", "room_from_json")

    # 3) Build and return
    agent_pos = [camera['position'] for camera in json_data['cameras'] if camera['id'] == 'agent'][0]
    agent_pos = np.array([agent_pos["x"], agent_pos["z"]])
    for obj in objects:
        obj.pos -= agent_pos
    return Room(objects=objects, name=room_name, agent=Agent())