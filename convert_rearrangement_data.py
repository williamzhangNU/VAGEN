#!/usr/bin/env python3
"""
Convert AI2-THOR Rearrangement data to VAGEN format
"""

import compress_pickle
import json
import os
from typing import Dict, List, Any

def generate_instruction(task_data: Dict[str, Any]) -> str:
    """Generate natural language instruction for rearrangement task"""
    
    # Count objects that need to be rearranged
    rearrange_count = task_data.get('object_rearrangement_count', 0)
    
    # Find objects that have changed positions
    starting_poses = task_data.get('starting_poses', [])
    target_poses = task_data.get('target_poses', [])
    
    changed_objects = []
    for start, target in zip(starting_poses, target_poses):
        if start != target:
            obj_name = start.get('name', '').split('_')[0]  # Remove instance ID
            changed_objects.append(obj_name)
    
    # Find objects that need opening/closing
    openable_data = task_data.get('openable_data', [])
    openable_changes = []
    for openable in openable_data:
        start_open = openable.get('start_openness', 0)
        target_open = openable.get('target_openness', 0)
        if abs(start_open - target_open) > 0.1:  # Significant change
            obj_name = openable.get('name', '').split('_')[0]
            if target_open > start_open:
                openable_changes.append(f"open the {obj_name}")
            else:
                openable_changes.append(f"close the {obj_name}")
    
    # Generate instruction based on task complexity
    if rearrange_count == 0 and len(openable_changes) == 0:
        return "observe the room and complete any necessary rearrangement tasks"
    
    instruction_parts = []
    
    # Add object rearrangement instructions
    if len(changed_objects) == 1:
        instruction_parts.append(f"rearrange the {changed_objects[0]} to its correct position")
    elif len(changed_objects) > 1:
        if len(changed_objects) <= 3:
            obj_list = ", ".join(changed_objects[:-1]) + f" and {changed_objects[-1]}"
            instruction_parts.append(f"rearrange the {obj_list} to their correct positions")
        else:
            instruction_parts.append(f"rearrange {len(changed_objects)} objects to their correct positions")
    
    # Add openable object instructions
    if len(openable_changes) > 0:
        instruction_parts.extend(openable_changes)
    
    # Combine instructions
    if len(instruction_parts) == 1:
        instruction = instruction_parts[0]
    elif len(instruction_parts) == 2:
        instruction = f"{instruction_parts[0]} and {instruction_parts[1]}"
    else:
        instruction = ", ".join(instruction_parts[:-1]) + f", and {instruction_parts[-1]}"
    
    # Add context
    instruction = f"In this room rearrangement task, {instruction}. " \
                 f"First observe the target state, then rearrange objects to match that state."
    
    return instruction

def convert_task_to_vagen_format(task_data: Dict[str, Any], scene: str, task_index: int) -> Dict[str, Any]:
    """Convert a single AI2-THOR rearrangement task to VAGEN format"""
    
    # Generate instruction
    instruction = generate_instruction(task_data)
    
    # Extract agent pose
    agent_position = task_data.get('agent_position', {})
    agent_rotation = task_data.get('agent_rotation', 0)
    
    # Find the primary object to rearrange (for targetObjectType)
    starting_poses = task_data.get('starting_poses', [])
    target_poses = task_data.get('target_poses', [])
    
    target_object_type = "Object"  # Default
    target_object_ids = ""
    target_position = {"x": 0, "y": 0, "z": 0}
    
    # Find first object that needs rearrangement
    for start, target in zip(starting_poses, target_poses):
        if start != target:
            target_object_type = start.get('name', '').split('_')[0]
            target_object_ids = start.get('name', '')
            target_position = target.get('position', {"x": 0, "y": 0, "z": 0})
            break
    
    # If no objects need rearrangement, use first object
    if target_object_type == "Object" and starting_poses:
        first_obj = starting_poses[0]
        target_object_type = first_obj.get('name', '').split('_')[0]
        target_object_ids = first_obj.get('name', '')
        target_position = first_obj.get('position', {"x": 0, "y": 0, "z": 0})
    
    # Create VAGEN format task
    vagen_task = {
        "targetObjectType": target_object_type,
        "targetObjectIds": target_object_ids,
        "target_position": target_position,
        "agentPose": {
            "position": agent_position,
            "rotation": agent_rotation,
            "horizon": 0.0  # Default horizon
        },
        "scene": scene,
        "object_to_hide": [],  # No objects to hide in rearrangement
        "instruction": instruction,
        # Additional rearrangement-specific fields
        "rearrangement_data": {
            "object_rearrangement_count": task_data.get('object_rearrangement_count', 0),
            "position_diff_count": task_data.get('position_diff_count', 0),
            "open_diff_count": task_data.get('open_diff_count', 0),
            "pose_diff_energy": task_data.get('pose_diff_energy', 0.0),
            "starting_poses": starting_poses,
            "target_poses": target_poses,
            "openable_data": task_data.get('openable_data', []),
            "task_index": task_index
        }
    }
    
    return vagen_task

def convert_rearrangement_to_vagen():
    """Convert AI2-THOR rearrangement data to VAGEN format"""
    
    # Load AI2-THOR rearrangement data
    input_path = "/home/zihanhuang/ai2thor-rearrangement/data/2023/combined.pkl.gz"
    output_path = "/home/zihanhuang/VAGEN/vagen/env/rearrangement/datasets/base.json"
    
    print(f"Loading AI2-THOR rearrangement data from: {input_path}")
    
    try:
        with open(input_path, 'rb') as f:
            rearrange_data = compress_pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loaded data with {len(rearrange_data)} scenes")
    
    # Convert to VAGEN format
    vagen_tasks = []
    
    for scene_name, episodes in rearrange_data.items():
        print(f"Processing scene {scene_name} with {len(episodes)} episodes...")
        
        for task_index, task_data in enumerate(episodes):
            try:
                vagen_task = convert_task_to_vagen_format(task_data, scene_name, task_index)
                vagen_tasks.append(vagen_task)
            except Exception as e:
                print(f"Error converting task {task_index} in scene {scene_name}: {e}")
                continue
    
    # Create output structure
    output_data = {
        "tasks": vagen_tasks
    }
    
    print(f"Converted {len(vagen_tasks)} tasks total")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    print(f"Saving to: {output_path}")
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✅ Successfully saved {len(vagen_tasks)} tasks to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        return
    
    # Print some statistics
    print(f"\n📊 Conversion Statistics:")
    print(f"Total tasks: {len(vagen_tasks)}")
    
    # Count by scene type
    scene_types = {}
    for task in vagen_tasks:
        scene = task['scene']
        scene_type = "Kitchen" if scene.startswith('FloorPlan') and int(scene[9:]) <= 30 else \
                    "Living Room" if scene.startswith('FloorPlan') and int(scene[9:]) <= 230 else \
                    "Bedroom" if scene.startswith('FloorPlan') and int(scene[9:]) <= 330 else \
                    "Bathroom"
        scene_types[scene_type] = scene_types.get(scene_type, 0) + 1
    
    for scene_type, count in scene_types.items():
        print(f"{scene_type}: {count} tasks")
    
    # Count by rearrangement complexity
    complexity_counts = {}
    for task in vagen_tasks:
        count = task['rearrangement_data']['object_rearrangement_count']
        complexity_counts[count] = complexity_counts.get(count, 0) + 1
    
    print(f"\nRearrangement complexity:")
    for count in sorted(complexity_counts.keys()):
        print(f"{count} objects: {complexity_counts[count]} tasks")

if __name__ == "__main__":
    convert_rearrangement_to_vagen()
