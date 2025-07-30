import json
import os
from typing import List, Dict, Optional

from vagen.env.spatial.env import SpatialGym
from vagen.env.spatial.utils.visualization import visualize_json
from vagen.env.spatial.Base.tos_base.core.room import Room
from vagen.env.spatial.Base.tos_base.utils.room_utils import set_initial_pos_as_origin

def plot_room(room_dict: Dict, out_dir: str, env_idx: int, turn_idx: int) -> Optional[str]:
    """Plot room from state and return image filename"""
    img_folder = os.path.join(out_dir, "images", f"env_{env_idx}")
    os.makedirs(img_folder, exist_ok=True)
    
    room = Room.from_dict(room_dict)
    transformed_room = set_initial_pos_as_origin(room)
    
    img_name = f"turn_{turn_idx+1}.png" if turn_idx > 0 else "initial_room.png"
    img_path = os.path.join(img_folder, img_name)
    transformed_room.plot(render_mode='img', save_path=img_path)
    
    return os.path.join("images", f"env_{env_idx}", img_name)

def save_results_to_disk(env_summaries: List[Dict], messages: List[List[Dict]], output_dir: str, **kwargs) -> None:
    """
    Save results to disk with image handling and create visualization.
    Args:
        env_summaries: List of env summary for each env
        messages: List of messages for each env
        output_dir: Directory to save results
        **kwargs: Additional keyword arguments
    """
    model_name = kwargs.get('model_name', 'unknown_model')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process results and save images from messages
    env_results = []
    for i, (env_summary, messages) in enumerate(zip(env_summaries, messages)):
        image_dir = os.path.join(output_dir, "images", f"env_{i}")
        os.makedirs(image_dir, exist_ok=True)
        env_summary['messages'] = messages
        turn_logs = env_summary.get('env_turn_logs', [])
        
        # Process message images and map to turns
        for turn_idx, turn_log in enumerate(turn_logs):
            # Save room image
            if turn_log.get('room_state'):
                room_img_path = plot_room(turn_log['room_state'], output_dir, i, turn_idx)
                turn_log['room_image'] = room_img_path
            
            # Find corresponding user message and save its images
            user_msg_idx = (turn_idx * 2) if messages and messages[0]['role'] == 'user' else turn_idx * 2 + 1
            if user_msg_idx < len(messages) and 'multi_modal_data' in messages[user_msg_idx]:
                message_images = {}
                for key, images in messages[user_msg_idx]['multi_modal_data'].items():
                    if 'image' in key.lower() and images:
                        image_paths = []
                        for img_idx, img in enumerate(images):
                            img_path = os.path.join(image_dir, f"turn_{turn_idx+1}_{img_idx}.png")
                            img.save(img_path)
                            image_paths.append(os.path.relpath(img_path, output_dir))
                        message_images[key] = image_paths
                turn_log['message_images'] = message_images
        
        # Plot initial room image
        if 'env_info' in env_summary and 'initial_room' in env_summary['env_info']:
            initial_room_img_path = plot_room(env_summary['env_info']['initial_room'], output_dir, i, -1)
            env_summary['initial_room_image'] = initial_room_img_path
        
        env_results.append(env_summary)
    
    # Aggregate results
    aggregated_data = SpatialGym.aggregate_env_data(env_results)
    aggregated_data['meta_info'] = {'model_name': model_name}
    
    # Save JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(aggregated_data, f, indent=2, default=str)
    
    # Create visualization
    html_path = os.path.join(output_dir, "visualization.html")
    visualize_json(json_path, html_path, plot_rooms=True)
    
    print(f"Results saved to: {output_dir}")
    print(f"Visualization: {html_path}")