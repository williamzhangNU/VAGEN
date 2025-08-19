import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from omegaconf import DictConfig, ListConfig, OmegaConf

from vagen.env.spatial.utils.visualization import visualize_json
import numpy as np
from vagen.env.spatial.Base.tos_base.core.object import Agent
from vagen.env.spatial.Base.tos_base.utils.room_utils import RoomPlotter
from vagen.env.spatial.Base.tos_base import ExplorationManager, EvaluationManager, CognitiveMapManager, Room


class SpatialEnvLogger:
    """Logger for spatial environment data aggregation and visualization."""
    
    @staticmethod
    def _convert_omegaconf_to_python(obj):
        """Recursively convert OmegaConf objects to standard Python types for JSON serialization."""
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {key: SpatialEnvLogger._convert_omegaconf_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [SpatialEnvLogger._convert_omegaconf_to_python(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _plot_room(room_dict: Dict, agent_dict: Dict, out_dir: str, config_name: str, sample_idx: int, turn_idx: int) -> Optional[str]:
        """Plot room from state and return image filename"""
        img_folder = os.path.join(out_dir, "images", config_name, f"sample_{sample_idx+1}")
        os.makedirs(img_folder, exist_ok=True)

        img_name = f"room_turn_{turn_idx+1}.png" if turn_idx > 0 else "room_initial.png"
        img_path = os.path.join(img_folder, img_name)
        RoomPlotter.plot(Room.from_dict(room_dict), Agent.from_dict(agent_dict), mode='img', save_path=img_path)

        return os.path.join("images", config_name, f"sample_{sample_idx+1}", img_name)

    @staticmethod
    def _validate_and_assign_messages(message: List[Dict], turn_logs: List[Dict]) -> bool:
        """Validate message structure and assign raw assistant messages to turn logs."""
        if not message:
            return False
        
        # Remove system messages and check turn structure
        filtered_msgs = [msg for msg in message if msg.get("role") != "system"]
        
        # Check alternating user/assistant pattern
        for i in range(0, len(filtered_msgs), 2):
            if i >= len(filtered_msgs) or filtered_msgs[i].get("role") != "user":
                print(f"User message missing at index {i}")
                return False
            if i + 1 >= len(filtered_msgs) or filtered_msgs[i + 1].get("role") != "assistant":
                print(f"Assistant message missing at index {i+1}")
                return False
        
        # Assign raw assistant messages to turn logs
        assistant_messages = [msg['content'] for msg in message if msg.get("role") == "assistant"]
        
        if len(assistant_messages) != len(turn_logs):
            print(f"Mismatch: {len(assistant_messages)} assistant messages vs {len(turn_logs)} turns")
            return False
        
        return True

    @staticmethod
    def _aggregate_env_data(env_summaries: List[Dict], messages: List[List[Dict]], output_dir: str, save_images: bool = True, **kwargs) -> Dict:
        """
        Aggregate environment data and create visualization.
        
        Args:
            env_summaries: List of environment summary dictionaries
            messages: List of message conversations for each environment
            output_dir: Output directory for saving results
            **kwargs: Additional arguments including model_name
        
        Returns:
            Aggregated data dictionary
        """
        # Group environments by config name
        config_groups = defaultdict(list)
        
        for env_summary, message in zip(env_summaries, messages):
            config_name = env_summary['env_info']['config']['name']
            
            # Validate and assign raw messages
            turn_logs = [turn_log for turn_log in env_summary.get('env_turn_logs', [])]
            if not SpatialEnvLogger._validate_and_assign_messages(message, turn_logs):
                continue
            
            env_data = {**env_summary, "message": message}
            config_groups[config_name].append(env_data)
        
        # Plot rooms and add image paths if requested
        if save_images:
            for config_name, group in config_groups.items():
                for sample_idx, env_data in enumerate(group):
                    # Plot initial room
                    initial_img_path = SpatialEnvLogger._plot_room(env_data["env_info"]["initial_room"], env_data["env_info"]["initial_agent"], output_dir, config_name, sample_idx, 0)
                    env_data["initial_room_image"] = initial_img_path

                    # Plot room for each turn
                    for turn_log in env_data["env_turn_logs"]:
                        if turn_log["room_state"]:
                            turn_idx = turn_log["turn_number"]
                            img_path = SpatialEnvLogger._plot_room(turn_log["room_state"], turn_log["agent_state"], output_dir, config_name, sample_idx, turn_idx)
                            turn_log["room_image"] = img_path
                        
                        # Find corresponding user message and save its images
                        message = env_data["message"]
                        img_folder = os.path.join(output_dir, "images", config_name, f"sample_{sample_idx+1}")
                        user_msg_idx = (turn_idx * 2) if message and message[0]['role'] == 'user' else turn_idx * 2 + 1
                        if user_msg_idx < len(message) and 'multi_modal_data' in message[user_msg_idx]:
                            message_images = {}
                            for key, images in message[user_msg_idx]['multi_modal_data'].items():
                                if 'image' in key.lower() and images:
                                    image_paths = []
                                    for img_idx, img in enumerate(images):
                                        img_name = f"obs_turn_{turn_idx+1}_{img_idx}.png"
                                        img.save(os.path.join(img_folder, img_name))
                                        image_paths.append(os.path.join("images", config_name, f"sample_{sample_idx+1}", img_name)) # relative path for visualization
                                    message_images[key] = image_paths
                            turn_log['message_images'] = message_images
                        env_data["message"] = [{k: v for k, v in msg.items() if k != 'multi_modal_data'} for msg in message]

        # Initialize result structure
        result = {
            "config_groups": dict(config_groups),
            "exp_summary": {"overall_performance": {}, "group_performance": {}},
            "eval_summary": {"overall_performance": {}, "group_performance": {}},
            "cogmap_summary": {"overall_performance": {}, "group_performance": {}}
        }
        
        # Calculate performance metrics
        all_exp_data, all_eval_data, all_cogmap_data = [], [], []
        
        for config_name, env_data_list in config_groups.items():
            result["config_groups"][config_name] = {"env_data": env_data_list}
            
            exp_summaries = [d['summary']['exp_summary'] for d in env_data_list]
            eval_summaries = [d['summary']['eval_summary'] for d in env_data_list]
            cogmap_summaries = [d['summary']['cogmap_summary'] for d in env_data_list]

            result["exp_summary"]["group_performance"][config_name] = ExplorationManager.aggregate_group_performance(exp_summaries)
            result["eval_summary"]["group_performance"][config_name] = EvaluationManager.aggregate_group_performance(eval_summaries)
            result["cogmap_summary"]["group_performance"][config_name] = CognitiveMapManager.aggregate_group_performance(cogmap_summaries)
            
            all_exp_data.extend(exp_summaries)
            all_eval_data.extend(eval_summaries)
            all_cogmap_data.extend(cogmap_summaries)

        # Calculate overall performance
        if all_exp_data:
            result["exp_summary"]["overall_performance"] = ExplorationManager.aggregate_group_performance(all_exp_data)
        if all_eval_data:
            result["eval_summary"]["overall_performance"] = EvaluationManager.aggregate_group_performance(all_eval_data)
        if all_cogmap_data:
            result["cogmap_summary"]["overall_performance"] = CognitiveMapManager.aggregate_group_performance(all_cogmap_data)

        return result

    @staticmethod
    def _save_data(aggregated_data: Dict, output_dir: str, **kwargs):
        """Save aggregated data to JSON and generate HTML dashboard."""
        saved_data = {
            'meta_info': {
                'model_name': kwargs.get('model_name', 'unknown'),
                'n_envs': sum(len(group['env_data']) for group in aggregated_data['config_groups'].values()),
            },
            **aggregated_data,
        }

        # Convert OmegaConf objects to standard Python types
        saved_data = SpatialEnvLogger._convert_omegaconf_to_python(saved_data)
        
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(os.path.join(output_dir, "env_data.json"), "w") as f:
            json.dump(saved_data, f, indent=2)

        # Generate HTML dashboard
        html_path = os.path.join(output_dir, "env_data.html")
        dashboard_path = visualize_json(os.path.join(output_dir, "env_data.json"), html_path, True)
        
        print(f"Environment data logged to {output_dir}")
        print(f"Dashboard written to {dashboard_path}")
        return output_dir
    


    @staticmethod
    def log_each_env_info(env_summaries: List[Dict], messages: List[Dict], output_dir: str, save_images: bool = True, **kwargs):
        """Logs detailed information for each environment and overall performance metrics."""

        # Aggregate data using the logger
        aggregated_data = SpatialEnvLogger._aggregate_env_data(
            env_summaries=env_summaries,
            messages=messages,
            output_dir=output_dir,
            save_images=save_images,
        )
        
        # Save aggregated data
        return SpatialEnvLogger._save_data(aggregated_data, output_dir, **kwargs)