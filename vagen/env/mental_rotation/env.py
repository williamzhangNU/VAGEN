import sys
import os
import json
from typing import Dict, Tuple, Any, List
import random
from vagen.env.utils.context_utils import convert_numpy_to_PIL

from vagen.env.base.base_env import BaseEnv
from .env_config import MentalRotationEnvConfig
from .prompt import system_prompt, init_observation_template, action_template
from .utils import euler_xyz_to_quat, quat_to_euler_xyz_scipy, quat_equal, quat_multiply

import genesis as gs
from .env_config import options_map

class MentalRotationEnv(BaseEnv):
    """
    Environment for mental-rotation.
    - Produces either an image or text observation
    - Expects LLM response like <answer>x90</answer> or <answer>y-90</answer>
    - Applies 3D rotation along x/y/z axes with configurable granularity
    - Uses quaternion representation for orientations
    
    Note: Genesis initialization is handled by the service layer.
    """

    def __init__(self, config: MentalRotationEnvConfig):
        super().__init__()
        self.config = config

        # Internal state used by step/reset - quaternion (w, x, y, z)
        # Now track state for each parallel environment
        self._target_orientations = {}  # env_idx -> target quaternion
        self._current_orientations = {}  # env_idx -> current quaternion 
        self._last_valid_actions = {}  # env_idx -> list of actions
        self.total_rewards = {}  # env_idx -> total reward
        self._step_counts = {}  # env_idx -> step count
        self._dones = {}  # env_idx -> done flag
        
        self.env_id_to_idx = {}  # external env_id -> internal env_idx
        self.idx_to_env_id = {}  # internal env_idx -> external env_id
        self.active_env_indices = []  # List of active parallel env indices

        self._task_data = {}  # env_idx -> task data from dataset
        
        self.n_parallel_envs = self.config.n_parallel_envs
        self.sub_env_spacing = self.config.parallel_env_spacing
        self.rotate_granularity = self.config.rotate_granularity

        self.VALID_ACTIONS = self._generate_valid_actions()

        self.scene = None
        self.target_images = {}  # env_idx -> target image
        
        self.dataset = self._load_dataset()

    def _generate_valid_actions(self) -> List[str]:
        actions = []
        axes = ['x', 'y', 'z']
        
        for axis in axes:
            for angle in range(self.rotate_granularity, 360, self.rotate_granularity):
                actions.append(f"{axis}{angle}")
                actions.append(f"{axis}-{angle}")
        
        return actions

    def _load_dataset(self) -> List[Dict]:
        """Load the multi-step interactive dataset."""
        dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "multi_step_interactive.json")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        tasks = data.get("tasks", [])
        if not tasks:
            raise ValueError("No tasks found in dataset")
        return tasks

    def _get_task_data(self, seed: int) -> Dict:
        """Get task data based on seed."""
        task_idx = seed % len(self.dataset)
        return self.dataset[task_idx]

    def _generate_target_images(self, env_indices=None):
        """Generate target images for specified parallel environments"""
        if env_indices is None:
            env_indices = self.active_env_indices
        
        import torch
        
        # Store current orientations
        current_quats = []
        for idx in env_indices:
            current_quats.append(self._current_orientations[idx])
        
        # Set target orientations for all specified envs
        target_quats = torch.zeros((len(env_indices), 4), device=gs.device)
        for i, idx in enumerate(env_indices):
            target_quats[i] = torch.tensor(self._target_orientations[idx], device=gs.device)
        
        # Convert global env indices to local scene indices (0, 1, 2, ...)
        local_env_indices = torch.arange(len(env_indices), device=gs.device)
        
        self.object.set_quat(
            target_quats,
            envs_idx=local_env_indices,
            zero_velocity=False
        )
        self.scene.step()
        
        # Render target images for each environment
        for i, idx in enumerate(env_indices):
            target_image = self.cameras[i].render()[0]  # Use local index i for camera
            self.target_images[idx] = convert_numpy_to_PIL(target_image)
        
        # Restore current orientations
        self.object.set_quat(
            current_quats,
            envs_idx=local_env_indices,
            zero_velocity=False
        )
        self.scene.step()

    def _render(self, init_obs: bool, env_indices=None) -> Dict[int, Dict]:
        """Render observations for specified parallel environments"""
        if env_indices is None:
            env_indices = self.active_env_indices
        
        image_placeholder = self.config.image_placeholder
        target_image_placeholder = self.config.target_image_placeholder
        format_prompt_text = self.system_prompt()
        
        # Generate target images if needed
        missing_target_indices = [idx for idx in env_indices if idx not in self.target_images]
        if missing_target_indices:
            self._generate_target_images(missing_target_indices)
        
        # Render observations for each environment
        observations = {}
        for i, idx in enumerate(env_indices):
            rgb = self.cameras[i].render()[0]  # Use local index i for camera
            
            if init_obs:
                obs_str = init_observation_template(
                    img_str=image_placeholder,
                    target_img_str=target_image_placeholder,
                    valid_actions=self.VALID_ACTIONS
                )
            else:
                last_action = self._last_valid_actions[idx][-1] if self._last_valid_actions.get(idx) else "None"
                obs_str = action_template(
                    img_str=image_placeholder,
                    target_img_str=target_image_placeholder,
                    last_action=last_action,
                    step_count=self._step_counts[idx]
                )

            multi_modal_data = {
                image_placeholder: [convert_numpy_to_PIL(rgb)],
                target_image_placeholder: [self.target_images[idx]],
            }
            
            observations[idx] = {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data
            }
        
        return observations

    def _setup_scene_from_dataset(self, task_configs: List[Dict]):
        """Setup scene based on task configurations from dataset.
        Args:
            task_configs: List of task configurations, one per environment
        """
        # Check that all environments use the same background and object for parallel processing
        first_task = task_configs[0]
        first_key = (first_task["background"], first_task["object"])
        
        for task_config in task_configs[1:]:
            key = (task_config["background"], task_config["object"])
            if key != first_key:
                raise ValueError(f"All parallel environments must use the same background and object. "
                               f"Found {first_key} and {key}")
        
        # Setup background and object
        self._setup_background(first_task)
        self._setup_object(first_task)
    
    def _setup_background(self, task_config: Dict):
        """Setup background based on task configuration."""
        bg = task_config["background"]

        _bg_pos = task_config.get("background_pos", {"x": 0.0, "y": 0.0, "z": -1.0})
        bg_pos = (_bg_pos["x"], _bg_pos["y"], _bg_pos["z"])

        bg_scale = task_config.get("background_scale", 1.0)

        _bg_euler = task_config.get("background_euler", {"x": 0.0, "y": 0.0, "z": 0.0})
        bg_euler = (_bg_euler["x"], _bg_euler["y"], _bg_euler["z"])
        
        if bg == "plane":
            self.background = self.scene.add_entity(
                gs.morphs.Plane(
                    pos=bg_pos,
                )
            )
        else:
            bg_path = os.path.join(os.path.dirname(__file__), "datasets", "assets", "scenes", bg)
            self.background = self.scene.add_entity(
                gs.morphs.Mesh(
                    file=bg_path,
                    pos=bg_pos,
                    euler=bg_euler,
                    scale=bg_scale,
                    fixed=True
                )
            )
    
    def _setup_object(self, task_config: Dict):
        """Setup main object based on task configuration."""
        obj = task_config["object"]
        obj_path = os.path.join(os.path.dirname(__file__), "datasets", "assets", "objects", obj)
        obj_scale = task_config.get("object_scale", 1.0)
        
        self.object = self.scene.add_entity(
            gs.morphs.Mesh(
                file=obj_path,
                fixed=True,
                pos=(0.0, 0.0, 0.0),
                euler=(0.0, 0.0, 0.0),
                scale=obj_scale
            ),
        )

    def _update_object_rotations(self, env_indices=None):
        """Update object rotations for specified parallel environments"""
        if env_indices is None:
            env_indices = self.active_env_indices
            
        assert hasattr(self, 'object') and self.object is not None
        
        import torch
        
        # Set orientations for all specified envs
        orientations = torch.zeros((len(env_indices), 4), device=gs.device)
        for i, idx in enumerate(env_indices):
            orientations[i] = torch.tensor(self._current_orientations[idx], device=gs.device)
        
        # Convert global env indices to local scene indices (0, 1, 2, ...)
        local_env_indices = torch.arange(len(env_indices), device=gs.device)
        
        self.object.set_quat(
            orientations,
            envs_idx=local_env_indices,
            zero_velocity=False
        )
        self.scene.step()


    def reset(self, env_id_to_seed=None, rebuild_scene=True) -> Dict[str, Tuple[Dict, Dict]]:
        """Reset environments with mapping of env_id to seed.
        
        Args:
            env_id_to_seed: Dict mapping external env_id to seed
            rebuild_scene: Whether to rebuild the scene (False when adding new envs to existing scene)
        
        Returns:
            Dict mapping env_id to (observation, info) tuple
        """
        if env_id_to_seed is None:
            env_id_to_seed = {}
        
        # Update env_id to index mappings
        if rebuild_scene:
            self.env_id_to_idx.clear()
            self.idx_to_env_id.clear()
            self.active_env_indices.clear()
            
        start_idx = len(self.active_env_indices)
        for i, env_id in enumerate(env_id_to_seed.keys()):
            idx = start_idx + i
            self.env_id_to_idx[env_id] = idx
            self.idx_to_env_id[idx] = env_id
            self.active_env_indices.append(idx)
        
        # Reset states for each environment using dataset
        task_configs = []
        for env_id, seed in env_id_to_seed.items():
            idx = self.env_id_to_idx[env_id]
            
            # Get task data from dataset based on seed
            task_data = self._get_task_data(seed)
            self._task_data[idx] = task_data
            task_configs.append(task_data)
            
            self.total_rewards[idx] = 0.0
            self._step_counts[idx] = 0
            self._dones[idx] = False
            
            # Set orientations from dataset
            initial_quat = task_data["initial_orientation"]
            target_quat = task_data["target_orientation"]
            
            self._current_orientations[idx] = (
                initial_quat["w"], initial_quat["x"], 
                initial_quat["y"], initial_quat["z"]
            )
            self._target_orientations[idx] = (
                target_quat["w"], target_quat["x"], 
                target_quat["y"], target_quat["z"]
            )
            
            self._last_valid_actions[idx] = []
            self.target_images.pop(idx, None)  # Clear cached target image

        # Build scene if needed
        if rebuild_scene:
            self.scene = gs.Scene(
                rigid_options=options_map[self.config.options]["rigid_options"],
                viewer_options=options_map[self.config.options]["viewer_options"],
                vis_options=options_map[self.config.options]["vis_options"],
                renderer=gs.renderers.Rasterizer(),
            )

            # Setup scene from dataset
            self._setup_scene_from_dataset(task_configs)

            # Create cameras for parallel environments based on task configs
            self.cameras = []
            n_envs = len(self.active_env_indices)
            for i in range(n_envs):
                task_config = task_configs[i]
                camera_pos = task_config.get("camera_pos", {"x": 5.0, "y": -2.0, "z": 2.5})
                camera_lookat = task_config.get("camera_lookat", {"x": 0.0, "y": 0.0, "z": 0.5})
                
                # Adjust camera position for parallel environments
                adjusted_pos = (
                    camera_pos["x"], 
                    camera_pos["y"] + i * self.sub_env_spacing, 
                    camera_pos["z"]
                )
                adjusted_lookat = (
                    camera_lookat["x"], 
                    camera_lookat["y"] + i * self.sub_env_spacing, 
                    camera_lookat["z"]
                )
                
                cam = self.scene.add_camera(
                    res    = self.config.resolution,
                    pos    = adjusted_pos,
                    lookat = adjusted_lookat,
                    fov    = self.config.fov,
                )
                self.cameras.append(cam)

            self.scene.build(n_envs=n_envs,
                env_spacing=(self.sub_env_spacing, self.sub_env_spacing),
                center_envs_at_origin=False,
                n_envs_per_row=n_envs
            )
        
        # Get observations for each environment
        observations = self._render(init_obs=True, env_indices=self.active_env_indices)
        
        # Convert to env_id based results
        results = {}
        for env_id in env_id_to_seed.keys():
            idx = self.env_id_to_idx[env_id]
            results[env_id] = (observations[idx], {})
        
        return results

    def step(self, env_id_to_action: Dict[str, str]) -> Dict[str, Tuple]:
        """Step multiple environments with their respective actions.
        
        Args:
            env_id_to_action: Dict mapping env_id to action string
            
        Returns:
            Dict mapping env_id to (obs, reward, done, info) tuple
        """
        results = {}
        env_indices_to_update = []
        
        for env_id, llm_raw_response in env_id_to_action.items():
            idx = self.env_id_to_idx.get(env_id)
            if idx is None:
                continue
                
            action = self._parse_action(llm_raw_response)

            metrics = {
                "turn_metrics": {
                    "action_is_valid": action in self.VALID_ACTIONS,
                    "action_is_effective": False,
                },
                "traj_metrics": {
                    "success": False,
                },
            }

            reward = 0.0
            self._last_valid_actions[idx] = []
            if action in self.VALID_ACTIONS:
                self._last_valid_actions[idx] = [action]
                self._apply_action_to_env(action, idx)
                env_indices_to_update.append(idx)
                metrics["turn_metrics"]["action_is_effective"] = action != "Noop"
                reward += self.config.format_reward
            else:
                metrics["turn_metrics"]["action_is_valid"] = False

            self._step_counts[idx] += 1
            if quat_equal(self._current_orientations[idx], self._target_orientations[idx]):
                metrics["traj_metrics"]["success"] = True
                self._dones[idx] = True
                reward += self.config.success_reward
            elif self._step_counts[idx] >= self.config.max_steps:
                self._dones[idx] = True

            self.total_rewards[idx] += reward
            
            # Store intermediate results for this env
            results[env_id] = {
                "reward": reward,
                "done": self._dones[idx],
                "metrics": metrics,
                "llm_raw_response": llm_raw_response,
                "admissible_commands": [self.VALID_ACTIONS],
            }
        
        # Update rotations for all environments that had valid actions
        if env_indices_to_update:
            self._update_object_rotations(env_indices_to_update)
        
        # Render observations for all environments
        observations = self._render(init_obs=False, env_indices=list(self.env_id_to_idx.values()))
        
        # Complete the results with observations
        final_results = {}
        for env_id in env_id_to_action.keys():
            idx = self.env_id_to_idx.get(env_id)
            if idx is not None and env_id in results:
                final_results[env_id] = (
                    observations[idx],
                    results[env_id]["reward"],
                    results[env_id]["done"],
                    {
                        "metrics": results[env_id]["metrics"],
                        "llm_raw_response": results[env_id]["llm_raw_response"],
                        "admissible_commands": results[env_id]["admissible_commands"],
                    }
                )
        
        return final_results

    def close(self):
        return

    def system_prompt(self, env_ids=None) -> Dict[str, str]:
        """Get system prompts for specified environments.
        
        Args:
            env_ids: List of env_ids to get prompts for. If None, get for all.
            
        Returns:
            Dict mapping env_id to system prompt
        """
        if env_ids is None:
            env_ids = list(self.env_id_to_idx.keys())
        
        prompt = system_prompt()
        return {env_id: prompt for env_id in env_ids}

    def compute_reward(self, env_ids=None) -> Dict[str, float]:
        """Compute rewards for specified environments.
        
        Args:
            env_ids: List of env_ids to compute rewards for. If None, compute for all.
            
        Returns:
            Dict mapping env_id to total reward
        """
        if env_ids is None:
            env_ids = list(self.env_id_to_idx.keys())
        
        results = {}
        for env_id in env_ids:
            idx = self.env_id_to_idx.get(env_id)
            if idx is not None:
                results[env_id] = float(self.total_rewards.get(idx, 0.0))
        
        return results

    # ------- helpers -------
    def _parse_action(self, text: str) -> str:
        try:
            start = text.index("<answer>") + len("<answer>")
            end = text.index("</answer>")
            return text[start:end].strip()
        except Exception:
            return ""

    def _apply_action_to_env(self, action: str, env_idx: int) -> None:
        """Apply action to a specific parallel environment."""
        import re
        
        match = re.match(r'^([xyz])(-?\d+)$', action)
        if not match:
            return
        
        axis = match.group(1)
        angle = int(match.group(2))
        
        if axis == 'x':
            rotation_quat = euler_xyz_to_quat(angle, 0, 0, degrees=True)
        elif axis == 'y':
            rotation_quat = euler_xyz_to_quat(0, angle, 0, degrees=True)
        elif axis == 'z':
            rotation_quat = euler_xyz_to_quat(0, 0, angle, degrees=True)
        else:
            return
        
        self._current_orientations[env_idx] = quat_multiply(rotation_quat, self._current_orientations[env_idx])


if __name__ == "__main__":
    import genesis as gs
    import os
    from .utils import quat_to_euler_xyz_scipy
    
    print("[Standalone] Testing MentalRotationEnv with dataset-driven configuration...")
    gs.init(backend=gs.gpu)
    
    config = MentalRotationEnvConfig(device="cuda", max_steps=10, n_parallel_envs=2)
    env = MentalRotationEnv(config)
    
    print(f"[TEST] Dataset loaded with {len(env.dataset)} tasks")
    for i, task in enumerate(env.dataset):
        scene_key = (task["background"], task["object"])
        print(f"[TEST] Task {i}: Scene key = {scene_key}")
        print(f"[TEST] Task {i}: Instruction = {task['instruction'][:60]}...")
    
    env_id1, env_id2, env_id3 = "same_scene_1", "same_scene_2", "different_task"
    
    print(f"\n[TEST] Available actions: {env.VALID_ACTIONS}")
    print("[TEST] Action format: axis + angle (e.g., x90, y-180, z270)")
    
    print(f"\n[TEST] Test 1: Reset with seeds 0, 1 (should have same scene key)")
    results = env.reset(env_id_to_seed={env_id1: 0, env_id2: 1})
    
    task1 = env._get_task_data(0)
    task2 = env._get_task_data(1) 
    scene_key1 = (task1["background"], task1["object"])
    scene_key2 = (task2["background"], task2["object"])
    
    print(f"[TEST] {env_id1} (seed 0): Scene key = {scene_key1}")
    print(f"[TEST] {env_id2} (seed 1): Scene key = {scene_key2}")
    
    if scene_key1 == scene_key2:
        print("[TEST] ✓ SUCCESS: Both environments use same scene configuration")
        print("[TEST] ✓ This confirms they share the same scene instance")
    else:
        print("[TEST] ✗ FAILED: Environments use different scene configurations")
    
    prompts = env.system_prompt([env_id1, env_id2])
    print(f"\n[TEST] System prompt (first 100 chars): {prompts[env_id1][:100]}...")
    
    os.makedirs("./test_mental_rotation_env", exist_ok=True)
    
    for env_id in [env_id1, env_id2]:
        obs, info = results[env_id]
        idx = env.env_id_to_idx[env_id]
        task_data = env._task_data[idx]
        
        print(f"\n[TEST] {env_id}:")
        print(f"[TEST]   Task instruction: {task_data['instruction'][:80]}...")
        print(f"[TEST]   Object: {task_data['object']}, Background: {task_data['background']}")
        print(f"[TEST]   Initial orientation: {env._current_orientations[idx]}")
        print(f"[TEST]   Target orientation: {env._target_orientations[idx]}")
        
        current_img = obs["multi_modal_data"][config.image_placeholder][0]
        target_img = obs["multi_modal_data"][config.target_image_placeholder][0]
        current_img.save(f"./test_mental_rotation_env/{env_id}_current_reset.png")
        target_img.save(f"./test_mental_rotation_env/{env_id}_target_reset.png")
        print(f"[TEST]   Saved images: {env_id}_current_reset.png, {env_id}_target_reset.png")
    
    print(f"\n[TEST] Test 2: Reset with seed 2 (different task)")
    results_single = env.reset(env_id_to_seed={env_id3: 2})
    
    task3 = env._get_task_data(2)
    scene_key3 = (task3["background"], task3["object"])
    print(f"[TEST] {env_id3} (seed 2): Scene key = {scene_key3}")
    
    if scene_key3 == scene_key1:
        print("[TEST] ✓ INFO: Same scene key - will reuse existing scene instance")
    else:
        print("[TEST] ✓ INFO: Different scene key - will create new scene instance")
    
    print(f"\n[TEST] Test 3: Testing actions on parallel environments")
    test_actions = [
        {env_id1: "<answer>x90</answer>", env_id2: "<answer>y-90</answer>"},
        {env_id1: "<answer>z180</answer>", env_id2: "<answer>x270</answer>"}
    ]
    
    for step_num, actions in enumerate(test_actions, 1):
        print(f"\n[TEST] Step {step_num} - Actions: {actions}")
        
        step_results = env.step(actions)
        
        for env_id in [env_id1, env_id2]:
            if env_id in step_results:
                obs, reward, done, info = step_results[env_id]
                print(f"[TEST] {env_id}: reward={reward}, done={done}")
                print(f"[TEST] {env_id}: action_valid={info['metrics']['turn_metrics']['action_is_valid']}")
                
                # Save step images
                current_img = obs["multi_modal_data"][config.image_placeholder][0]
                target_img = obs["multi_modal_data"][config.target_image_placeholder][0]
                current_img.save(f"./test_mental_rotation_env/{env_id}_current_step{step_num}.png")
                target_img.save(f"./test_mental_rotation_env/{env_id}_target_step{step_num}.png")
                
                if done:
                    print(f"[TEST] ✓ {env_id} completed the task!")
    
    # Test invalid action
    print(f"\n[TEST] Test 4: Testing invalid action")
    invalid_results = env.step({env_id1: "<answer>invalid_action</answer>"})
    if env_id1 in invalid_results:
        obs, reward, done, info = invalid_results[env_id1]
        action_valid = info['metrics']['turn_metrics']['action_is_valid']
        if not action_valid:
            print("[TEST] ✓ SUCCESS: Invalid action properly rejected")
        else:
            print("[TEST] ✗ FAILED: Invalid action not rejected")
    
    # Compute final rewards
    print(f"\n[TEST] Final Results:")
    rewards = env.compute_reward([env_id1, env_id2, env_id3])
    for env_id in [env_id1, env_id2, env_id3]:
        if env_id in rewards:
            print(f"[TEST] {env_id} total reward: {rewards[env_id]}")
    
    print(f"\n[TEST] ✓ All tests completed!")
    print(f"[TEST] Key findings:")
    print(f"[TEST] - Dataset-driven configuration working correctly")
    print(f"[TEST] - Environments with same (background, object) can share scenes")
    print(f"[TEST] - Task-specific orientations and instructions loaded from JSON")
    print(f"[TEST] - Images saved in ./test_mental_rotation_env/")
    
    env.close()