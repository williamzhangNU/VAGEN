import sys
import os
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
        # Public-ish step tracking (kept for compatibility)
        self.step_count = 0
        self.done = False

        # Internal state used by step/reset - quaternion (w, x, y, z)
        self._target_orientation = (1.0, 0.0, 0.0, 0.0)  # target quaternion
        self._current_orientation = (1.0, 0.0, 0.0, 0.0)  # current quaternion
        self._last_valid_actions = []
        self.total_reward = 0.0
        self._step_count = 0
        self._done = False

        self.n_sub_scenes = 1 # TODO
        self.sub_env_spacing = 10 # TODO
        self.rotate_granularity = 90 # TODO

        # 动态生成valid actions基于granularity
        self.VALID_ACTIONS = self._generate_valid_actions()

        self.scene = None
        
        # Genesis should already be initialized by the service layer
        # If used standalone, user must call genesis.init() before creating environments

    def _generate_valid_actions(self) -> List[str]:
        actions = []
        axes = ['x', 'y', 'z']
        
        for axis in axes:
            for angle in range(self.rotate_granularity, 360, self.rotate_granularity):
                actions.append(f"{axis}{angle}")
                actions.append(f"{axis}-{angle}")
        
        return actions


    def _generate_target_image(self):
        current_quat = self._current_orientation
        
        self.object.set_quat(self._target_orientation, zero_velocity=False)
        self.scene.step()
        target_image = self.cameras[0].render()[0]
        
        self.object.set_quat(current_quat, zero_velocity=False)
        self.scene.step()
        
        self.target_image = convert_numpy_to_PIL(target_image)

    def _render(self, init_obs: bool) -> Dict:
        image_placeholder = self.config.image_placeholder
        target_image_placeholder = self.config.target_image_placeholder
        format_prompt_text = self.system_prompt()
        
        rgb = self.cameras[0].render()[0] # TODO: render all cameras
        
        if self.target_image is None:
            self._generate_target_image()

        if init_obs:
            obs_str = init_observation_template(
                img_str=image_placeholder,
                target_img_str=target_image_placeholder,
                valid_actions=self.VALID_ACTIONS
            )
        else:
            last_action = self._last_valid_actions[-1] if self._last_valid_actions else "None"
            obs_str = action_template(
                img_str=image_placeholder,
                target_img_str=target_image_placeholder,
                last_action=last_action,
                step_count=self.step_count
            )

        multi_modal_data = {
            image_placeholder: [convert_numpy_to_PIL(rgb)],
            target_image_placeholder: [self.target_image],
        }
        
        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data
        }

    def _add_object(self, seed=None):
        # TODO: randomize the object
        
        background = self.scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -1)))

        self.object = self.scene.add_entity(
            gs.morphs.Mesh(
                file="/workspace/genesis-data-generator/assets/objects/airplane.glb",
                fixed=True,
                pos=(0.0, 0.0, 0.0),
                euler=(0.0, 0.0, 0.0),
                scale=0.1
            ),
        )

    def _update_object_rotation(self):
        assert hasattr(self, 'object') and self.object is not None
        self.object.set_quat(self._current_orientation, zero_velocity=False)
        self.scene.step()


    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        random.seed(seed)
        # Reset counters
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self._step_count = 0

        # Reset task state
        self._done = False
        self._current_orientation = (1.0, 0.0, 0.0, 0.0)
        self._target_orientation = euler_xyz_to_quat(
            random.choice([0, 90, 180, 270]),
            random.choice([0, 90, 180, 270]), 
            random.choice([0, 90, 180, 270]),
            degrees=True
        )
        self._last_valid_actions = []

        self.target_image = None

        self.scene = gs.Scene(
            rigid_options=options_map[self.config.options]["rigid_options"],
            viewer_options=options_map[self.config.options]["viewer_options"],
            vis_options=options_map[self.config.options]["vis_options"],
            renderer=self.config.renderer,
        )

        self._add_object(seed=seed)

        self.cameras = [] 
        for env_idx in range(self.n_sub_scenes):
            cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (5, -2 + env_idx * self.sub_env_spacing, 2.5),
                lookat = (0, env_idx * self.sub_env_spacing, 0.5),
                fov    = 30,
            )
            self.cameras.append(cam)

        self.scene.build(n_envs=self.n_sub_scenes,
            env_spacing=(self.sub_env_spacing, self.sub_env_spacing),
            center_envs_at_origin=False,
            n_envs_per_row=self.n_sub_scenes
        )

        return self._render(init_obs=True), {}

    def step(self, llm_raw_response: str):
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
        self._last_valid_actions = []
        if action in self.VALID_ACTIONS:
            self._last_valid_actions = [action]
            self._apply_action(action)
            self._update_object_rotation()
            metrics["turn_metrics"]["action_is_effective"] = action != "Noop"
            reward += self.config.format_reward
        else:
            metrics["turn_metrics"]["action_is_valid"] = False

        self._step_count += 1
        self.step_count = self._step_count
        if quat_equal(self._current_orientation, self._target_orientation):
            metrics["traj_metrics"]["success"] = True
            self._done = True
            reward += self.config.success_reward
        elif self._step_count >= self.config.max_steps:
            self._done = True

        self.done = self._done
        self.total_reward += reward
        obs = self._render(init_obs=False)
        info = {
            "metrics": metrics,
            "llm_raw_response": llm_raw_response,
            "admissible_commands": [self.VALID_ACTIONS],
        }

        return obs, reward, self._done, info

    def close(self):
        return

    def system_prompt(self) -> str:
        return system_prompt()

    def compute_reward(self) -> float:
        return float(self.total_reward)

    # ------- helpers -------
    def _parse_action(self, text: str) -> str:
        try:
            start = text.index("<answer>") + len("<answer>")
            end = text.index("</answer>")
            return text[start:end].strip()
        except Exception:
            return ""

    def _apply_action(self, action: str) -> None:
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
        
        self._current_orientation = quat_multiply(rotation_quat, self._current_orientation)


if __name__ == "__main__":
    import genesis as gs
    from .utils import quat_to_euler_xyz_scipy
    
    # 当standalone使用时，需要手动初始化Genesis
    print("[Standalone] Initializing Genesis...")
    gs.init(backend=gs.gpu)  # 或者 gs.cpu
    
    config = MentalRotationEnvConfig(device="cuda", max_steps=100)
    env = MentalRotationEnv(config)
    print(env.system_prompt())
    print(f"[TEST] Available actions: {env.VALID_ACTIONS}")
    print("[TEST] Action format: axis + angle (e.g., x90, y-180, z270)")
    
    obs, info = env.reset(seed=3712037021)
    print(obs["obs_str"])
    i = 0
    os.makedirs("./test_mental_rotation", exist_ok=True)
    
    current_img = obs["multi_modal_data"][config.image_placeholder][0]
    target_img = obs["multi_modal_data"][config.target_image_placeholder][0]
    current_img.save(f"./test_mental_rotation/current_{i}.png")
    target_img.save(f"./test_mental_rotation/target_{i}.png")
    
    done = False
    
    while not done:
        i += 1
        print(f"[TEST] Current orientation (euler): {quat_to_euler_xyz_scipy(env._current_orientation)}")
        print(f"[TEST] Target orientation (euler): {quat_to_euler_xyz_scipy(env._target_orientation)}")
        action = input("[TEST] Enter action (e.g., x90, y-90, z180): ")
        action = f"<answer>{action}</answer>"

        obs, reward, done, info = env.step(action)
        print(f"[TEST] Reward: {reward}, Done: {done}")
        print(obs["obs_str"])
        
        current_img = obs["multi_modal_data"][config.image_placeholder][0]
        target_img = obs["multi_modal_data"][config.target_image_placeholder][0]
        current_img.save(f"./test_mental_rotation/current_{i}.png")
        target_img.save(f"./test_mental_rotation/target_{i}.png")
        
        print(f"[TEST] Action valid: {info['metrics']['turn_metrics']['action_is_valid']}")
        print(f"[TEST] Action effective: {info['metrics']['turn_metrics']['action_is_effective']}")
        
        if done:
            break
    
    print(f"[TEST] Total reward: {env.compute_reward()}")
    env.close()