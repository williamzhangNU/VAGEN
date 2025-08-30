from vagen.env.base.base_env import BaseEnv
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .prompt import system_prompt, init_observation_template, action_template, format_prompt
from .env_config import MentalRotation2DEnvConfig

class MentalRotation2DEnv(BaseEnv):
    """
    Mental Rotation 2D Environment for training and evaluating language models.
    
    This environment presents an agent with an initial image of a 2D shape and a target
    image showing the same shape at a different rotation angle. The agent must determine
    the rotation angle needed to transform the initial shape to match the target.
    """
    
    def __init__(self, config: MentalRotation2DEnvConfig):
        """
        Initialize the Mental Rotation 2D environment.
        
        Args:
            config (MentalRotation2DEnvConfig): Configuration parameters for the environment
        """
        BaseEnv.__init__(self)
        self.config = config
        
        # Load dataset
        self.tasks = self._load_dataset()
        
        # Environment state
        self.current_task = None
        self.current_rotation = 0  # Current rotation angle
        self.target_rotation = 0   # Target rotation angle
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        
        # Images
        self.base_image = None
        self.current_image = None
        self.target_image = None
        
        # Valid actions (rotation angles)
        self.valid_actions = self._generate_valid_actions()
        
        # Store the format prompt function for later use
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load dataset from JSON file.
        
        Returns:
            List of task dictionaries containing img_name, initial_rotation, target_rotation
        """
        dataset_path = os.path.join(
            os.path.dirname(__file__), 
            "dataset", 
            f"{self.config.task_name}.json"
        )
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Store img_dir for later use
        self.img_dir = data.get("img_dir", "imgs")
        
        return data.get("tasks", [])
    
    def _load_image(self, img_name: str) -> Image.Image:
        """
        Load an image from the dataset directory.
        
        Args:
            img_name: Name of the image file (without extension)
            
        Returns:
            PIL Image object
        """
        img_path = os.path.join(
            os.path.dirname(__file__), 
            "dataset", 
            self.img_dir, 
            f"{img_name}.png"
        )
        
        # Try different extensions if .png doesn't exist
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.bmp', '.gif']:
                alt_path = os.path.join(
                    os.path.dirname(__file__), 
                    "dataset", 
                    self.img_dir, 
                    f"{img_name}{ext}"
                )
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        return Image.open(img_path)
    
    def _generate_valid_actions(self) -> List[int]:
        """Generate list of valid rotation angles based on granularity."""
        actions = []
        granularity = self.config.rotation_granularity
        
        # Generate positive angles (clockwise)
        for angle in range(granularity, 360, granularity):
            actions.append(angle)
        
        # Generate negative angles (counterclockwise)  
        for angle in range(granularity, 360, granularity):
            actions.append(-angle)
        
        # Add 0 as a valid action (no rotation)
        actions.append(0)
        
        return sorted(actions)
    
    def _rotate_image(self, image: Image.Image, angle: float, bg_color: str = 'white') -> Image.Image:
        """
        Rotate an image by the specified angle.
        
        Args:
            image: PIL Image to rotate
            angle: Rotation angle in degrees (positive = clockwise)
            bg_color: Background color for rotation fill
            
        Returns:
            Rotated PIL Image
        """
        # PIL rotate() uses counter-clockwise as positive, so we negate the angle
        # to make positive angles clockwise as expected in the task
        return image.rotate(-angle, expand=True, fillcolor=bg_color)
    

    
    def _get_task_by_seed(self, seed: int) -> Dict:
        """Get a task based on the seed."""
        if not self.tasks:
            raise ValueError("No tasks available in dataset")
        
        task_idx = seed % len(self.tasks)
        return self.tasks[task_idx]
    
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for task selection
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is None:
            seed = np.random.randint(0, len(self.tasks))
        
        # Select task based on seed
        self.current_task = self._get_task_by_seed(seed)
        
        # Reset environment state
        self.current_rotation = self.current_task["initial_rotation"]
        self.target_rotation = self.current_task["target_rotation"]
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        
        # Load base image and generate rotated versions
        shape_name = self.current_task["img_name"]
        bg_color = self.current_task.get("bg_color", "white")
        self.base_image = self._load_image(shape_name)
        self.current_image = self._rotate_image(self.base_image, self.current_rotation, bg_color)
        self.target_image = self._rotate_image(self.base_image, self.target_rotation, bg_color)
        
        return self._render(init_obs=True), {}
    
    def step(self, action_str: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment based on the agent's action.
        
        Args:
            action_str: Raw string from LLM containing rotation angle
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Parse the LLM's raw response to extract actions
        rst = self.parse_func(
            response=action_str,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step
        )
        
        action_list = rst['actions']
        
        # Initialize metrics for this step
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }
        
        reward = 0
        valid_action = None
        
        # Process the first valid action
        if action_list:
            action_str_parsed = action_list[0].strip()
            
            # Try to parse as integer (rotation angle)
            try:
                rotation_angle = int(action_str_parsed)
                if rotation_angle in self.valid_actions:
                    valid_action = rotation_angle
                    
                    # Apply rotation
                    self.current_rotation = (self.current_rotation + rotation_angle) % 360
                    
                    # Update current image
                    bg_color = self.current_task.get("bg_color", "white")
                    self.current_image = self._rotate_image(self.base_image, self.current_rotation, bg_color)
                    
                    metrics["turn_metrics"]["action_is_effective"] = True
                    
                    # Check if we've reached the target
                    angle_diff = abs(self.current_rotation - self.target_rotation)
                    # Handle wrap-around (e.g., 359 vs 1 degree)
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    
                    if angle_diff <= self.config.angle_tolerance:
                        reward += self.config.success_reward
                        metrics["traj_metrics"]["success"] = True
                        self.done = True
                    else:
                        # Small penalty for each step
                        reward += self.config.step_penalty
                        
            except ValueError:
                # Invalid action format
                reward += self.config.step_penalty * 2  # Larger penalty for invalid format
        else:
            # No valid action parsed
            reward += self.config.step_penalty * 2
        
        self.step_count += 1
        self.total_reward += reward
        
        # Check if max steps reached
        if self.step_count >= self.config.max_steps:
            self.done = True
        
        # Prepare info
        info = {
            "metrics": metrics,
            "llm_raw_response": action_str,
            "llm_response": rst,
            "valid_action": valid_action,
            "current_rotation": self.current_rotation,
            "target_rotation": self.target_rotation,
            "angle_difference": abs(self.current_rotation - self.target_rotation),
            "step_count": self.step_count,
            "total_reward": self.total_reward
        }
        
        return self._render(), reward, self.done, info
    
    def _render(self, init_obs: bool = False) -> Dict:
        """
        Render the current state of the environment in vision mode.
        
        Args:
            init_obs: Whether this is the initial observation
            
        Returns:
            Dictionary containing observation string and multi-modal data
        """
        # Create side-by-side image of current and target
        current_img = self.current_image
        target_img = self.target_image
        
        # Create combined image without padding
        gap = 30  # Gap between the two images
        
        # Use max dimensions for consistent sizing
        max_img_width = max(current_img.width, target_img.width)
        max_img_height = max(current_img.height, target_img.height)
        
        combined_width = (max_img_width * 2) + gap  # two max-width slots + gap
        combined_height = max_img_height
        bg_color = self.current_task.get("bg_color", "white")
        combined_img = Image.new('RGB', (combined_width, combined_height), bg_color)
        
        # Paste images with center alignment (no padding)
        
        # Current image: left side with center alignment
        current_x = (max_img_width - current_img.width) // 2
        current_y = (max_img_height - current_img.height) // 2
        combined_img.paste(current_img, (current_x, current_y))
        
        # Target image: right side with center alignment
        target_x = max_img_width + gap + (max_img_width - target_img.width) // 2
        target_y = (max_img_height - target_img.height) // 2
        combined_img.paste(target_img, (target_x, target_y))
        
        # No labels needed - will be explained in prompt
        
        obs_str = f"<image>"
        
        multi_modal_data = {
            "<image>": [combined_img]
        }
        
        if init_obs:
            template_obs = init_observation_template(
                observation=obs_str,
                valid_actions=self.valid_actions
            )
        else:
            template_obs = obs_str
        
        return {
            "obs_str": template_obs,
            "multi_modal_data": multi_modal_data
        }
    
    def system_prompt(self) -> str:
        """Get the system prompt for the environment."""
        return system_prompt()
    
    def close(self):
        """Close the environment."""
        # Clean up any resources if needed
        pass
    
    def compute_reward(self) -> float:
        """Compute final reward (already accumulated in step rewards)."""
        return 0.0


if __name__ == "__main__":
    """Interactive test interface for the Mental Rotation 2D environment."""
    from .env_config import MentalRotation2DEnvConfig
    
    print("=== Mental Rotation 2D Environment Interactive Test ===")
    print("This is an interactive test of the mental rotation 2D environment.")
    print("You can interact with the environment by providing rotation angles.")
    print()
    
    # Create environment
    config = MentalRotation2DEnvConfig()
    config.rotation_granularity = 30  # Use 30-degree increments to match dataset
    config.prompt_format = "no_think"
    
    try:
        env = MentalRotation2DEnv(config)
        
        print("System Prompt:")
        print(env.system_prompt())
        print("\n" + "="*60 + "\n")
        
        print(f"Loaded {len(env.tasks)} tasks from dataset")
        print(f"Valid actions: {env.valid_actions}")
        print()
        
        # Interactive loop
        while True:
            # Ask for seed or use random
            seed_input = input("Enter seed (0-5) or press Enter for seed 0, 'q' to quit: ").strip()
            
            if seed_input.lower() == 'q':
                print("Goodbye!")
                break
            
            try:
                seed = int(seed_input) if seed_input else 0
                seed = max(0, min(seed, len(env.tasks) - 1))  # Clamp to valid range
            except ValueError:
                seed = 0
            
            # Reset environment
            obs, info = env.reset(seed=seed)
            
            print(f"\n{'='*60}")
            print(f"New Episode (Seed: {seed})")
            print(f"{'='*60}")
            print(f"Task: {env.current_task}")
            print()
            print("Observation:")
            print(obs["obs_str"])
            
            # Save and show image
            if "<image>" in obs["multi_modal_data"]:
                combined_img = obs["multi_modal_data"]["<image>"][0]
                combined_img.save(f"/workspace/VAGEN/mental_rotation_2d_step0.png")
                print(f"Image saved to: /workspace/VAGEN/mental_rotation_2d_step0.png")
                print("Left side: Current shape | Right side: Target shape")
            
            print()
            
            # Episode loop
            step_count = 0
            while not env.done and step_count < config.max_steps:
                user_input = input(f"Step {step_count + 1}: Enter rotation angle (or 'n' for new episode, 'q' to quit): ").strip()
                
                if user_input.lower() == 'q':
                    print("Goodbye!")
                    exit()
                elif user_input.lower() == 'n':
                    break
                
                # Format as expected LLM response
                action_str = f"<answer>{user_input}</answer>"
                
                # Take step
                obs, reward, done, info = env.step(action_str)
                step_count += 1
                
                print(f"\nStep {step_count} Result:")
                print(f"  Action: {user_input}")
                print(f"  Valid action: {info.get('valid_action', 'None')}")
                print(f"  Reward: {reward:.2f}")
                print(f"  Current rotation: {info['current_rotation']}°")
                print(f"  Target rotation: {info['target_rotation']}°")
                print(f"  Angle difference: {info['angle_difference']}°")
                print(f"  Success: {info['metrics']['traj_metrics']['success']}")
                print(f"  Done: {done}")
                
                # Save updated image
                if "<image>" in obs["multi_modal_data"]:
                    combined_img = obs["multi_modal_data"]["<image>"][0]
                    combined_img.save(f"/workspace/VAGEN/mental_rotation_2d_step{step_count}.png")
                    print(f"  Updated image: /workspace/VAGEN/mental_rotation_2d_step{step_count}.png")
                
                if done:
                    if info['metrics']['traj_metrics']['success']:
                        print("\n🎉 SUCCESS! You matched the target rotation!")
                    else:
                        print(f"\n❌ Episode ended. Maximum steps ({config.max_steps}) reached.")
                    print(f"Total reward: {info['total_reward']:.2f}")
                    break
                
                print()
            
            print(f"\n{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure you have:")
        print("1. Created a dataset JSON file")
        print("2. Added corresponding image files")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
