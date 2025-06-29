from vagen.env.base.base_env import BaseEnv
from typing import Dict, List, Tuple
import json
import os
import random
from PIL import Image
from .env_config import ChartAgentEnvConfig
from .utils import apply_bounding_boxes, response_extraction


class ChartAgentEnv(BaseEnv):
    def __init__(self, config: ChartAgentEnvConfig):
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.script_dir, config.image_path)
        self.split = config.split
        
        # Set data paths
        data_filename = f"chart_agent_{self.split}.jsonl"
        self.data_path = os.path.join(self.script_dir, "ChartAgent", "json_data", data_filename)
        
        # Load dataset
        with open(self.data_path, "r") as f:
            self.dataset = [json.loads(line) for line in f]
        
        # Initialize episode state
        self.current_data = None
        self.current_seed = None
        self.chart_image = None
        self.done = False
        self.total_reward = 0
        
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset environment with new seed"""
        if seed is not None:
            self.current_seed = seed
            random.seed(seed)
        
        self.done = False
        self.total_reward = 0
        
        # Select data point
        idx = self.current_seed % len(self.dataset) if self.current_seed is not None else random.randint(0, len(self.dataset) - 1)
        self.current_data = self.dataset[idx]
        
        # Create initial observation
        obs = self._get_initial_observation()
        info = {
            "ground_truth": self.current_data["gt_answer"],
            "question_id": self.current_data["id"],
        }
        
        return obs, info
    
    def _get_initial_observation(self) -> Dict:
        """Create initial observation with question and chart image"""
        # Load and resize image
        image_path = self.current_data["chart_image"]
        question_str = self.current_data["question_str"]

        full_path = os.path.join(self.image_dir, image_path)
        img = Image.open(full_path)
        self.chart_image = img.resize(self.config.image_size, Image.LANCZOS)
        
        return {
            'obs_str': question_str,
            'multi_modal_data': {
                self.config.image_placeholder: [self.chart_image]
            }
        }
    
    def step(self, llm_raw_response: str) -> Tuple[Dict, float, bool, Dict]:
        """Process the LLM's response and compute reward"""
        info = {
            "metrics": {
                "turn_metrics": {
                    "action_is_effective": False,
                    "action_is_valid": False,
                },
                "traj_metrics": {
                    "success": False,
                }
            },
            "llm_raw_response": llm_raw_response,
        }
        
        # Extract response information
        parsed_response = response_extraction(llm_raw_response)
        if not parsed_response:
            # TODO: different fail message for different cases
            return {'obs_str': "Failed to parse response"}, 0.0, False, info
        
        # Response was parsed successfully
        reward = 1.0
        info["metrics"]["turn_metrics"]["action_is_effective"] = True
        info["metrics"]["turn_metrics"]["action_is_valid"] = True
        
        # Check if we have a final answer
        if parsed_response.get("answer"):
            obs = {'obs_str': "Task completed"}
            if parsed_response["answer"].lower() == self.current_data["gt_answer"].lower():
                self.done = True
                reward += 5.0
                info["metrics"]["traj_metrics"]["success"] = True
            return obs, reward, self.done, info
        
        # Apply bounding boxes and create observation
        img_with_boxes, _ = apply_bounding_boxes(
            self.chart_image, 
            parsed_response.get("bounding_boxes", []),
            return_cropped=False,
            highlight_boxes=True
        )
        
        obs = {
            'obs_str': f"The cropped regions are shown in {self.config.image_placeholder}",
            'multi_modal_data': {
                self.config.image_placeholder: [img_with_boxes]
            }
        }
        
        return obs, reward, self.done, info
    
    def close(self):
        """Close the environment"""
        pass
    
    def system_prompt(self) -> str:
        return "You are an AI assistant that answers visual questions based on images."


if __name__ == "__main__":
    # Create config
    config = ChartAgentEnvConfig()
    
    # Create environment
    env = ChartAgentEnv(config)
    
    print("System prompt:")
    print(env.system_prompt())
    print("\n" + "-"*50 + "\n")
    
    i = 0
    # Reset environment and get first observation for one data sample
    obs, info = env.reset(seed=0)
    print("Question:")
    print(obs["obs_str"])
    print("\nGround truth:", info["ground_truth"])
    
    # Save initial image
    if config.image_placeholder in obs["multi_modal_data"] and obs["multi_modal_data"][config.image_placeholder]:
        os.makedirs("./test_chartagent", exist_ok=True)
        for j, img in enumerate(obs["multi_modal_data"][config.image_placeholder]):
            img.save(f"./test_chartagent/chartagent_initial_{j}.png")
    print(f"\nSaved {len(obs['multi_modal_data'][config.image_placeholder])} images to ./test_chartagent/")
    
    # Multi-round interaction with the same data sample
    round_num = 0
    while not env.done and round_num < 5:  # Limit to 5 rounds max
        round_num += 1
        print(f"\n{'='*20} Round {round_num} {'='*20}")
        
        # Get user input
        answer = input("\nEnter your answer or action (or press Enter for default): ")
        if not answer:
            if round_num == 1:
                answer = "I need to examine the chart more closely. Let me look at the data points."
            elif round_num == 2:
                answer = "Based on my analysis, the answer is 42"
            else:
                answer = "Let me try a different approach to answer this question."
        
        # Step the environment
        obs, reward, done, info = env.step(answer)
        
        # Display results
        print("\nAction Result:")
        print(f"Observation: {obs['obs_str']}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Success: {info['metrics']['traj_metrics']['success']}")
        
        # Save any new images from this round
        if 'multi_modal_data' in obs and config.image_placeholder in obs["multi_modal_data"]:
            for j, img in enumerate(obs["multi_modal_data"][config.image_placeholder]):
                img.save(f"./test_chartagent/chartagent_round{round_num}_{j}.png")
            print(f"Saved {len(obs['multi_modal_data'][config.image_placeholder])} images from round {round_num}")
    
    print(f"\nCompleted after {round_num} rounds")
    print(f"Final success: {info['metrics']['traj_metrics']['success']}")
  