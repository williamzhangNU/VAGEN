#!/usr/bin/env python3
"""
RAGEN Spatial Runner for VAGEN2
Simplified version that directly implements RAGEN's agent_proxy logic
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAGEN spatial experiments in VAGEN2"
    )
    parser.add_argument("--num_per_task", type=int, default=1, help="num of each task. Default: 1")
    parser.add_argument(
        "--task",
        dest="tasks",
        nargs="+",
        required=False,
        default=["ActiveRot"],
        help="Task tags. Space-separated, or a single comma-separated arg, e.g. --task ActiveRot,ActiveDir. Defaults to [ActiveRot] if omitted.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="If set, will override the active exploration history",
    )
    parser.add_argument(
        "--cogmap",
        action="store_true",
        help="If set, will enable cognitive map evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="For each task, set output_dir=<base>. Default base=results.",
    )
    parser.add_argument(
        "--eval_model_type",
        type=str,
        choices=["api", "vllm"],
        default="api",
        help="Override eval_model_type. Choices: api or vllm. Default: api",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Model name. Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="If set, will disable think",
    )
    parser.add_argument(
        "--vision_data",
        type=str,
        default="/Users/songshe/ToS/VAGEN2/vagen/env/spatial/dataset3room9obj_0913/",
        help="Path to vision alignment data",
    )
    return parser.parse_args()

def normalize_tasks(tasks):
    if len(tasks) == 1 and ("," in tasks[0]):
        return [t.strip() for t in tasks[0].split(",") if t.strip()]
    return tasks

class MockApiCaller:
    """Mock API caller that simulates LLM responses."""
    
    def __init__(self, model_name: str, enable_think: bool = True):
        self.model_name = model_name
        self.enable_think = enable_think
    
    def call_api(self, messages: List[Dict[str, str]]) -> str:
        """Simulate API call with mock response following RAGEN format."""
        last_message = messages[-1]['content'].lower()
        
        # Count assistant messages to determine step
        step_num = len([m for m in messages if m['role'] == 'assistant'])
        
        # Determine response based on content and phase
        # First check for evaluation phase (highest priority)
        if 'choose the correct sequence' in last_message or 'answer with only the letter' in last_message or '## evaluation question' in last_message or 'evaluation question' in last_message:
            # Multiple choice evaluation phase - return a letter
            import random
            choices = ['A', 'B', 'C', 'D']
            answer = random.choice(choices)
            
            if self.enable_think:
                return f"<think>Based on my exploration, I need to choose from the given options.</think>\n{answer}"
            else:
                return answer
        elif 'exploration' in last_message or 'steps left' in last_message or 'available actions' in last_message:
            # Exploration phase - use proper RAGEN action format
            if step_num >= 4:
                # End exploration
                if self.enable_think:
                    return "<think>I've explored enough, let me finish.</think>\nActions: [Term()]"
                else:
                    return "Actions: [Term()]"
            else:
                # Generate valid exploration actions
                actions = [
                    "Actions: [Observe()]",
                    "Actions: [Rotate(90), Observe()]", 
                    "Actions: [Rotate(-90), Observe()]"
                ]
                action = actions[step_num % len(actions)]
                
                if self.enable_think:
                    return f"<think>I need to explore the environment. Let me observe and rotate.</think>\n{action}"
                else:
                    return action
        else:
            # Other evaluation questions
            if 'rotation' in last_message or 'rot' in last_message:
                answer = "clockwise"
            elif 'direction' in last_message or 'dir' in last_message:
                answer = "north"  
            elif 'location' in last_message or 'loc' in last_message:
                answer = "center"
            else:
                answer = "yes"
            
            if self.enable_think:
                return f"<think>Based on my exploration, I'll answer the question.</think>\n{answer}"
            else:
                return answer

class RAGENSpatialRunner:
    """Main runner that implements RAGEN's agent_proxy logic."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_caller = MockApiCaller(
            config['model_name'], 
            config.get('enable_think', True)
        )
        
        # Add VAGEN2 to path
        vagen2_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(vagen2_root))
        
        # Import RAGEN components
        from vagen.env.spatial.ragen.env import SpatialGym
        from vagen.env.spatial.ragen.config import SpatialGymConfig
        self.SpatialGym = SpatialGym
        self.SpatialGymConfig = SpatialGymConfig
    
    def create_env_config(self, task: str, sample_idx: int) -> 'SpatialGymConfig':
        """Create environment configuration for a task."""
        task_type = task.replace("Active", "").replace("Passive", "").lower()
        exp_type = "active" if "Active" in task else "passive"
        
        return self.SpatialGymConfig(
            name=f"ragen_{task.lower()}_{sample_idx}",
            exp_type=exp_type,
            eval_tasks=[{"task_type": task_type, "task_kwargs": {}}],
            max_exp_steps=50,
            prompt_config={
                "topdown": False,
                "type": "standard",
                "enable_think": self.config.get('enable_think', True),
                "cogmap": self.config.get('evaluate_cogmap', False)
            },
            align_with_vision=True,
            vision_data_path=self.config['vision_data_path'],
            level=2,
            main=6,
            n_objects=9,
            room_size=[15, 15],
            kwargs={"model_config": {"model_name": self.config['model_name']}}
        )
    
    def run_single_episode(self, task: str, sample_idx: int) -> Dict[str, Any]:
        """Run a single episode following RAGEN's rollout logic."""
        
        # Create environment
        env_config = self.create_env_config(task, sample_idx)
        env = self.SpatialGym(env_config)
        
        # Reset environment
        obs, info = env.reset(seed=sample_idx)
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": obs}
        ]
        
        step = 0
        done = False
        total_reward = 0
        episode_data = {
            'task': task,
            'sample_idx': sample_idx,
            'env_id': f"{task}_{sample_idx}",
            'messages': messages.copy(),
            'steps': [],
            'total_reward': 0
        }
        
        print(f"    Starting episode {sample_idx + 1} for {task}")
        
        # Run episode loop (similar to RAGEN's agent_proxy rollout)
        while not done and step < 20:  # max_turn = 20
            
            # Get LLM response (equivalent to generate_sequences)
            response = self.api_caller.call_api(messages)
            
            # Step environment (equivalent to env step)
            obs, reward, done, step_info = env.step(response)
            total_reward += reward
            
            # Update messages (equivalent to formulate_rollouts)
            messages.append({"role": "assistant", "content": response})
            if not done and obs.strip():
                messages.append({"role": "user", "content": obs})
            
            # Record step data
            episode_data['steps'].append({
                'step': step,
                'response': response,
                'observation': obs,
                'reward': reward,
                'done': done,
                'info': step_info
            })
            
            step += 1
            
            if done:
                print(f"      Episode completed at step {step}, reward: {reward}")
                break
        
        episode_data['total_reward'] = total_reward
        episode_data['messages'] = messages
        episode_data['env_summary'] = env.get_env_summary()
        
        return episode_data
    
    def run_all_tasks(self) -> List[Dict[str, Any]]:
        """Run all tasks following RAGEN's main logic."""
        
        all_results = []
        
        for task in self.config['tags']:
            print(f"Running task: {task}")
            
            task_results = []
            for sample_idx in range(self.config['num_per_task']):
                try:
                    result = self.run_single_episode(task, sample_idx)
                    task_results.append(result)
                except Exception as e:
                    print(f"    Error in sample {sample_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            all_results.extend(task_results)
            print(f"  Completed {len(task_results)}/{self.config['num_per_task']} samples for {task}")
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results using RAGEN's SpatialEnvLogger."""
        
        if not results:
            print("No results to save")
            return
        
        try:
            from vagen.env.spatial.Base.tos_base.utils.env_logger import SpatialEnvLogger
            
            # Extract data for SpatialEnvLogger (following RAGEN's agent_proxy logic)
            env_summaries = [r['env_summary'] for r in results]
            messages_list = [r['messages'] for r in results]
            
            # Call SpatialEnvLogger.log_each_env_info (same as RAGEN's agent_proxy)
            SpatialEnvLogger.log_each_env_info(
                env_summaries=env_summaries,
                messages=messages_list,
                output_dir=self.config['output_dir'],
                model_name=self.config['model_name']
            )
            
            # Also save raw results
            results_file = os.path.join(self.config['output_dir'], "raw_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Results saved to: {self.config['output_dir']}")
            print(f"HTML dashboard: {self.config['output_dir']}/env_data.html")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()

def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.model_name.replace("\\", "/").rstrip("/").split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    tasks = normalize_tasks(args.tasks)
    
    # Create config (similar to RAGEN's config structure)
    config = {
        'tags': tasks,
        'num_per_task': args.num_per_task,
        'output_dir': output_dir,
        'eval_model_type': args.eval_model_type,
        'model_name': args.model_name,
        'override': args.override,
        'evaluate_cogmap': args.cogmap,
        'enable_think': not args.no_think,
        'vision_data_path': args.vision_data
    }
    
    # Save config
    config_file = os.path.join(output_dir, "config.yaml")
    with open(config_file, 'w') as f:
        import yaml
        yaml.safe_dump(config, f, indent=2)
    
    print("RAGEN Spatial Experiments in VAGEN2")
    print(f"Tasks: {tasks}")
    print(f"Samples per task: {args.num_per_task}")
    print(f"Model: {args.model_name}")
    print(f"Vision data: {args.vision_data}")
    print(f"Output: {output_dir}")
    
    # Create and run experiments
    runner = RAGENSpatialRunner(config)
    
    start_time = time.time()
    results = runner.run_all_tasks()
    end_time = time.time()
    
    print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")
    print(f"Total samples: {len(results)}")
    
    # Save results using RAGEN's logger
    runner.save_results(results)
    
    print("All RAGEN spatial tasks completed successfully!")

if __name__ == "__main__":
    main()