"""
Preprocess dataset for genereal tasks
"""

import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--start_seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/sokoban", help="Output file to save the trajectories (default: 'data/sokoban').")
    parser.add_argument("--train_size", type=int, default=300, help="Number of trajectories to generate (default: 3000).")
    parser.add_argument("--test_size", type=int, default=10, help="Number of trajectories to generate (default: 100).")
    parser.add_argument("--config", type=str, default="config/sokoban.yaml", help="Config file to use (default: 'config/sokoban.yaml').")

    args = parser.parse_args()
    
    assert args.env == "sokoban", "Unsupported environment: {args.env}"
    assert args.algo == "bfs", "Unsupported algorithm: {args.algo}"
    data_source = args.env
    
    dim_x, dim_y, num_boxes, max_steps, search_depth = os.environ.get("DIM_X"), os.environ.get("DIM_Y"), os.environ.get("NUM_BOXES"), os.environ.get("MAX_STEPS"), os.environ.get("SEARCH_DEPTH")
    dim_x, dim_y, num_boxes, max_steps, search_depth = int(dim_x), int(dim_y), int(num_boxes), int(max_steps), int(search_depth)

    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    instructions = []
    for seed in seeds:
        env = SokobanEnv(
            dim_room=(dim_x, dim_y),
            num_boxes=num_boxes,
            max_steps=max_steps,
            search_depth=search_depth
        )
        observation = env.reset(seed=seed, mode='tiny_rgb_array')
        instruction = INSTRUCTION_TEMPLATE.format(observation=observation)
        instructions.append(instruction)
    
    
    def _create_instance(idx, instruction):
        prompt_formatted = templates[args.prefix].format(prompt=instruction)

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": idx}
        }
    train_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i]) for i in range(args.train_size)])
    test_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i]) for i in range(args.train_size, args.train_size + args.test_size)])


    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn

    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))

if __name__ == "__main__":
    main()