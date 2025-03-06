import numpy as np
import yaml
from datasets import Dataset, load_dataset
import os

class DatasetCreator:

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.env_name = self.config['env']['name']
        self.env_kwargs = self.config['env']['env_kwargs']
        self.data_dir = self.config['env']['data_dir']
        
        

    def create_dataset(self, start_seed, train_size, test_size):
        seeds = range(start_seed, start_seed + train_size + test_size)
        instructions = []
        for seed in seeds:
            # Generate instruction based on environment
            # This is a placeholder - actual implementation would depend on the environment
            instruction = f"Instruction for seed {seed}"
            instructions.append(instruction)
            
        def _create_instance(seed_idx, instruction):
            split = "train" if seed_idx < start_seed + train_size else "test"
            env_settings = {
                'env_name': self.env_name,
                'env_kwargs': self.env_kwargs,
                'seed': seed_idx
            }
            return {
                "data_source": self.env_name,
                "prompt": [{"role": "user", "content": instruction}],
                "ability": "bfs",
                "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
                "extra_info": {"split": split, **env_settings}
            }

        train_instances = [_create_instance(start_seed + i, '') for i in range(train_size)]
        test_instances = [_create_instance(start_seed + train_size + i, '') for i in range(test_size)]
        
        train_dataset = Dataset.from_list(train_instances)
        test_dataset = Dataset.from_list(test_instances)

        def make_map_fn(split):
            def process_fn(example, idx):
                return example
            return process_fn

        
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        train_dataset.to_parquet(os.path.join(self.data_dir, 'train.parquet'))
        test_dataset.to_parquet(os.path.join(self.data_dir, 'test.parquet'))
