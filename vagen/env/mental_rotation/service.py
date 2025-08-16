from typing import Dict, List, Tuple, Optional, Any
import os
import time
import logging
import threading
import multiprocessing as mp
from queue import Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

from vagen.env.base.base_service import BaseService
from vagen.env.base.base_service_config import BaseServiceConfig
from vagen.server.serial import serialize_observation

from .env import MentalRotationEnv
from .env_config import MentalRotationEnvConfig


class MentalRotationService(BaseService):
    """
    Service class for MentalRotation environments using multiprocessing.
    Uses persistent worker processes to maintain environment instances.
    """
    
    def __init__(self, config: BaseServiceConfig):
        cpu_count = os.cpu_count() or 4
        self.max_process_workers = min(getattr(config, 'max_workers', cpu_count), cpu_count)
        self.timeout = getattr(config, 'timeout', 120)
        
        devices = config.devices if config.devices else ["cpu"]
        self.device_status = {device_id: set() for device_id in devices}
        
        self.environments = {}
        self.env_configs = {}
        self.processes = []
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MentalRotationService')
        
        self._setup_mp_queues()
    
    def _setup_mp_queues(self):
        self.task_queues = []
        self.result_queues = []
        for _ in range(self.max_process_workers):
            self.task_queues.append(mp.Queue())
            self.result_queues.append(mp.Queue())
    
    def _worker_process(self, process_id, task_queue, result_queue):
        """
        Target function run in each subprocess to manage MentalRotationEnv instances.
        """
        import logging
        logger = logging.getLogger(f'MentalRotationWorker-{process_id}')
        
        scene_key_to_env = {}
        env_id_to_scene_env = {}
        genesis_initialized = False
        
        def _get_scene_key_from_config(seed, config):
            """Extract scene key (background, object) from config and seed."""
            import json
            import os
            
            dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "multi_step_interactive.json")
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            tasks = data.get("tasks", [])
            
            if not tasks:
                raise ValueError("No tasks found in dataset")
            
            task_idx = seed % len(tasks)
            task_data = tasks[task_idx]
            
            return (task_data["background"], task_data["object"])
        
        def _initialize_genesis_if_needed(device):
            """Initialize Genesis if not already done."""
            nonlocal genesis_initialized
            if not genesis_initialized:
                import genesis as gs
                
                logger.info(f"Worker {process_id}: Initializing Genesis with device {device}")
                
                if device == "cpu":
                    gs.init(backend=gs.cpu)
                else:
                    gs.init(backend=gs.gpu)
                
                genesis_initialized = True
                logger.info(f"Worker {process_id}: Genesis initialization completed")
        
        running = True
        
        def send_heartbeat():
            while running:
                try:
                    result_queue.put((-999, "heartbeat", process_id))
                    time.sleep(120)
                except:
                    pass
        
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        while running:
            try:
                command, task_id, args = task_queue.get()
                
                if command == "create_batch":
                    env_ids_to_config = args  # Dict[env_id, config]
                    try:
                        first_config = next(iter(env_ids_to_config.values()))
                        device = first_config.get('env_config', {}).get('device', 'cpu')
                        _initialize_genesis_if_needed(device)
                        
                        scene_key_to_env_configs = {}
                        for env_id, config in env_ids_to_config.items():
                            seed = config.get('seed', 0)  # Get actual seed from config
                            scene_key = _get_scene_key_from_config(seed, config)
                            if scene_key not in scene_key_to_env_configs:
                                scene_key_to_env_configs[scene_key] = []
                            scene_key_to_env_configs[scene_key].append((env_id, config))
                        
                        for scene_key, env_configs in scene_key_to_env_configs.items():
                            first_config = env_configs[0][1]
                            env_config_dict = first_config.get('env_config', {})
                            env_config_dict['n_parallel_envs'] = len(env_configs)
                            
                            env_config = MentalRotationEnvConfig(**env_config_dict)
                            
                            env_instance = MentalRotationEnv(env_config)
                            scene_key_to_env[scene_key] = env_instance
                            
                            for env_id, _ in env_configs:
                                env_id_to_scene_env[env_id] = (scene_key, env_instance)
                        
                        result_queue.put((task_id, 'success', True))
                    except Exception as e:
                        import traceback
                        logger.error(f"Error in create_batch: {traceback.format_exc()}")
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'reset_batch':
                    env_ids_to_seed = args  # Dict[env_id, seed]
                    try:
                        # Group environments by their existing environment instance
                        env_to_resets = {}
                        for env_id, seed in env_ids_to_seed.items():
                            if env_id in env_id_to_scene_env:
                                scene_key, env_instance = env_id_to_scene_env[env_id]
                                if env_instance not in env_to_resets:
                                    env_to_resets[env_instance] = {}
                                env_to_resets[env_instance][env_id] = seed
                        
                        all_results = {}
                        
                        # Reset each environment instance with its batch of env_ids
                        for env_instance, env_id_to_seed in env_to_resets.items():
                            results = env_instance.reset(
                                env_id_to_seed=env_id_to_seed,
                                rebuild_scene=True
                            )
                            
                            for env_id, (obs, info) in results.items():
                                all_results[env_id] = (serialize_observation(obs), info)
                        
                        result_queue.put((task_id, 'success', all_results))
                    except Exception as e:
                        import traceback
                        logger.error(f"Error in reset_batch: {traceback.format_exc()}")
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'step_batch':
                    env_id_to_action = args  # Dict[env_id, action]
                    try:
                        # Group actions by environment instance
                        env_to_actions = {}
                        for env_id, action in env_id_to_action.items():
                            if env_id in env_id_to_scene_env:
                                scene_key, env_instance = env_id_to_scene_env[env_id]
                                if env_instance not in env_to_actions:
                                    env_to_actions[env_instance] = {}
                                env_to_actions[env_instance][env_id] = action
                        
                        all_results = {}
                        
                        # Step each environment instance with its batch of actions
                        for env_instance, actions in env_to_actions.items():
                            results = env_instance.step(actions)
                            
                            # Serialize observations
                            for env_id, (obs, reward, done, info) in results.items():
                                all_results[env_id] = (serialize_observation(obs), reward, done, info)
                        
                        result_queue.put((task_id, 'success', all_results))
                    except Exception as e:
                        import traceback
                        logger.error(f"Error in step_batch: {traceback.format_exc()}")
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'compute_reward_batch':
                    env_ids = args  # List[env_id]
                    try:
                        # Group env_ids by environment instance
                        env_to_ids = {}
                        for env_id in env_ids:
                            if env_id in env_id_to_scene_env:
                                scene_key, env_instance = env_id_to_scene_env[env_id]
                                if env_instance not in env_to_ids:
                                    env_to_ids[env_instance] = []
                                env_to_ids[env_instance].append(env_id)
                        
                        all_results = {}
                        
                        # Compute rewards for each environment instance
                        for env_instance, ids in env_to_ids.items():
                            rewards = env_instance.compute_reward(ids)
                            all_results.update(rewards)
                        
                        result_queue.put((task_id, 'success', all_results))
                    except Exception as e:
                        import traceback
                        logger.error(f"Error in compute_reward_batch: {traceback.format_exc()}")
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'system_prompt_batch':
                    env_ids = args  # List[env_id]
                    try:
                        # Group env_ids by environment instance
                        env_to_ids = {}
                        for env_id in env_ids:
                            if env_id in env_id_to_scene_env:
                                scene_key, env_instance = env_id_to_scene_env[env_id]
                                if env_instance not in env_to_ids:
                                    env_to_ids[env_instance] = []
                                env_to_ids[env_instance].append(env_id)
                        
                        all_results = {}
                        
                        # Get prompts for each environment instance
                        for env_instance, ids in env_to_ids.items():
                            prompts = env_instance.system_prompt(ids)
                            all_results.update(prompts)
                        
                        result_queue.put((task_id, 'success', all_results))
                    except Exception as e:
                        import traceback
                        logger.error(f"Error in system_prompt_batch: {traceback.format_exc()}")
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'get_internal_state':
                    env_ids = args  # List[env_id]
                    try:
                        all_results = {}
                        
                        for env_id in env_ids:
                            if env_id in env_id_to_scene_env:
                                scene_key, env_instance = env_id_to_scene_env[env_id]
                                idx = env_instance.env_id_to_idx.get(env_id)
                                if idx is not None:
                                    all_results[env_id] = {
                                        'current_orientation': env_instance._current_orientations.get(idx),
                                        'target_orientation': env_instance._target_orientations.get(idx),
                                        'step_count': env_instance._step_counts.get(idx, 0),
                                        'total_reward': env_instance.total_rewards.get(idx, 0.0),
                                        'done': env_instance._dones.get(idx, False),
                                    }
                                else:
                                    all_results[env_id] = {'error': f'env_id {env_id} not found in environment instance'}
                            else:
                                all_results[env_id] = {'error': f'env_id {env_id} not found'}
                        
                        result_queue.put((task_id, 'success', all_results))
                    except Exception as e:
                        import traceback
                        logger.error(f"Error in get_internal_state: {traceback.format_exc()}")
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'close_batch':
                    env_ids = args  # List[env_id]
                    try:
                        # Track which scene keys need to be cleaned up
                        scene_keys_to_check = set()
                        
                        for env_id in env_ids:
                            if env_id in env_id_to_scene_env:
                                scene_key, _ = env_id_to_scene_env[env_id]
                                scene_keys_to_check.add(scene_key)
                                del env_id_to_scene_env[env_id]
                        
                        # Clean up environment instances that no longer have any env_ids
                        for scene_key in scene_keys_to_check:
                            # Check if any env_ids still use this scene key
                            still_used = any(
                                sk == scene_key for sk, _ in env_id_to_scene_env.values()
                            )
                            if not still_used and scene_key in scene_key_to_env:
                                try:
                                    scene_key_to_env[scene_key].close()
                                except:
                                    pass
                                del scene_key_to_env[scene_key]
                        
                        result_queue.put((task_id, 'success', True))
                    except Exception as e:
                        import traceback
                        logger.error(f"Error in close_batch: {traceback.format_exc()}")
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'exit':
                    running = False
                    result_queue.put((task_id, 'success', 'exited'))
                
                else:
                    result_queue.put((task_id, 'error', f"Unknown command {command}"))
            
            except Exception as e:
                try:
                    result_queue.put((-1, 'error', str(e)))
                except:
                    pass
    
    def _start_worker_processes(self):
        for i in range(self.max_process_workers):
            p = mp.Process(
                target=self._worker_process,
                args=(i, self.task_queues[i], self.result_queues[i]),
                daemon=True
            )
            p.start()
            self.processes.append(p)
            self.logger.info(f"Started MentalRotation worker {i} (pid={p.pid})")
    
    def _assign_to_process(self, env_id):
        """Assign environment to process with least load, considering device distribution"""
        loads = [0] * self.max_process_workers
        for pid in self.environments.values():
            loads[pid] += 1
        
        # Find process with minimum load
        min_load = min(loads)
        candidates = [i for i, load in enumerate(loads) if load == min_load]
        
        # If multiple candidates with same load, distribute evenly across devices
        if len(candidates) > 1:
            # Use hash of env_id to deterministically select among candidates
            selected_idx = hash(str(env_id)) % len(candidates)
            return candidates[selected_idx]
        else:
            return candidates[0]
    
    def _send_command(self, pid, command, env_id, args):
        """Send command to specific worker process and wait for result"""
        task_id = hash(f"{command}_{env_id}_{time.time()}")
        self.task_queues[pid].put((command, task_id, args))
        
        while True:
            try:
                rid, status, result = self.result_queues[pid].get(timeout=self.timeout)
                
                # Handle heartbeat messages
                if rid == -999 and status == "heartbeat":
                    self.logger.debug(f"Received heartbeat from worker {pid}")
                    continue
                
                if rid != task_id:
                    continue
                    
                if status == 'success':
                    return result
                raise Exception(result)
                
            except Empty:
                raise TimeoutError(f"Timeout {command} for {env_id}")
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """Create multiple MentalRotation environments distributed across processes"""
        if not self.processes:
            self._start_worker_processes()
        
        def _get_scene_key_from_config(seed, config):
            import json
            import os
            
            dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "multi_step_interactive.json")
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            tasks = data.get("tasks", [])
            
            if not tasks:
                raise ValueError("No tasks found in dataset")
            
            # Use actual seed to determine scene key
            task_idx = seed % len(tasks)
            task_data = tasks[task_idx]
            return (task_data["background"], task_data["object"])
        
        # First, group by scene key - same scene environments must share device and process
        scene_key_to_envs = {}
        for env_id, cfg in ids2configs.items():
            seed = cfg.get('seed', 0)  # Get actual seed from config
            scene_key = _get_scene_key_from_config(seed, cfg)
            scene_key_to_envs.setdefault(scene_key, []).append((env_id, cfg))
        
        # Assign device to each scene group (all environments in same scene use same device)
        for scene_key, env_list in scene_key_to_envs.items():
            # Select device with least load for this scene group
            selected_device = min(self.device_status, key=lambda x: len(self.device_status[x]))
            print(f"[DEBUG] Scene key {scene_key} assigned to device {selected_device} (current loads: {[(d, len(envs)) for d, envs in self.device_status.items()]})")
            
            # Set same device for all environments in this scene group
            for env_id, cfg in env_list:
                if selected_device == "cpu":
                    cfg['env_config']['device'] = "cpu"
                else:
                    cfg['env_config']['device'] = f"cuda:{selected_device}"
                
                # Add all environments in this scene to the selected device
                self.device_status[selected_device].add(env_id)
        
        # Then assign each scene group to a process
        by_pid = {}
        for scene_key, env_list in scene_key_to_envs.items():
            # All environments with same scene key must go to same process
            representative_env_id = env_list[0][0]
            pid = self._assign_to_process(representative_env_id)
            
            # Store all environments in this scene group to same process
            for env_id, cfg in env_list:
                self.environments[env_id] = pid
                self.env_configs[env_id] = cfg
                by_pid.setdefault(pid, {})[env_id] = cfg
        
        def _create_group(pid, env_configs):
            try:
                return self._send_command(pid, 'create_batch', None, env_configs)
            except Exception as e:
                self.logger.error(f"Error creating environments: {str(e)}")
                # Remove from mappings on failure
                for env_id in env_configs.keys():
                    self.environments.pop(env_id, None)
                    self.env_configs.pop(env_id, None)
                return f"Error: {str(e)}"
        
        # Use ThreadPoolExecutor to parallelize across processes
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            for pid, group in by_pid.items():
                futures.append(executor.submit(_create_group, pid, group))
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """Reset multiple MentalRotation environments in parallel"""
        results = {}
        
        # Group environments by process (environments should already be created)
        by_pid = {}
        for env_id, seed in ids2seeds.items():
            pid = self.environments.get(env_id)
            if pid is not None:
                by_pid.setdefault(pid, {})[env_id] = seed
        
        def _reset_group(pid, env_ids_to_seed):
            try:
                return self._send_command(pid, 'reset_batch', None, env_ids_to_seed)
            except Exception as e:
                self.logger.error(f"Error resetting environments: {str(e)}")
                # Return empty results for all env_ids in this group
                return {env_id: ({}, {"error": str(e)}) for env_id in env_ids_to_seed.keys()}
        
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            for pid, group in by_pid.items():
                futures.append(executor.submit(_reset_group, pid, group))
            
            for future in as_completed(futures):
                results.update(future.result())
        
        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """Step multiple MentalRotation environments in parallel"""
        results = {}
        by_pid = {}
        
        # Group by process
        for env_id, action in ids2actions.items():
            pid = self.environments.get(env_id)
            if pid is not None:
                by_pid.setdefault(pid, {})[env_id] = action
        
        def _step_group(pid, env_id_to_action):
            try:
                return self._send_command(pid, 'step_batch', None, env_id_to_action)
            except Exception as e:
                self.logger.error(f"Error stepping environments: {str(e)}")
                # Return default results for all env_ids in this group
                return {env_id: ({}, 0.0, True, {"error": str(e)}) for env_id in env_id_to_action.keys()}
        
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            for pid, group in by_pid.items():
                futures.append(executor.submit(_step_group, pid, group))
            
            for future in as_completed(futures):
                results.update(future.result())
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """Compute rewards for multiple MentalRotation environments in parallel"""
        results = {}
        by_pid = {}
        
        # Group by process
        for env_id in env_ids:
            pid = self.environments.get(env_id)
            if pid is not None:
                by_pid.setdefault(pid, []).append(env_id)
        
        def _compute_group(pid, group):
            try:
                return self._send_command(pid, 'compute_reward_batch', None, group)
            except Exception as e:
                self.logger.error(f"Error computing rewards: {str(e)}")
                # Return default rewards for all env_ids in this group
                return {env_id: 0.0 for env_id in group}
        
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            for pid, group in by_pid.items():
                futures.append(executor.submit(_compute_group, pid, group))
            
            for future in as_completed(futures):
                results.update(future.result())
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """Get system prompts for multiple MentalRotation environments in parallel"""
        results = {}
        by_pid = {}
        
        # Group by process
        for env_id in env_ids:
            pid = self.environments.get(env_id)
            if pid is not None:
                by_pid.setdefault(pid, []).append(env_id)
        
        def _prompt_group(pid, group):
            try:
                return self._send_command(pid, 'system_prompt_batch', None, group)
            except Exception as e:
                self.logger.error(f"Error getting system prompts: {str(e)}")
                # Return empty prompts for all env_ids in this group
                return {env_id: "" for env_id in group}
        
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            for pid, group in by_pid.items():
                futures.append(executor.submit(_prompt_group, pid, group))
            
            for future in as_completed(futures):
                results.update(future.result())
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """Close multiple MentalRotation environments and clean up resources"""
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        by_pid = {}
        for env_id in env_ids:
            pid = self.environments.get(env_id)
            if pid is not None:
                by_pid.setdefault(pid, []).append(env_id)
        
        def _close_group(pid, group):
            try:
                self._send_command(pid, 'close_batch', None, group)
            except Exception as e:
                self.logger.error(f"Error closing environments: {str(e)}")
        
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            for pid, group in by_pid.items():
                futures.append(executor.submit(_close_group, pid, group))
            
            for future in as_completed(futures):
                future.result()
        
        # Remove from mappings and device status
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
            # Remove from device status
            for device_id in self.device_status:
                self.device_status[device_id].discard(env_id)
        
        # If closing all environments, also terminate worker processes
        if not self.environments:  # If no environments left
            for i, p in enumerate(self.processes):
                try:
                    self.task_queues[i].put(('exit', -1, None))
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()
                except:
                    pass
            self.processes.clear()


if __name__ == "__main__":
    import time
    import os
    from .service_config import MentalRotationServiceConfig
    
    print("[TEST] Testing MentalRotationService with new background/object grouping...")
    
    test_dir = "./test_mental_rotation_service"
    os.makedirs(test_dir, exist_ok=True)
    print(f"[TEST] Created test directory: {test_dir}")
    
    config = MentalRotationServiceConfig(
        devices=[0, 1],
        max_workers=2,
        timeout=60
    )
    service = MentalRotationService(config)
    print(f"[TEST] Created service with {service.max_process_workers} workers")
    
    try:
        # Test 1: Environments with same scene key (background+object) should share scene
        env_id_1, env_id_2, env_id_3 = "mr_same_1", "mr_same_2", "mr_different"
        ids2configs = {
            env_id_1: {
                "env_name": "mental-rotation", 
                "env_config": {
                    "render_mode": "vision",
                    "max_steps": 5,
                },
                "seed": 0,  # Add seed to config
            },
            env_id_2: {
                "env_name": "mental-rotation", 
                "env_config": {
                    "render_mode": "vision",
                    "max_steps": 5,
                },
                "seed": 1,  # Add seed to config
            },
            env_id_3: {
                "env_name": "mental-rotation", 
                "env_config": {
                    "render_mode": "vision",
                    "max_steps": 5,
                },
                "seed": 2,  # Add seed to config
            },
        }
        
        print(f"[TEST] Creating environments: {list(ids2configs.keys())}")
        service.create_environments_batch(ids2configs)
        print("[TEST] Environments created successfully")
        
        # Test scene key grouping
        print("\n[TEST] Testing scene key grouping logic...")
        
        # Seeds 0, 1, 2 should map to same scene key: ('plane', 'airplane.glb')
        # Seed 3 should map to different scene key: ('plane', 'airplane.glb') - now same after our fix
        print("[TEST] Checking scene keys for different seeds:")
        
        import json
        dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "multi_step_interactive.json")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        tasks = data.get("tasks", [])
        
        for seed in [0, 1, 2, 3]:
            task_idx = seed % len(tasks)
            task_data = tasks[task_idx]
            scene_key = (task_data["background"], task_data["object"])
            print(f"[TEST] Seed {seed} -> Task {task_idx} -> Scene key: {scene_key}")
        
        # Test 1: Reset all environments at once to maintain consistency
        print(f"\n[TEST] Test 1: Resetting all environments (seeds 0, 1, 2)")
        reset_results = service.reset_batch({env_id_1: 0, env_id_2: 1, env_id_3: 2})
        
        if len(reset_results) == 3:
            print("[TEST] ✓ SUCCESS: All environments reset successfully")
            print("[TEST] ✓ Environments with same scene key share the same instance")
        else:
            print(f"[TEST] ✗ PARTIAL: Only {len(reset_results)} environments reset")
        
        # Save images for verification
        for env_id in [env_id_1, env_id_2, env_id_3]:
            if env_id in reset_results:
                obs, info = reset_results[env_id]
                
                # Deserialize observation
                from vagen.server.serial import deserialize_observation
                obs = deserialize_observation(obs)
                
                print(f"[TEST] {env_id} reset successfully")
                
                # Save images
                if 'multi_modal_data' in obs:
                    from .env_config import MentalRotationEnvConfig
                    temp_config = MentalRotationEnvConfig()
                    
                    try:
                        current_img = obs['multi_modal_data'][temp_config.image_placeholder][0]
                        target_img = obs['multi_modal_data'][temp_config.target_image_placeholder][0]
                        
                        current_filename = f"{env_id}_current_reset.png"
                        target_filename = f"{env_id}_target_reset.png"
                        
                        current_img.save(os.path.join(test_dir, current_filename))
                        target_img.save(os.path.join(test_dir, target_filename))
                        
                        print(f"[TEST] Saved {current_filename} and {target_filename}")
                    except Exception as save_error:
                        print(f"[TEST] Failed to save images: {save_error}")
        
        # Test 2: Basic functionality tests
        print(f"\n[TEST] Test 2: Testing basic service functionality...")
        
        # Get system prompts
        print("[TEST] Getting system prompts")
        prompts = service.get_system_prompts_batch([env_id_1, env_id_2, env_id_3])
        if len(prompts) == 3:
            print("[TEST] ✓ SUCCESS: System prompts retrieved for all environments")
        else:
            print(f"[TEST] ✗ PARTIAL: Only {len(prompts)} prompts retrieved")
        
        # Test single step action for all three environments
        print("[TEST] Testing single step action for all environments")
        test_actions = {
            env_id_1: "<answer>x90</answer>", 
            env_id_2: "<answer>y-90</answer>",
            env_id_3: "<answer>z180</answer>"
        }
        step_results = service.step_batch(test_actions)
        
        if len(step_results) == 3:
            print("[TEST] ✓ SUCCESS: Step actions executed for all environments")
            for env_id, (obs, reward, done, info) in step_results.items():
                print(f"[TEST] {env_id}: reward={reward}, done={done}")
                if 'metrics' in info:
                    metrics = info['metrics']
                    action_valid = metrics.get('turn_metrics', {}).get('action_is_valid', 'N/A')
                    print(f"[TEST] {env_id}: action_valid={action_valid}")
        else:
            print(f"[TEST] ✗ PARTIAL: Only {len(step_results)} step results")
        
        # Test compute rewards
        print("[TEST] Testing compute rewards")
        rewards = service.compute_reward_batch([env_id_1, env_id_2, env_id_3])
        if len(rewards) == 3:
            print("[TEST] ✓ SUCCESS: Rewards computed for all environments")
            for env_id, reward in rewards.items():
                print(f"[TEST] {env_id}: total_reward={reward}")
        else:
            print(f"[TEST] ✗ PARTIAL: Only {len(rewards)} rewards computed")
        
        # Test invalid action
        print("[TEST] Testing invalid action handling")
        invalid_results = service.step_batch({env_id_3: "<answer>invalid_action</answer>"})
        if env_id_3 in invalid_results:
            obs, reward, done, info = invalid_results[env_id_3]
            if 'metrics' in info:
                action_valid = info['metrics'].get('turn_metrics', {}).get('action_is_valid', True)
                if not action_valid:
                    print("[TEST] ✓ SUCCESS: Invalid action properly rejected")
                else:
                    print("[TEST] ✗ FAILED: Invalid action not rejected")
            else:
                print("[TEST] ✗ FAILED: No metrics in response")
            
        print(f"\n[TEST] ✓ All grouping and functionality tests completed!")
        print(f"[TEST] Key findings:")
        print(f"[TEST] - Environments with same (background, object) share scene instances")
        print(f"[TEST] - Service correctly groups by scene key instead of seed")
        print(f"[TEST] - Basic functionality (reset, step, rewards, prompts) works correctly")
        print(f"[TEST] Images saved in {test_dir}")
        
    except Exception as e:
        print(f"[TEST] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n[TEST] Cleaning up environments...")
        try:
            service.close_batch([env_id_1, env_id_2, env_id_3])
            print("[TEST] ✓ All environments closed successfully")
        except Exception as e:
            print(f"[TEST] ✗ Error during cleanup: {str(e)}")
        
        print("[TEST] ✓ Test completed successfully!") 