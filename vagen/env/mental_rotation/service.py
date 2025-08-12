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
        # Determine CPU count and max workers
        cpu_count = os.cpu_count() or 4
        self.max_process_workers = min(getattr(config, 'max_workers', cpu_count), cpu_count)
        self.timeout = getattr(config, 'timeout', 120)
        
        # Mapping: env_id -> process_id
        self.environments = {}
        # Store raw configs
        self.env_configs = {}
        # Worker processes list
        self.processes = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MentalRotationService')
        
        # Setup MP queues
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
        # 在worker进程中创建独立的logger
        import logging
        logger = logging.getLogger(f'MentalRotationWorker-{process_id}')
        
        local_environments = {}
        local_env_configs = {}
        genesis_initialized = False
        
        running = True
        
        # Heartbeat thread to indicate process is alive
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
                
                if command == "create":
                    env_id, config = args
                    try:
                        env_name = config.get('env_name', 'mental-rotation')
                        if env_name != 'mental-rotation':
                            result_queue.put((task_id, "error", f"Expected environment type 'mental-rotation', got '{env_name}'"))
                            continue
                        
                        env_config_dict = config.get('env_config', {})
                        env_config = MentalRotationEnvConfig(**env_config_dict)
                        
                        if not genesis_initialized:
                            import genesis as gs
                            device = env_config.device
                            logger.info(f"Worker {process_id}: Initializing Genesis with device {device}")
                            
                            if device == "cpu":
                                gs.init(backend=gs.cpu)
                            elif 'cuda' in device:
                                gs.init(backend=gs.gpu)
                            else:
                                gs.init(backend=gs.cpu)
                            
                            genesis_initialized = True
                            logger.info(f"Worker {process_id}: Genesis initialization completed")
                        
                        env = MentalRotationEnv(env_config)
                        
                        local_environments[env_id] = env
                        local_env_configs[env_id] = env_config
                        
                        result_queue.put((task_id, "success", env_id))
                    except Exception as e:
                        result_queue.put((task_id, "error", f"Error creating environment {env_id}: {str(e)}"))
                
                elif command == 'reset':
                    env_id, seed = args
                    if env_id not in local_environments:
                        result_queue.put((task_id, 'error', f"Env {env_id} not found"))
                        continue
                    try:
                        obs, info = local_environments[env_id].reset(seed)
                        s_obs = serialize_observation(obs)
                        result_queue.put((task_id, 'success', (s_obs, info)))
                    except Exception as e:
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'step':
                    env_id, action = args
                    if env_id not in local_environments:
                        result_queue.put((task_id, 'error', f"Env {env_id} not found"))
                        continue
                    try:
                        obs, reward, done, info = local_environments[env_id].step(action)
                        s_obs = serialize_observation(obs)
                        result_queue.put((task_id, 'success', (s_obs, reward, done, info)))
                    except Exception as e:
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'compute_reward':
                    env_id = args
                    if env_id not in local_environments:
                        result_queue.put((task_id, 'error', f"Env {env_id} not found"))
                        continue
                    try:
                        reward = local_environments[env_id].compute_reward()
                        result_queue.put((task_id, 'success', reward))
                    except Exception as e:
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'system_prompt':
                    env_id = args
                    if env_id not in local_environments:
                        result_queue.put((task_id, 'error', f"Env {env_id} not found"))
                        continue
                    try:
                        prompt = local_environments[env_id].system_prompt()
                        result_queue.put((task_id, 'success', prompt))
                    except Exception as e:
                        result_queue.put((task_id, 'error', str(e)))
                
                elif command == 'close':
                    env_id = args
                    if env_id in local_environments:
                        try:
                            local_environments[env_id].close()
                        except:
                            pass
                        local_environments.pop(env_id, None)
                        local_env_configs.pop(env_id, None)
                    result_queue.put((task_id, 'success', True))
                
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
        """Assign environment to process with least load"""
        loads = [0] * self.max_process_workers
        for pid in self.environments.values():
            loads[pid] += 1
        return loads.index(min(loads))
    
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
        
        # Group environments by assigned process
        by_pid = {}
        for env_id, cfg in ids2configs.items():
            pid = self._assign_to_process(env_id)
            by_pid.setdefault(pid, []).append((env_id, cfg))
            # Store assigned process ID and config for later use
            self.environments[env_id] = pid
            self.env_configs[env_id] = cfg
        
        # Define function to create environments within a process group
        def _create_group(pid, group):
            results = {}
            
            # Process environments sequentially within each process group
            for env_id, cfg in group:
                try:
                    result = self._send_command(pid, 'create', env_id, (env_id, cfg))
                    results[env_id] = result
                except Exception as e:
                    error_msg = f"Error creating environment {env_id}: {str(e)}"
                    self.logger.error(error_msg)
                    # Remove from mappings on failure
                    self.environments.pop(env_id, None)
                    self.env_configs.pop(env_id, None)
                    results[env_id] = f"Error: {error_msg}"
            
            return results
        
        # Use ThreadPoolExecutor to parallelize across processes
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            # Submit creation tasks for each process group
            for pid, group in by_pid.items():
                futures.append(executor.submit(_create_group, pid, group))
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """Reset multiple MentalRotation environments in parallel"""
        results = {}
        by_pid = {}
        
        # Group by process
        for env_id, seed in ids2seeds.items():
            pid = self.environments.get(env_id)
            if pid is not None:
                by_pid.setdefault(pid, []).append((env_id, seed))
        
        def _reset_group(pid, group):
            out = {}
            for env_id, seed in group:
                try:
                    obs, info = self._send_command(pid, 'reset', env_id, (env_id, seed))
                    out[env_id] = (obs, info)
                except Exception as e:
                    self.logger.error(f"Error resetting environment {env_id}: {str(e)}")
                    out[env_id] = ({}, {"error": str(e)})
            return out
        
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
                by_pid.setdefault(pid, []).append((env_id, action))
        
        def _step_group(pid, group):
            out = {}
            for env_id, action in group:
                try:
                    obs, reward, done, info = self._send_command(pid, 'step', env_id, (env_id, action))
                    out[env_id] = (obs, reward, done, info)
                except Exception as e:
                    self.logger.error(f"Error stepping environment {env_id}: {str(e)}")
                    out[env_id] = ({}, 0.0, True, {"error": str(e)})
            return out
        
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
            out = {}
            for env_id in group:
                try:
                    reward = self._send_command(pid, 'compute_reward', env_id, env_id)
                    out[env_id] = reward
                except Exception as e:
                    self.logger.error(f"Error computing reward for environment {env_id}: {str(e)}")
                    out[env_id] = 0.0
            return out
        
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
            out = {}
            for env_id in group:
                try:
                    prompt = self._send_command(pid, 'system_prompt', env_id, env_id)
                    out[env_id] = prompt
                except Exception as e:
                    self.logger.error(f"Error getting system prompt for environment {env_id}: {str(e)}")
                    out[env_id] = ""
            return out
        
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
            for env_id in group:
                try:
                    self._send_command(pid, 'close', env_id, env_id)
                except Exception as e:
                    self.logger.error(f"Error closing environment {env_id}: {str(e)}")
        
        with ThreadPoolExecutor(max_workers=self.max_process_workers) as executor:
            futures = []
            for pid, group in by_pid.items():
                futures.append(executor.submit(_close_group, pid, group))
            
            for future in as_completed(futures):
                future.result()
        
        # Remove from mappings
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
        
        # If closing all environments, also terminate worker processes
        if set(env_ids) == set(self.environments.keys()):
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
    
    print("[TEST] Testing MentalRotationService functionality...")
    
    test_dir = "./test_mental_rotation_service"
    os.makedirs(test_dir, exist_ok=True)
    print(f"[TEST] Created test directory: {test_dir}")
    
    config = MentalRotationServiceConfig(
        max_workers=4,
        timeout=60
    )
    service = MentalRotationService(config)
    print(f"[TEST] Created service with {service.max_process_workers} workers")
    
    try:
        env_id_1, env_id_2 = "mr_test_1", "mr_test_2"
        ids2configs = {
            env_id_1: {
                "env_name": "mental-rotation",
                "env_config": {
                    "render_mode": "vision", 
                    "max_steps": 5,
                    "device": "cuda"
                },
            },
            env_id_2: {
                "env_name": "mental-rotation", 
                "env_config": {
                    "render_mode": "vision",
                    "max_steps": 5,
                    "device": "cuda"
                },
            },
        }
        
        print(f"[TEST] Creating environments: {list(ids2configs.keys())}")
        service.create_environments_batch(ids2configs)
        print("[TEST] Environments created successfully")
        
        # 3. Reset environments
        seed = 42
        print(f"[TEST] Resetting environments with seed={seed}")
        reset_results = service.reset_batch({env_id_1: seed, env_id_2: seed})
        
        step_counter = 0
        for env_id in [env_id_1, env_id_2]:
            if env_id in reset_results:
                obs, info = reset_results[env_id]
                
                # 反序列化observation以获取真正的PIL图片对象
                from vagen.server.serial import deserialize_observation
                obs = deserialize_observation(obs)
                
                print(f"[TEST] {env_id} reset - obs keys: {list(obs.keys())}")
                if 'multi_modal_data' in obs:
                    mm_keys = list(obs['multi_modal_data'].keys())
                    print(f"[TEST] {env_id} multi_modal_data keys: {mm_keys}")
                    for k in mm_keys:
                        print(f"[TEST] {env_id} multi_modal_data[{k}] length: {len(obs['multi_modal_data'][k])}")
                    
                    # 保存重置后的图片 - 使用类似env.py的直接方法
                    from .env_config import MentalRotationEnvConfig
                    temp_config = MentalRotationEnvConfig()
                    
                    try:
                        current_img = obs['multi_modal_data'][temp_config.image_placeholder][0]
                        target_img = obs['multi_modal_data'][temp_config.target_image_placeholder][0]
                        
                        current_filename = f"{env_id}_current_step{step_counter}.png"
                        target_filename = f"{env_id}_target_step{step_counter}.png"
                        
                        current_img.save(os.path.join(test_dir, current_filename))
                        target_img.save(os.path.join(test_dir, target_filename))
                        
                        print(f"[TEST] Saved {current_filename}")
                        print(f"[TEST] Saved {target_filename}")
                    except Exception as save_error:
                        print(f"[TEST] Failed to save images: {save_error}")
                
                print(f"[TEST] {env_id} obs_str (first 100 chars): {obs['obs_str'][:100]}...")
            else:
                print(f"[TEST] ERROR: {env_id} not in reset results")
        
        # 4. Get system prompts
        print("[TEST] Getting system prompts")
        prompts = service.get_system_prompts_batch([env_id_1, env_id_2])
        for env_id in [env_id_1, env_id_2]:
            if env_id in prompts:
                print(f"[TEST] {env_id} system prompt (first 100 chars): {prompts[env_id][:100]}...")
            else:
                print(f"[TEST] ERROR: {env_id} not in prompt results")
        
        # 5. Step with valid actions (multiple steps)
        test_actions_sequence = [
            {env_id_1: "<answer>x90</answer>", env_id_2: "<answer>y-90</answer>"},
            {env_id_1: "<answer>z180</answer>", env_id_2: "<answer>x270</answer>"},
            {env_id_1: "<answer>y90</answer>", env_id_2: "<answer>z-180</answer>"}
        ]
        
        for step_num, test_actions in enumerate(test_actions_sequence, 1):
            step_counter += 1
            print(f"[TEST] Step {step_num} - Stepping with actions: {test_actions}")
            step_results = service.step_batch(test_actions)
            
            for env_id in [env_id_1, env_id_2]:
                if env_id in step_results:
                    obs, reward, done, info = step_results[env_id]
                    
                    # 反序列化observation以获取真正的PIL图片对象
                    from vagen.server.serial import deserialize_observation
                    obs = deserialize_observation(obs)
                    
                    print(f"[TEST] {env_id} step {step_num} result - reward: {reward}, done: {done}")
                    if 'metrics' in info:
                        metrics = info['metrics']
                        print(f"[TEST] {env_id} action_valid: {metrics.get('turn_metrics', {}).get('action_is_valid', 'N/A')}")
                        print(f"[TEST] {env_id} action_effective: {metrics.get('turn_metrics', {}).get('action_is_effective', 'N/A')}")
                    
                    # 保存步骤后的图片 - 使用类似env.py的直接方法
                    if 'multi_modal_data' in obs:
                        from .env_config import MentalRotationEnvConfig
                        temp_config = MentalRotationEnvConfig()
                        
                        try:
                            current_img = obs['multi_modal_data'][temp_config.image_placeholder][0]
                            target_img = obs['multi_modal_data'][temp_config.target_image_placeholder][0]
                            
                            current_filename = f"{env_id}_current_step{step_counter}.png"
                            target_filename = f"{env_id}_target_step{step_counter}.png"
                            
                            current_img.save(os.path.join(test_dir, current_filename))
                            target_img.save(os.path.join(test_dir, target_filename))
                            
                            print(f"[TEST] Saved {current_filename}")
                            print(f"[TEST] Saved {target_filename}")
                        except Exception as save_error:
                            print(f"[TEST] Failed to save step images: {save_error}")
                    
                    if done:
                        print(f"[TEST] {env_id} completed the task!")
                        break
                else:
                    print(f"[TEST] ERROR: {env_id} not in step results")
        
        # 6. Compute rewards
        print("[TEST] Computing rewards")
        rewards = service.compute_reward_batch([env_id_1, env_id_2])
        for env_id in [env_id_1, env_id_2]:
            if env_id in rewards:
                print(f"[TEST] {env_id} total reward: {rewards[env_id]}")
            else:
                print(f"[TEST] ERROR: {env_id} not in reward results")
        
        # 7. Test invalid action
        print("[TEST] Testing invalid action")
        step_counter += 1
        invalid_step_results = service.step_batch({env_id_1: "<answer>invalid_action</answer>"})
        if env_id_1 in invalid_step_results:
            obs, reward, done, info = invalid_step_results[env_id_1]
            
            # 反序列化observation以获取真正的PIL图片对象
            from vagen.server.serial import deserialize_observation
            obs = deserialize_observation(obs)
            
            print(f"[TEST] Invalid action - reward: {reward}")
            if 'metrics' in info:
                metrics = info['metrics']
                print(f"[TEST] Invalid action - action_valid: {metrics.get('turn_metrics', {}).get('action_is_valid', 'N/A')}")
            
            # 保存无效动作后的图片 - 使用类似env.py的直接方法
            if 'multi_modal_data' in obs:
                from .env_config import MentalRotationEnvConfig
                temp_config = MentalRotationEnvConfig()
                
                try:
                    current_img = obs['multi_modal_data'][temp_config.image_placeholder][0]
                    target_img = obs['multi_modal_data'][temp_config.target_image_placeholder][0]
                    
                    current_filename = f"{env_id_1}_current_invalid_step{step_counter}.png"
                    target_filename = f"{env_id_1}_target_invalid_step{step_counter}.png"
                    
                    current_img.save(os.path.join(test_dir, current_filename))
                    target_img.save(os.path.join(test_dir, target_filename))
                    
                    print(f"[TEST] Saved {current_filename}")
                    print(f"[TEST] Saved {target_filename}")
                except Exception as save_error:
                    print(f"[TEST] Failed to save invalid action images: {save_error}")
        
        print(f"[TEST] All tests completed successfully! Images saved in {test_dir}")
        
    except Exception as e:
        print(f"[TEST] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 8. Clean up
        print("[TEST] Cleaning up environments...")
        try:
            service.close_batch([env_id_1, env_id_2])
            print("[TEST] Environments closed")
        except Exception as e:
            print(f"[TEST] Error during cleanup: {str(e)}")
        
        print("[TEST] Test completed.") 