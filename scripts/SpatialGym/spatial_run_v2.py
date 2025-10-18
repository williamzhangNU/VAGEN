#!/usr/bin/env python3
"""
Split SpatialGym runner: separate exploration, evaluation, and cogmap phases.

Phases:
- exploration: Run dataset creation only (generates exploration histories)
- evaluation: Build eval messages and run inference
- cogmap: Build cogmap messages and run inference
"""
import argparse
import os
import sys
import shlex
import subprocess
import time
import socket
import json
from pathlib import Path
from typing import Dict, Any, List
import yaml as pyyaml
import urllib.request
import threading
from datetime import datetime
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


def is_port_available(port: int, host: str = '0.0.0.0') -> bool:
    """Check if a port is available (not in use)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0
    except (socket.error, OSError):
        return True


def find_available_port(start_port: int = 5000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")


def get_adaptive_port(user_port: int = None, default_start: int = 5000) -> int:
    """Get a port for the server. If user_port is specified and available, use it.
    Otherwise, find an available port starting from default_start."""
    if user_port is not None:
        if is_port_available(user_port):
            return user_port
        else:
            print(f"Warning: User-specified port {user_port} is not available, finding alternative...")
            return find_available_port(default_start)
    return find_available_port(default_start)


def parse_args():
    p = argparse.ArgumentParser(
        description="SpatialGym runner with separated phases: exploration, evaluation, cogmap."
    )
    # Phase selection
    p.add_argument("--phase", type=str, default="all", 
                   choices=['exploration', 'evaluation', 'cogmap', 'all'],
                   help="Which phase to run: exploration, evaluation, cogmap, or all")
    
    # Common parameters
    p.add_argument("--exp-type", type=str, dest="exp_type", 
                   choices=["active", "passive"], default="active",
                   help="Experiment type: active or passive. Default: active")
    p.add_argument("--model-name", type=str, default="gpt-4o-mini",
                   help="Model identifier. Default: gpt-4o-mini")
    p.add_argument("--data-dir", type=str, dest="data_dir", default=None, 
                   help="Data directory root. Default: data")
    p.add_argument("--output-root", type=str, dest="output_root", default="results", 
                   help="Root dir for output. Default: results")
    
    # Exploration phase parameters
    p.add_argument("--num", type=int, default=1, 
                   help="Number of samples per task (exploration phase). Default: 1")
    p.add_argument("--render-mode", type=str, dest="render_mode", default="vision", 
                   help="Environment render mode (vision or text). Default: vision")
    p.add_argument("--seed-range", type=str, dest="seed_range", default=None, 
                   help="Seed range 'start-end' (0-based), e.g., 0-24")
    p.add_argument("--enable-think", type=int, dest="enable_think", choices=[0,1], default=1, 
                   help="1 to enable think, 0 to disable (default: 1)")
    p.add_argument("--proxy-agent", type=str, dest="proxy_agent", default=None, 
                   choices=["scout","strategist","oracle"], 
                   help="Proxy agent for passive tasks (required if exp-type is passive)")
    p.add_argument("--all-override", action="store_true", dest="all_override", 
                   help="Override all history (delete whole sample path)")
    
    # Evaluation/Cogmap phase parameters
    p.add_argument("--eval-task-counts", type=str, dest="eval_task_counts", default=None,
                   help='JSON string for eval task counts, e.g., {"qa": 2, "dir": 1}. If omitted, use inference_config.yaml eval_task_counts or default {"qa": 1}')
    p.add_argument("--inference-seed", type=int, dest="inference_seed", default=0,
                   help="Seed for evaluation task generation. Default: 0")
    p.add_argument("--eval-override", action="store_true", dest="eval_override", 
                   help="Override evaluation history (delete evaluation json only)")
    p.add_argument("--cogmap-override", action="store_true", dest="cogmap_override", 
                   help="Override cognitive map cache (regenerate cogmap prompts)")
    p.add_argument("--cogmap-reevaluate", action="store_true", dest="cogmap_reevaluate",
                   help="Re-evaluate existing cognitive maps (pass to CognitiveMapManager)")
    
    # Inference parameters
    p.add_argument("--inference-mode", type=str, dest="inference_mode", 
                   choices=['batch', 'direct'], default='direct',
                   help="Inference mode: batch (OpenAI batch API) or direct. Default: direct")
    
    # Server options
    p.add_argument("--no-server", action="store_true", dest="no_server", 
                   help="Do not start internal env server (assume external server is running)")
    p.add_argument("--server-host", type=str, dest="server_host", default="127.0.0.1", 
                   help="Server host to bind/connect")
    p.add_argument("--server-port", type=int, dest="server_port", default=5000, 
                   help="Server port to bind/connect")
    
    # Base config paths
    p.add_argument("--base-env", type=str, dest="base_env", 
                   default=str(SCRIPT_DIR / "base_env_config.yaml"))
    p.add_argument("--base-infer", type=str, dest="base_infer", 
                   default=str(SCRIPT_DIR / "inference_config.yaml"))
    p.add_argument("--base-model", type=str, dest="base_model", 
                   default=str(SCRIPT_DIR / "base_model_config.yaml"))
    
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return pyyaml.safe_load(f)


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        pyyaml.safe_dump(data, f, sort_keys=False)


def model_segment(model_name: str) -> str:
    return model_name.replace("\\", "/").rstrip("/").split("/")[-1]


def build_tmp_paths(run_id: str, task_key: str) -> Dict[str, Path]:
    base = SCRIPT_DIR / "tmp" / run_id / task_key
    return {
        "base": base,
        "env": base / "env.yaml",
        "infer": base / "inference.yaml",
        "model": base / "model.yaml",
    }


def patch_env_yaml(env_cfg: Dict[str, Any], exp_type: str, num: int, render_mode="vision", 
                   seed_opts: tuple[int, int] | None = None, enable_think: int | None = None, 
                   data_dir: str | None = None, proxy_agent: str | None = None) -> Dict[str, Any]:
    """Return env config by selecting based on exp_type from custom_envs.
    
    Args:
        env_cfg: Base environment config
        exp_type: 'active' or 'passive'
        num: Number of samples
        render_mode: Render mode
        seed_opts: Seed range tuple
        enable_think: Enable thinking
        data_dir: Data directory
        proxy_agent: Proxy agent for passive mode
        
    Returns:
        Dict with selected task config
    """
    custom_envs = env_cfg.get("custom_envs", {}) or {}
    
    # Select a default task based on exp_type
    # For active: use ActiveRot, for passive: use PassiveRot
    task_key = "ActiveRot" if exp_type == "active" else "PassiveRot"
    
    if task_key not in custom_envs:
        raise ValueError(f"Task {task_key} not found in custom_envs")
    
    selected = dict(custom_envs[task_key])
    selected["test_size"] = int(num)
    selected["env_config"]['render_mode'] = render_mode
    selected["env_config"]['exp_type'] = exp_type
    
    if data_dir:
        selected["env_config"]["data_dir"] = data_dir
    if seed_opts:
        selected["env_config"].setdefault("kwargs", {})
        selected["env_config"]["kwargs"]["seed_start"] = int(seed_opts[0])
        selected["env_config"]["kwargs"]["seed_end"] = int(seed_opts[1])
        selected["test_size"] = int(seed_opts[1] - seed_opts[0] + 1)
    if enable_think is not None:
        selected["env_config"].setdefault("prompt_config", {})
        selected["env_config"]["prompt_config"]["enable_think"] = bool(enable_think)
    if proxy_agent and exp_type == "passive":
        selected["env_config"]["proxy_agent"] = proxy_agent
        
    return {task_key: selected}


def patch_model_yaml(model_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Pick a single entry from base_model_config.yaml's `models`."""
    models = model_cfg.get("models", {}) or {}
    if model_name in models:
        model_cfg["models"] = {model_name: dict(models[model_name])}
        return model_cfg
    for k, v in models.items():
        if isinstance(v, Dict) and v.get("model_name") == model_name:
            model_cfg["models"] = {k: dict(v)}
            return model_cfg
    available = ", ".join(models.keys())
    print(f"[ERROR] Model '{model_name}' not found. Available model keys: {available}", file=sys.stderr)
    sys.exit(2)


def patch_infer_yaml(
    infer_cfg: Dict[str, Any], 
    output_dir: str, 
    server_url: str | None = None,
    all_override: bool = False
) -> Dict[str, Any]:
    """Patch inference yaml to set output directory, server URL, and override flags."""
    infer_cfg = dict(infer_cfg or {})
    infer_cfg["output_dir"] = output_dir
    if server_url:
        infer_cfg["server_url"] = server_url
    if all_override:
        infer_cfg["all_override"] = True
    return infer_cfg


def run_cmd(cmd: List[str], cwd: Path | None = None) -> int:
    print("Running:", " ".join(shlex.quote(c) for c in cmd), f"(cwd={cwd or Path.cwd()})", flush=True)
    cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return cp.returncode


def _wait_for_http(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.getcode() == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _stream_process_output(proc: subprocess.Popen, prefix: str = "server") -> None:
    def _reader():
        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                print(f"[{prefix}] {line}", end='')
        except Exception as e:
            print(f"[WARN] log stream error: {e}")
    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def start_env_server(host: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vagen.server.server",
        f"server.host={host}",
        f"server.port={port}",
        "use_state_reward=false",
    ]
    print("Starting env server:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    _stream_process_output(proc, prefix="server")
    health_url = f"http://{host}:{port}/health"
    if not _wait_for_http(health_url, timeout=40.0):
        raise RuntimeError(f"Env server failed to start at {health_url}")
    print(f"Env server is up at {health_url}")
    return proc


def stop_env_server(proc: subprocess.Popen) -> None:
    if not proc:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        print(f"[WARN] Failed to stop server: {e}")


def compute_combo_paths(
    output_root: str,
    model_name: str,
    exp_type: str,
    seed_range: tuple[int, int] | None,
    render_mode: str,
    enable_think: bool,
    data_dir: str,
    proxy_agent: str | None = None,
) -> List[str]:
    """Compute expected combo directory paths based on parameters.
    
    This replicates the logic from HistoryManager to determine where
    exploration results should be stored.
    
    Args:
        output_root: Base output directory
        model_name: Model name
        exp_type: 'active' or 'passive'
        seed_range: Tuple of (start_seed, end_seed) or None
        render_mode: 'vision' or 'text'
        enable_think: Whether thinking is enabled
        data_dir: Data directory containing room data
        proxy_agent: Proxy agent for passive mode
        
    Returns:
        List of combo directory paths
    """
    from vagen.env.spatial.Base.tos_base.utils.utils import hash as compute_hash
    from vagen.env.spatial.Base.tos_base.utils.image_handler import ImageHandler
    from vagen.env.spatial.Base.tos_base.utils.room_utils import initialize_room_from_json
    
    # Determine seed list
    if seed_range:
        seeds = list(range(seed_range[0], seed_range[1] + 1))
    else:
        seeds = [0]  # Default single seed
    
    combo_paths = []
    
    for seed in seeds:
        # Load room/agent data to compute hash
        try:
            image_handler = ImageHandler(data_dir, seed, image_size=(512, 512), preload_images=False)
            room, agent = initialize_room_from_json(image_handler.json_data)
            
            # Compute room hash (same as HistoryManager._generate_room_key)
            room_str = json.dumps(
                {**room.to_dict(), **agent.to_dict()},
                sort_keys=True
            )
            room_hash = compute_hash(room_str)
            
            # Build path following HistoryManager structure
            # model_name/room_hash/render_mode/exp_type/think_or_nothink/[proxy_agent]
            think_str = "think" if enable_think else "nothink"
            
            path_parts = [
                output_root,
                model_name,
                room_hash,
                render_mode,
                exp_type,
                think_str,
            ]
            
            if exp_type == "passive":
                path_parts.append(proxy_agent if proxy_agent else "scout")
            
            combo_path = os.path.join(*path_parts)
            combo_paths.append(combo_path)
            
        except Exception as e:
            print(f"Warning: Failed to compute combo path for seed={seed}: {e}", 
                  file=sys.stderr)
            continue
    
    return combo_paths


def run_exploration_phase(args, run_id: str, server_url: str | None):
    """Run exploration phase: create dataset and run inference once.
    
    Note: All seeds are processed in a single run via seed_opts.
    Uses exp_type to determine environment configuration.
    """
    print("\n" + "="*60)
    print("PHASE: EXPLORATION")
    print("="*60 + "\n")
    
    data_train = f"data/{run_id}/train.parquet"
    data_test = f"data/{run_id}/test.parquet"
    
    base_env = Path(args.base_env)
    base_infer = Path(args.base_infer)
    base_model = Path(args.base_model)
    
    seed_opts = None
    if args.seed_range:
        try:
            s, e = [int(x) for x in args.seed_range.split('-', 1)]
            seed_opts = (s, e)
        except Exception:
            print(f"[ERROR] Bad --seed_range '{args.seed_range}'. Use 'start-end'.", file=sys.stderr)
            sys.exit(2)
    
    # Validate passive mode requires proxy_agent
    if args.exp_type == "passive" and not args.proxy_agent:
        print(f"[ERROR] --proxy-agent is required when --exp-type is passive", file=sys.stderr)
        sys.exit(2)
    
    # Use exp_type for env config setup
    tmp_paths = build_tmp_paths(run_id, "exploration")
    
    env_cfg = load_yaml(base_env)
    infer_cfg = load_yaml(base_infer)
    model_cfg = load_yaml(base_model)
    
    # Create env config with seed range (all seeds processed together)
    env_cfg = patch_env_yaml(env_cfg, args.exp_type, args.num, args.render_mode, 
                             seed_opts, args.enable_think, data_dir=args.data_dir,
                             proxy_agent=args.proxy_agent)
    
    model_cfg = patch_model_yaml(model_cfg, args.model_name)
    
    # Patch inference config with all_override flag if specified
    patched_infer_cfg = patch_infer_yaml(
        infer_cfg,
        args.output_root,
        server_url=server_url,
        all_override=args.all_override,
    )
    
    dump_yaml(env_cfg, tmp_paths["env"])
    dump_yaml(model_cfg, tmp_paths["model"])
    dump_yaml(patched_infer_cfg, tmp_paths["infer"])
    
    # Create dataset
    print(f"Creating dataset for {args.exp_type} exploration...")
    rc = run_cmd([
        sys.executable, "-m", "vagen.env.create_dataset",
        "--yaml_path", str(tmp_paths["env"]),
        "--train_path", data_train,
        "--test_path", data_test,
        "--force_gen",
    ])
    if rc != 0:
        sys.exit(rc)
    
    # Run inference
    print(f"Running exploration inference...")
    val_path = data_test
    wandb_path_name = "spatial_gym"
    cmd = [
        sys.executable, "-m", "vagen.inference.run_inference",
        f"--inference_config_path={tmp_paths['infer']}",
        f"--model_config_path={tmp_paths['model']}",
        f"--val_files_path={val_path}",
        f"--wandb_path_name={wandb_path_name}",
    ]
    rc = run_cmd(cmd)
    if rc != 0:
        sys.exit(rc)
    
    print(f"\nExploration completed. Results in: {args.output_root}")


def run_evaluation_phase(args, seed_opts: tuple[int, int] | None = None):
    """Run evaluation phase: build eval messages and run inference."""
    print("\n" + "="*60)
    print("PHASE: EVALUATION")
    print("="*60 + "\n")
    
    # Parse eval_task_counts from CLI argument or use inference_config.yaml default
    eval_task_counts = None
    if args.eval_task_counts:
        try:
            eval_task_counts = json.loads(args.eval_task_counts)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON for --eval-task-counts: {e}", file=sys.stderr)
            sys.exit(2)
    
    # If not provided via CLI, load from inference_config.yaml
    if eval_task_counts is None:
        base_infer = Path(args.base_infer)
        if base_infer.exists():
            infer_cfg = load_yaml(base_infer)
            eval_task_counts = infer_cfg.get("eval_task_counts")
            if eval_task_counts:
                print(f"Using eval_task_counts from inference_config.yaml: {eval_task_counts}")
            else:
                # Default fallback
                raise FileNotFoundError("eval_task_counts not found in inference_config.yaml")
        else:
            raise FileNotFoundError(f"Base inference config not found: {base_infer}")

    model_name = load_yaml(Path(args.base_model))['models'][args.model_name]['model_name']
    # Compute combo paths from parameters
    print("Computing combo directory paths...")
    combo_paths = compute_combo_paths(
        output_root=args.output_root,
        model_name=model_name,
        exp_type=args.exp_type,
        seed_range=seed_opts,
        render_mode=args.render_mode,
        enable_think=bool(args.enable_think),
        data_dir=args.data_dir,
        proxy_agent=args.proxy_agent,
    )
    
    if not combo_paths:
        print("[ERROR] No valid combo paths computed", file=sys.stderr)
        sys.exit(2)
    
    print(f"Found {len(combo_paths)} combo directories to evaluate")
    
    # Run inference using the function interface with override flags
    from vagen.env.spatial.llm_inference import run_inference_for_combo_dirs
    
    run_inference_for_combo_dirs(
        combo_dirs=combo_paths,
        model_name=model_name,
        mode="eval",
        eval_task_counts=eval_task_counts,
        seed=args.inference_seed,
        inference_mode=args.inference_mode,
        eval_override=args.eval_override,  # Use override flag
    )
    
    print("\nEvaluation completed.")


def run_cogmap_phase(args, seed_opts: tuple[int, int] | None = None):
    """Run cogmap phase: build cogmap messages and run inference."""
    print("\n" + "="*60)
    print("PHASE: COGNITIVE MAP")
    print("="*60 + "\n")
    
    # Compute combo paths from parameters
    print("Computing combo directory paths...")
    model_name = load_yaml(Path(args.base_model))['models'][args.model_name]['model_name']
    combo_paths = compute_combo_paths(
        output_root=args.output_root,
        model_name=model_name,
        exp_type=args.exp_type,
        seed_range=seed_opts,
        render_mode=args.render_mode,
        enable_think=bool(args.enable_think),
        data_dir=args.data_dir,
        proxy_agent=args.proxy_agent,
    )
    
    if not combo_paths:
        print("[ERROR] No valid combo paths computed", file=sys.stderr)
        sys.exit(2)
    
    print(f"Found {len(combo_paths)} combo directories for cogmap")
    
    # Run inference using the function interface with override flags
    from vagen.env.spatial.llm_inference import run_inference_for_combo_dirs
    
    run_inference_for_combo_dirs(
        combo_dirs=combo_paths,
        model_name=model_name,                                             
        mode="cogmap",
        inference_mode=args.inference_mode,
        cogmap_override=args.cogmap_override,  # Regenerate cogmap prompts
        cogmap_reevaluate=args.cogmap_reevaluate,  # Re-evaluate existing cogmaps
    )
    
    print("\nCognitive map evaluation completed.")


def main():
    args = parse_args()
    
    # Environment variables
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    
    # Check base config files exist
    base_env = Path(args.base_env)
    base_infer = Path(args.base_infer)
    base_model = Path(args.base_model)
    if not base_env.exists() or not base_infer.exists() or not base_model.exists():
        print(f"Base YAML missing: env={base_env.exists()} infer={base_infer.exists()} model={base_model.exists()}", 
              file=sys.stderr)
        sys.exit(2)
    
    # Parse seed range
    seed_opts = None
    if args.seed_range:
        try:
            s, e = [int(x) for x in args.seed_range.split('-', 1)]
            seed_opts = (s, e)
        except Exception:
            print(f"[ERROR] Bad --seed_range '{args.seed_range}'. Use 'start-end'.", file=sys.stderr)
            sys.exit(2)
    else:
        seed_opts = (0, 0 + args.num - 1)
    
    run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    server_proc: subprocess.Popen | None = None
    
    try:
        server_url: str | None = None
        
        # Start server only for exploration phase
        if args.phase in ['exploration', 'all'] and not args.no_server:
            actual_port = get_adaptive_port(args.server_port, 5000)
            if actual_port != args.server_port:
                print(f"Using port {actual_port} instead of requested {args.server_port}")
            server_proc = start_env_server(args.server_host, actual_port)
            server_url = f"http://{args.server_host}:{actual_port}"
        
        # Run requested phase(s)
        if args.phase == 'exploration':
            run_exploration_phase(args, run_id, server_url)
        elif args.phase == 'evaluation':
            run_evaluation_phase(args, seed_opts)
        elif args.phase == 'cogmap':
            run_cogmap_phase(args, seed_opts)
        elif args.phase == 'all':
            run_exploration_phase(args, run_id, server_url)
            run_evaluation_phase(args, seed_opts)
            run_cogmap_phase(args, seed_opts)
    
    except Exception as e:
        raise e
    
    finally:
        if server_proc is not None:
            stop_env_server(server_proc)
    
    print("\n" + "="*60)
    print("ALL PHASES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
