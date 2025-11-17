#!/usr/bin/env python3
import argparse
import os
import sys
import shlex
import subprocess
import time
import socket
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
        description="Batch run SpatialGym: per-task tmp YAML generation, dataset then inference (no Hydra)."
    )
    p.add_argument("--tasks", nargs="+", default=['ActiveRot'],
                   help="Tasks (space or comma separated). Examples: ActiveRot PassiveRot or 'ActiveRot,PassiveLoc'. Default: ActiveRot")
    p.add_argument("--num", type=int, default=1, help="Number of samples per task. Default: 1")
    p.add_argument("--model_name", type=str, default="gpt-4.1-mini",
                   help="Model identifier. Default: gpt-4.1-mini")
    p.add_argument("--data-dir", type=str, dest="data_dir", default=None, help="Data directory root. Default: data")
    p.add_argument("--render-mode", type=str, dest="render_mode", default="vision", help="Environment render mode (vision or text). Default: vision")
    p.add_argument("--output-root", type=str, dest="output_root", default="results", help="Root dir for inference output_dir. Default: results")
    p.add_argument("--seed-range", type=str, dest="seed_range", default=None, help="Seed range 'start-end' (0-based), e.g., 0-24")
    p.add_argument("--enable-think", type=int, dest="enable_think", choices=[0,1], default=1, help="1 to enable think, 0 to disable (default: 1)")
    p.add_argument("--cogmap-reevaluate", action="store_true", dest="cogmap_reevaluate", help="If set, will re-evaluate existing cognitive maps")
    # New granular override flags
    p.add_argument("--eval-override", action="store_true", dest="eval_override", help="Override evaluation history (delete evaluation json only)")
    p.add_argument("--cogmap-override", action="store_true", dest="cogmap_override", help="Override cognitive map cache")
    p.add_argument("--all-override", action="store_true", dest="all_override", help="Override all history (delete whole sample path)")
    p.add_argument("--cogmap", action="store_true", help="If set, will enable cognitive map evaluation")
    # Eval repetition controls: CLI overrides YAML eval_task_counts
    p.add_argument("--eval_counts", type=str, default=None,
                   help="Per-task eval run counts, e.g., 'PassiveRot=3,ActiveDir=2'. If omitted, use inference_config.yaml eval_task_counts or default 1")
    # Choose which evaluation tasks to override
    p.add_argument("--eval-override-tasks", type=str, dest="eval_override_tasks", default=None,
                   help="Comma/space separated eval task keys to override (short names or class names), e.g., 'dir,RotEvaluationTask'")
    # Optional: override base yaml paths (env/model now default to base_*.yaml)
    p.add_argument("--base_env", type=str, dest="base_env", default=str(SCRIPT_DIR / "base_env_config.yaml"))
    p.add_argument("--base_infer", type=str, dest="base_infer", default=str(SCRIPT_DIR / "inference_config.yaml"))
    p.add_argument("--base_model", type=str, dest="base_model", default=str(SCRIPT_DIR / "base_model_config.yaml"))
    # Server options: server is ON by default, use --no_server to skip starting it
    p.add_argument("--no-server", action="store_true", dest="no_server", help="Do not start internal env server (assume an external server is running)")
    p.add_argument("--server-host", type=str, dest="server_host", default="127.0.0.1", help="Server host to bind/connect")
    p.add_argument("--server-port", type=int, dest="server_port", default=5000, help="Server port to bind/connect")
    # Proxy agent selection (for passive tasks)
    p.add_argument("--proxy-agent", type=str, dest="proxy_agent", default=None, choices=["scout","strategist","oracle"], help="Proxy agent for passive tasks")
    p.add_argument("--inference-only", action="store_true", dest="inference_only", help="If set, skip SpatialEnvLogger logging after inference")
    p.add_argument("--aggregate-only", action="store_true", dest="aggregate_only", help="If set, skip individual task logging and only log aggregate results")
    # Exploration tuning knobs
    p.add_argument("--use-real-relations", action="store_true", dest="use_real_relations", default=False,
                   help="Report precise (real-value) spatial relations in observations")
    p.add_argument("--query-cost", type=int, dest="query_cost", default=None,
                   help="Override Query() action cost (default from base config).")
    p.add_argument("--max-exp-steps", type=int, dest="max_exp_steps", default=None,
                   help="Override maximum exploration steps before forced termination.")
    # Ground-truth testing options
    p.add_argument("--gt-cogmap-eval", action="store_true", dest="gt_cogmap_eval",
                   help="If set, provide ground-truth cogmap and test evaluation tasks")
    p.add_argument("--gt-local-cogmap", action="store_true", dest="gt_local_cogmap",
                   help="If set, provide ground-truth local cogmap at each step and test cogmap")
    # Cognitive map before evaluation option
    p.add_argument("--cogmap-before-eval", action="store_true", dest="cogmap_before_eval",
                   help="If set, request model to output cognitive map before answering evaluation questions")

    return p.parse_args()


def normalize_tasks(tasks_arg: List[str]) -> List[str]:
    if len(tasks_arg) == 1 and "," in tasks_arg[0]:
        return [t.strip() for t in tasks_arg[0].split(",") if t.strip()]
    return [t.strip() for t in tasks_arg]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return pyyaml.safe_load(f)


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        pyyaml.safe_dump(data, f, sort_keys=False)


def compute_experiment_name(script_dir: Path) -> str:
    # Match run.sh behavior: last two parts joined by '-'
    parts = [p for p in script_dir.parts if p]
    if len(parts) >= 2:
        return f"{parts[-2]}-{parts[-1]}"
    return parts[-1] if parts else "exp"


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



def parse_eval_counts_arg(arg: str | None) -> Dict[str, int]:
    """Parse CLI eval counts string into a dict, e.g., 'PassiveRot=3,ActiveDir=2'."""
    result: Dict[str, int] = {}
    if not arg:
        return result
    # Split by comma or spaces
    parts: List[str] = []
    for token in arg.replace(" ", ",").split(","):
        t = token.strip()
        if t:
            parts.append(t)
    for item in parts:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        try:
            result[k] = int(v.strip())
        except Exception:
            continue
    return result


def parse_task_list_arg(arg: str | None) -> List[str]:
    """Parse CLI list string into a list, splitting on commas/spaces."""
    if not arg:
        return []
    parts: List[str] = []
    for token in arg.replace(" ", ",").split(","):
        t = token.strip()
        if t:
            parts.append(t)
    return parts


def resolve_eval_runs_count(task_key: str, infer_cfg: Dict[str, Any], eval_counts_cli: Dict[str, int] | None) -> int:
    """Decide how many times to run inference for a given task.

    Priority: CLI --eval_counts > inference_config.yaml eval_task_counts > 1.
    """
    if eval_counts_cli and task_key in eval_counts_cli:
        return max(1, int(eval_counts_cli[task_key]))
    yaml_counts = (infer_cfg or {}).get("eval_task_counts") or {}
    if isinstance(yaml_counts, dict) and task_key in yaml_counts:
        try:
            return max(1, int(yaml_counts[task_key]))
        except Exception:
            pass
    return 1


def patch_env_yaml(env_cfg: Dict[str, Any], task_key: str, num: int, render_mode = "vision", seed_opts: tuple[int, int] | None = None,
                   enable_think: int | None = None, eval_num: int | None = None, data_dir: str | None = None,
                   use_real_relations: bool | None = None, query_cost: int | None = None, max_exp_steps: int | None = None,
                   gt_cogmap_eval: bool = False, gt_local_cogmap: bool = False, cogmap_before_eval: bool = False) -> Dict[str, Any]:
    """Return {TaskKey: {...}} by selecting the entry from custom_envs and overriding sizes.

    Behavior:
    - Select the env config by key from env_cfg['custom_envs'].
    - Shallow-copy the entry and set test_size to `num`.
    - Wrap it under the CamelCase task key for create_dataset.
    """
    custom_envs = env_cfg.get("custom_envs", {}) or {}
    selected = dict(custom_envs[task_key])
    selected["test_size"] = int(num)
    selected["env_config"]['render_mode'] = render_mode
    if use_real_relations is not None:
        selected["env_config"]["use_real_relations"] = bool(use_real_relations)
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
    if query_cost is not None:
        selected["env_config"]["query_action_cost"] = int(query_cost)
    if max_exp_steps is not None:
        selected["env_config"]["max_exp_steps"] = int(max_exp_steps)
    if eval_num is not None:
        # Pass desired evaluation repetitions to EvaluationManager via env config
        tasks = selected["env_config"].get("eval_tasks") or []
        if tasks:
            tasks[0]["num"] = int(eval_num)
    # Ground-truth testing options
    if gt_cogmap_eval:
        selected["env_config"]["gt_cogmap_eval"] = True
    if gt_local_cogmap:
        selected["env_config"]["gt_local_cogmap"] = True
    # Cognitive map before evaluation option
    if cogmap_before_eval:
        selected["env_config"]["cogmap_before_eval"] = True
    return {task_key: selected}



def patch_model_yaml(model_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Pick a single entry from base_model_config.yaml's `models`.

    Selection rules:
    1) If `model_name` matches a key in `models`, use that key.
    2) Else if any entry has v['model_name'] == `model_name`, use that entry's key.
    3) Else exit with an error listing available keys.
    """
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


def patch_infer_yaml(infer_cfg: Dict[str, Any], output_dir: str, eval_override: bool, cogmap_override: bool, all_override: bool, evaluate_cogmap: bool, cogmap_reevaluate: bool = False, server_url: str | None = None, eval_override_tasks: List[str] | None = None) -> Dict[str, Any]:
    """Patch inference yaml to set output directory and override flags and optional server_url. Split remains as in base config."""
    infer_cfg = dict(infer_cfg or {})
    infer_cfg["output_dir"] = output_dir
    if eval_override:
        infer_cfg["eval_override"] = True
    if cogmap_override:
        infer_cfg["cogmap_override"] = True
    if all_override:
        infer_cfg["all_override"] = True
    if server_url:
        infer_cfg["server_url"] = server_url
    if evaluate_cogmap:
        infer_cfg["evaluate_cogmap"] = True
    if cogmap_reevaluate:
        infer_cfg["cogmap_reevaluate"] = True
    if eval_override_tasks:
        infer_cfg["eval_override_tasks"] = list(eval_override_tasks)
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


def main():
    args = parse_args()
    tasks = normalize_tasks(args.tasks)
    eval_counts_cli = parse_eval_counts_arg(args.eval_counts)
    eval_override_tasks_cli = parse_task_list_arg(args.eval_override_tasks)
    use_real_relations = bool(args.use_real_relations)
    query_cost = args.query_cost
    max_exp_steps = args.max_exp_steps

    # Environment variables similar to run.sh
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    # Paths
    base_env = Path(args.base_env)
    base_infer = Path(args.base_infer)
    base_model = Path(args.base_model)
    if not base_env.exists() or not base_infer.exists() or not base_model.exists():
        print(f"Base YAML missing: env={base_env.exists()} infer={base_infer.exists()} model={base_model.exists()}", file=sys.stderr)
        sys.exit(2)

    # Compute run id and experiment/data paths
    run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    exp_name = compute_experiment_name(SCRIPT_DIR)
    data_train = f"data/{run_id}/train.parquet"
    data_test = f"data/{run_id}/test.parquet"

    output_root = args.output_root
    seed_opts = None
    if args.seed_range:
        try:
            s, e = [int(x) for x in args.seed_range.split('-', 1)]
            seed_opts = (s, e)
        except Exception:
            print(f"[ERROR] Bad --seed_range '{args.seed_range}'. Use 'start-end', e.g., 2-5.", file=sys.stderr)
            sys.exit(2)

    created_tmp_dirs: List[Path] = []
    server_proc: subprocess.Popen | None = None
    try:
        server_url: str | None = None
        if not args.no_server:
            # Use adaptive port selection
            actual_port = get_adaptive_port(args.server_port, 5000)
            if actual_port != args.server_port:
                print(f"Using port {actual_port} instead of requested {args.server_port}")
            server_proc = start_env_server(args.server_host, actual_port)
            server_url = f"http://{args.server_host}:{actual_port}"

        for task in tqdm(tasks, desc="Running tasks"):
            tmp_paths = build_tmp_paths(run_id, f"{task}")
            created_tmp_dirs.append(tmp_paths["base"])

            env_cfg = load_yaml(base_env)
            infer_cfg = load_yaml(base_infer)
            model_cfg = load_yaml(base_model)

            # Decide repetition count per task and embed into env config for EvaluationManager
            repeat = resolve_eval_runs_count(task, infer_cfg, eval_counts_cli)

            env_cfg = patch_env_yaml(
                env_cfg,
                task,
                args.num,
                args.render_mode,
                seed_opts,
                args.enable_think,
                eval_num=repeat,
                data_dir=args.data_dir,
                use_real_relations=use_real_relations,
                query_cost=query_cost,
                max_exp_steps=max_exp_steps,
                gt_cogmap_eval=args.gt_cogmap_eval,
                gt_local_cogmap=args.gt_local_cogmap,
                cogmap_before_eval=args.cogmap_before_eval,
            )
            if args.proxy_agent:
                if (env_cfg[task]["env_config"].get("exp_type") == "passive"):
                    env_cfg[task]["env_config"]["proxy_agent"] = args.proxy_agent
            model_cfg = patch_model_yaml(model_cfg, args.model_name)
            dump_yaml(env_cfg, tmp_paths["env"])
            dump_yaml(model_cfg, tmp_paths["model"])
            rc = run_cmd([
                sys.executable, "-m", "vagen.env.create_dataset",
                "--yaml_path", str(tmp_paths["env"]),
                "--train_path", data_train,
                "--test_path", data_test,
                "--force_gen",
            ])
            if rc != 0:
                sys.exit(rc)
            # for question_idx in range(num_questions):
            # Only pass eval_override on first question when num_question > 1

            for i in range(repeat):
                # Apply overrides only on the first repetition to avoid wiping between repeats
                patched_infer_cfg = patch_infer_yaml(
                    infer_cfg,
                    output_root,
                    bool(args.eval_override and i == 0),
                    bool(args.cogmap_override and i == 0),
                    bool(args.all_override and i == 0),
                    args.cogmap,
                    args.cogmap_reevaluate,
                    server_url,
                    eval_override_tasks=(eval_override_tasks_cli if i == 0 else None),
                )
                dump_yaml(patched_infer_cfg, tmp_paths["infer"])

                # Run inference
                val_path = data_test
                wandb_path_name = "spatial_gym"
                cmd = [
                    sys.executable, "-m", "vagen.inference.run_inference",
                    f"--inference_config_path={tmp_paths['infer']}",
                    f"--model_config_path={tmp_paths['model']}",
                    f"--val_files_path={val_path}",
                    f"--wandb_path_name={wandb_path_name}",
                ]
                if args.inference_only:
                    cmd.append("--inference-only")
                if args.aggregate_only:
                    cmd.append("--aggregate-only")
                rc = run_cmd(cmd)
                if rc != 0:
                    sys.exit(rc)
                if args.aggregate_only:
                    break

    except Exception as e:
        raise e

    finally:
        # top_tmp = SCRIPT_DIR / "tmp" / run_id
        # if top_tmp.exists():
        #     import shutil
        #     try:
        #         shutil.rmtree(top_tmp)
        #     except Exception as e:
        #         print(f"[WARN] Failed to remove tmp dir {top_tmp}: {e}")

        if server_proc is not None:
            stop_env_server(server_proc)

    print("All tasks completed.")


if __name__ == "__main__":
    main()
