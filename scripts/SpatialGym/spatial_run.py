#!/usr/bin/env python3
import argparse
import os
import sys
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
import yaml as pyyaml
import urllib.request
import threading

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch run SpatialGym: per-task tmp YAML generation, dataset then inference (no Hydra)."
    )
    p.add_argument("--tasks", nargs="+", default=['ActiveRot'],
                   help="Tasks (space or comma separated). Examples: ActiveRot PassiveRot or 'ActiveRot,PassiveLoc'. Default: ActiveRot")
    p.add_argument("--num", type=int, default=1, help="Number of samples per task. Default: 1")
    p.add_argument("--model_name", type=str, default="gpt-4.1-mini",
                   help="Model identifier. Default: gpt-4.1-mini")
    p.add_argument("--data-dir", type=str, default=None, help="Data directory root. Default: data")
    p.add_argument("--render-mode", type=str, default="vision", help="Environment render mode (vision or text). Default: vision")
    p.add_argument("--output_root", type=str, default="results", help="Root dir for inference output_dir. Default: results")
    p.add_argument("--seed_range", type=str, default=None, help="Seed range 'start-end' (0-based), e.g., 0-24")
    p.add_argument("--enable_think", type=int, choices=[0,1], default=1, help="1 to enable think, 0 to disable (default: 1)")
    # New granular override flags
    p.add_argument("--eval-override", action="store_true", dest="eval_override", help="Override evaluation history (delete evaluation json only)")
    p.add_argument("--cogmap-override", action="store_true", dest="cogmap_override", help="Override cognitive map cache")
    p.add_argument("--all-override", action="store_true", dest="all_override", help="Override all history (delete whole sample path)")
    p.add_argument("--cogmap", action="store_true", help="If set, will enable cognitive map evaluation")
    # Eval repetition controls: CLI overrides YAML eval_task_counts
    p.add_argument("--eval_counts", type=str, default=None,
                   help="Per-task eval run counts, e.g., 'PassiveRot=3,ActiveDir=2'. If omitted, use inference_config.yaml eval_task_counts or default 1")
    # Optional: override base yaml paths (env/model now default to base_*.yaml)
    p.add_argument("--base_env", type=str, default=str(SCRIPT_DIR / "base_env_config.yaml"))
    p.add_argument("--base_infer", type=str, default=str(SCRIPT_DIR / "inference_config.yaml"))
    p.add_argument("--base_model", type=str, default=str(SCRIPT_DIR / "base_model_config.yaml"))
    # Server options: server is ON by default, use --no_server to skip starting it
    p.add_argument("--no_server", action="store_true", help="Do not start internal env server (assume an external server is running)")
    p.add_argument("--server_host", type=str, default="127.0.0.1", help="Server host to bind/connect")
    p.add_argument("--server_port", type=int, default=5000, help="Server port to bind/connect")
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
                   enable_think: int | None = None, eval_num: int | None = None, data_dir: str | None = None) -> Dict[str, Any]:
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
    if eval_num is not None:
        # Pass desired evaluation repetitions to EvaluationManager via env config
        tasks = selected["env_config"].get("eval_tasks") or []
        if tasks:
            tasks[0]["num"] = int(eval_num)
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


def patch_infer_yaml(infer_cfg: Dict[str, Any], output_dir: str, eval_override: bool, cogmap_override: bool, all_override: bool, evaluate_cogmap: bool, server_url: str | None = None) -> Dict[str, Any]:
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
    run_id = time.strftime("%Y-%m-%d/%H-%M-%S")
    exp_name = compute_experiment_name(SCRIPT_DIR)
    data_train = f"data/{exp_name}/train.parquet"
    data_test = f"data/{exp_name}/test.parquet"

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
            server_proc = start_env_server(args.server_host, args.server_port)
            server_url = f"http://{args.server_host}:{args.server_port}"

        for task in tasks:
            tmp_paths = build_tmp_paths(run_id.replace('/', '-'), f"{task}")
            created_tmp_dirs.append(tmp_paths["base"])

            env_cfg = load_yaml(base_env)
            infer_cfg = load_yaml(base_infer)
            model_cfg = load_yaml(base_model)

            # Decide repetition count per task and embed into env config for EvaluationManager
            repeat = resolve_eval_runs_count(task, infer_cfg, eval_counts_cli)

            env_cfg = patch_env_yaml(env_cfg, task, args.num, args.render_mode, seed_opts, args.enable_think, eval_num=repeat, data_dir=args.data_dir)
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
                    server_url,
                )
                dump_yaml(patched_infer_cfg, tmp_paths["infer"])

                # Run inference
                val_path = data_test
                wandb_path_name = "spatial_gym"
                rc = run_cmd([
                    sys.executable, "-m", "vagen.inference.run_inference",
                    f"--inference_config_path={tmp_paths['infer']}",
                    f"--model_config_path={tmp_paths['model']}",
                    f"--val_files_path={val_path}",
                    f"--wandb_path_name={wandb_path_name}",
                ])
                if rc != 0:
                    sys.exit(rc)

    except Exception as e:
        raise e

    finally:
        # Always clean up tmp dir
        top_tmp = SCRIPT_DIR / "tmp" / run_id.replace('/', '-')
        if top_tmp.exists():
            import shutil
            try:
                shutil.rmtree(top_tmp)
            except Exception as e:
                print(f"[WARN] Failed to remove tmp dir {top_tmp}: {e}")

        if server_proc is not None:
            stop_env_server(server_proc)

    print("All tasks completed.")


if __name__ == "__main__":
    main()
