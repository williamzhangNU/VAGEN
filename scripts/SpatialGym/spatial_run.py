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
    p.add_argument("--model_name", type=str, default="gpt-5-mini",
                   help="Model identifier. Default: gpt-5-mini")
    p.add_argument("--output_root", type=str, default="results", help="Root dir for inference output_dir. Default: results")
    p.add_argument("--override", action="store_true", help="If set, will override the active exploration history")
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


def to_task_key(name: str) -> str:
    # Accept forms like ActiveRot, active-rot, active_rot -> active_rot
    s = name.strip().replace("-", "_")
    # CamelCase to snake
    out = []
    for ch in s:
        if ch.isupper() and out and out[-1] != "_":
            out.append("_")
        out.append(ch.lower())
    s2 = "".join(out)
    s2 = s2.replace("__", "_")
    return s2



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



def patch_env_yaml(env_cfg: Dict[str, Any], task_key: str, num: int) -> Dict[str, Any]:
    """Return {TaskKey: {...}} by selecting the entry from custom_envs and overriding sizes.

    Behavior:
    - Select the env config by key from env_cfg['custom_envs'].
    - Shallow-copy the entry and set test_size to `num`.
    - Wrap it under the CamelCase task key for create_dataset.
    """
    custom_envs = env_cfg.get("custom_envs", {}) or {}
    selected = dict(custom_envs[task_key])
    selected["test_size"] = int(num)
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
        picked_key = model_name
        model_cfg["models"] = {picked_key: dict(models[picked_key])}
        return model_cfg

    for k, v in models.items():
        if isinstance(v, Dict) and v.get("model_name") == model_name:
            model_cfg["models"] = {k: dict(v)}
            return model_cfg

    available = ", ".join(models.keys())
    print(f"[ERROR] Model '{model_name}' not found. Available model keys: {available}", file=sys.stderr)
    sys.exit(2)


def patch_infer_yaml(infer_cfg: Dict[str, Any], output_dir: Path, override: bool, server_url: str | None = None) -> Dict[str, Any]:
    """Patch inference yaml to set output directory and optional server_url. Split remains as in base config."""
    infer_cfg = dict(infer_cfg or {})
    infer_cfg["output_dir"] = str(output_dir)
    if override:
        infer_cfg["override"] = True
    if server_url:
        infer_cfg["server_url"] = server_url
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

    output_root = Path(args.output_root)

    created_tmp_dirs: List[Path] = []
    server_proc: subprocess.Popen | None = None
    try:
        server_url: str | None = None
        if not args.no_server:
            server_proc = start_env_server(args.server_host, args.server_port)
            server_url = f"http://{args.server_host}:{args.server_port}"

        for task in tasks:
            snake_task = to_task_key(task)
            tmp_paths = build_tmp_paths(run_id.replace('/', '-'), snake_task)
            created_tmp_dirs.append(tmp_paths["base"])

            env_cfg = load_yaml(base_env)
            infer_cfg = load_yaml(base_infer)
            model_cfg = load_yaml(base_model)

            env_cfg = patch_env_yaml(env_cfg, task, args.num)
            model_cfg = patch_model_yaml(model_cfg, args.model_name)
            task_output_dir = output_root / snake_task
            infer_cfg = patch_infer_yaml(infer_cfg, task_output_dir, args.override, server_url)

            dump_yaml(env_cfg, tmp_paths["env"])
            dump_yaml(infer_cfg, tmp_paths["infer"])
            dump_yaml(model_cfg, tmp_paths["model"])

            # Create dataset
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
