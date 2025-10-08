from typing import List, Dict, Tuple, Any
import os
import json
import re
import numpy as np
import argparse

# Reuse existing components
from vagen.env.spatial.Base.tos_base import Room, Agent
from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType
from vagen.env.spatial.Base.tos_base.evaluation.tasks import BaseEvaluationTask
from vagen.env.spatial.Base.tos_base.prompts.cogmap_prompts import get_cogmap_prompt
from vagen.env.spatial.Base.tos_base.utils.utils import hash

# Shared common utilities/constants
from vagen.env.spatial.common import (
    MESSAGES_BASENAME,
    EXPLORATION_LOG_BASENAME,
    CONFIG_BASENAME,
    read_json,
    resolve_built_root,
    paths_for_mode,
    generate_message_id,
    iter_combo_dirs,
    load_history_manager,
)


def _load_exploration_artifacts(combo_dir: str) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    messages_path = os.path.join(combo_dir, MESSAGES_BASENAME)
    turn_logs_path = os.path.join(combo_dir, EXPLORATION_LOG_BASENAME)
    messages = read_json(messages_path)
    turn_logs = read_json(turn_logs_path) if os.path.exists(turn_logs_path) else []
    cfg_path = os.path.join(combo_dir, CONFIG_BASENAME)
    sample_cfg = read_json(cfg_path)
    return messages, turn_logs, sample_cfg


def _detect_exp_type(combo_dir: str) -> str:
    parts = os.path.abspath(combo_dir).split(os.sep)
    return "active" if "active" in parts else "passive"


def _get_turn_number(seq: List[Dict]) -> int:
    return len([m for m in seq if m.get('role') == 'user' and (seq[0].get('role') != 'system' or True)])


def _iter_user_indices(messages: List[Dict]) -> List[int]:
    idxs: List[int] = []
    for i, m in enumerate(messages):
        if i == 0 and m.get("role") == "system":
            continue
        if m.get("role") == "user":
            idxs.append(i)
    return idxs


def _clone_until_inclusive(messages: List[Dict], end_idx: int) -> List[Dict]:
    return [m.copy() for m in messages[: end_idx + 1]]


def _format_eval_question(task: BaseEvaluationTask) -> Tuple[str, str]:
    q = task.generate_question()
    choices = getattr(task, "choices", []) or getattr(task.eval_data, "choices", []) or []
    if choices:
        lines = [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
        q_text = q + "\n" + "\n".join(lines)
    else:
        q_text = q
    qid = hash(q)
    return q_text, qid


def _add_message(out_msgs: List[List[Dict]], out_meta: List[Dict], msgs: List[Dict], meta: Dict[str, Any]) -> None:
    out_msgs.append(msgs)
    out_meta.append(meta)


def build_evaluation_from_combo(
    combo_dir: str,
    eval_task_counts: Dict[str, int],
    seed: int | None = None,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Create evaluation message lists from exploration history for one sample combo dir.

    Returns (messages_list, meta_list) with meta including sample_id, task_type, question_id, message_id.
    """
    messages, _turn_logs, sample_cfg = _load_exploration_artifacts(combo_dir)
    sample_id = os.path.basename(sample_cfg.get("image_dir", "sample"))
    base_msgs = [m.copy() for m in messages]

    # Prefer the saved run seed for reproducibility
    hm = load_history_manager(combo_dir)
    run_seed = hm.run_seed if hm and hm.run_seed is not None else seed
    out_msgs: List[List[Dict]] = []
    meta: List[Dict] = []

    for task_short, count in (eval_task_counts or {}).items():
        for i in range(int(count)):
            room = Room.from_dict(sample_cfg["room_dict"]).copy()
            agent = Agent.from_dict(sample_cfg["agent_dict"]).copy()
            task = EvalTaskType.create_task(task_short, np.random.default_rng(None if run_seed is None else int(run_seed)), room, agent, {}, None)
            q_text, qid = _format_eval_question(task)
            msg = {"role": "user", "content": q_text}
            new_list = [m.copy() for m in base_msgs] + [msg]
            meta_obj = {
                "type": "evaluation",
                "sample_id": sample_id,
                "task_type": task_short,
                "question_id": qid,
                "combo_dir": os.path.abspath(combo_dir),
            }
            meta_obj["message_id"] = generate_message_id(meta_obj)
            _add_message(out_msgs, meta, new_list, meta_obj)

    return out_msgs, meta


def build_cogmap_from_combo(combo_dir: str) -> Tuple[List[List[Dict]], List[Dict]]:
    """Create cogmap message lists strictly following cog_utils logic (local/global only)."""
    messages, turn_logs, sample_cfg = _load_exploration_artifacts(combo_dir)
    sample_id = os.path.basename(sample_cfg.get("image_dir", "sample"))
    enable_think = ("think" in os.path.abspath(combo_dir).split(os.sep))
    exp_type = _detect_exp_type(combo_dir)

    out_msgs: List[List[Dict]] = []
    meta: List[Dict] = []

    user_idxs = _iter_user_indices(messages)

    if exp_type == "active":
        # For each turn after the first action, use previous turn index for decision
        for i in range(1, len(turn_logs)):
            t_idx = i - 1
            if t_idx >= len(user_idxs):
                break
            types = ["local", "global"] if (turn_logs[t_idx].get("exploration_log", {}) or {}).get("visible_objects") else ["global"]
            end_idx = user_idxs[t_idx]
            seq = _clone_until_inclusive(messages, end_idx)
            assert seq[-1]["role"] == "user"
            base_user = re.sub(r"You have a maximum of\s*\d+\s*exploration steps left.*", "", seq[-1]["content"], flags=re.DOTALL)
            for mtype in types:
                mod_seq = [m.copy() for m in seq]
                mod_seq[-1]["content"] = base_user + get_cogmap_prompt(mtype, enable_think)
                turn_number = _get_turn_number(mod_seq)
                meta_obj = {
                    "type": "cogmap",
                    "sample_id": sample_id,
                    "turn_number": turn_number,
                    "map_type": mtype,
                    "combo_dir": os.path.abspath(combo_dir),
                }
                meta_obj["message_id"] = generate_message_id(meta_obj)
                _add_message(out_msgs, meta, mod_seq, meta_obj)

    else:  # passive
        if user_idxs:
            end_idx = user_idxs[0]
            seq = _clone_until_inclusive(messages, end_idx)
            base_user = re.sub(r"You have a maximum of\s*\d+\s*exploration steps left.*", "", seq[-1]["content"], flags=re.DOTALL)
            mod_seq = [m.copy() for m in seq]
            mod_seq[-1]["content"] = base_user + get_cogmap_prompt("global", enable_think)
            meta_obj = {
                "type": "cogmap",
                "sample_id": sample_id,
                "turn_number": _get_turn_number(mod_seq),
                "map_type": "global",
                "combo_dir": os.path.abspath(combo_dir),
            }
            meta_obj["message_id"] = generate_message_id(meta_obj)
            _add_message(out_msgs, meta, mod_seq, meta_obj)

    return out_msgs, meta


def save_messages_jsonl(messages_list: List[List[Dict]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for msgs in messages_list:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")


def save_meta_jsonl(meta_list: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for meta in meta_list:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")







# ========================= Root-level Builders =========================

def build_all_under_root(
    root_dir: str,
    mode: str = "eval",
    eval_task_counts: Dict[str, int] | None = None,
    out: str | None = None,
    seed: int | None = 0,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Aggregate and write messages/meta for all combo dirs into a single built folder under root_dir."""
    all_msgs: List[List[Dict]] = []
    all_meta: List[Dict] = []
    for combo in iter_combo_dirs(root_dir):
        if mode == "eval":
            msgs, meta = build_evaluation_from_combo(combo, eval_task_counts or {"qa": 1}, seed=seed)
        else:
            msgs, meta = build_cogmap_from_combo(combo)
        all_msgs.extend(msgs)
        all_meta.extend(meta)

    built_root = resolve_built_root(root_dir, out)
    os.makedirs(built_root, exist_ok=True)
    msg_path, meta_path = paths_for_mode(built_root, mode)
    save_messages_jsonl(all_msgs, msg_path)
    save_meta_jsonl(all_meta, meta_path)
    return all_msgs, all_meta


# ========================= CLI Entrypoint =========================

def main_builder() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation/cogmap message lists into a single built folder under root")
    parser.add_argument("--root-dir", required=True, help="Root dir to scan for all combo dirs")
    parser.add_argument("--mode", choices=["eval", "cogmap"], default="eval")
    parser.add_argument("--eval-task-counts", default='{"qa": 1}', help='JSON string, e.g., {"qa": 2}')
    parser.add_argument("--out", default=None, help="Output directory (default: <root>/built_messages)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_all_under_root(
        root_dir=args.root_dir,
        mode=args.mode,
        eval_task_counts=json.loads(args.eval_task_counts),
        out=args.out,
        seed=args.seed,
    )
    print("Built messages and meta under:", resolve_built_root(args.root_dir, args.out))


if __name__ == "__main__":
    main_builder()

