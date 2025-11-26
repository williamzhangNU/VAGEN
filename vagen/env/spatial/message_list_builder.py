from typing import List, Dict, Tuple, Any
import os
import json
import re
import numpy as np
import argparse

# Reuse existing components
from vagen.env.spatial.Base.tos_base import Room, Agent
from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType
from vagen.env.spatial.Base.tos_base.prompts.cogmap_prompts import get_cogmap_prompt
from vagen.env.spatial.Base.tos_base.utils.utils import hash, numpy_to_python

# Shared common utilities/constants
from vagen.env.spatial.common import (
    MESSAGES_BASENAME,
    EXPLORATION_LOG_BASENAME,
    STATE_BASENAME,
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
    state_path = os.path.join(combo_dir, STATE_BASENAME)
    sample_cfg = read_json(state_path)
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


"""Builder utilities for evaluation and cogmap message lists."""


def _add_message(out_msgs: List[List[Dict]], out_meta: List[Dict], msgs: List[Dict], meta: Dict[str, Any]) -> None:
    out_msgs.append(msgs)
    out_meta.append(meta)


def build_evaluation_from_combo(
    combo_dir: str,
    eval_task_counts: Dict[str, int],
    eval_override: bool = False,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Create evaluation message lists from exploration history for one sample combo dir.

    Returns (messages_list, meta_list) with meta including sample_id, task_type, question_id, message_id.

    Args:
        combo_dir: Directory containing exploration history
        eval_task_counts: Dict mapping task types to count
        seed: Seed for task generation
        eval_override: If True, ignore existing evaluation history and regenerate all questions
    """
    messages, _turn_logs, sample_cfg = _load_exploration_artifacts(combo_dir)

    base_msgs = [m.copy() for m in messages]

    # Load history manager with eval_override flag
    hm = load_history_manager(combo_dir, eval_override=eval_override, all_tasks=list(eval_task_counts.keys()))
    out_msgs: List[List[Dict]] = []
    meta: List[Dict] = []

    # Get existing eval counts (will be empty if eval_override=True)
    existing_ids = hm.get_eval_ids()
    room = Room.from_dict(sample_cfg["room_dict"]).copy()
    agent = Agent.from_dict(sample_cfg["agent_dict"]).copy()
    image_dir = sample_cfg.get("image_dir")
    # Track message_ids to ensure uniqueness
    seen_message_ids = set()

    for task_short, count in (eval_task_counts or {}).items():
        # Skip false_belief_exp as it requires running a full environment
        if task_short == 'false_belief_exp':
            continue
            
        is_vision_question = False
        if 'vision' in task_short:
            if hm.observation_config['render_mode'] == "text":
                raise ValueError('cannot use vision question in text mode')
            else:
                is_vision_question = True

        task = EvalTaskType.create_task(task_short, np.random.default_rng(hm.seed), room, agent, {"image_dir": image_dir if is_vision_question else None}, None)
        task_class_name = task.__class__.__name__

        # Calculate how many questions still needed
        existing_id_for_task = existing_ids.get(task_class_name, [])
        for i in range(count - len(existing_id_for_task)):
            # retry
            q_text = task.generate_question()
            retry = 20
            while task.eval_data.id in existing_id_for_task and retry:
                q_text = task.generate_question()
                retry -= 1
            assert task.eval_data.id not in existing_id_for_task, f"Failed to generate unique question for {task_short} in {combo_dir}"
            existing_id_for_task.append(task.eval_data.id)
            assert base_msgs[-1]["role"] == "user"
            new_list = [m.copy() for m in base_msgs]
            new_list[-1]['content'] = new_list[-1]['content'] + "\n" + q_text
            if is_vision_question:
                if "images" not in new_list[-1]:
                    new_list[-1]["images"] = []
                assert os.path.exists(os.path.join(image_dir, f"{task.eval_data.id}.png"))
                new_list[-1]["images"] += [os.path.join(image_dir, f"{task.eval_data.id}.png")]
            meta_obj = {
                "type": "evaluation",
                "task_type": task_short,
                "task_class": task.__class__.__name__,
                "question_id": task.eval_data.id,
                "combo_dir": os.path.abspath(combo_dir),
                "message_images": new_list[-1].get("images", []),
                "evaluation_data": task.eval_data.to_dict(),
            }
            meta_obj["message_id"] = hash(json.dumps(meta_obj, sort_keys=True))
            if meta_obj["message_id"] in seen_message_ids:
                raise ValueError(f"Duplicate message_id detected: {meta_obj['message_id']} for combo_dir={combo_dir}, task={task_short}, question_id={task.eval_data.id}")
            seen_message_ids.add(meta_obj["message_id"])
            _add_message(out_msgs, meta, new_list, meta_obj)

    return out_msgs, meta


def build_cogmap_from_combo(
    combo_dir: str,
    cogmap_override: bool = False,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Create cogmap message lists strictly following cog_utils logic (local/global only).

    Args:
        combo_dir: Directory containing exploration history
        cogmap_override: If True, regenerate all cogmaps; if False, skip turns with existing cogmaps
    """
    messages, turn_logs, sample_cfg = _load_exploration_artifacts(combo_dir)
    # Derive sample_id from combo_dir path instead of image_dir
    # Format: room_hash (parent of vision/text directory)
    combo_abs = os.path.abspath(combo_dir)
    parts = combo_abs.split(os.sep)
    try:
        render_idx = max(i for i, p in enumerate(parts) if p in ("vision", "text"))
        room_hash_idx = render_idx - 1
        sample_id = parts[room_hash_idx]
    except (ValueError, IndexError):
        # Fallback if path structure is unexpected
        sample_id = os.path.basename(sample_cfg.get("image_dir", "sample"))
    
    hm = load_history_manager(combo_dir)
    enable_think = hm.get_enable_think()
    exp_type = getattr(hm, "exp_type", _detect_exp_type(combo_dir))

    out_msgs: List[List[Dict]] = []
    meta: List[Dict] = []

    user_idxs = _iter_user_indices(messages)

    if exp_type == "active":
        # For each turn after the first action, use previous turn index for decision
        for t_idx in range(1, len(turn_logs)):
            # Check if cogmap already exists for this turn (unless override)
            if not cogmap_override:
                existing_cogmap = hm.get_cogmap(t_idx)
                if existing_cogmap:
                    print(f"Skipping turn {t_idx} in {combo_dir}: cogmap already exists")
                    continue
            
            types = ["local", "global"] if (turn_logs[t_idx].get("exploration_log", {}) or {}).get("visible_objects") else ["global"]
            end_idx = user_idxs[t_idx]
            seq = _clone_until_inclusive(messages, end_idx)
            assert seq[-1]["role"] == "user"
            base_user = re.sub(r"You have a maximum of\s*\d+\s*exploration steps left.*", "", seq[-1]["content"], flags=re.DOTALL) 
            mod_seq = [m.copy() for m in seq]
            # current turn cogmap question => previous turn number !!!
            turn_number = _get_turn_number(mod_seq) -1
            for mtype in types:
                mod_seq[-1]["content"] = base_user + get_cogmap_prompt(mtype, enable_think)
                meta_obj = {
                    "type": "cogmap",
                    "sample_id": sample_id,
                    "turn_number": turn_number,
                    "map_type": mtype,
                    "combo_dir": os.path.abspath(combo_dir),
                }
                meta_obj["message_id"] = hash(json.dumps(meta_obj, sort_keys=True, default=numpy_to_python))
                _add_message(out_msgs, meta, mod_seq, meta_obj)

    else:  # passive
        if user_idxs:
            # Check if cogmap already exists (unless override)
            if not cogmap_override:
                existing_cogmap = hm.get_cogmap(0)
                if existing_cogmap:
                    print(f"Skipping passive cogmap in {combo_dir}: cogmap already exists")
                    return out_msgs, meta
            
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
            meta_obj["message_id"] = hash(json.dumps(meta_obj, sort_keys=True))
            _add_message(out_msgs, meta, mod_seq, meta_obj)

    return out_msgs, meta


def save_messages_jsonl(messages_list: List[List[Dict]], out_path: str, meta_list: List[Dict] | None = None) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for i, msgs in enumerate(messages_list):
            mid = None
            if meta_list and i < len(meta_list):
                mid = (meta_list[i] or {}).get("message_id")
            obj = {"messages": msgs}
            if mid is not None:
                obj["message_id"] = mid
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_meta_jsonl(meta_list: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for meta in meta_list:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")







# ========================= Root-level Builders =========================

def build_all_for_combo_dirs(
    combo_dirs: List[str],
    mode: str = "eval",
    eval_task_counts: Dict[str, int] | None = None,
    eval_override: bool = False,
    cogmap_override: bool = False,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Build messages/meta for a specific list of combo directories.
    
    Args:
        combo_dirs: List of combo directory paths to process
        mode: 'eval' or 'cogmap'
        eval_task_counts: Dict mapping task types to count (for eval mode)
        seed: Seed for task generation (for eval mode)
        eval_override: If True, ignore existing evaluation history and regenerate all
        cogmap_override: If True, regenerate all cogmaps; if False, skip existing cogmaps
    
    Returns:
        Tuple of (messages_list, meta_list)
    """
    all_msgs: List[List[Dict]] = []
    all_meta: List[Dict] = []
    
    for combo in combo_dirs:
        if mode == "eval":
            assert eval_task_counts is not None, "eval_task_counts must be provided for eval mode"
            msgs, meta = build_evaluation_from_combo(
                combo, eval_task_counts, 
                eval_override=eval_override
            )
        else:
            msgs, meta = build_cogmap_from_combo(combo, cogmap_override=cogmap_override)
        all_msgs.extend(msgs)
        all_meta.extend(meta)
    
    return all_msgs, all_meta


def build_all_under_root(
    root_dir: str,
    mode: str = "eval",
    eval_task_counts: Dict[str, int] | None = None,
    out: str | None = None,
    seed: int | None = 0,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Aggregate and write messages/meta for all combo dirs into a single built folder under root_dir.
    
    Args:
        root_dir: Root directory to scan for combo dirs
        mode: 'eval' or 'cogmap'
        eval_task_counts: Dict mapping task types to count (for eval mode)
        out: Output directory path
        seed: Seed for task generation (for eval mode)
    """
    all_msgs: List[List[Dict]] = []
    all_meta: List[Dict] = []
    for combo in iter_combo_dirs(root_dir):
        if mode == "eval":
            msgs, meta = build_evaluation_from_combo(combo, eval_task_counts or {"qa": 1})
        else:
            msgs, meta = build_cogmap_from_combo(combo)
        all_msgs.extend(msgs)
        all_meta.extend(meta)

    built_root = resolve_built_root(root_dir, out)
    os.makedirs(built_root, exist_ok=True)
    msg_path, meta_path = paths_for_mode(built_root, mode)
    save_messages_jsonl(all_msgs, msg_path, all_meta)
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

