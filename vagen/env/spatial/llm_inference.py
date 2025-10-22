import os
import json
import time
from typing import List, Dict, Tuple, Any
import argparse

from openai import OpenAI

from vagen.env.spatial.Base.tos_base.managers.cognitive_map_manager import CognitiveMapManager
from vagen.env.spatial.Base.tos_base.utils.cog_utils import _evaluate_cogmaps
from vagen.env.spatial.Base.tos_base.evaluation.tasks import evaluate_from_dict
from vagen.inference.model_interface.openai.model import OpenAIModelInterface
from vagen.inference.model_interface.openai.model_config import OpenAIModelConfig
from vagen.env.spatial.Base.tos_base.utils.utils import parse_llm_response
import dotenv
dotenv.load_dotenv()

from vagen.env.spatial.common import (
    resolve_built_root,
    paths_for_mode,
    list_built_pairs,
    load_messages_and_meta_jsonl as common_load_messages_and_meta_jsonl,
    responses_jsonl_path,
    iter_combo_dirs,
    load_history_manager,
)


"""Root-only inference runner: reads built files once and maps responses back via HistoryManager.load_from_dir."""


def submit_openai_batch(
    client: OpenAI,
    messages_list: List[List[Dict[str, Any]]],
    metas: List[Dict[str, Any]],
    model_config: dict,
    jsonl_path: str
) -> str:
    """Create JSONL, upload file, and start a batch job. Returns batch_id."""
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "w") as f:
        for i, msgs in enumerate(messages_list):
            mid = (metas[i] or {}).get("message_id", f"req_{i}")
            line = {
                "custom_id": str(mid),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_config["model_name"],
                    "messages": OpenAIModelInterface._convert_qwen_to_openai_format(msgs),
                    "max_completion_tokens": model_config["max_completion_tokens"],
                    "temperature": model_config["temperature"],
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch_input = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_input.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def collect_openai_batch(client: OpenAI, batch_id: str, poll_seconds: int = 10) -> List[Dict[str, Any]]:
    """Poll until batch completes. Returns list of {message_id, text, usage}."""
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in ("failed", "expired", "canceled"):
            raise RuntimeError(f"Batch {batch_id} status={b.status} reason={b.errors}")
        if b.status == "completed":
            break
        time.sleep(poll_seconds)

    out_file_id = b.output_file_id
    content = client.files.content(out_file_id)
    text = getattr(content, "text", None) or getattr(content, "content", None)
    if hasattr(text, "decode"):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        text = content.read().decode("utf-8")

    results: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj.get("custom_id")
        body = ((obj.get("response") or {}).get("body") or {})
        choices = body.get("choices") or []
        llm_text = ""
        if choices:
            msg = choices[0].get("message") or {}
            llm_text = msg.get("content", "")
        usage = body.get("usage") or {}
        results.append({"message_id": custom_id, "text": llm_text, "usage": usage})

    # Order by req index
    return results


# ========================= Direct Generate via Model Interface =========================

def generate_with_model_interface(
    model_config: dict,
    messages_list: List[List[Dict[str, Any]]],
    metas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cfg = OpenAIModelConfig(**model_config)
    interface = OpenAIModelInterface(cfg)
    results = interface.generate(messages_list)
    outputs: List[Dict[str, Any]] = []
    for i, r in enumerate(results):
        outputs.append({
            "message_id": (metas[i] or {}).get("message_id"),
            "text": r.get("text", ""),
            "usage": r.get("usage", {}),
        })
    return outputs


# ========================= Output Utilities =========================

def index_meta_by_id(metas: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(m.get("message_id")): m for m in metas}


def save_outputs_jsonl(outputs: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


# ========================= Input Loading (prebuilt) =========================
def load_prebuilt_inputs(built_root: str) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Load prebuilt inputs from built root directory."""
    pairs = list_built_pairs(built_root)
    messages_list: List[List[Dict[str, Any]]] = []
    metas: List[Dict[str, Any]] = []
    for msgs_path, meta_path in pairs:
        msgs, meta = common_load_messages_and_meta_jsonl(msgs_path, meta_path)
        messages_list.extend(msgs)
        metas.extend(meta)
    return messages_list, metas


def load_prebuilt_inputs_under_root(root_dir: str, out_dir: str | None) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Load prebuilt inputs from root directory."""
    built_root = resolve_built_root(root_dir, out_dir)
    return load_prebuilt_inputs(built_root)


def get_done_ids(responses_path: str) -> set:
    if not os.path.exists(responses_path):
        return set()
    done = set()
    with open(responses_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                mid = str(obj.get("message_id"))
                if mid:
                    done.add(mid)
            except Exception:
                continue
    return done


# ========================= Mapping LLM Responses back to History =========================

def map_llm_responses(
    combo_dir: str,
    metas: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    cogmap_config: Dict[str, Any] | None = None,
) -> None:
    """
    Map responses (aligned to metas) into the correct history files.
    - evaluation: write per-question via HistoryManager.update_eval_turn_log
    - cogmap: evaluate and write via HistoryManager.update_cogmap
    """
    history = load_history_manager(combo_dir)
    sample_cfg = json.load(open(history.sample_config_path))

    meta_by_id = index_meta_by_id(metas)

    # Group for cogmap: (turn_idx) -> {map_type: text}
    cogmap_groups: Dict[int, Dict[str, str]] = {}

    for out in outputs:
        mid = str(out.get("message_id"))
        meta = meta_by_id.get(mid)
        if not meta:
            continue
        text = out.get("text", "")
        if (meta.get("type") or "").lower() == "evaluation":
            qid = meta["question_id"]
            if history.has_question(qid):
                print("question repeated:", qid)
                continue  # Skip existing
            eval_data = (meta.get("evaluation_data") or {})
            # Evaluate using same logic as in env runtime
            _, answer, _ = parse_llm_response(text)
            score, info = evaluate_from_dict(eval_data, answer)
            task_class = meta.get("task_class") or meta.get("task_type")
            turn_log = {
                "is_exploration_phase": False,
                "evaluation_log": {
                    "task_type": task_class,
                    "user_answer": text,
                    "score": score,
                    "evaluation_info": info or {},
                    "evaluation_data": eval_data,
                },
                "assistant_raw_message": text,
                "room_state": sample_cfg["room_dict"],
                "agent_state": sample_cfg["agent_dict"],
                "turn_number": 1,
            }
            history.update_eval_turn_log(turn_log)

        elif (meta.get("type") or "").lower() == "cogmap":
            tnum = int(meta.get("turn_number", 1))
            t_idx = tnum - 1
            cogmap_groups.setdefault(t_idx, {})[meta.get("map_type", "global")] = text

    # Process cogmap groups
    if cogmap_groups:
        cm_cfg = cogmap_config or {"cogmap_type": "standard", "pos_allow_scale": False, "scope": "all"}
        cm = CognitiveMapManager(**cm_cfg)

        for t_idx, resp_by_type in cogmap_groups.items():
            if not (0 <= t_idx < len(history.exploration_turn_logs)):
                continue
            turn_log = history.exploration_turn_logs[t_idx]
            try:
                cogmap_log = _evaluate_cogmaps(cm, resp_by_type, turn_log)
                result = cogmap_log.to_dict() if cogmap_log else {}
            except Exception:
                result = {k: {"original_response": v} for k, v in resp_by_type.items()}
            history.update_cogmap({
                "is_exploration_phase": True,
                "turn_number": t_idx + 1,
                "cogmap_log": result,
            })

    history.save()


# ========================= Re-evaluation of existing answers =========================

def reevaluate_combo_dir(combo_dir: str) -> int:
    """Re-evaluate all existing evaluation answers in a combo directory.
    
    Args:
        combo_dir: Path to combo directory
        
    Returns:
        Number of answers re-evaluated
    """
    history = load_history_manager(combo_dir)
    
    # Get all evaluation turn logs
    eval_logs = history.evaluation_turn_logs
    if not eval_logs:
        print(f"No evaluation logs found in {combo_dir}")
        return 0
    
    count = 0
    for questions in eval_logs.values():
        for question in questions.values():
            eval_log = question.get("evaluation_log", {})
            if not eval_log:
                continue

            # Get existing data
            eval_data = eval_log.get("evaluation_data")
            user_answer_raw = eval_log.get("user_answer", "")
            
            if not eval_data:
                continue
            
            # Parse answer from raw message (same as in map_llm_responses)
            _, answer, _ = parse_llm_response(user_answer_raw)
            
            # Re-evaluate
            score, info = evaluate_from_dict(eval_data, answer)
            
            # Update the log
            eval_log["score"] = score
            eval_log["evaluation_info"] = info or {}
            
            count += 1
    
    # Save updated history
    history.save()
    print(f"Re-evaluated {count} answers in {combo_dir}")
    return count


def reevaluate_combo_dirs(combo_dirs: List[str]) -> None:
    """Re-evaluate all existing evaluation answers in multiple combo directories.
    
    Args:
        combo_dirs: List of combo directory paths
    """
    total_count = 0
    for combo_dir in combo_dirs:
        try:
            count = reevaluate_combo_dir(combo_dir)
            total_count += count
        except Exception as e:
            print(f"Error re-evaluating {combo_dir}: {e}")
            continue
    
    print(f"\nRe-evaluation completed: {total_count} answers re-evaluated across {len(combo_dirs)} combo directories.")


# ========================= Combo-level inference =========================

def run_inference_for_combo_dirs(
    combo_dirs: List[str],
    model_config: dict,
    mode: str = "eval",
    eval_task_counts: Dict[str, int] | None = None,
    inference_mode: str = "direct",
    eval_override: bool = False,
    cogmap_override: bool = False,
    cogmap_reevaluate: bool = False,
) -> None:
    """Run inference for a specific list of combo directories.
    
    Args:
        combo_dirs: List of combo directory paths
        model_name: Model name for inference
        mode: 'eval' or 'cogmap'
        eval_task_counts: Evaluation task counts (for eval mode)
        seed: Seed for task generation (for eval mode)
        inference_mode: 'batch' or 'direct'
        eval_override: If True, ignore existing evaluation history and regenerate all
        cogmap_override: If True, regenerate all cogmaps; if False, skip existing cogmaps
        cogmap_reevaluate: If True, re-evaluate existing cognitive maps (passed to CognitiveMapManager)
    """
    from vagen.env.spatial.message_list_builder import build_all_for_combo_dirs
    
    # Build messages for the specified combos (override logic handled in builder)
    all_msgs, all_meta = build_all_for_combo_dirs(
        combo_dirs=combo_dirs,
        mode=mode,
        eval_task_counts=eval_task_counts,
        eval_override=eval_override,
        cogmap_override=cogmap_override,
    )
    
    if not all_msgs or not all_meta:
        print(f"No messages generated for {mode} mode")
        return
    
    # Run inference
    if inference_mode == "batch":
        client = OpenAI()
        # Use a temporary directory for batch files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_jsonl = os.path.join(tmpdir, "batch_input.jsonl")
            batch_id = submit_openai_batch(client, all_msgs, all_meta, model_config, batch_jsonl)
            print(f"Submitted batch: {batch_id}, Messages: {len(all_msgs)}")
            outputs = collect_openai_batch(client, batch_id)
    else:
        outputs = generate_with_model_interface(model_config, all_msgs, all_meta)
    
    # Map responses back to histories
    meta_by_id = index_meta_by_id(all_meta)
    combo_data: Dict[str, Dict[str, List]] = {}
    
    for out in outputs:
        mid = str(out.get("message_id"))
        m = meta_by_id.get(mid)
        if not m:
            continue
        cdir = m.get("combo_dir")
        if cdir not in combo_data:
            combo_data[cdir] = {"outputs": [], "metas": []}
        combo_data[cdir]["outputs"].append(out)
        combo_data[cdir]["metas"].append(m)
    
    # Update history for each combo
    cogmap_config = {"cogmap_reevaluate": cogmap_reevaluate} if cogmap_reevaluate and mode == "cogmap" else None
    for cdir, data in combo_data.items():
        map_llm_responses(cdir, data["metas"], data["outputs"], cogmap_config=cogmap_config)
    
    print(f"Completed {mode} inference for {len(combo_data)} combos, processed {len(outputs)} responses.")


# ========================= __main__ demos =========================

def main_infer() -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference on prebuilt messages and map results by message_id")
    parser.add_argument("--root-dir", required=True, help="Root dir to scan for combo dirs")
    parser.add_argument("--model-name", default="gpt-4o-mini")
    parser.add_argument("--mode", choices=["batch", "direct"], default="batch")
    parser.add_argument("--out-dir", default=None, help="Built output dir under root (single folder)")
    parser.add_argument("--override", action="store_true", help="If set, resend all and overwrite responses")
    args = parser.parse_args()

    # Gather inputs
    all_msgs, all_meta = load_prebuilt_inputs_under_root(args.root_dir, args.out_dir)
    built_dir = resolve_built_root(args.root_dir, args.out_dir)

    os.makedirs(built_dir, exist_ok=True)
    responses_path = responses_jsonl_path(built_dir, args.model_name)

    assert all_msgs and all_meta and len(all_msgs) == len(all_meta), "No inputs loaded or mismatched lengths"

    # Use message_id from builder meta directly
    metas_with_ids: List[Dict[str, Any]] = []
    for meta in all_meta:
        meta = dict(meta)
        assert "message_id" in meta, "Message ID is required in meta"
        metas_with_ids.append(meta)

    # Skip already processed unless override
    if not args.override:
        done_ids = get_done_ids(responses_path)
        filtered = [(msg, meta) for msg, meta in zip(all_msgs, metas_with_ids) if str(meta["message_id"]) not in done_ids]
        if filtered:
            all_msgs, metas_with_ids = zip(*filtered)
            all_msgs, metas_with_ids = list(all_msgs), list(metas_with_ids)
        else:
            all_msgs, metas_with_ids = [], []

    if not all_msgs or not metas_with_ids:
        print("No new messages to process. Skipping.")
        raise SystemExit(0)

    # Run inference
    if args.mode == "batch":
        client = OpenAI()
        batch_jsonl = os.path.join(built_dir, "batch_input.jsonl")
        os.makedirs(os.path.dirname(batch_jsonl) or ".", exist_ok=True)
        batch_id = submit_openai_batch(client, all_msgs, metas_with_ids, args.model_name, batch_jsonl)
        print(f"Submitted batch: {batch_id}")
        outputs = collect_openai_batch(client, batch_id)
    else:
        outputs = generate_with_model_interface(args.model_name, all_msgs, metas_with_ids)

    # Append responses
    save_outputs_jsonl(outputs, responses_path)

    # Map back to histories per combo
    # Group outputs by combo_dir
    meta_by_id = index_meta_by_id(metas_with_ids)
    combo_to_outputs: Dict[str, List[Dict[str, Any]]] = {}
    combo_to_metas: Dict[str, List[Dict[str, Any]]] = {}
    for out in outputs:
        mid = str(out.get("message_id"))
        m = meta_by_id.get(mid)
        if not m:
            continue
        cdir = m.get("combo_dir")
        combo_to_outputs.setdefault(cdir, []).append(out)
        combo_to_metas.setdefault(cdir, []).append(m)

    for cdir, outs in combo_to_outputs.items():
        metas = combo_to_metas.get(cdir, [])
        if metas:
            map_llm_responses(cdir, metas, outs)

    print(f"Appended {len(outputs)} responses to {responses_path} and updated histories for {len(combo_to_outputs)} combos.")


if __name__ == "__main__":
    main_infer()


