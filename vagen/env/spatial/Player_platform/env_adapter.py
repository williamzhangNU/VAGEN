from __future__ import annotations
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import os, json
import yaml
import re
from omegaconf import OmegaConf
from vagen.env.spatial.Base.tos_base.utils.utils import THINK_LABEL, ANSWER_LABEL, format_llm_output
from vagen.env.spatial.env import SpatialGym                  
from vagen.env.spatial.env_config import SpatialGymConfig     
@dataclass
class TurnRecord:
    t: int
    action: str
    obs_text: str
    reward: float
    done: bool
    info: Dict[str, Any]
    obs_raw: Dict[str, Any] = None 

def load_cfg_from_yaml(path: str) -> SpatialGymConfig:
    """
    Load SpatialGymConfig from a YAML file.
    Converts eval_task_counts to eval_tasks format.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    del raw['seed-range']
    # Convert eval_task_counts to eval_tasks if present
    if "eval_task_counts" in raw and not raw.get("eval_tasks"):
        eval_tasks = []
        for task_type, count in raw["eval_task_counts"].items():
            eval_tasks.append({
                "task_type": task_type,
                "num": int(count)
            })
        raw["eval_tasks"] = eval_tasks
        # Remove eval_task_counts to avoid confusion
        del raw["eval_task_counts"]

    # OmegaConf to ensure compatibility with ListConfig, etc.
    conf = OmegaConf.create(raw)

    # Convert into dataclass
    return SpatialGymConfig(**OmegaConf.to_container(conf, resolve=True))

def _wrap_user_action_for_env(user_text: str, enable_think: bool = True) -> str:
    """
    Convert human input into the expected LLM-style format (header-based).
    """
    action = user_text.strip()
    think_content = "Placeholder" if enable_think else ""
    # Case 1: single letter answer (A, B, C, etc.)
    if re.fullmatch(r"[A-Za-z]", action):
        return format_llm_output(think_content, action, enable_think=enable_think)

    # Case 2: action sequence (contains "()")
    if "(" in action and ")" in action:
        return format_llm_output(think_content, f"Actions: [{action}]", enable_think=enable_think)
    return format_llm_output(think_content, action, enable_think=enable_think)

def save_episode(user_id: str, episode_id: int, trajectory: list, analytics: dict, correct_answers: dict = None):
    log_dir = os.path.join("vagen/env/spatial/Player_platform/logs", user_id)
    os.makedirs(log_dir, exist_ok=True)

    out_path = os.path.join(log_dir, f"episode_{episode_id}.json")

    clean_traj = []
    for r in trajectory:
        # If it's a dataclass or object with __dict__, copy it
        if hasattr(r, "__dict__"):
            d = r.__dict__.copy()
        else:
            d = dict(r)  # already a dict

        #Drop obs_raw (contains PIL images)
        d.pop("obs_raw", None)
        clean_traj.append(d)

    payload = {
        "user_id": user_id,
        "episode_id": episode_id,
        "trajectory": clean_traj,
        "analytics": analytics,
        "correct_answers": correct_answers,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


class SpatialEnvAdapter:
    """
    Thin wrapper exposing a stable API to the Streamlit pages.
    """
    def __init__(
        self,
        cfg: SpatialGymConfig,
    ):
        self.cfg = cfg
        self.env = SpatialGym(cfg)
        self.turn = 0
        self._last_obs = None
        self._last_info = {}

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs, info = self.env.reset(seed=seed)
        if isinstance(obs, str):
            obs = {"obs_str": obs}
        self.turn = 0
        self._last_obs = obs
        self._last_info = info or {}
        return obs

    def step(self, user_action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Streamlit calls this with human action text. We wrap it for the env.
        Your env.step returns: (obs: dict, reward: float, done: bool, step_info: dict)
        """
        enable_think = bool(self.cfg.prompt_config.get("enable_think", True))
        llm_response = _wrap_user_action_for_env(user_action, enable_think=enable_think)

        obs, reward, done, step_info = self.env.step(llm_response)
        self.turn += 1
        self._last_obs = obs
        self._last_info = step_info or {}
        return obs, reward, done, step_info

    def get_eval_answers(self):
        answers = []
        for task in self.env.evaluation_manager.tasks:
            answers.append(task.answer)
        return answers

    def get_room_objects_str(self):
        return ";".join(sorted(self.env.exploration_manager.observed_items))

    def max_turn(self) -> int:
        return int(self.cfg.max_exp_steps or 20)

    # ---- optional helpers surfaced to UI pages ----
    def get_env_summary(self) -> Dict[str, Any]:
        env_summary = self.env.get_env_summary()
        env_summary["exploration_summary"] = self.env.get_exp_summary()
        env_summary["evaluation_summary"] = self.env.get_eval_summary()
        return env_summary

    def render_cache(self) -> Dict[str, Any]:
        return self.env.render() 

def format_obs(obs: Dict[str, Any]) -> str:
    """Pick the text to display to the user (your env places FORMAT_PROMPT at the end already)."""
    return obs.get("obs_str", "").strip()

def summarize_turn(t: int, action: str, obs: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> TurnRecord:
    return TurnRecord(
        t=t,
        action=action,
        obs_text=format_obs(obs),   
        reward=reward,
        done=done,
        info=info,
        obs_raw=obs                 
    )

