# pages/03_📈_Analytics.py
import streamlit as st
import pandas as pd
from vagen.env.spatial.Player_platform.env_adapter import (
    get_user_session_state,
    require_user_id,
    get_base_user_id,
)

st.set_page_config(page_title="Analytics", layout="wide", page_icon="📈")
st.title("📈 Analytics")

session_id = require_user_id("Set your participant ID on the Home page before viewing analytics.")
base_id = get_base_user_id()
st.caption(f"Participant ID: {base_id} · Session ID: {session_id}")

# Share the same per-user bucket the Play page uses.
state = get_user_session_state(session_id)
env = state.get("env")
if not env:
    st.info("No environment yet.")
else:
    cfg = env.cfg.to_dict()
    st.subheader("Environment Config")
    st.json(cfg)
    summary = env.get_env_summary()
    st.subheader("Exploration Summary")
    st.json(summary["exploration_summary"])
    st.subheader("Evaluation Summary")
    st.json(summary["evaluation_summary"])
    # Use env_adapter's evaluation_manager instead of core_env
    eval_mgr = env.evaluation_manager
    if eval_mgr and eval_mgr.turn_logs:
        per_task = {}
        for log in eval_mgr.turn_logs:
            task_type = log.get("task_type") if isinstance(log, dict) else getattr(log, "task_type", None)
            score = log.get("score") if isinstance(log, dict) else getattr(log, "score", False)
            if task_type:
                total, cur_score = per_task.get(task_type, (0, 0))
                per_task[task_type] = (total + 1, cur_score + (score if score else 0))
        rows = [
            {
                "task": task,
                "accuracy": cur_score / total if total else 0.0,
                "score": cur_score,
                "total": total,
            }
            for task, (total, cur_score) in sorted(per_task.items())
        ]
        st.subheader("Evaluation Accuracy by Task")
        st.json(rows)
    st.subheader("Turn Logs")
    st.json(summary["env_turn_logs"])
    correct_answers = env.get_eval_answers()
    st.subheader("Evaluation Answers")
    st.write(correct_answers)


hist = state.get("history", [])
if not hist:
    st.info("No data yet.")
else:
    df = pd.DataFrame([{"t": h.t, "reward": h.reward} for h in hist])
    st.subheader("Reward per turn")
    st.line_chart(df.set_index("t"))
    st.subheader("Cumulative reward")
    df["cum_reward"] = df["reward"].cumsum()
    st.line_chart(df.set_index("t")[["cum_reward"]])
