# pages/02_🧭_Trajectory.py
import streamlit as st
import pandas as pd
from env_adapter import TurnRecord

st.set_page_config(page_title="Trajectory", layout="wide", page_icon="🧭")
st.title("🧭 Trajectory Log")

if "history" not in st.session_state or not st.session_state.history:
    st.info("No turns yet. Go to **Play** and start interacting.")
else:
    data = [{
        "t": r.t,
        "action": r.action,
        "obs": r.obs_text,
        "reward": r.reward,
        "done": r.done
    } for r in st.session_state.history]
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, height=500)
