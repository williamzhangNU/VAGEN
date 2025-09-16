# pages/04_⚙️_Admin.py
import streamlit as st
from env_adapter import SpatialEnvAdapter
from vagen.env.spatial.env_config import SpatialGymConfig

st.set_page_config(page_title="Admin", layout="wide", page_icon="⚙️")

st.title("⚙️ Admin")

if "env" not in st.session_state:
    st.warning("⚠️ No environment loaded yet. Go to Play first.")
    st.stop()

seed = st.number_input("Seed", min_value=0, value=0, step=1)
if st.button("Reset Episode"):
    obs, info = st.session_state.env.env.reset(seed=int(seed))

    # Normalize obs into dict
    if isinstance(obs, str):
        init_obs = {"obs_str": obs}
    else:
        init_obs = obs

    # Reset bookkeeping
    st.session_state.turn = 0
    st.session_state.episode_id += 1
    st.session_state.last_obs = init_obs

    # ✅ Add back system message
    from env_adapter import summarize_turn
    st.session_state.history = [
        summarize_turn(
            t=0,
            action="(system)",
            obs=init_obs,
            reward=0.0,
            done=False,
            info=info or {}
        )
    ]

    st.success("Episode reset. Initial prompt added to chat history.")
