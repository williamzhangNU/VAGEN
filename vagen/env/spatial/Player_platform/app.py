import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

st.set_page_config(page_title="Spatial Exploration Dashboard", layout="wide", page_icon="🧭")

st.title("🧭 Spatial Exploration Dashboard")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

st.session_state.user_id = st.text_input(
    "Enter your participant ID",
    value=st.session_state.user_id or "",
    placeholder="e.g., user123"
)

if not st.session_state.user_id:
    st.warning("⚠️ Please enter your ID before proceeding to Play.")
else:
    st.success(f"Using ID: {st.session_state.user_id}")

st.markdown("""
Welcome! Use the **Play** page to interact with the environment turn by turn.
- Type an action → press **Submit** (or hit Enter)
- See the new observation, reward, and whether the episode is done
- **Trajectory** page shows a turn-by-turn log
- **Analytics** gives quick summaries
- **Admin** lets you set seed, max turns, and reset
""")

st.markdown("---")
st.subheader("Quick Tips")
st.markdown("""
- You can reset the episode any time from the **Admin** page.
- Your session is isolated—no server-side persistence unless you add it.
""")
