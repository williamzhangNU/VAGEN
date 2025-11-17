# vagen/env/spatial/Player_platform/pages/Play.py
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import streamlit as st
import streamlit.components.v1 as components
import yaml

from vagen.env.spatial.Player_platform.env_adapter import (
    SpatialEnvAdapter,
    summarize_turn,
    format_obs,
    save_episode,
    load_cfg_from_yaml,
)
st.set_page_config(page_title="Play", layout="wide", page_icon="🎮")

st.title("🎮 Exploration (Chat Mode)")
config_path = "vagen/env/spatial/Player_platform/config.yaml"

# ---- Parse seed-range from config ----
def parse_seed_range(config_path):
    """Parse seed-range from config file and return list of available seeds."""
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    seed_range_str = raw_config.get('seed-range', '0-0')
    try:
        start, end = map(int, seed_range_str.split('-'))
        return list(range(start, end + 1))
    except:
        return [0]

# Get available seeds
available_seeds = parse_seed_range(config_path)

# ---- Sample selector in sidebar ----
with st.sidebar:
    st.subheader("Sample Selection")
    selected_seed = st.selectbox(
        "Select Sample (Seed)",
        options=available_seeds,
        index=0,
        key="seed_selector"
    )

    # Reset button to reload with new seed
    if st.button("🔄 Load Selected Sample", key="load_sample"):
        # Clear the environment to force reload
        if "env" in st.session_state:
            del st.session_state.env
        st.session_state.force_reload = True

# ---- Bootstrap env once ----
if "env" not in st.session_state or st.session_state.get("loaded_config_path") != config_path or st.session_state.get("current_seed") != selected_seed or st.session_state.get("force_reload", False):
    cfg = load_cfg_from_yaml(config_path)
    st.session_state.env = SpatialEnvAdapter(cfg)
    st.session_state.loaded_config_path = config_path
    st.session_state.current_seed = selected_seed
    st.session_state.force_reload = False

    # Reset env and get initial observation with selected seed
    obs, info = st.session_state.env.env.reset(seed=selected_seed)
    if isinstance(obs, str):
        init_obs = {"obs_str": obs}
    else:
        init_obs = obs

    st.session_state.last_obs = init_obs
    st.session_state.history = [
        summarize_turn(
            t=0,
            action="(system)",
            obs=init_obs,
            reward=0.0,
            done=False,
            info=info or {},
        )
    ]
    st.session_state.episode_id = 1
    st.session_state.turn = 0
    print(f"Loaded sample with seed: {selected_seed}")

# ---- Initialize action buffer ----
if "action_buffer" not in st.session_state:
    st.session_state.action_buffer = ["Observe()"]

# ---- Render chat history ----
chat_container = st.container()
with chat_container:
    for rec in st.session_state.history:
        if rec.action == "(system)":
            with st.chat_message("assistant", avatar="🛠️"):
                st.markdown(rec.obs_text)
                if rec.obs_raw and "multi_modal_data" in rec.obs_raw:
                    imgs = rec.obs_raw["multi_modal_data"].get("<image>", [])
                    if imgs:
                        st.image(imgs, clamp=True, width=512, channels="RGB")
        else:
            with st.chat_message("user"):
                st.markdown(rec.action)
            with st.chat_message("assistant"):
                st.markdown(rec.obs_text)
                if rec.obs_raw and "multi_modal_data" in rec.obs_raw:
                    imgs = rec.obs_raw["multi_modal_data"].get("<image>", [])
                    if imgs:
                        st.image(imgs, clamp=True, width=512, channels="RGB")

# Auto-scroll to latest observation after sending action
if st.session_state.get("scroll_to_bottom", False):
    components.html(
        """
        <script>
            setTimeout(function() {
                var chatMessages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                if (chatMessages.length > 0) {
                    var lastMessage = chatMessages[chatMessages.length - 1];
                    lastMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 100);
        </script>
        """,
        height=0,
    )
    st.session_state.scroll_to_bottom = False

# ---- Action Builder (moved to bottom) ----
st.divider()
st.subheader("Action Builder")

built_action = None

# Check if in evaluation phase (after Term has been sent)
if not st.session_state.env.env.is_exploration_phase:
    # In evaluation phase, only show MCQ answer options
    eval_manager = st.session_state.env.env.evaluation_manager
    current_idx = eval_manager.current_index
    total_questions = len(eval_manager.tasks)

    st.info(f"Evaluation Question {current_idx + 1} of {total_questions}")
    choices = ["A", "B", "C", "D"]
    answer = st.radio("Select your answer:", choices, key=f"mcq_answer_{current_idx}")
    if answer:
        built_action = answer
else:
    # In exploration phase, show action type selector
    action_type = st.selectbox(
        "Action type",
        ["-- select --", "JumpTo", "Rotate", "Query", "Term"],
        key="action_type"
    )

    if action_type == "JumpTo":
        room_objects = st.session_state.env.env.initial_room.all_objects

        obj_names = [o.name for o in room_objects]
        target = st.selectbox("Jump to object", obj_names, key="jumpto_target")
        if target:
            built_action = f"JumpTo({target})"

    elif action_type == "Rotate":
        deg = st.selectbox("Degrees", [-270, -180, -90, 0, 90, 180, 270], key="rotate_deg")
        built_action = f"Rotate({deg})"

    elif action_type == "Query":
        room_objects = st.session_state.env.env.initial_room.all_objects
        obj_names = [o.name for o in room_objects]
        target = st.selectbox("Query which object?", obj_names, key="query_target")
        if target:
            built_action = f"Query({target})"

    elif action_type == "Term":
        built_action = "Term()"

# --- Render buttons ---
# Different buttons for exploration vs evaluation phase
if not st.session_state.env.env.is_exploration_phase:
    # Evaluation phase: show Next or Submit button
    eval_manager = st.session_state.env.env.evaluation_manager
    current_idx = eval_manager.current_index
    total_questions = len(eval_manager.tasks)
    is_last_question = (current_idx == total_questions - 1)

    if is_last_question:
        send_clicked = st.button("✅ Submit Final Answer", key="submit_final")
    else:
        send_clicked = st.button("➡️ Next Question", key="next_question")
    add_clicked = False
    clear_clicked = False
else:
    # Exploration phase: show normal action buttons
    add_clicked = st.button("➕ Add to Action Sequence", key="add_to_buffer")
    send_clicked = st.button("🚀 Send Action Sequence", key="send_buffer")
    clear_clicked = st.button("🗑️ Clear Action Sequence", key="clear_buffer")

# Handle add
if add_clicked and built_action:
    if built_action.startswith("Term()"):
        # If Term is selected, remove Observe() and add Term
        st.session_state.action_buffer = [built_action]
    elif built_action in ["A", "B", "C", "D"]:
        # If it's an MCQ answer, replace buffer with just the answer
        st.session_state.action_buffer = [built_action]
    else:
        # For non-Term actions, insert before Observe()
        if "Observe()" in st.session_state.action_buffer:
            # Insert before Observe()
            observe_index = st.session_state.action_buffer.index("Observe()")
            st.session_state.action_buffer.insert(observe_index, built_action)
        else:
            # If no Observe(), just append
            st.session_state.action_buffer.append(built_action)

# Handle clear
if clear_clicked:
    st.session_state.action_buffer = ["Observe()"]

# Show buffer (only in exploration phase)
if st.session_state.env.env.is_exploration_phase and st.session_state.action_buffer:
    st.info(" | ".join(st.session_state.action_buffer))

# Handle send
if send_clicked:
    # In evaluation phase, use built_action directly; in exploration phase, use action_buffer
    if not st.session_state.env.env.is_exploration_phase:
        # Evaluation phase: send the selected answer
        if built_action:
            full_action = built_action
        else:
            st.warning("Please select an answer before proceeding.")
            st.stop()
    else:
        # Exploration phase: send action buffer
        if st.session_state.action_buffer:
            full_action = " | ".join(st.session_state.action_buffer)
        else:
            st.warning("Action buffer is empty.")
            st.stop()

    st.session_state.turn += 1

    with st.chat_message("user"):
        st.markdown(full_action)

    obs, reward, done, info = st.session_state.env.step(full_action)
    rec = summarize_turn(
        t=st.session_state.turn,
        action=full_action,
        obs=obs,
        reward=reward,
        done=done,
        info=info,
    )
    st.session_state.history.append(rec)
    st.session_state.last_obs = obs

    with st.chat_message("assistant"):
        st.markdown(rec.obs_text)
        if rec.obs_raw and "multi_modal_data" in rec.obs_raw:
            imgs = rec.obs_raw["multi_modal_data"].get("<image>", [])
            if imgs:
                st.image(imgs, clamp=True, width=512, channels="RGB")

    # Reset buffer to default Observe() (only in exploration phase)
    if st.session_state.env.env.is_exploration_phase:
        st.session_state.action_buffer = ["Observe()"]

    # Set flag to trigger auto-scroll
    st.session_state.scroll_to_bottom = True

    # Save + feedback if done
    if done:
        user_id = st.session_state.get("user_id", "anon")
        episode_id = st.session_state.episode_id
        trajectory = [r.__dict__ for r in st.session_state.history]
        analytics = st.session_state.env.get_env_summary()
        correct_answers = st.session_state.env.get_eval_answers()
        out_path = save_episode(user_id, episode_id, trajectory, analytics, correct_answers)

        st.success(f"✅ Evaluation complete. Saved to {out_path}")    # Rerun to trigger scroll
    else:
        st.rerun()
