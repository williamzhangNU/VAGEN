# pages/01_🎮_Play.py
import streamlit as st
from env_adapter import SpatialEnvAdapter, summarize_turn, format_obs, save_episode, load_cfg_from_yaml

st.set_page_config(page_title="Play", layout="wide", page_icon="🎮")

st.title("🎮 Exploration (Chat Mode)")
config_path = "vagen/env/spatial/Player_platform/config.yaml"
# ---- Bootstrap env once ----
if "env" not in st.session_state or st.session_state.get("loaded_config_path") != config_path:
    cfg = load_cfg_from_yaml(config_path)
    st.session_state.env = SpatialEnvAdapter(cfg)
    st.session_state.loaded_config_path = config_path

    # Reset env and get initial observation
    obs, info = st.session_state.env.env.reset(seed=75)  # you can also pull a seed from cfg
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

# ---- Render chat history ----
for rec in st.session_state.history:
    if rec.action == "(system)":
        with st.chat_message("assistant", avatar="🛠️"):
            st.markdown(rec.obs_text)
            if rec.obs_raw and "multi_modal_data" in rec.obs_raw:
                imgs = rec.obs_raw["multi_modal_data"].get("<image>", [])
                if imgs:
                    st.image(imgs, clamp=True, use_container_width=True)
    else:
        with st.chat_message("user"):
            st.markdown(rec.action)
        with st.chat_message("assistant"):
            st.markdown(rec.obs_text)
            if rec.obs_raw and "multi_modal_data" in rec.obs_raw:
                imgs = rec.obs_raw["multi_modal_data"].get("<image>", [])
                if imgs:
                    st.image(imgs, clamp=True, use_container_width=True)
# ---- Chat input ----
if "action_buffer" not in st.session_state:
    st.session_state.action_buffer = []

st.subheader("Action Builder")

action_type = st.selectbox(
    "Action type",
    ["-- select --", "JumpTo", "Rotate", "Observe", "Query", "Term", "Answer (MCQ)"],
    key="action_type"
)

built_action = None
if action_type == "JumpTo":
    room_objects = st.session_state.env.env.initial_room.all_objects
        
    obj_names = [o.name for o in room_objects]
    target = st.selectbox("Jump to object", obj_names, key="jumpto_target")
    if target:
        built_action = f"JumpTo({target})"

elif action_type == "Rotate":
    deg = st.selectbox("Degrees", [-270, -180, -90, 0, 90, 180, 270], key="rotate_deg")
    built_action = f"Rotate({deg})"

elif action_type == "Observe":
    built_action = "Observe()"

elif action_type == "Query":
    room_objects = st.session_state.env.env.initial_room.all_objects
    obj_names = [o.name for o in room_objects]
    target = st.selectbox("Query which object?", obj_names, key="query_target")
    if target:
        built_action = f"Query({target})"

elif action_type == "Term":
    built_action = "Term()"

elif action_type == "Answer (MCQ)":
    # Only show if in evaluation phase
    if not st.session_state.env.env.is_exploration_phase:
        # Suppose env provides current question + options in the last obs string
        obs_text = st.session_state.last_obs.get("obs_str", "")
        choices = ["A", "B", "C", "D"]
        answer = st.radio("Select your answer:", choices, key="mcq_answer")
        if answer:
            built_action = answer
    else:
        st.info("MCQ answers are only available during evaluation phase.")

# --- Render buttons ---
add_clicked = st.button("➕ Add to Action Sequence", key="add_to_buffer")
send_clicked = st.button("🚀 Send Action Sequence", key="send_buffer")
clear_clicked = st.button("🗑️ Clear Action Sequence", key="clear_buffer")

# Handle add
if add_clicked and built_action:
    st.session_state.action_buffer.append(built_action)

# Handle clear
if clear_clicked:
    st.session_state.action_buffer = []

# Show buffer
if st.session_state.action_buffer:
    st.info(" | ".join(st.session_state.action_buffer))

# Handle send
if send_clicked and st.session_state.action_buffer:
    full_action = " | ".join(st.session_state.action_buffer)
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
                st.image(imgs, clamp=True, use_container_width=True)

    # Clear buffer
    st.session_state.action_buffer = []

    # Save + feedback if done
    if done:
        user_id = st.session_state.get("user_id", "anon")
        episode_id = st.session_state.episode_id
        trajectory = [r.__dict__ for r in st.session_state.history]
        analytics = st.session_state.env.get_env_summary()
        correct_answers = st.session_state.env.get_eval_answers()
        out_path = save_episode(user_id, episode_id, trajectory, analytics, correct_answers)

        st.success(f"✅ Evaluation complete. Saved to {out_path}")
