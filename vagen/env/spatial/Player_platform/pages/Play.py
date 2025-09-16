# pages/01_🎮_Play.py
import streamlit as st
from env_adapter import SpatialEnvAdapter, summarize_turn, format_obs, save_episode, make_cfg, load_cfg_from_yaml

st.set_page_config(page_title="Play", layout="wide", page_icon="🎮")

st.title("🎮 Exploration (Chat Mode)")
config_path = "vagen/env/spatial/Player_platform/config.yaml"
# ---- Step 2: Bootstrap env once ----
if "env" not in st.session_state or st.session_state.get("loaded_config_path") != config_path:
    cfg = load_cfg_from_yaml(config_path)
    st.session_state.env = SpatialEnvAdapter(cfg)
    st.session_state.loaded_config_path = config_path

    # Reset env and get initial observation
    obs, info = st.session_state.env.env.reset(seed=0)  # you can also pull a seed from cfg
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

# ---- Step 3: Render chat history ----
for rec in st.session_state.history:
    if rec.action == "(system)":
        with st.chat_message("assistant", avatar="🛠️"):
            st.markdown(rec.obs_text)
    else:
        with st.chat_message("user"):
            st.markdown(rec.action)
        with st.chat_message("assistant"):
            st.markdown(rec.obs_text)
            if rec.obs_raw and "multi_modal_data" in rec.obs_raw:
                imgs = rec.obs_raw["multi_modal_data"].get("<image>", [])
                if imgs:
                    st.image(imgs, clamp=True, use_container_width=True)
# ---- Step 4: Chat input ----
#objects = st.session_state.env.get_room_objects_str()
if prompt := st.chat_input("Type your action (e.g., Move(chair), Rotate(90), Observe())."):
    st.session_state.turn += 1
    with st.chat_message("user"):
        st.markdown(prompt)

    obs, reward, done, info = st.session_state.env.step(prompt)
    rec = summarize_turn(
        t=st.session_state.turn,
        action=prompt,
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


    # ---- Save + feedback if done ----
    if done:
        user_id = st.session_state.get("user_id", "anon")
        episode_id = st.session_state.episode_id
        trajectory = [r.__dict__ for r in st.session_state.history]
        analytics = st.session_state.env.get_env_summary()
        correct_answers = st.session_state.env.get_eval_answers()
        out_path = save_episode(user_id, episode_id, trajectory, analytics, correct_answers)

        if reward > 0:
            st.success(f"✅ Correct answer! Saved to {out_path}")
        else:
            st.error(f"❌ Incorrect answer. Saved to {out_path}")
