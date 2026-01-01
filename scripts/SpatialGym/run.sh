#!/bin/bash
#ablation
#512*512
tmux new-session -d -s "gemini-3-pro_vision_512" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  25 \
  --output-root results_arxiv_512/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type active \
  --render-mode vision \
  --proxy-agent scout \
  --cogmap \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_vision_512.log; bash"

#1024*1024
tmux new-session -d -s "gemini-3-pro_vision_1024" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  10 \
  --output-root results_arxiv_1024/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type active \
  --render-mode vision \
  --proxy-agent scout \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_vision_1024.log; bash"

#variance
tmux new-session -d -s "gpt-5.2_2room_1" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --max-exp-steps 15 \
  --output-root results_arxiv_2room_1/ \
  --data-dir data-2room/tos_dataset_1217_2room_25runs   \
  --exp-type active,passive \
  --render-mode text,vision \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_2room_1.log; bash"

tmux new-session -d -s "gpt-5.2_2room_2" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --max-exp-steps 15 \
  --output-root results_arxiv_2room_2/ \
  --data-dir data-2room/tos_dataset_1217_2room_25runs   \
  --exp-type active,passive \
  --render-mode text,vision \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_2room_2.log; bash"

tmux new-session -d -s "gpt-5.2_4room_1" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --max-exp-steps 25 \
  --output-root results_arxiv_4room_1/ \
  --data-dir data-4room/tos_dataset_1214_4room_25runs   \
  --exp-type active,passive \
  --render-mode text,vision \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_4room_1.log; bash"

tmux new-session -d -s "gpt-5.2_4room_2" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --max-exp-steps 25 \
  --output-root results_arxiv_4room_2/ \
  --data-dir data-4room/tos_dataset_1214_4room_25runs   \
  --exp-type active,passive \
  --render-mode text,vision \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_4room_2.log; bash"

tmux new-session -d -s "gemini-3-pro_4room_1" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview  \
  --num  10 \
  --max-exp-steps 25 \
  --output-root results_arxiv_4room_1/ \
  --data-dir data-4room/tos_dataset_1214_4room_25runs   \
  --exp-type active,passive \
  --render-mode text,vision \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_4room_1.log; bash"

tmux new-session -d -s "gemini-3-pro_4room_2" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview  \
  --num  10 \
  --max-exp-steps 25 \
  --output-root results_arxiv_4room_2/ \
  --data-dir data-4room/tos_dataset_1214_4room_25runs   \
  --exp-type active,passive \
  --render-mode text,vision \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_4room_2.log; bash"

#2room

tmux new-session -d -s "gpt-5.2_text_2room" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --max-exp-steps 15 \
  --output-root results_arxiv_2room/ \
  --data-dir data-2room/tos_dataset_1217_2room_25runs   \
  --exp-type active,passive \
  --render-mode text \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_text_2room.log; bash"
sleep 5s
tmux new-session -d -s "gemini-3-pro_text_2room" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  25 \
  --max-exp-steps 15 \
  --output-root results_arxiv_2room/ \
  --data-dir data-2room/tos_dataset_1217_2room_25runs   \
  --exp-type active,passive \
  --render-mode text \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_text_2room.log; bash"
sleep 5s
tmux new-session -d -s "gpt-5.2_vision_2room" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --max-exp-steps 15 \
  --output-root results_arxiv_2room/ \
  --data-dir data-2room/tos_dataset_1217_2room_25runs   \
  --exp-type passive \
  --render-mode vision \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_vision_2room.log; bash"
sleep 5s
tmux new-session -d -s "gemini-3-pro_vision_2room" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  25 \
  --max-exp-steps 15 \
  --output-root results_arxiv_2room/ \
  --data-dir data-2room/tos_dataset_1217_2room_25runs   \
  --exp-type passive \
  --render-mode vision \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_vision_2room.log; bash"

#4room
tmux new-session -d -s "gpt-5.2_text_4room" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --max-exp-steps 25 \
  --output-root results_arxiv_4room/ \
  --data-dir data-4room/tos_dataset_1214_4room_25runs   \
  --exp-type active,passive \
  --render-mode text \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_text_4room.log; bash"

tmux new-session -d -s "gemini-3-pro_text_4room" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  25 \
  --max-exp-steps 25 \
  --output-root results_arxiv_4room/ \
  --data-dir data-4room/tos_dataset_1214_4room_25runs   \
  --exp-type active,passive \
  --render-mode text \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_text_4room.log; bash"

tmux new-session -d -s "gpt-5.2_vision_4room" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --max-exp-steps 25 \
  --output-root results_arxiv_4room/ \
  --data-dir data-4room/tos_dataset_1214_4room_25runs   \
  --exp-type passive \
  --render-mode vision \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_vision_4room.log; bash"

tmux new-session -d -s "gemini-3-pro_vision_4room" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  25 \
  --max-exp-steps 25 \
  --output-root results_arxiv_4room/ \
  --data-dir data-4room/tos_dataset_1214_4room_25runs   \
  --exp-type passive \
  --render-mode vision \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_vision_4room.log; bash"

# 3room top1
tmux new-session -d -s "gpt-5.2_text_3room_top1" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --output-root results_arxiv_top1/ \
  --data-dir data-3room/tos_dataset_1224_3room_25runs_top1   \
  --exp-type active,passive \
  --render-mode text \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_text_3room_top1.log; bash"

tmux new-session -d -s "gpt-5.2_vision_3room_top1" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --output-root results_arxiv_top1/ \
  --data-dir data-3room/tos_dataset_1224_3room_25runs_top1   \
  --exp-type active,passive \
  --render-mode vision \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_vision_3room_top1.log; bash"

tmux new-session -d -s "gemini-3-pro_text_3room_top1" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  25 \
  --output-root results_arxiv_top1/ \
  --data-dir data-3room/tos_dataset_1224_3room_25runs_top1   \
  --exp-type active,passive \
  --render-mode text \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_text_3room_top1.log; bash"

tmux new-session -d -s "gemini-3-pro_vision_3room_top1" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  25 \
  --output-root results_arxiv_top1/ \
  --data-dir data-3room/tos_dataset_1224_3room_25runs_top1   \
  --exp-type active,passive \
  --render-mode vision \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_vision_3room_top1.log; bash"

# false belief
python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs   \
  --exp-type active \
  --render-mode text \
  --false-belief-exp \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_text.log

python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  25 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1217_3room_fbexp_25runs  \
  --exp-type active \
  --render-mode vision \
  --false-belief-exp \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_vision_fb.log

python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  25 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1217_3room_fbexp_25runs   \
  --exp-type active \
  --render-mode vision,text \
  --false-belief-exp \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_fb.log

# replay explore
tmux new-session -d -s "gpt-5.2_cogmap" "python scripts/SpatialGym/spatial_run.py \
  --phase explore \
  --model-name gpt-5.2 \
  --num  50 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type active \
  --render-mode vision,text \
  --replay \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_cogmap.log; bash"

tmux new-session -d -s "gemini-3-pro_cogmap" "python scripts/SpatialGym/spatial_run.py \
  --phase explore \
  --model-name gemini-3-pro-preview \
  --num  50 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type active \
  --render-mode vision,text \
  --replay \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_cogmap.log; bash"

# cogmap 
tmux new-session -d -s "gpt-5.2_cogmap" "python scripts/SpatialGym/spatial_run.py \
  --phase cogmap \
  --model-name gpt-5.2 \
  --num  50 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type active \
  --render-mode vision,text \
  --cogmap-override \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_cogmap.log; bash"

tmux new-session -d -s "gemini-3-pro_cogmap" "python scripts/SpatialGym/spatial_run.py \
  --phase cogmap \
  --model-name gemini-3-pro-preview \
  --num  50 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type active \
  --render-mode vision,text \
  --cogmap-override \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_cogmap.log; bash"

# gpt-5.2 vision
tmux new-session -d -s "gpt-5.2_vision" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type passive,active \
  --render-mode vision \
  --proxy-agent scout \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_vision.log; bash"

echo "Started task 1 in tmux session: gpt-5.2_vision "

# gpt-5.2 text
tmux new-session -d -s "gpt-5.2_text" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num  100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs   \
  --exp-type passive,active \
  --render-mode text \
  --inference-mode batch 2>&1 | tee logs/gpt-5.2_text.log; bash"

echo "Started task 2 in tmux session: gpt-5.2_text"

# gemini-3-pro vision
tmux new-session -d -s "gemini-3-pro_vision" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type passive,active \
  --render-mode vision \
  --proxy-agent scout \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_vision.log; bash"

echo "Started task 3 in tmux session: gemini-3-pro_vision"

# gemini-3-pro text
tmux new-session -d -s "gemini-3-pro_text" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gemini-3-pro-preview \
  --num  100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs   \
  --exp-type passive,active \
  --render-mode text \
  --inference-mode batch 2>&1 | tee logs/gemini-3-pro_text.log; bash"

echo "Started task 4 in tmux session: gemini-3-pro_text"

# claude-sonnet-4-5 vision
tmux new-session -d -s "claude-sonnet-4-5_vision" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name claude-sonnet-4-5 \
  --num  100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type passive,active \
  --render-mode vision \
  --proxy-agent scout \
  --inference-mode direct 2>&1 | tee logs/claude-sonnet-4-5_vision.log; bash"


# claude-sonnet-4-5 text
tmux new-session -d -s "claude-sonnet-4-5_text" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name claude-sonnet-4-5 \
  --num  100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs   \
  --exp-type passive,active \
  --render-mode text \
  --inference-mode direct 2>&1 | tee logs/claude-sonnet-4-5_text.log; bash"
  
# GLM-4.6V vision
tmux new-session -d -s "glm-4_6v_vision" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name glm-4.6v \
  --num 100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type passive,active \
  --render-mode vision \
  --proxy-agent scout \
  --inference-mode direct 2>&1 | tee logs/glm-4_6v_vision.log; bash"

echo "Started task 5 in tmux session: glm-4_6v_vision"

# GLM-4.6V text
tmux new-session -d -s "glm-4_6v_text" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name glm-4.6v \
  --num  100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs   \
  --exp-type passive,active \
  --render-mode text \
  --inference-mode direct 2>&1 | tee logs/glm-4_6v_text.log; bash"

echo "Started task 6 in tmux session: glm-4_6v_text"


# internvl3.5-241b-a28b text
tmux new-session -d -s "internvl3.5-241b-a28b_text" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name internvl3.5-241b-a28b \
  --num  100 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs   \
  --exp-type passive,active \
  --render-mode text \
  --inference-mode direct 2>&1 | tee logs/internvl3.5-241b-a28b_text.log; bash"

echo "Started task 10 in tmux session: internvl3.5-241b-a28b_text"

# qwen3-vl-235b-a22b-thinking vision  
tmux new-session -d -s "qwen3-vl-235b-a22b-thinking_vision" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name qwen3-vl-235b-a22b-thinking \
  --seed-range  50-99 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs  \
  --exp-type passive,active \
  --render-mode vision \
  --proxy-agent scout \
  --inference-mode direct 2>&1 | tee logs/qwen3-vl-235b-a22b-thinking_vision.log; bash"

echo "Started task 11 in tmux session: qwen3-vl-235b-a22b-thinking_vision"

# qwen3-vl-235b-a22b-thinking text
tmux new-session -d -s "qwen3-vl-235b-a22b-thinking_text" "python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name qwen3-vl-235b-a22b-thinking \
  --seed-range  50-99 \
  --output-root results_arxiv/ \
  --data-dir data-3room/tos_dataset_1214_3room_100runs   \
  --exp-type passive,active \
  --render-mode text \
  --inference-mode direct 2>&1 | tee logs/qwen3-vl-235b-a22b-thinking_text.log; bash"

echo "Started task 12 in tmux session: qwen3-vl-235b-a22b-thinking_text"
