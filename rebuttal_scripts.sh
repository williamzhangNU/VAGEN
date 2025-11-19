# real-value relations
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/real-value \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --relation-mode real 2>&1 | tee logs/real-value.log

# another bin system relations
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/bin-2 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --relation-mode bin_system2 2>&1 | tee logs/bin2

# another bin system relations
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name claude-4-sonnet \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/bin-2 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --relation-mode bin_system2 2>&1 | tee logs/bin2-claude-4-sonnet

# another bin system relations
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gemini-2_5-pro \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/bin-2 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --relation-mode bin_system2 2>&1 | tee logs/bin2-gemini-2_5-pro

# query cost
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/query-cost-0.5 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --query-cost 0.5 2>&1 | tee logs/query-cost-0.5






# Text: max steps 3 and 5
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/max-steps-3 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --max-exp-steps 3 2>&1 | tee logs/text-max-steps-3

python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/max-steps-5 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --max-exp-steps 5 2>&1 | tee logs/text-max-steps-5

# Vision: max steps 3 and 5
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/max-steps-3 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --max-exp-steps 3 2>&1 | tee logs/vision-max-steps-3

python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/max-steps-5 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data \
    --max-exp-steps 5 2>&1 | tee logs/vision-max-steps-5





# Random light
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/random-light \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_2_room_jitter/ 2>&1 | tee logs/random-light

# given gt cogmap, test performance
python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassiveRot,PassiveRotDual,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/vision-gt-cogmap \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data/ \
    --gt-cogmap-eval 2>&1 | tee logs/vision-gt-cogmap

# given gt local cogmap
## test evaluation performance
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --cogmap \
    --output-root results_rebuttal/vision-gt-cogmap \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data/ \
    --gt-local-cogmap 2>&1 | tee logs/vision-gt-local-cogmap

# cognitive map then answer
# cp -r /home/pingyue/work/VAGEN/results_submit/gpt-5/run{00..19} results_rebuttal/vision-cogmap-before-eval # reuse the same results
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/vision-cogmap-before-eval \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data/ \
    --cogmap-before-eval 2>&1 | tee logs/vision-cogmap-before-eval


# 3-room
## text
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/3-room \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_3_room 2>&1 | tee logs/text-3-room-active.log

python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassiveRot,PassiveRotDual,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/3-room \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_3_room 2>&1 | tee logs/text-3-room-passive.log

## vision
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/3-room \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_3_room 2>&1 | tee logs/vision-3-room-active.log

python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassiveRot,PassiveRotDual,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/3-room \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_3_room 2>&1 | tee logs/vision-3-room-passive.log


# 4-room
## text
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/4-room \
    --max-exp-steps 25 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_4_room 2>&1 | tee logs/text-4-room-active.log

python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassiveRot,PassiveRotDual,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/4-room \
    --max-exp-steps 25 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_4_room 2>&1 | tee logs/text-4-room-passive.log

## vision
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/4-room \
    --max-exp-steps 25 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_4_room 2>&1 | tee logs/vision-4-room-active.log

python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassiveRot,PassiveRotDual,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/4-room \
    --max-exp-steps 25 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_4_room 2>&1 | tee logs/vision-4-room-passive.log


# 4-room-loop
## text
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/4-room-loop \
    --max-exp-steps 25 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_4_room_loop 2>&1 | tee logs/text-4-room-loop-active.log

python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassiveRot,PassiveRotDual,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode text \
    --inference-only  \
    --output-root results_rebuttal/4-room-loop \
    --max-exp-steps 25 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_4_room_loop 2>&1 | tee logs/text-4-room-loop-passive.log

## vision
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/4-room-loop \
    --max-exp-steps 25 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_4_room_loop 2>&1 | tee logs/vision-4-room-loop-active.log

python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassiveRot,PassiveRotDual,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 0-19 \
    --render-mode vision \
    --inference-only  \
    --output-root results_rebuttal/4-room-loop \
    --max-exp-steps 25 \
    --data-dir /home/pingyue/work/VAGEN/vagen/env/spatial/room_data_4_room_loop 2>&1 | tee logs/vision-4-room-loop-passive.log