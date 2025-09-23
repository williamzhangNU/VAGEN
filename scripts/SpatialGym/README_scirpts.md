# SpatialGym Scripts


### Prepare the environment
add api into .bashrc:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
- SELF_HOSTED_API_KEY
- TOGETHER_API_KEY



### Examples

Please use ***tmux*** on a server to run the spatial_run.py.

```bash
mkdir logs
python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveRot,PassiveRotDual,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 75-99 \
    --render-mode text \
    --inference-only \
    --proxy-agent strategist 2>&1 | tee logs/gpt-5-passive-text.log

python scripts/SpatialGym/spatial_run.py \
    --tasks PassiveDir,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveRot,PassiveRotDual,PassiveFwdLoc,PassiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 75-99 \
    --render-mode vision \
    --inference-only \
    --proxy-agent scout 2>&1 | tee logs/gpt-5-passive-vision.log
```
if you want to run multiple spatial_run.py, you need to switch server-port for each run.
- **NOTE** seed-range: corresponds to run: seed 0 corresponds to run00, change it to your share
- **NOTE** for passive or active, different evaluation tasks can NOT be run in parallel (if same model, same render, same passive/active). E.g., no two parallel spatial run of PassiveRot and PassiveE2A for gpt-5, text.
- **NOTE** strategist for text, scout for vision.
- Run different models, different render mode (can be in parallel)

```bash
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActivePov,ActiveBwdPov,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 75-99 \
    --inference-only \
    --render-mode text 2>&1 | tee logs/gpt-5-active-text.log

python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir,ActivePov,ActiveBwdPov,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc \
    --model_name gpt-5 \
    --seed-range 75-99 \
    --inference-only \
    --render-mode vision 2>&1 | tee logs/gpt-5-active-vision.log
```


```bash
python scripts/SpatialGym/spatial_run.py \
    --tasks ActiveDir \
    --model_name gpt-5 \
    --seed-range 75-99 \
    --render-mode text \
    --inference-only \
    --cogmap 2>&1 | tee logs/gpt-5-active-text.log
```
- cogmap: to run cogmap extraction and evaluation, only need to run cogmap on one active task.




### Settings
- in `base_model_config.yaml`, you can change model settings like temperature, max_workers (parallel workers), for openai, anthropic, together ai, you can increase max_workers to speed up the inference.

### Models:
- gemini-2.5-pro
- gemini-2.5-flash
- gpt-5
- claude-4-sonnet
- gpt-oss-120b (text-only)
- internvl3.5-241b-a28b
- GLM-4.5V


### Available Tasks

**Active Tasks:**
- ActiveDir
- ActivePov
- ActiveBwdPov
- ActiveFwdFov
- ActiveBwdNav
- ActiveE2A
- ActiveRot
- ActiveRotDual
- ActiveFwdLoc
- ActiveBwdLoc
- ActiveFalseBelief (exclude)
- ActiveDirAnchor (exclude)

**Passive Tasks:**
- PassiveDir
- PassivePov
- PassiveBwdPov
- PassiveFwdFov
- PassiveBwdNav
- PassiveE2A
- PassiveRot
- PassiveRotDual
- PassiveFwdLoc
- PassiveBwdLoc
- PassiveFalseBelief (exclude)
- PassiveDirAnchor (exclude)
