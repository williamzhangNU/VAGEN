# SpatialGym Scripts

### Examples

```bash
python scripts/SpatialGym/spatial_run.py --tasks ActiveRot,ActiveDir --model_name gpt-5-mini --seed-range 0-4 --server-port 5000 --render-mode text
```
if you want to run multiple spatial_run.py, you need to switch server-port for each run.
- **NOTE** seed-range: corresponds to run: seed 0 corresponds to run00

```bash
python scripts/SpatialGym/spatial_run.py --tasks ActiveRot --model_name gpt-5-mini --seed-range 0-4 --server-port 5000 --render-mode text --cogmap
```
- cogmap: to run cogmap extraction and evaluation, only need to run cogmap on one active task.

```bash
python scripts/SpatialGym/spatial_run.py --tasks PassiveRot,PassiveDir --model_name gpt-5-mini --seed-range 0-4 --server-port 5000 --render-mode text --proxy-agent strategist
```
- proxy-agent: to run passive tasks with different proxy agents, use strategist for text, scout for vision.
- no cogmap for passive tasks.


### Settings
- in `base_model_config.yaml`, you can change model settings like temperature, max_workers (parallel workers), for openai, anthropic, together ai, you can increase max_workers to speed up the inference.

### Models:
- gemini-2.5-pro
- gemini-2.5-flash
- gpt-5-mini
- gpt-5
- claude-4-sonnet
- gpt-oss-20b (text-only)
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


```bash
python spatial_run.py --tasks ActiveRot PassiveRot PassiveLoc
```

```bash
python spatial_run.py --tasks "ActiveRot,PassiveRot"
```