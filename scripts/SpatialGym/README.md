# SpatialGym Runner

A modular execution script for running SpatialGym experiments with separated phases: exploration, evaluation, and cognitive mapping.

## Overview

`spatial_run.py` is a comprehensive automation tool that orchestrates spatial reasoning experiments through distinct phases:
- **Exploration**: Generate datasets and run initial spatial exploration
- **Evaluation**: Build evaluation messages and run inference on exploration results
- **Cognitive Map (CogMap)**: Generate and evaluate cognitive map representations
- **Aggregation**: Aggregate logs and images from all phases
- **Re-evaluation**: Re-run evaluation or cogmap on existing exploration data

The script supports multiple experiment types (active/passive), render modes (vision/text), and flexible seed ranges.

## Usage

```bash
python spatial_run.py [OPTIONS]
```

### Basic Examples

```bash
# Run all phases (exploration + evaluation + aggregation)
python spatial_run.py (--phase all)

# Run only exploration phase
python spatial_run.py --phase explore --num 10

# Run only evaluation phase on existing data
python spatial_run.py --phase eval --seed-range 0-9

# Run with specific model and multiple experiment types
python spatial_run.py --phase all --model-name gpt-5.2 --exp-type active,passive --num 5

# Run with text and vision render modes
python spatial_run.py --phase explore --render-mode vision,text

# Run cognitive mapping for active experiments
python spatial_run.py --phase cogmap --exp-type active

# Re-evaluate existing results
python spatial_run.py --phase reeval --seed-range 0-24

# Run aggregation only
python spatial_run.py --phase aggregate
```

### Commands to reproduce the results
```bash
# passive
python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --exp-type passive \
  --num 5 \
  --output-root results_arxiv/ \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --inference-mode batch \
  --render-mode vision \
  --proxy-agent scout 2>&1 | tee logs/passive_vision_gpt-5.2.log

# active vision (exploration only)
python scripts/SpatialGym/spatial_run.py \
  --phase explore \
  --model-name gpt-5.2 \
  --exp-type active \
  --num 5 \
  --output-root results_arxiv/ \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --inference-mode direct \
  --render-mode vision 2>&1 | tee logs/active_vision_explore_gpt-5.2.log

# eval (after exploration)
python scripts/SpatialGym/spatial_run.py \
  --phase eval \
  --model-name gpt-5.2 \
  --exp-type active \
  --num 5 \
  --output-root results_arxiv/ \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --inference-mode batch \
  --render-mode vision 2>&1 | tee logs/active_vision_eval_gpt-5.2.log

# cogmap (after exploration)
python scripts/SpatialGym/spatial_run.py \
  --phase cogmap \
  --model-name gpt-5.2 \
  --exp-type active \
  --num 5 \
  --output-root results_arxiv/ \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --inference-mode batch \
  --render-mode vision 2>&1 | tee logs/cogmap_gpt-5.2.log

# false-belief-exp (after exploration, text only)
python scripts/SpatialGym/spatial_run.py \
  --phase explore \
  --model-name gpt-5.2 \
  --exp-type active \
  --num 5 \
  --data-dir vagen/env/spatial/room_data_3_room/  \
  --output-root results_arxiv/ \
  --data-dir vagen/env/spatial/room_data_3_room/ \
  --render-mode text \
  --false-belief-exp 2>&1 | tee logs/fb-exp_gpt-5.2.log
```


## Command Line Arguments

### Phase Selection
- `--phase`: Which phase to run (default: `all`)
  - `explore`: Dataset creation and exploration inference
  - `eval`: Evaluation inference on exploration results
  - `cogmap`: Cognitive map generation and evaluation
  - `reeval`: Re-run evaluation on existing exploration data
  - `cogmap_reeval`: Re-run cognitive map evaluation on existing data
  - `aggregate`: Aggregate logs and images
  - `all`: Run exploration + evaluation + aggregation (+ cogmap if `--cogmap` is set)

### Core Options
- `--exp-type`: Experiment type (default: `active`)
  - Single value: `active` or `passive`
  - Multiple values: `active,passive` (comma-separated)
- `--model-name`: Model identifier (default: `gpt-4o-mini`)
- `--render-mode`: Environment render mode (default: `vision`)
  - Single value: `vision` or `text`
  - Multiple values: `vision,text` (comma-separated)
- `--num`: Number of samples per task (default: `1`)
- `--seed-range`: Seed range in format `start-end` (e.g., `0-24`)
  - If not specified, uses `0` to `num-1`
- `--data-dir`: Data directory root (default: `data`)
- `--output-root`: Root directory for output (default: `results`)

### Thinking and Agent Options
- `--enable-think`: Enable/disable thinking mode (default: `1`)
  - `1`: Enable thinking
  - `0`: Disable thinking
- `--proxy-agent`: Proxy agent for passive tasks (default: `scout`)
  - Choices: `scout`, `strategist`, `oracle`
  - Required when `--exp-type` includes `passive`

### Override Options
- `--all-override`: Override all history (delete entire sample path)
- `--eval-override`: Override evaluation history only
- `--cogmap-override`: Override cognitive map cache only
- `--cogmap`: Enable cognitive map phase in `all` mode

### Evaluation Options
- `--eval-task-counts`: JSON string specifying evaluation task counts
  - Example: `'{"dir": 1, "loc": 2}'`
  - If omitted, uses `eval_task_counts` from `inference_config.yaml`

### Inference Options
- `--inference-mode`: Inference execution mode (default: `direct`)
  - `direct`: Direct API calls
  - `batch`: OpenAI batch API

### Server Options
- `--no-server`: Don't start internal environment server (assume external server running)
- `--server-host`: Server host (default: `127.0.0.1`)
- `--server-port`: Server port (default: `5000`)
  - Automatically finds available port if specified port is in use

### Configuration Files
- `--base-env`: Path to base environment config (default: `base_env_config.yaml`)
- `--base-infer`: Path to inference config (default: `inference_config.yaml`)
- `--base-model`: Path to base model config (default: `base_model_config.yaml`)

## Workflow

### Phase: Exploration
1. Load base configurations and room config
2. Start environment server (if not disabled)
3. For each combination of `exp_type` and `render_mode`:
   - Generate temporary YAML configurations
   - Create dataset using `vagen.env.create_dataset`
   - Run exploration inference
4. Stop environment server

### Phase: Evaluation
1. Compute combo directory paths from previous exploration
2. Load evaluation task counts
3. Build evaluation messages for all combo directories
4. Run evaluation inference

### Phase: Cognitive Map
1. Compute combo directory paths from previous exploration
2. Build cognitive map messages for all combo directories
3. Run cognitive map inference

### Phase: Aggregation
1. Aggregate all logs and images using `SpatialEnvLogger`
2. Generate consolidated results

## Configuration Structure

### Environment Config (`base_env_config.yaml`)
Must contain `room_config` for spatial environment setup:
```yaml
room_config:
  n_objects: 9
  room_num: 1
  topology: "single"
  room_size: [10, 10]
```

### Inference Config (`inference_config.yaml`)
Must contain inference parameters and optional `eval_task_counts`:
```yaml
output_dir: "results"
eval_task_counts:
  dir: 1
  loc: 2
```

### Model Config (`base_model_config.yaml`)
Must contain a `models` section:
```yaml
models:
  gpt-4o-mini:
    model_name: "gpt-4o-mini"
    # ... other model parameters
```

## Output Structure

Results are organized as:
```
{output_root}/
  {model_name}/
    {room_hash}/
      {render_mode}/
        {exp_type}/
          {think|nothink}/
            [proxy_agent]/  # Only for passive exp_type
              config.json
              exploration.json
              evaluation.json
              iamges/
```
