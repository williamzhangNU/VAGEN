# RAGEN Integration Guide for VAGEN2

This guide provides comprehensive instructions for using the RAGEN spatial reasoning functionality integrated into VAGEN2.

## ✅ Integration Status

All major tasks have been completed:
- ✅ Analyzed VAGEN and RAGEN project structures
- ✅ Identified key differences between the two projects (prompts, roomutils, etc.)
- ✅ Created VAGEN2 copy as working directory
- ✅ Merged RAGEN spatial components into VAGEN2
- ✅ Implemented config switching functionality for text-based experiments
- ✅ Tested and verified the merged functionality works correctly

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate the environment
conda activate vagen

# Run the installation script
bash install_ragen_deps.sh
```

### 2. Run RAGEN Spatial Tasks

```bash
# Run single RAGEN task (ActiveRot)
python scripts/spatial_run_ragen.py --tasks ActiveRot --num 2 --model_name gpt-4.1-mini

# Run multiple RAGEN tasks
python scripts/spatial_run_ragen.py --tasks ActiveRot PassiveRot --num 5 --model_name gpt-4.1-mini

# Disable thinking process to simplify output
python scripts/spatial_run_ragen.py --tasks ActiveRot --num 3 --no_think
```

### 3. Use VAGEN Standard Workflow

```bash
# Use the provided example configuration
python -m vagen.env.create_dataset \
  --yaml_path vagen/env/spatial/ragen_config_example.yaml \
  --test_path data/ragen_test.parquet \
  --force_gen

# Run inference (requires corresponding model and inference configs)
python -m vagen.inference.run_inference \
  --inference_config_path your_inference_config.yaml \
  --model_config_path your_model_config.yaml \
  --val_files_path data/ragen_test.parquet
```

## 📁 File Structure Overview

```
VAGEN2/
├── vagen/env/spatial/
│   ├── ragen/                    # RAGEN spatial components
│   │   ├── Base/                 # RAGEN base components
│   │   ├── prompts/              # RAGEN prompt system
│   │   ├── config.py             # RAGEN configuration
│   │   └── env.py                # RAGEN spatial environment
│   ├── ragen_adapter.py          # Adapter (core integration component)
│   ├── ragen_config_example.yaml # Example configuration
│   ├── env.py                    # Modified VAGEN environment
│   └── env_config.py             # Support for text_based_mode
├── scripts/
│   └── spatial_run_ragen.py      # RAGEN run script
├── test_ragen_integration.py     # Integration test
└── install_ragen_deps.sh         # Dependency installation script
```

## 🔧 Key Features

### Dual Mode Support
- **Text mode** (`text_based_mode: true`): Uses RAGEN's pure text spatial reasoning
- **Visual mode** (`text_based_mode: false`): Uses VAGEN's multimodal processing

### Configuration Examples

**RAGEN Text-based mode:**
```yaml
env_config:
  text_based_mode: true     # Enable RAGEN
  exp_type: "active"        # or "passive"
  eval_tasks:
    - task_type: "rot"
      task_kwargs: {}
```

**VAGEN Visual mode:**
```yaml  
env_config:
  text_based_mode: false    # Use VAGEN original functionality
  exp_type: "passive"
  eval_tasks:
    - task_type: "rot"
      task_kwargs: {}
```

## 🎯 Supported Task Types

- `ActiveRot` - Active rotation task
- `PassiveRot` - Passive rotation task
- `ActiveDir` - Active direction task
- `PassiveDir` - Passive direction task
- `ActiveE2A` - Active E2A task

## 🔍 Verify Integration

Run the test to ensure everything works correctly:
```bash
python test_ragen_integration.py
```

Expected output should show all tests passing:
```
✓ All integration tests passed!
RAGEN spatial reasoning is successfully integrated into VAGEN2!
```

## 📊 Example Output

RAGEN's text-based mode produces observations like:
```
# Spatial Understanding Task

You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates.

Facing: forward/backward/right/left. When facing north: forward=north, back=south, right=east, left=west.

Observation: For visible objects you receive (direction, signed degree, distance).
- direction uses <vertical>-<horizontal> with front|back|same and left|right|same and
- degree is clockwise from your facing; distance is Euclidean
...
```

## 🚨 Important Notes

1. **Dependencies**: Ensure all RAGEN and VAGEN dependencies are installed
2. **Models**: Text-based mode uses different prompts, more suitable for pure text reasoning
3. **Performance**: Text mode is typically faster than visual mode
4. **Compatibility**: Both modes can be mixed in the same environment configuration

## 🛠️ Troubleshooting

If you encounter issues:

1. **Import errors**: Check Python path, ensure all modules can be found
2. **Configuration errors**: Verify YAML configuration format is correct
3. **Task failures**: Check detailed log output

## 🎉 Summary

You now have a powerful unified framework that can:
- Run RAGEN's text-based spatial reasoning within VAGEN
- Maintain VAGEN's original multimodal capabilities
- Switch between two modes through simple configuration
- Use unified API and toolchain

You can now easily conduct comparative experiments between text-based and visual-based spatial reasoning without maintaining two separate codebases!
