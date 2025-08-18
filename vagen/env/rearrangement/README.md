# Rearrangement Environment

This directory contains the implementation of a two-phase rearrangement task environment based on AI2-THOR, designed to test an agent's ability to observe, remember, and restore object arrangements.

## Overview

The rearrangement task consists of two phases:

1. **Walkthrough Phase**: The agent observes the target state of the environment and records the positions and states of movable objects.
2. **Unshuffle Phase**: Objects are shuffled, and the agent must restore them to their original positions based on the recorded memory.

## Key Features

- **Two-phase workflow**: Walkthrough → Unshuffle
- **Memory persistence**: Walkthrough observations are maintained throughout the task
- **AI2-THOR integration**: Uses AI2-THOR for realistic 3D environments
- **Multiple action types**: Navigation, manipulation, and object interaction
- **Flexible prompting**: Supports different reasoning formats (grounding, world modeling, etc.)

## File Structure

```
rearrangement/
├── __init__.py              # Package initialization
├── env.py                   # Main environment implementation
├── env_config.py            # Environment configuration
├── service.py               # Batch service for multiple environments
├── service_config.py        # Service configuration
├── prompt.py                # Prompt templates and system messages
├── utils.py                 # Utility functions for object handling
├── datasets/                # Task datasets
│   └── base.json           # Base dataset with rearrangement tasks
└── README.md               # This file
```

## Usage

### Basic Environment Usage

```python
from vagen.env.rearrangement import RearrangementEnv, RearrangementEnvConfig

# Create configuration
config = RearrangementEnvConfig(
    eval_set='base',
    resolution=256,
    prompt_format='grounding_worldmodeling'
)

# Create environment
env = RearrangementEnv(config)

# Reset to start a task
initial_state = env.reset(task_id=0)
print(f"Phase: {initial_state['phase']}")  # Should be 'walkthrough'
print(f"Instruction: {initial_state['instruction']}")

# Execute actions during walkthrough
result = env.step("moveahead, rotateright, lookup")

# Complete walkthrough phase
walkthrough_result = env.step("done")
print(f"New phase: {walkthrough_result['phase']}")  # Should be 'unshuffle'

# Execute actions during unshuffle
result = env.step("pickup, moveright, putdown")

# Complete unshuffle phase
final_result = env.step("done")
```

### Service Usage (Batch Processing)

```python
from vagen.env.rearrangement import RearrangementService, RearrangementServiceConfig

# Create service
service_config = RearrangementServiceConfig(max_workers=4, devices=[0, 1])
service = RearrangementService(service_config)

# Create multiple environments
env_configs = {
    'env_1': {
        'env_name': 'rearrangement',
        'env_config': {'eval_set': 'base', 'resolution': 256}
    },
    'env_2': {
        'env_name': 'rearrangement', 
        'env_config': {'eval_set': 'base', 'resolution': 256}
    }
}

service.create_environments_batch(env_configs)

# Reset environments
reset_results = service.reset_batch({'env_1': 0, 'env_2': 1})
# reset_results format: {'env_1': (observation, info), 'env_2': (observation, info)}

# Execute actions in parallel
step_results = service.step_batch({'env_1': 'moveahead', 'env_2': 'rotateright'})
# step_results format: {'env_1': (observation, reward, done, info), 'env_2': (observation, reward, done, info)}
```

## Available Actions

### Navigation Actions
- `moveahead`: Move forward by 0.5 meter
- `moveback`: Move backward by 0.5 meter
- `moveright`: Move rightward by 0.5 meter
- `moveleft`: Move leftward by 0.5 meter
- `rotateright`: Rotate right by 90 degrees
- `rotateleft`: Rotate left by 90 degrees
- `lookup`: Tilt camera upward by 30 degrees
- `lookdown`: Tilt camera downward by 30 degrees

### Manipulation Actions
- `pickup`: Pick up the nearest pickupable object
- `putdown`: Put down the currently held object
- `open`: Open the nearest openable object
- `close`: Close the nearest closeable object

### Phase Control Actions
- `done`: Complete the current phase (agent decides when task is finished)

## Configuration Options

### RearrangementEnvConfig

- `eval_set`: Dataset to use ('base')
- `resolution`: Image resolution (default: 255)
- `fov`: Field of view in degrees (default: 100)
- `prompt_format`: Reasoning format ('grounding_worldmodeling', 'free_think', etc.)
- `success_threshold`: Distance threshold for success (default: 0.5)
- `step_length`: Movement step size (default: 0.5)
- `max_actions_per_step`: Maximum actions per step (default: 5)

## Dataset Format

The dataset contains tasks with the following structure:

```json
{
  "tasks": [
    {
      "targetObjectType": "Box",
      "targetObjectIds": "Box_09bf8add",
      "target_position": {"x": -1.68, "y": 0.72, "z": 4.05},
      "agentPose": {
        "position": {"x": -2.5, "y": 0.90, "z": 0.5},
        "rotation": 90,
        "horizon": 0.0
      },
      "scene": "FloorPlan204",
      "instruction": "Rearrange objects to match the target state",
      "rearrangement_data": {
        "starting_poses": [
          {
            "name": "Box_09bf8add",
            "objectName": "Box_09bf8add", 
            "position": {"x": -1.68, "y": 0.72, "z": 4.05},
            "rotation": {"x": 0, "y": 0, "z": 0}
          }
        ]
      }
    }
  ]
}
```

## Memory Format

During the walkthrough phase, agents should record object states in JSON format:

```json
[
  {
    "name": "Box_09bf8add",
    "type": "Box",
    "position": {"x": -1.68, "y": 0.72, "z": 4.05},
    "rotation": {"y": 0.0},
    "openness": null
  },
  {
    "name": "Laptop_dfbbea5a", 
    "type": "Laptop",
    "position": {"x": -1.09, "y": 0.74, "z": 1.47},
    "rotation": {"y": 0.0},
    "openness": null
  }
]
```

## Success Metrics

- **Position Accuracy**: Objects within `success_threshold` distance of target positions
- **Overall Success Rate**: Percentage of objects correctly positioned
- **Phase Completion**: Successful transition from walkthrough to unshuffle phase

## Integration with AI2-THOR Rearrangement

This implementation is designed to work with the AI2-THOR rearrangement dataset and follows similar conventions to the original rearrangement challenge, while adapting the interface to work with the VAGEN framework.

## Testing

Run the test script to verify the implementation:

```bash
python test_rearrangement.py
```

This will test basic environment functionality, service operations, and phase transitions.
