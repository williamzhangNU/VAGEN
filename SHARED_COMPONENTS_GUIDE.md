# Shared Components Architecture in VAGEN2

## Overview

VAGEN2 now uses a unified shared component architecture where RAGEN and VAGEN spatial functionality share the same base components, eliminating duplication and ensuring consistency.

## Architecture Changes

### Before (Duplicated Components)
```
vagen/env/spatial/
├── Base/tos_base/          # VAGEN's tos_base
└── ragen/
    └── Base/tos_base/      # RAGEN's duplicate tos_base (REMOVED)
```

### After (Shared Components)
```
vagen/env/spatial/
├── Base/tos_base/          # Shared tos_base for both VAGEN and RAGEN
└── ragen/
    ├── config.py           # Uses ../Base/tos_base
    ├── env.py              # Uses ../Base/tos_base  
    ├── prompts/            # Uses ../../Base/tos_base
    └── vision_alignment.py # Uses ../Base/tos_base
```

## Key Benefits

1. **No Duplication**: Single source of truth for `tos_base` components
2. **Consistency**: Both VAGEN and RAGEN use identical base functionality
3. **Easier Maintenance**: Updates to core components benefit both systems
4. **Reduced Storage**: Eliminates ~60 duplicate files

## Import Path Changes

All RAGEN components now use relative imports to access the shared `tos_base`:

### RAGEN Environment (`ragen/env.py`)
```python
# Changed from: from .Base.tos_base import ...
from ..Base.tos_base import (
    EvaluationManager,
    ActionSequence,
    ExplorationManager,
    # ... other imports
)
```

### RAGEN Configuration (`ragen/config.py`)
```python
# Changed from: from .Base.tos_base import ...
from ..Base.tos_base import CANDIDATE_OBJECTS
from ..Base.tos_base.evaluation.task_types import EvalTaskType
```

### RAGEN Prompter (`ragen/prompts/prompter.py`)
```python
# Changed from: from ..Base.tos_base import ...
from ...Base.tos_base import ActionSequence, EvaluationManager
from ...Base.tos_base import Room, Agent
```

### RAGEN Vision Alignment (`ragen/vision_alignment.py`)
```python
# Changed from: from .Base.tos_base import ...
from ..Base.tos_base.core.room import Room
from ..Base.tos_base.core.object import Object, Agent, Gate
```

## Shared Components Location

The shared `tos_base` is located at:
```
/Users/songshe/ToS/VAGEN2/vagen/env/spatial/Base/tos_base/
```

This contains all core functionality used by both VAGEN and RAGEN:
- `actions/` - Action definitions and handlers
- `core/` - Core data structures (Room, Agent, Object, etc.)
- `evaluation/` - Evaluation tasks and metrics
- `managers/` - Exploration, evaluation, and history managers
- `utils/` - Utility functions and helpers

## Verification

The integration has been tested and confirmed working:
```bash
python test_ragen_integration.py
# ✓ All integration tests passed!
```

Both text-based (RAGEN) and visual (VAGEN) modes work correctly with the shared architecture.

## Development Guidelines

When modifying core spatial functionality:

1. **Shared Components**: Edit files in `Base/tos_base/` 
2. **VAGEN-Specific**: Edit files in `spatial/` (excluding `ragen/`)
3. **RAGEN-Specific**: Edit files in `spatial/ragen/`
4. **Test Both Modes**: Always run integration tests after changes

This ensures changes benefit both systems while maintaining their distinct capabilities.
