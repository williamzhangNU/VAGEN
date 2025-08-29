#!/bin/bash

# Script to run mental-rotation inference
# This creates dataset, starts server, runs inference, and cleans up

set -e

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Extract experiment name from the path
# This will take the last part of the path: mental-rotation
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1 | rev)

echo "=== Mental Rotation Inference ==="
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Experiment name: $EXPERIMENT_NAME"

cd "$PROJECT_ROOT"



# Step 3: Run inference
echo ""
echo "Step 3: Running inference..."
python -m vagen.inference.run_inference \
    --inference_config_path="$SCRIPT_DIR/inference_config.yaml" \
    --model_config_path="$SCRIPT_DIR/model_config.yaml" \
    --val_files_path="data/$EXPERIMENT_NAME/test.parquet" \
    --wandb_path_name="mental_rotation"

# Step 4: Cleanup
echo ""
echo "Step 4: Cleaning up..."
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Stopping server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
    echo "✓ Server stopped"
fi

echo ""
echo "=== Inference completed successfully! ==="
echo "Results saved in: results/$EXPERIMENT_NAME/"
