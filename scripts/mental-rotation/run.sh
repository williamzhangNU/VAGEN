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

# Create data directory
mkdir -p "data/$EXPERIMENT_NAME"

# Step 1: Create dataset (force regenerate to ensure clean config)
echo ""
echo "Step 1: Creating dataset from YAML config..."
python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet" \
    --force_gen
echo "✓ Dataset created successfully"

# Step 2: Check for existing server and start new one
echo ""
echo "Step 2: Starting VAGEN server..."

# Check if port 5000 is already in use and kill existing process
echo "Checking for existing server on port 5000..."
EXISTING_PID=$(lsof -ti:5000 || echo "")
if [ ! -z "$EXISTING_PID" ]; then
    echo "Found existing server with PID: $EXISTING_PID, stopping it..."
    kill $EXISTING_PID 2>/dev/null || true
    sleep 3
    # Force kill if still running
    if kill -0 $EXISTING_PID 2>/dev/null; then
        echo "Force killing server..."
        kill -9 $EXISTING_PID 2>/dev/null || true
        sleep 2
    fi
    echo "✓ Existing server stopped"
fi

echo "The server is configured to use devices [0, 1] for mental-rotation environments"
python -m vagen.server.server server.port=5000 use_state_reward=False &
SERVER_PID=$!
echo "✓ Server started with PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 10

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
