#!/bin/bash

# Script to run mental-rotation service benchmark
# This tests the MentalRotationService integration with the VAGEN framework

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=== Mental Rotation Service Benchmark ==="
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Create data directory if it doesn't exist
mkdir -p data/mental-rotation-vision-benchmark

# Step 1: Create dataset if it doesn't exist
echo ""
echo "Step 1: Creating dataset from YAML config..."
if [ ! -f "data/mental-rotation-vision-benchmark/train.parquet" ] || [ ! -f "data/mental-rotation-vision-benchmark/test.parquet" ]; then
    python -m vagen.env.create_dataset \
        --yaml_path "$SCRIPT_DIR/dataset_config.yaml" \
        --train_path "data/mental-rotation-vision-benchmark/train.parquet" \
        --test_path "data/mental-rotation-vision-benchmark/test.parquet" \
        --seed 42
    echo "✓ Dataset created successfully"
else
    echo "✓ Dataset already exists, skipping creation"
fi

# Step 2: Start the server in background
echo ""
echo "Step 2: Starting VAGEN server..."
python -m vagen.server.server server.port=5000 use_state_reward=False &
SERVER_PID=$!
echo "✓ Server started with PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 10

# Step 3: Run the service benchmark
echo ""
echo "Step 3: Running service benchmark..."
python -m vagen.env.verify_service \
    --config "$SCRIPT_DIR/benchmark_config.yaml"

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
echo "=== Benchmark completed successfully! ==="
echo "Results saved in: $SCRIPT_DIR/benchmark_results/"
