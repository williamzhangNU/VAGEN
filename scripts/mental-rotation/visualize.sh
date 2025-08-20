#!/bin/bash

# Visualize debug results from mental-rotation inference
# Usage: ./visualize.sh [debug_dir] [output_dir]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default parameters
DEBUG_DIR="${1:-debug_results}"
OUTPUT_DIR="${2:-debug_html}"

echo "Visualizing debug results..."
echo "Debug directory: $DEBUG_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the visualization script
python3 "$SCRIPT_DIR/visualize_debug.py" \
    --debug_dir "$DEBUG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_file "mental_rotation_debug.html"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Visualization complete!"
    echo "✓ Open $OUTPUT_DIR/mental_rotation_debug.html in your browser"
else
    echo "✗ Visualization failed!"
fi
