#!/bin/bash

# RAGEN Integration Dependencies Installation Script
# Run this script after activating the vagen environment

set -e

echo "🚀 Starting RAGEN integration dependency installation..."
echo "Current Python version: $(python --version)"
echo "Current environment: $CONDA_DEFAULT_ENV"

# Check if we're in the correct environment
if [[ "$CONDA_DEFAULT_ENV" != "vagen" ]]; then
    echo "⚠️  Warning: You are not currently in the vagen environment"
    echo "Please run first: conda activate vagen"
    exit 1
fi

echo "📦 Installing RAGEN-specific dependencies..."

# Install missing dependencies
pip install IPython
pip install codetiming  
pip install pyarrow>=15.0.0
pip install pylatexenc
pip install torchdata
pip install debugpy

# API related dependencies
echo "🔌 Installing API dependencies..."
pip install together
pip install anthropic
pip install openai

# Ensure core dependency versions are correct
echo "🔧 Ensuring core dependency versions..."
pip install "hydra-core>=1.3.0"
pip install "wandb"

# Install vagen2 package
echo "📦 Installing VAGEN2 package..."
cd /Users/songshe/ToS/VAGEN2
pip install -e .

echo "✅ Dependencies installation completed!"

# Run tests
echo "🧪 Running integration tests..."
python test_ragen_integration.py

echo ""
echo "🎉 RAGEN integration installation completed!"
echo ""
echo "Now you can run:"
echo "  python scripts/spatial_run_ragen.py --tasks ActiveRot --num 2"
echo ""
