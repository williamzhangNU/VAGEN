#!/bin/bash
set -e

echo "Starting installation process..."

# echo "Initializing git submodules..."
# git submodule update --init

# echo "Installing verl package..."
# cd verl
# pip install -e .
# cd ../

echo "Installing vagen dependencies..."
uv pip install 'qwen-vl-utils'
uv pip install 'mathruler'
uv pip install 'matplotlib'
uv pip install 'flask'


# echo "Installing flash-attn with no build isolation..."
# pip install flash-attn --no-build-isolation

echo "Installing vagen package..."
uv pip install -e .

echo "Installing Sokoban dependencies"
uv pip install 'gym'
uv pip install 'gym-sokoban'

echo "Installing Frozenlake dependencies"
uv pip install 'gymnasium'
uv pip install 'pygame'
uv pip install "gym-toytext"

echo "Installation complete, to install dependencies for other environments, refer to env/readme!"
