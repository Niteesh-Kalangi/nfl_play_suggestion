#!/bin/bash
# Training script for diffusion model

set -e

# Get script directory and navigate to diffusion directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Default paths (relative to diffusion directory)
CONFIG="${1:-src/football_diffusion/config/train.yaml}"
CACHE_DIR="${2:-../data/cache}"
OUTPUT_DIR="${3:-../artifacts/diffusion}"

echo "Training Football Diffusion Model"
echo "Running from: $(pwd)"
echo "Config: $CONFIG"
echo "Cache Dir: $CACHE_DIR"
echo "Output Dir: $OUTPUT_DIR"

python train_main.py \
    --config "$CONFIG" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --gpus 1 \
    --max_epochs 50

echo "Training complete! Checkpoints saved to $OUTPUT_DIR"

