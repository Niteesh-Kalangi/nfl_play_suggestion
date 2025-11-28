#!/bin/bash
# Evaluation script for diffusion model

set -e

CHECKPOINT="${1:-../../artifacts/diffusion/last.ckpt}"
CACHE_DIR="${2:-../../data/cache}"
CONFIG="${3:-src/football_diffusion/config/eval.yaml}"
SPLIT="${4:-test}"

echo "Evaluating Football Diffusion Model"
echo "Checkpoint: $CHECKPOINT"
echo "Cache Dir: $CACHE_DIR"
echo "Split: $SPLIT"

python -m football_diffusion.eval.eval_diffusion \
    --checkpoint "$CHECKPOINT" \
    --cache_dir "$CACHE_DIR" \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --num_samples 8 \
    --sample_steps 20 50 100

echo "Evaluation complete! Results saved to $(dirname $CHECKPOINT)/eval_results.json"

