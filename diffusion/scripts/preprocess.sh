#!/bin/bash
# Preprocessing script

set -e

RAW_DIR="${1:-../../data/nfl-big-data-bowl-2023}"
CACHE_DIR="${2:-../../data/cache}"
CONFIG="${3:-src/football_diffusion/config/default.yaml}"

echo "Preprocessing NFL tracking data"
echo "Raw Dir: $RAW_DIR"
echo "Cache Dir: $CACHE_DIR"

python -m football_diffusion.data.preprocess \
    --config "$CONFIG" \
    --raw_dir "$RAW_DIR" \
    --cache_dir "$CACHE_DIR"

echo "Preprocessing complete! Cached data saved to $CACHE_DIR"

