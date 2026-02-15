#!/bin/bash

# =============================================================================
# DDPM Sampling Script for CMU 10799 Diffusion Homework
# =============================================================================
# 
# Usage:
#   # Sample with default settings (1000 steps)
#   bash scripts/sample_ddpm.sh
#
#   # Sample with custom steps
#   bash scripts/sample_ddpm.sh --num_steps 500
#
#   # Generate grid instead of individual images
#   bash scripts/sample_ddpm.sh --grid
#
# =============================================================================

# Default checkpoint path (update with your checkpoint)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt}"

# Default settings
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_STEPS="${NUM_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-samples_ddpm}"
DEVICE="${DEVICE:-cuda}"

echo "=============================================="
echo "DDPM Sampling"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Number of samples: $NUM_SAMPLES"
echo "Number of steps: $NUM_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=============================================="

# Activate virtual environment if available
if [ -d ".venv-cuda129" ]; then
    source .venv-cuda129/bin/activate
elif [ -d ".venv-cuda126" ]; then
    source .venv-cuda126/bin/activate
elif [ -d ".venv-cuda121" ]; then
    source .venv-cuda121/bin/activate
elif [ -d ".venv-cuda118" ]; then
    source .venv-cuda118/bin/activate
elif [ -d ".venv-rocm" ]; then
    source .venv-rocm/bin/activate
elif [ -d ".venv-cpu" ]; then
    source .venv-cpu/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run sampling
python sample.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --method ddpm \
    --num_samples "$NUM_SAMPLES" \
    --num_steps "$NUM_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --sampler ddpm \
    "$@"

echo ""
echo "=============================================="
echo "DDPM Sampling completed at $(date)"
echo "=============================================="
