#!/bin/bash
# =============================================================================
# Torch-Fidelity Evaluation Script
# =============================================================================
#
# Usage:
#   # Basic usage with DDPM
#   ./scripts/evaluate_torch_fidelity.sh \
#       --checkpoint checkpoints/ddpm/ddpm_final.pt \
#       --method ddpm \
#       --dataset-path data/celeba \
#       --metrics kid
#
#   # With DDIM sampler (faster)
#   ./scripts/evaluate_torch_fidelity.sh \
#       --checkpoint checkpoints/ddpm/ddpm_final.pt \
#       --method ddpm \
#       --dataset-path data/celeba \
#       --num-steps 100 \
#       --sampler ddim \
#       --eta 0.0 \
#       --metrics kid
#
# =============================================================================

set -e

# Defaults
METHOD="ddpm" # (right now you only have ddpm but you will be implementing more methods as hw progresses)
CHECKPOINT="YOUR_PATH"
DATASET_PATH="data/celeba"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=1000
SAMPLER=""        # Optional: ddpm or ddim
ETA=""            # Optional: eta for DDIM (0.0 = deterministic, 1.0 = stochastic)
GENERATED_DIR=""  # Will be set based on checkpoint location
CACHE_DIR=""      # Will be set based on checkpoint location

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --sampler) SAMPLER="$2"; shift 2 ;;
        --eta) ETA="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

# Set output directories based on checkpoint location
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
GENERATED_DIR="${CHECKPOINT_DIR}/samples/generated"
CACHE_DIR="${CHECKPOINT_DIR}/samples/cache"

echo "=========================================="
echo "Torch-Fidelity Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Method: $METHOD"
echo "Dataset: $DATASET_PATH"
echo "Metrics: $METRICS"
echo "Num samples: $NUM_SAMPLES"
[ -n "$NUM_STEPS" ] && echo "Num steps: $NUM_STEPS"
[ -n "$SAMPLER" ] && echo "Sampler: $SAMPLER"
[ -n "$ETA" ] && echo "Eta: $ETA"
echo "Output: $GENERATED_DIR"
echo "=========================================="

# Step 1: Generate samples
echo ""
echo "[1/2] Generating samples..."
rm -rf "$GENERATED_DIR"

SAMPLE_CMD="python sample.py \
    --checkpoint $CHECKPOINT \
    --method $METHOD \
    --output_dir $GENERATED_DIR \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE"

[ -n "$NUM_STEPS" ] && SAMPLE_CMD="$SAMPLE_CMD --num_steps $NUM_STEPS"
[ -n "$SAMPLER" ] && SAMPLE_CMD="$SAMPLE_CMD --sampler $SAMPLER"
[ -n "$ETA" ] && SAMPLE_CMD="$SAMPLE_CMD --eta $ETA"

eval $SAMPLE_CMD

# Step 2: Run fidelity
echo ""
echo "[2/2] Computing metrics..."
mkdir -p "$CACHE_DIR"

FIDELITY_CMD="fidelity --gpu 0 --batch-size $BATCH_SIZE --cache-root $CACHE_DIR \
    --input1 $GENERATED_DIR --input2 $DATASET_PATH"

[[ "$METRICS" == *"fid"* ]] && FIDELITY_CMD="$FIDELITY_CMD --fid"
[[ "$METRICS" == *"kid"* ]] && FIDELITY_CMD="$FIDELITY_CMD --kid"
[[ "$METRICS" == *"is"* ]] && FIDELITY_CMD="$FIDELITY_CMD --isc"

eval $FIDELITY_CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
