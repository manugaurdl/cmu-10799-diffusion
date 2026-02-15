#!/bin/bash
# =============================================================================
# Modal Torch-Fidelity Evaluation Script
# =============================================================================
#
# Submits torch-fidelity evaluation jobs to Modal cloud.
#
# Usage:
#   # Basic usage
#   ./scripts/evaluate_modal_torch_fidelity.sh \
#       --method ddpm \
#       --checkpoint checkpoints/ddpm/ddpm_final.pt
#
#   # With DDIM sampler
#   ./scripts/evaluate_modal_torch_fidelity.sh \
#       --method ddpm \
#       --checkpoint checkpoints/ddpm/ddpm_final.pt \
#       --num-steps 100 \
#       --sampler ddim \
#       --eta 0.0
#
# =============================================================================

set -e

# Defaults optimized for KID evaluation
METHOD="ddpm" # (right now you only have ddpm but you will be implementing more methods as hw progresses)
CHECKPOINT="YOUR_PATH"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=1000
SAMPLER=""        # Optional: ddpm or ddim
ETA=""            # Optional: eta for DDIM (0.0 = deterministic, 1.0 = stochastic)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
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
    echo ""
    echo "Usage: $0 --method ddpm --checkpoint <path> [options]"
    echo ""
    echo "Options:"
    echo "  --metrics <fid,kid,is>    Metrics to compute (default: kid)"
    echo "  --num-samples <N>         Number of samples (default: 1000)"
    echo "  --batch-size <N>          Batch size (default: 256)"
    echo "  --num-steps <N>           Sampling steps (default: 1000)"
    echo "  --sampler <ddpm|ddim>     Sampler to use (optional)"
    echo "  --eta <float>             DDIM eta parameter (default: 0.0)"
    exit 1
fi

echo "=========================================="
echo "Modal Torch-Fidelity Evaluation"
echo "=========================================="
echo "Method: $METHOD"
echo "Checkpoint: $CHECKPOINT"
echo "Metrics: $METRICS"
echo "Num samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Num steps: $NUM_STEPS"
[ -n "$SAMPLER" ] && echo "Sampler: $SAMPLER"
[ -n "$ETA" ] && echo "Eta: $ETA"
echo "=========================================="
echo ""
echo "Submitting to Modal..."
echo ""

# Build Modal command
MODAL_CMD="modal run modal_app.py::main --action evaluate_torch_fidelity \
    --method $METHOD \
    --checkpoint $CHECKPOINT \
    --metrics $METRICS \
    --num-samples $NUM_SAMPLES \
    --batch-size $BATCH_SIZE \
    --num-steps $NUM_STEPS"

[ -n "$SAMPLER" ] && MODAL_CMD="$MODAL_CMD --sampler $SAMPLER"
[ -n "$ETA" ] && MODAL_CMD="$MODAL_CMD --eta $ETA"

# Run Modal command
eval $MODAL_CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
