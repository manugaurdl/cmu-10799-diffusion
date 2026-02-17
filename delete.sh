#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the array of steps to test
STEPS=(1 5 10 50 100 200 1000)

# Create a log file with a timestamp
LOGFILE="evaluation_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting evaluation run. Logging to: $LOGFILE" | tee -a "$LOGFILE"
echo "----------------------------------------" | tee -a "$LOGFILE"

for STEP in "${STEPS[@]}"; do
    echo "Running evaluation for num-steps = $STEP..." | tee -a "$LOGFILE"
    
    # Run the command and append both stdout and stderr to the logfile
    # We also use 'tee' to show the output in the console simultaneously
    CUDA_VISIBLE_DEVICES=3 ./scripts/evaluate_torch_fidelity.sh \
        --checkpoint /home/manu/cmu-10799-diffusion/logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
        --dataset-path /home/manu/cmu-10799-diffusion/data/celeba-subset/train/images \
        --num-steps "$STEP" \
        --sampler ddim \
        --eta 0.0 \
        --metrics kid 2>&1 | tee -a "$LOGFILE"

    echo "Finished num-steps = $STEP" | tee -a "$LOGFILE"
    echo "----------------------------------------" | tee -a "$LOGFILE"
done

echo "All evaluations completed successfully." | tee -a "$LOGFILE"
