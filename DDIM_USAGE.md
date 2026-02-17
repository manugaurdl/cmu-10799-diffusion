# DDIM Sampling Guide

DDIM (Denoising Diffusion Implicit Models) is a faster, deterministic sampling method for DDPM models.

## Key Features

- **Faster sampling**: Use as few as 50 steps instead of 1000
- **Deterministic**: With `eta=0.0`, same seed produces same output
- **Flexible**: Adjust `eta` to control stochasticity (0.0 = deterministic, 1.0 = DDPM-like)

## Quick Start

### Using the Script

Sample from your trained DDPM model with DDIM (50 steps):

```bash
bash scripts/sample_ddim.sh
```

Or with custom settings:

```bash
CHECKPOINT_PATH=/home/manu/cmu-10799-diffusion/logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
NUM_STEPS=50 \
ETA=0.0 \
bash scripts/sample_ddim.sh
```

### Using Python Directly

```bash
# Deterministic DDIM (50 steps)
python sample.py \
    --checkpoint /home/manu/cmu-10799-diffusion/logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
    --method ddpm \
    --sampler ddim \
    --num_steps 50 \
    --eta 0.0 \
    --num_samples 64

# Stochastic DDIM (more like DDPM)
python sample.py \
    --checkpoint /home/manu/cmu-10799-diffusion/logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
    --method ddpm \
    --sampler ddim \
    --num_steps 100 \
    --eta 0.5 \
    --num_samples 64

# Generate grid instead of individual images
python sample.py \
    --checkpoint /home/manu/cmu-10799-diffusion/logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
    --method ddpm \
    --sampler ddim \
    --num_steps 50 \
    --num_samples 64 \
    --grid \
    --output samples_ddim_grid.png
```

## Configuration File

The `configs/ddim.yaml` file contains DDIM-specific settings:

```yaml
sampling:
  num_steps: 50      # Much fewer steps than DDPM (1000)
  sampler: "ddim"    # Use DDIM sampler
  eta: 0.0          # 0.0 = deterministic, 1.0 = stochastic
```

## Parameters

### `--sampler`
- `ddpm`: Original stochastic sampling (slow, ~1000 steps)
- `ddim`: Deterministic/faster sampling (50-250 steps)

### `--num_steps`
Number of sampling steps:
- DDPM: typically 1000
- DDIM: 50-250 (more steps = better quality)

### `--eta`
Controls stochasticity in DDIM (only used with `--sampler ddim`):
- `0.0`: Fully deterministic (recommended)
- `0.5`: Semi-stochastic
- `1.0`: Similar to DDPM

## Comparison: DDPM vs DDIM

| Feature | DDPM | DDIM |
|---------|------|------|
| Steps | ~1000 | 50-250 |
| Speed | Slow | Fast (10-20x) |
| Deterministic | No | Yes (eta=0) |
| Quality | High | Comparable |

## Examples

### Example 1: Fast Generation (50 steps)
```bash
python sample.py \
    --checkpoint logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
    --method ddpm \
    --sampler ddim \
    --num_steps 50 \
    --num_samples 64 \
    --output_dir samples_ddim_fast
```

### Example 2: High Quality (100 steps)
```bash
python sample.py \
    --checkpoint logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
    --method ddpm \
    --sampler ddim \
    --num_steps 100 \
    --num_samples 64 \
    --output_dir samples_ddim_quality
```

### Example 3: Comparison with DDPM
```bash
# DDIM (fast)
python sample.py --checkpoint logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
    --method ddpm --sampler ddim --num_steps 50 --output_dir samples_ddim

# DDPM (slow)
python sample.py --checkpoint logs/ddpm_20260124_180058/checkpoints/ddpm_final.pt \
    --method ddpm --sampler ddpm --num_steps 1000 --output_dir samples_ddpm
```

## Technical Details

DDIM uses a different reverse process than DDPM:

**DDPM**: `x_{t-1} = μ + σ * noise` (stochastic)

**DDIM**: `x_{t-1} = sqrt(α_{t-1}) * x_0_pred + sqrt(1 - α_{t-1} - σ²) * ε_pred + σ * noise`

Where:
- `σ = eta * sqrt((1 - α_{t-1}) / (1 - α_t) * (1 - α_t / α_{t-1}))`
- When `eta = 0`, the process is deterministic
- When `eta = 1`, it recovers DDPM

## References

- Paper: [Denoising Diffusion Implicit Models (Song et al., 2020)](https://arxiv.org/abs/2010.02502)
- Original DDPM: [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
