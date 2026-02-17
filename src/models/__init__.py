"""
Models module for cmu-10799-diffusion.

This module contains the neural network architectures used for
diffusion models and flow matching.
"""

from .unet import UNet, create_model_from_config as create_unet_from_config
from .dit import DiT, DiT_models, create_dit_from_config
from .blocks import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)

def create_model_from_config(config: dict):
    """
    Factory for model creation from config.

    Supports:
      - model.type: "unet" (default)
      - model.type: "dit"
    """
    model_type = config.get("model", {}).get("type", "unet").lower()
    if model_type == "unet":
        return create_unet_from_config(config)
    if model_type == "dit":
        return create_dit_from_config(config)
    raise ValueError(f"Unknown model.type '{model_type}'. Expected one of: unet, dit.")


__all__ = [
    "UNet",
    "DiT",
    "DiT_models",
    "create_model_from_config",
    "create_unet_from_config",
    "create_dit_from_config",
    "SinusoidalPositionalEmbedding",
    "TimestepEmbedding",
    "ResBlock",
    "AttentionBlock",
    "Downsample",
    "Upsample",
    "GroupNorm32",
]
