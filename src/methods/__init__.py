"""
Methods module for cmu-10799-diffusion.

This module contains implementations of generative modeling methods:
- DDPM (Denoising Diffusion Probabilistic Models)
- FlowMatching (Flow Matching with Optimal Transport)
- JiTFlowMatching (Flow Matching with x-prediction / JiT-style)
"""

from .base import BaseMethod
from .ddpm import DDPM
from .flow_matching import FlowMatching
from .jit import FlowMatching as JiTFlowMatching

__all__ = [
    'BaseMethod',
    'DDPM',
    'FlowMatching',
    'JiTFlowMatching',
]
