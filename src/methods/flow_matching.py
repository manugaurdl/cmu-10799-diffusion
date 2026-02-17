import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base import BaseMethod

class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_inference_steps: int = 50,
        sigma_min: float = 0.0,
    ):
        """
        Flow Matching with Optimal Transport path (Linear Interpolation).
        path: x_t = t * x_1 + (1 - t) * x_0
        where x_1 is data, x_0 is noise.
        """
        super().__init__(model, device)
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the Flow Matching loss.
        
        Args:
            x_0: Clean data samples (Target x_1 in FM notation)
            **kwargs: Additional method-specific arguments
        """
        batch_size = x_0.shape[0]

        # x_1 is clean data and x_0_src is Gaussian source noise in FM notation.
        x_1 = x_0
        x_0_src = torch.randn_like(x_1)

        # Sample t in [0, 1], build interpolant x_t, and target velocity.
        t = torch.rand((batch_size,), device=self.device)
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_expand) * x_0_src + t_expand * x_1
        target_velocity = x_1 - x_0_src

        # Match model timestep interface (DDPM-style integer index in [0, 999]).
        t_scaled = (t * 999).long()
        predicted_velocity = self.model(x_t, t_scaled)

        # Compute per-sample FM MSE and then average across batch.
        pv = predicted_velocity.float()
        tv = target_velocity.float()
        mse_loss = F.mse_loss(pv, tv, reduction="none")
        mse_loss = mse_loss.mean(dim=[1, 2, 3])
        loss = mse_loss.mean()

        metrics = {
            "loss": loss.item(),
            "mse": loss.item(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using an ODE solver (Euler method).
        Starts from noise (t=0) and integrates to data (t=1).
        
        Args:
            batch_size: Number of images
            image_shape: (C, H, W)
            steps: Number of Euler integration steps (defaults to self.num_inference_steps)
        """
        self.eval_mode()
        if steps is None:
            steps = self.num_inference_steps
            
        # 1. Start from pure noise at t=0
        x = torch.randn((batch_size, *image_shape), device=self.device)
        
        # 2. Setup time grid
        # We go from 0 to 1
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        dt = 1.0 / steps

        # 3. Euler Integration Loop
        for i in range(steps):
            # Current time t
            t_curr = times[i]
            
            # Broadcast t for the batch
            t_batch = torch.full((batch_size,), t_curr, device=self.device)
            
            # Predict vector field v at current point
            v_pred = self.model(x, t_batch)
            
            # Euler step: x_{t+1} = x_t + v(x_t, t) * dt
            x = x + v_pred * dt

        return x

    def to(self, device: torch.device) -> "FlowMatching":
        super().to(device)
        self.device = device
        return self
    
    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        fm_config = config.get("flow_matching", config)
        return cls(
            model=model,
            device=device,
            num_inference_steps=fm_config.get("num_inference_steps", 50),
            sigma_min=fm_config.get("sigma_min", 0.0),
        )