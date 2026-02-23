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
        clip_denom: float = 0.05,  # prevents division by zero near t=1
    ):
        """
        Flow Matching with x-prediction (JiT-style).
        path: x_t = t * x_1 + (1 - t) * x_noise
        Network directly predicts clean data x_1, loss is computed in v-space.
        """
        super().__init__(model, device)
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
        self.clip_denom = clip_denom

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the Flow Matching loss with x-prediction.
        Network predicts x_1 (clean data) directly; loss is MSE in v-space.
        
        Args:
            x_0: Clean data samples (x_1 in FM notation)
        """
        batch_size = x_0.shape[0]

        # 1. Sample time steps t uniformly from [0, 1]
        t = torch.rand((batch_size,), device=self.device)
        t_view = t.view(batch_size, *([1] * (x_0.ndim - 1)))

        # 2. Sample noise and rename for clarity
        x_1 = x_0
        x_noise = torch.randn_like(x_1)

        # 3. Interpolate: x_t = t * x_1 + (1 - t) * x_noise
        x_t = t_view * x_1 + (1 - t_view) * x_noise

        # 4. True velocity: v = x_1 - x_noise
        v_true = x_1 - x_noise

        # 5. Network predicts clean x_1 directly
        x_1_pred = self.model(x_t, t)

        # 6. Derive v_pred from x_1_pred via reparameterization:
        #    v_pred = (x_1_pred - x_t) / (1 - t)
        #    This is equivalent to a reweighted x-loss: 1/(1-t)^2 * ||x_1_pred - x_1||^2
        denom = (1 - t_view).clamp(min=self.clip_denom)
        v_pred = (x_1_pred - x_t) / denom

        # 7. MSE loss in v-space
        loss = F.mse_loss(v_pred, v_true)

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
        Generate samples using Euler integration.
        At each step, network predicts x_1, which is converted to v for the ODE step.
        """
        self.eval_mode()
        if steps is None:
            steps = self.num_inference_steps

        # 1. Start from pure noise at t=0
        x = torch.randn((batch_size, *image_shape), device=self.device)

        # 2. Time grid from 0 -> 1
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        dt = 1.0 / steps

        # 3. Euler integration
        for i in range(steps):
            t_curr = times[i]
            t_batch = torch.full((batch_size,), t_curr, device=self.device)
            t_view = t_batch.view(batch_size, *([1] * len(image_shape)))

            # Network predicts clean x_1
            x_1_pred = self.model(x, t_batch)

            # Derive v_pred: (x_1_pred - x_t) / (1 - t)
            denom = (1 - t_view).clamp(min=self.clip_denom)
            v_pred = (x_1_pred - x) / denom

            # Euler step
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
            clip_denom=fm_config.get("clip_denom", 0.05),
        )