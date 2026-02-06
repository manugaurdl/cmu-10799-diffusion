"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt(1.0 - alpha_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod),
        )

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        a= a.cuda()
        t= t.cuda()
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.view(batch_size, *([1] * (len(x_shape) - 1)))

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement the forward (noise adding) process of DDPM
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alpha_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alpha_cumprod, t, x_0.shape
        )
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        
        # We still generate noise to create x_t, but we don't need it for the loss target
        x_t, _ = self.forward_process(x_0, t)
        
        # CHANGE 1: The model is now parametrized to output the clean image (pred_x0)
        pred_x0 = self.model(x_t, t)
        
        # CHANGE 2: The loss target is the original clean image (x_0)
        t_max_threshold = int(self.num_timesteps * 0.95)
        mask = (t < t_max_threshold).float().view(-1, 1, 1, 1)
        per_elem_loss = F.mse_loss(pred_x0, x_0, reduction="none")
        masked_loss = per_elem_loss * mask
        denom = mask.sum() * per_elem_loss[0].numel()
        loss = masked_loss.sum() / denom
        
        metrics = {
            "loss": loss,
            "mse": loss,
        }
        return loss, metrics

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        if t.dim() == 0:
            t = t[None].repeat(x_t.shape[0])

        pred_noise = self.model(x_t, t)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alpha_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alpha_cumprod, t, x_t.shape
        )
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t

        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_pred
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(x_t.shape[0], *([1] * (x_t.dim() - 1)))
        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        num_steps = kwargs.get("num_steps", self.num_timesteps)
        if num_steps is None:
            num_steps = self.num_timesteps

        x_t = torch.randn((batch_size, *image_shape), device=self.device)
        if num_steps == self.num_timesteps:
            timesteps = range(self.num_timesteps - 1, -1, -1)
        else:
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, num_steps, device=self.device
            ).long()

        for step in timesteps:
            t = torch.full((batch_size,), int(step), device=self.device, dtype=torch.long)
            x_t = self.reverse_process(x_t, t)

        return x_t

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )
