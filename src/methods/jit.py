import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base import BaseMethod


# ---------------------------------------------------------------------------
# iREPA alignment module
# ---------------------------------------------------------------------------

class iREPAAlignment(nn.Module):
    """Improved Representation Alignment (iREPA) auxiliary module.

    Two improvements over vanilla REPA:
    1. Conv2d projection (3×3, padding=1) instead of an MLP — preserves spatial
       structure when mapping DiT hidden tokens → encoder feature dim.
    2. Spatial normalisation on encoder features: mean-centre across the T
       spatial token dimension, then divide by std.  A learnable scalar `gamma`
       (init 1) controls the strength of mean subtraction.

    Reference: https://github.com/End2End-Diffusion/iREPA
    """

    def __init__(self, d_dit: int, d_enc: int):
        """
        Args:
            d_dit: DiT hidden dimension (D_in for projection).
            d_enc: Encoder feature dimension (D_out for projection).
        """
        super().__init__()
        # iREPA improvement 1: Conv2d projection instead of MLP
        self.proj = nn.Conv2d(d_dit, d_enc, kernel_size=3, padding=1)
        # Learnable scale for the mean-subtraction in spatial normalisation
        self.gamma = nn.Parameter(torch.ones(1))

        # Initialise projection like a linear layer (zero bias, small weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def spatial_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """iREPA improvement 2: Spatial normalisation on encoder features.

        Args:
            x: Encoder tokens of shape [B, T, D].
        Returns:
            Normalised tokens of shape [B, T, D].
        """
        # Mean-centre across the T (spatial) dimension
        x = x - self.gamma * x.mean(dim=1, keepdim=True)
        # Standardise
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return x

    def forward(
        self,
        dit_tokens: torch.Tensor,
        enc_tokens: torch.Tensor,
        grid_size: int,
    ) -> torch.Tensor:
        """Compute the cosine alignment loss between projected DiT tokens and
        spatially-normalised encoder tokens.

        Args:
            dit_tokens: DiT intermediate tokens [B, T_dit, D_dit].
            enc_tokens:  Encoder feature tokens  [B, T_enc, D_enc].
            grid_size:   Spatial grid edge length for DiT tokens (sqrt(T_dit)).
        Returns:
            Scalar alignment loss (mean cosine distance).
        """
        B, T_dit, D_dit = dit_tokens.shape

        # Reshape DiT tokens to spatial feature map for Conv2d
        # [B, T, D] -> [B, D, H, W]
        H = W = grid_size
        dit_spatial = dit_tokens.permute(0, 2, 1).reshape(B, D_dit, H, W)

        # Project via conv: [B, D_dit, H, W] -> [B, D_enc, H, W]
        proj_spatial = self.proj(dit_spatial)

        # Flatten back to token sequence: [B, T_dit, D_enc]
        proj_tokens = proj_spatial.flatten(2).permute(0, 2, 1)  # [B, T, D_enc]

        # --- Interpolate encoder tokens to match DiT spatial resolution ---
        # Encoder may have a different number of tokens (T_enc ≠ T_dit).
        T_enc = enc_tokens.shape[1]
        if T_enc != T_dit:
            # Reshape enc features to spatial and interpolate
            H_enc = W_enc = int(math.isqrt(T_enc))
            enc_spatial = enc_tokens.permute(0, 2, 1).reshape(
                B, enc_tokens.shape[2], H_enc, W_enc
            )
            enc_spatial = F.interpolate(
                enc_spatial.float(), size=(H, W), mode="bilinear", align_corners=False
            )
            enc_tokens = enc_spatial.flatten(2).permute(0, 2, 1)  # [B, T_dit, D_enc]

        # Spatial normalisation on encoder features (iREPA key contribution)
        enc_norm = self.spatial_normalize(enc_tokens.float())

        # Cosine similarity loss: 1 - cosine_sim, averaged over B and T
        proj_tokens = F.normalize(proj_tokens.float(), dim=-1)
        enc_norm = F.normalize(enc_norm, dim=-1)
        align_loss = (1.0 - (proj_tokens * enc_norm).sum(dim=-1)).mean()
        return align_loss


# ---------------------------------------------------------------------------
# DINOv2 encoder wrapper (frozen)
# ---------------------------------------------------------------------------

class DINOv2Encoder(nn.Module):
    """Thin wrapper around a frozen DINOv2 hub model.

    Exposes `encode(x) -> tokens [B, T, D]` for use inside the alignment loss.
    Images should be in approximately [-1, 1] (standard diffusion normalisation);
    this wrapper rescales to DINOv2's expected ImageNet normalisation.
    """

    # ImageNet mean/std expected by DINOv2
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD  = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    PATCH_SIZE = 14  # DINOv2 patch size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to patch tokens.

        Args:
            x: Images [B, 3, H, W] in [-1, 1] (diffusion normalisation).
        Returns:
            Patch tokens [B, T, D] (no CLS token).
        """
        device = x.device
        # Rescale from [-1, 1] → [0, 1] → ImageNet normalisation
        x = (x * 0.5 + 0.5).clamp(0, 1)
        mean = self.MEAN.to(device, x.dtype).view(1, 3, 1, 1)
        std  = self.STD.to(device, x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        # DINOv2 requires H and W to be multiples of patch_size (14).
        # Round up to the nearest valid resolution.
        H, W = x.shape[-2], x.shape[-1]
        p = self.PATCH_SIZE
        H_new = math.ceil(H / p) * p
        W_new = math.ceil(W / p) * p
        if H_new != H or W_new != W:
            x = F.interpolate(x, size=(H_new, W_new), mode="bilinear", align_corners=False)

        # DINOv2 forward_features returns a dict with 'x_norm_patchtokens'
        with torch.no_grad():
            features = self.model.forward_features(x)
        # shape: [B, T, D]  (spatial tokens only, no CLS)
        return features["x_norm_patchtokens"]


# ---------------------------------------------------------------------------
# JiT Flow Matching
# ---------------------------------------------------------------------------

class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_inference_steps: int = 50,
        sigma_min: float = 0.0,
        clip_denom: float = 0.05,  # prevents division by zero near t=1
        # iREPA arguments
        use_irepa: bool = False,
        irepa_weight: float = 0.5,
        irepa_layer: int = 8,
        encoder_name: str = "dinov2_vits14",
    ):
        """
        Flow Matching with x-prediction (JiT-style) + optional iREPA alignment.

        path: x_t = t * x_1 + (1 - t) * x_noise
        Network directly predicts clean data x_1, loss is computed in v-space.

        iREPA adds a cosine alignment auxiliary loss between intermediate DiT
        tokens and frozen DINOv2 encoder features, with:
        - Conv2d projection (instead of MLP)
        - Spatial normalisation on encoder features
        """
        super().__init__(model, device)
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
        self.clip_denom = clip_denom

        # iREPA
        self.use_irepa = use_irepa
        self.irepa_weight = irepa_weight
        self.irepa_layer = irepa_layer

        if use_irepa:
            # Infer DiT hidden dim from the model
            d_dit = self._get_dit_hidden_dim()
            # DINOv2 ViT-S/14 → 384-dim; ViT-B/14 → 768-dim etc.
            encoder_dim_map = {
                "dinov2_vits14": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitg14": 1536,
            }
            d_enc = encoder_dim_map.get(encoder_name, 384)

            self.irepa_align = iREPAAlignment(d_dit=d_dit, d_enc=d_enc).to(device)
            self.encoder = DINOv2Encoder(model_name=encoder_name).to(device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_dit_hidden_dim(self) -> int:
        """Extract hidden_size from the underlying DiT model."""
        # unwrap DDP if needed
        m = self.model
        while hasattr(m, "module"):
            m = m.module
        if hasattr(m, "hidden_size"):
            return m.hidden_size
        # Fallback: read from x_embedder projection weight
        if hasattr(m, "x_embedder"):
            return m.x_embedder.proj.out_channels
        raise AttributeError(
            "Cannot determine DiT hidden_size. Ensure model is a DiT instance."
        )

    def _get_dit_grid_size(self) -> int:
        """Return the spatial grid edge length of DiT tokens (sqrt(num_patches))."""
        m = self.model
        while hasattr(m, "module"):
            m = m.module
        if hasattr(m, "x_embedder"):
            return m.x_embedder.grid_size
        raise AttributeError("Cannot determine DiT grid_size.")

    def parameters(self):
        """Return all trainable parameters: DiT model + iREPA projection."""
        params = list(self.model.parameters())
        if self.use_irepa:
            params += list(self.irepa_align.parameters())
        return iter(params)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Computes the Flow Matching loss with x-prediction, plus optional
        iREPA alignment loss.

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
        #    When iREPA is on, also return intermediate tokens from irepa_layer
        if self.use_irepa:
            x_1_pred, dit_tokens = self.model(x_t, t, return_layer=self.irepa_layer)
        else:
            x_1_pred = self.model(x_t, t)

        # 6. Derive v_pred from x_1_pred via reparameterization:
        #    v_pred = (x_1_pred - x_t) / (1 - t)
        denom = (1 - t_view).clamp(min=self.clip_denom)
        v_pred = (x_1_pred - x_t) / denom

        # 7. MSE loss in v-space
        mse_loss = F.mse_loss(v_pred, v_true)

        metrics: Dict = {
            "loss": mse_loss.item(),
            "mse_loss": mse_loss.item(),
        }

        # 8. iREPA alignment loss
        if self.use_irepa:
            # Encode clean images with frozen DINOv2
            # Images are already in ~[-1,1] from the dataloader
            enc_tokens = self.encoder.encode(x_0.float())

            # Compute alignment loss
            grid_size = self._get_dit_grid_size()
            align_loss = self.irepa_align(dit_tokens, enc_tokens, grid_size)

            total_loss = mse_loss + self.irepa_weight * align_loss
            metrics["align_loss"] = align_loss.item()
            metrics["loss"] = total_loss.item()
        else:
            total_loss = mse_loss

        return total_loss, metrics

    # ------------------------------------------------------------------
    # Sampling (unchanged)
    # ------------------------------------------------------------------

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

            # Network predicts clean x_1 (no intermediate tokens needed at inference)
            x_1_pred = self.model(x, t_batch)

            # Derive v_pred: (x_1_pred - x_t) / (1 - t)
            denom = (1 - t_view).clamp(min=self.clip_denom)
            v_pred = (x_1_pred - x) / denom

            # Euler step
            x = x + v_pred * dt

        return x

    # ------------------------------------------------------------------
    # Device / state
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "FlowMatching":
        super().to(device)
        self.device = device
        if self.use_irepa:
            self.irepa_align = self.irepa_align.to(device)
            self.encoder = self.encoder.to(device)
        return self

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        fm_config = config.get("flow_matching", config)
        irepa_config = config.get("irepa", {})
        return cls(
            model=model,
            device=device,
            num_inference_steps=fm_config.get("num_inference_steps", 50),
            sigma_min=fm_config.get("sigma_min", 0.0),
            clip_denom=fm_config.get("clip_denom", 0.05),
            use_irepa=irepa_config.get("enabled", False),
            irepa_weight=irepa_config.get("weight", 0.5),
            irepa_layer=irepa_config.get("layer", 8),
            encoder_name=irepa_config.get("encoder", "dinov2_vits14"),
        )