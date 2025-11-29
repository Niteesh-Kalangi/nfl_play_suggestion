"""
Diffusion model wrapper with DDPM/DDIM sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

from .diffusion_unet import TemporalUNet
from .context_encoder import ContextEncoder
from ..utils.tensor_ops import (
    cosine_beta_schedule,
    linear_beta_schedule,
    extract,
    q_sample
)


class FootballDiffusion(nn.Module):
    """
    Conditional diffusion model for football play generation.
    """
    
    def __init__(
        self,
        num_players: int = 22,
        num_features: int = 3,  # x, y, s
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        condition_dim: int = 256,
        model_channels: int = 256,
        model_depth: int = 4,
        temporal_kernel: int = 3,
        player_heads: int = 4,
        guidance_scale: float = 2.0,
        context_drop_prob: float = 0.1
    ):
        """
        Args:
            num_players: Number of players (22 = 11 offense + 11 defense)
            num_features: Features per player (3 = x, y, s)
            num_timesteps: Number of diffusion timesteps
            beta_schedule: 'cosine' or 'linear'
            condition_dim: Dimension of condition embedding
            model_channels: Base channels for UNet
            model_depth: Depth of UNet
            temporal_kernel: Kernel size for temporal convolutions
            player_heads: Number of player groups
            guidance_scale: Classifier-free guidance scale
            context_drop_prob: Probability of dropping context during training
        """
        super().__init__()
        
        self.num_players = num_players
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale
        self.context_drop_prob = context_drop_prob
        
        # Context encoder
        self.context_encoder = ContextEncoder(output_dim=condition_dim)
        
        # Diffusion model
        in_channels = num_players * num_features
        self.model = TemporalUNet(
            in_channels=in_channels,
            condition_dim=condition_dim,
            channels=model_channels,
            depth=model_depth,
            temporal_kernel=temporal_kernel,
            player_heads=player_heads
        )
        
        # Noise schedule
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = linear_beta_schedule(num_timesteps)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Pre-compute values for sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        ))
        self.register_buffer('posterior_mean_coef1', 
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer('posterior_mean_coef2',
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context_categorical: list,
        context_continuous: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        drop_context: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            x: [B, T, P, F] - input trajectories
            context_categorical: List of categorical context dicts
            context_continuous: [B, 2] - continuous context features
            timestep: [B] - timestep indices (if None, random timesteps sampled)
            drop_context: Whether to drop context for classifier-free guidance
            
        Returns:
            (predicted_noise, actual_noise)
        """
        B, T, P, F = x.shape
        
        # Sample random timesteps if not provided
        if timestep is None:
            timestep = torch.randint(0, self.num_timesteps, (B,), device=x.device)
        
        # Encode context
        context_emb = self.context_encoder(context_categorical, context_continuous)
        
        # Classifier-free guidance: randomly drop context during training
        if drop_context is None:
            drop_context = self.training and torch.rand(1).item() < self.context_drop_prob
        
        if drop_context:
            # Use null context
            context_emb = torch.zeros_like(context_emb)
        
        # Reshape for model: [B, T, P, F] -> [B, P*F, T]
        # CRITICAL: Model expects [B, P*F, T] shape (e.g., [B, 66, 60] for 22 players * 3 features)
        # This must match sampling which uses torch.randn(B, P*F, T)
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # [B, T, P, F] -> [B, P, F, T]
        B, P, F, T = x_reshaped.shape
        x_reshaped = x_reshaped.reshape(B, P * F, T)  # [B, P, F, T] -> [B, P*F, T]
        
        # Add noise
        noisy, noise = q_sample(
            x_reshaped,
            timestep,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod
        )
        
        # Predict noise (model expects [B, P*F, T])
        predicted_noise = self.model(noisy, timestep, context_emb)
        
        # Reshape back: [B, P*F, T] -> [B, T, P, F]
        # Model outputs [B, P*F, T], reshape to [B, P, F, T], then permute
        predicted_noise = predicted_noise.reshape(B, P, F, T).permute(0, 3, 1, 2).contiguous()  # [B, T, P, F]
        noise_target = noise.reshape(B, P, F, T).permute(0, 3, 1, 2).contiguous()  # [B, T, P, F]
        
        return predicted_noise, noise_target
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],  # (T, P, F)
        context_categorical: list,
        context_continuous: torch.Tensor,
        num_steps: int = 50,
        ddim: bool = False,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Sample from the diffusion model.
        
        Args:
            shape: (T, P, F) - shape of output trajectory
            context_categorical: List of categorical context dicts
            context_continuous: [B, 2] - continuous context features
            num_steps: Number of sampling steps
            ddim: Use DDIM sampling if True, DDPM otherwise
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)
            
        Returns:
            [B, T, P, F] - generated trajectories
        """
        B = context_continuous.shape[0]
        T, P, F = shape
        device = context_continuous.device
        
        # Encode context
        context_emb = self.context_encoder(context_categorical, context_continuous)
        
        # Classifier-free guidance: use both conditioned and unconditioned
        if self.guidance_scale > 1.0:
            null_context = torch.zeros_like(context_emb)
            context_emb = torch.cat([context_emb, null_context], dim=0)
        
        # Start from pure noise
        x = torch.randn(B, P * F, T, device=device)
        
        # Sampling loop
        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).long()
        timesteps = timesteps.flip(0)  # Reverse order
        
        for i, t in enumerate(timesteps):
            # Expand timestep for batch
            t_batch = t.repeat(B)
            
            if self.guidance_scale > 1.0:
                # Classifier-free guidance: predict with and without context
                t_batch_guidance = t_batch.repeat(2)
                x_guidance = x.repeat(2, 1, 1)
                pred_noise = self.model(x_guidance, t_batch_guidance, context_emb)
                pred_noise_cond, pred_noise_uncond = pred_noise.chunk(2)
                pred_noise = pred_noise_uncond + self.guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = self.model(x, t_batch, context_emb)
            
            if ddim:
                # DDIM sampling
                alpha_t = extract(self.alphas_cumprod, t_batch, x.shape)
                alpha_prev = extract(
                    self.alphas_cumprod,
                    (t - step_size).clamp(min=0) if i < len(timesteps) - 1 else torch.zeros_like(t_batch),
                    x.shape
                )
                
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                dir_xt = torch.sqrt(1 - alpha_prev - eta**2 * (1 - alpha_prev)) * pred_noise
                
                if i < len(timesteps) - 1:
                    noise = eta * torch.sqrt(1 - alpha_prev) * torch.randn_like(x)
                else:
                    noise = 0
                
                x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + noise
            else:
                # DDPM sampling
                alpha_t = extract(self.alphas, t_batch, x.shape)
                alpha_cumprod_t = extract(self.alphas_cumprod, t_batch, x.shape)
                beta_t = extract(self.betas, t_batch, x.shape)
                
                if i < len(timesteps) - 1:
                    # Predict x0
                    pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
                    
                    # Sample from q(x_{t-1} | x_t, x_0)
                    posterior_mean_coef1_t = extract(self.posterior_mean_coef1, t_batch, x.shape)
                    posterior_mean_coef2_t = extract(self.posterior_mean_coef2, t_batch, x.shape)
                    
                    posterior_mean = posterior_mean_coef1_t * pred_x0 + posterior_mean_coef2_t * x
                    posterior_variance_t = extract(self.posterior_variance, t_batch, x.shape)
                    
                    noise = torch.randn_like(x)
                    x = posterior_mean + torch.sqrt(posterior_variance_t) * noise
                else:
                    # Last step: predict x0 directly
                    x = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Reshape: [B, P*F, T] -> [B, T, P, F]
        # Use same reshape method as training for consistency
        x = x.reshape(B, P, F, T).permute(0, 3, 1, 2).contiguous()  # [B, T, P, F]
        
        return x

