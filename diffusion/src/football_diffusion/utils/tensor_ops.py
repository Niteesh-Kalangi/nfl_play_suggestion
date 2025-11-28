"""
Tensor operations utilities for diffusion model.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """
    Extract values from tensor a at indices t, then reshape to match x_shape.
    
    Used for extracting beta, alpha, etc. at specific timesteps.
    """
    batch_size = t.shape[0]
    # Ensure both tensors are on the same device (MPS compatibility)
    device = t.device
    # Move schedule tensor to the same device as timesteps
    a = a.to(device)
    # Convert indices to long and ensure on same device
    t_long = t.long().to(device)
    
    # Use indexing instead of gather for better MPS compatibility
    # Clamp indices to valid range
    t_clamped = torch.clamp(t_long, 0, a.shape[-1] - 1)
    # Extract values using indexing (works better with MPS than gather)
    out = a[t_clamped]
    
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule for diffusion.
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset for numerical stability
        
    Returns:
        Tensor of shape [timesteps] with beta values
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear noise schedule for diffusion.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Tensor of shape [timesteps] with beta values
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I).
    
    Args:
        x_start: Starting tensor [B, ...]
        t: Timestep indices [B]
        sqrt_alphas_cumprod: sqrt(alpha_bar_t) for each timestep [T]
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar_t) for each timestep [T]
        
    Returns:
        Noisy tensor and noise tensor
    """
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    noise = torch.randn_like(x_start)
    
    noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    return noisy, noise


def q_posterior_mean_variance(
    x_start: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    sqrt_recip_alphas_cumprod: torch.Tensor,
    sqrt_recipm1_alphas_cumprod: torch.Tensor,
    posterior_mean_coef1: torch.Tensor,
    posterior_mean_coef2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute q(x_{t-1} | x_t, x_0) mean and variance.
    """
    posterior_mean = (
        extract(posterior_mean_coef1, t, x_t.shape) * x_start +
        extract(posterior_mean_coef2, t, x_t.shape) * x_t
    )
    
    # Posterior variance (simplified - would need proper computation)
    posterior_variance = extract(
        sqrt_recipm1_alphas_cumprod ** 2,
        t,
        x_t.shape
    )
    
    return posterior_mean, posterior_variance

