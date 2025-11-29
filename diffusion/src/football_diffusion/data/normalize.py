"""
Coordinate normalization utilities for diffusion model.
"""
import numpy as np
from typing import Dict, Tuple


def denormalize_tensor(
    tensor: np.ndarray,
    normalization_stats: Dict
) -> np.ndarray:
    """
    Denormalize tensor from normalized space (mean=0, std=1) back to original coordinates.
    
    Args:
        tensor: Normalized tensor [..., F] where F is number of features
        normalization_stats: Dict with 'means' and 'stds' lists for each feature
        
    Returns:
        Denormalized tensor in original coordinate space
    """
    means = np.array(normalization_stats['means'])
    stds = np.array(normalization_stats['stds'])
    
    # Denormalize: x = x_norm * std + mean
    # Apply to last dimension (features)
    tensor_np = np.asarray(tensor)
    denormalized = tensor_np.copy()
    
    # Handle different tensor shapes: [T, P, F] or [B, T, P, F]
    if tensor_np.ndim == 3:
        # [T, P, F]
        for f_idx in range(tensor_np.shape[-1]):
            denormalized[:, :, f_idx] = tensor_np[:, :, f_idx] * stds[f_idx] + means[f_idx]
    elif tensor_np.ndim == 4:
        # [B, T, P, F]
        for f_idx in range(tensor_np.shape[-1]):
            denormalized[:, :, :, f_idx] = tensor_np[:, :, :, f_idx] * stds[f_idx] + means[f_idx]
    else:
        # Generic: apply to last dimension
        for f_idx in range(tensor_np.shape[-1]):
            denormalized[..., f_idx] = tensor_np[..., f_idx] * stds[f_idx] + means[f_idx]
    
    return denormalized

