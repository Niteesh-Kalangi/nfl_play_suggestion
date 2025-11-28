"""
Evaluation metrics for trajectory generation models.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def compute_ade(predicted: torch.Tensor, ground_truth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Average Displacement Error (ADE).
    
    ADE = mean L2 distance across all timesteps and players.
    
    Args:
        predicted: [B, T, P, 2] - predicted trajectories (x, y)
        ground_truth: [B, T, P, 2] - ground truth trajectories
        mask: [B, T] - optional mask for valid timesteps
        
    Returns:
        Average ADE across batch
    """
    # Extract x, y coordinates (first 2 features)
    if predicted.shape[-1] >= 2:
        pred_xy = predicted[:, :, :, :2]
        gt_xy = ground_truth[:, :, :, :2]
    else:
        pred_xy = predicted
        gt_xy = ground_truth
    
    # Compute L2 distance per timestep, per player
    errors = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, T, P]
    
    # Apply mask if provided
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand_as(errors)  # [B, T, P]
        errors = errors * mask_expanded
        valid_count = mask_expanded.sum()
    else:
        valid_count = errors.numel()
    
    ade = errors.sum() / valid_count if valid_count > 0 else 0.0
    return ade.item()


def compute_fde(predicted: torch.Tensor, ground_truth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Final Displacement Error (FDE).
    
    FDE = L2 distance at the last timestep.
    
    Args:
        predicted: [B, T, P, 2] - predicted trajectories
        ground_truth: [B, T, P, 2] - ground truth trajectories
        mask: [B, T] - optional mask for valid timesteps
        
    Returns:
        Average FDE across batch
    """
    # Extract x, y coordinates
    if predicted.shape[-1] >= 2:
        pred_xy = predicted[:, :, :, :2]
        gt_xy = ground_truth[:, :, :, :2]
    else:
        pred_xy = predicted
        gt_xy = ground_truth
    
    # Get last valid timestep for each sequence
    if mask is not None:
        # Find last valid timestep for each batch item
        last_valid = mask.long().argmax(dim=1)  # [B]
        last_valid = torch.clamp(last_valid, max=pred_xy.shape[1] - 1)
        
        pred_final = pred_xy[torch.arange(pred_xy.shape[0]), last_valid]  # [B, P, 2]
        gt_final = gt_xy[torch.arange(gt_xy.shape[0]), last_valid]  # [B, P, 2]
    else:
        pred_final = pred_xy[:, -1, :, :]  # [B, P, 2]
        gt_final = gt_xy[:, -1, :, :]  # [B, P, 2]
    
    # Compute L2 distance
    errors = torch.norm(pred_final - gt_final, dim=-1)  # [B, P]
    fde = errors.mean().item()
    
    return fde


def compute_validity_rate(
    trajectories: torch.Tensor,
    field_bounds: List[float] = [0, 120, 0, 53.3],
    speed_cap: float = 12.0
) -> float:
    """
    Compute validity rate: % of points within field bounds and below speed cap.
    
    Args:
        trajectories: [B, T, P, F] - trajectories (F includes x, y, s)
        field_bounds: [x_min, x_max, y_min, y_max]
        speed_cap: Maximum allowed speed in yd/s
        
    Returns:
        Validity rate (0-1)
    """
    x_min, x_max, y_min, y_max = field_bounds
    
    # Extract positions and speeds
    if trajectories.shape[-1] >= 3:
        x_pos = trajectories[:, :, :, 0]
        y_pos = trajectories[:, :, :, 1]
        speeds = trajectories[:, :, :, 2]
    elif trajectories.shape[-1] >= 2:
        x_pos = trajectories[:, :, :, 0]
        y_pos = trajectories[:, :, :, 1]
        speeds = torch.zeros_like(x_pos)  # No speed info
    else:
        return 0.0
    
    # Check bounds
    x_valid = (x_pos >= x_min) & (x_pos <= x_max)
    y_valid = (y_pos >= y_min) & (y_pos <= y_max)
    speed_valid = speeds <= speed_cap
    
    # All conditions must be met
    all_valid = x_valid & y_valid & speed_valid
    
    validity_rate = all_valid.float().mean().item()
    return validity_rate


def compute_diversity(
    samples: torch.Tensor,
    context_groups: Optional[List[int]] = None
) -> float:
    """
    Compute diversity: pairwise endpoint distance for samples under same context.
    
    Args:
        samples: [N, T, P, 2] - generated samples
        context_groups: [N] - context group indices (samples with same context have same index)
        
    Returns:
        Average pairwise distance
    """
    # Get final positions
    final_positions = samples[:, -1, :, :2]  # [N, P, 2]
    
    if context_groups is not None:
        # Compute diversity within each context group
        unique_groups = torch.unique(torch.tensor(context_groups))
        diversities = []
        
        for group in unique_groups:
            group_mask = torch.tensor(context_groups) == group
            group_samples = final_positions[group_mask]  # [N_g, P, 2]
            
            if len(group_samples) < 2:
                continue
            
            # Flatten to [N_g, P*2] for pairwise distance
            group_flat = group_samples.reshape(len(group_samples), -1)
            
            # Compute pairwise distances
            dists = torch.cdist(group_flat, group_flat)  # [N_g, N_g]
            
            # Average distance (excluding diagonal)
            mask = ~torch.eye(len(group_samples), dtype=torch.bool)
            diversity = dists[mask].mean()
            diversities.append(diversity.item())
        
        return np.mean(diversities) if diversities else 0.0
    else:
        # Compute overall diversity
        samples_flat = final_positions.reshape(len(samples), -1)  # [N, P*2]
        dists = torch.cdist(samples_flat, samples_flat)  # [N, N]
        mask = ~torch.eye(len(samples), dtype=torch.bool)
        diversity = dists[mask].mean().item()
        return diversity


def compute_speed_distribution_distance(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor
) -> float:
    """
    Compute Wasserstein distance between predicted and real speed distributions.
    
    Args:
        predicted: [B, T, P, F] - predicted trajectories (with speed as 3rd feature)
        ground_truth: [B, T, P, F] - ground truth trajectories
        
    Returns:
        Wasserstein distance
    """
    if predicted.shape[-1] < 3 or ground_truth.shape[-1] < 3:
        return 0.0
    
    pred_speeds = predicted[:, :, :, 2].cpu().numpy().flatten()
    gt_speeds = ground_truth[:, :, :, 2].cpu().numpy().flatten()
    
    # Remove invalid speeds
    pred_speeds = pred_speeds[pred_speeds >= 0]
    gt_speeds = gt_speeds[gt_speeds >= 0]
    
    if len(pred_speeds) == 0 or len(gt_speeds) == 0:
        return 0.0
    
    # Compute Wasserstein distance
    wd = wasserstein_distance(pred_speeds, gt_speeds)
    return wd


def compute_collision_rate(trajectories: torch.Tensor, collision_threshold: float = 1.0) -> float:
    """
    Compute collision rate: fraction of timesteps with player collisions.
    
    Args:
        trajectories: [B, T, P, 2] - trajectories (x, y positions)
        collision_threshold: Distance threshold for collision (yards)
        
    Returns:
        Collision rate (0-1)
    """
    if trajectories.shape[-1] < 2:
        return 0.0
    
    positions = trajectories[:, :, :, :2]  # [B, T, P, 2]
    B, T, P, _ = positions.shape
    
    collisions = 0
    total_pairs = 0
    
    for b in range(B):
        for t in range(T):
            pos = positions[b, t]  # [P, 2]
            
            # Compute pairwise distances
            dists = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)  # [P, P]
            
            # Count collisions (distance < threshold, excluding self)
            mask = ~torch.eye(P, dtype=torch.bool, device=dists.device)
            collisions += (dists[mask] < collision_threshold).sum().item()
            total_pairs += mask.sum().item()
    
    collision_rate = collisions / total_pairs if total_pairs > 0 else 0.0
    return collision_rate


def compute_all_metrics(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    field_bounds: List[float] = [0, 120, 0, 53.3],
    speed_cap: float = 12.0
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Returns:
        Dict with metric names and values
    """
    metrics = {
        'ade': compute_ade(predicted, ground_truth, mask),
        'fde': compute_fde(predicted, ground_truth, mask),
        'validity_rate': compute_validity_rate(predicted, field_bounds, speed_cap),
        'collision_rate': compute_collision_rate(predicted),
        'speed_dist': compute_speed_distribution_distance(predicted, ground_truth)
    }
    
    return metrics

