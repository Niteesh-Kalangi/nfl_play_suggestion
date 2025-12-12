"""
Unit tests for evaluation metrics.
"""
import pytest
import torch
import numpy as np

from src.football_diffusion.eval.metrics import (
    compute_ade,
    compute_fde,
    compute_validity_rate,
    compute_diversity,
    compute_all_metrics
)


def test_ade_synthetic():
    """Test ADE computation on synthetic data."""
    # Create simple trajectories
    B, T, P, F = 2, 10, 5, 2
    
    # Ground truth: straight line
    gt = torch.zeros(B, T, P, F)
    gt[:, :, :, 0] = torch.linspace(0, 10, T).unsqueeze(0).unsqueeze(-1).expand(B, T, P)
    
    # Predicted: same but with small offset
    pred = gt.clone()
    pred[:, :, :, 0] += 1.0  # 1 yard offset
    
    ade = compute_ade(pred, gt)
    
    # ADE should be approximately 1.0
    assert abs(ade - 1.0) < 0.1


def test_fde_synthetic():
    """Test FDE computation on synthetic data."""
    B, T, P, F = 2, 10, 5, 2
    
    # Ground truth
    gt = torch.zeros(B, T, P, F)
    gt[:, -1, :, 0] = 10.0  # Final position at x=10
    
    # Predicted: offset by 2 yards
    pred = gt.clone()
    pred[:, -1, :, 0] = 12.0
    
    fde = compute_fde(pred, gt)
    
    # FDE should be 2.0
    assert abs(fde - 2.0) < 0.1


def test_validity_rate():
    """Test validity rate computation."""
    B, T, P, F = 2, 10, 5, 3
    
    # Create valid trajectories (within bounds, speed < 12)
    valid_traj = torch.zeros(B, T, P, F)
    valid_traj[:, :, :, 0] = torch.uniform(10, 110)  # x in [10, 110]
    valid_traj[:, :, :, 1] = torch.uniform(5, 48)    # y in [5, 48.3]
    valid_traj[:, :, :, 2] = torch.uniform(0, 10)    # speed < 12
    
    validity = compute_validity_rate(
        valid_traj,
        field_bounds=[0, 120, 0, 53.3],
        speed_cap=12.0
    )
    
    # Should be close to 1.0 (all valid)
    assert validity > 0.95
    
    # Create invalid trajectories (out of bounds)
    invalid_traj = valid_traj.clone()
    invalid_traj[:, :, :, 0] = 150  # x out of bounds
    
    validity_invalid = compute_validity_rate(
        invalid_traj,
        field_bounds=[0, 120, 0, 53.3],
        speed_cap=12.0
    )
    
    # Should be 0.0 (all invalid)
    assert validity_invalid == 0.0


def test_diversity():
    """Test diversity computation."""
    N, T, P, F = 4, 10, 5, 2
    
    # Create diverse samples
    samples = torch.randn(N, T, P, F) * 10
    
    diversity = compute_diversity(samples)
    
    # Diversity should be positive
    assert diversity > 0
    
    # Identical samples should have zero diversity
    identical = samples[0:1].expand(N, T, P, F)
    diversity_zero = compute_diversity(identical)
    
    assert diversity_zero == 0.0


def test_metrics_on_simple_case():
    """Test all metrics on a simple case."""
    B, T, P, F = 1, 5, 3, 3
    
    # Simple trajectories
    gt = torch.zeros(B, T, P, F)
    gt[:, :, :, 0] = torch.linspace(0, 10, T).unsqueeze(-1).expand(B, T, P)
    gt[:, :, :, 1] = 25.0  # y = 25
    gt[:, :, :, 2] = 2.0   # speed = 2
    
    pred = gt.clone()
    pred[:, :, :, 0] += 0.5  # Small offset
    
    metrics = compute_all_metrics(
        pred, gt,
        field_bounds=[0, 120, 0, 53.3],
        speed_cap=12.0
    )
    
    # Check all metrics are computed
    assert 'ade' in metrics
    assert 'fde' in metrics
    assert 'validity_rate' in metrics
    assert 'collision_rate' in metrics
    assert 'speed_dist' in metrics
    
    # ADE should be around 0.5
    assert abs(metrics['ade'] - 0.5) < 0.1


if __name__ == '__main__':
    pytest.main([__file__])

