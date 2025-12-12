"""
Play quality evaluation utilities.
"""
import numpy as np
import torch
from typing import Dict, List, Optional
from ..eval.metrics import (
    compute_validity_rate, compute_diversity
)


def evaluate_play_quality(
    trajectory: np.ndarray,
    field_bounds: List[float] = [0, 120, 0, 53.3],
    speed_cap: float = 12.0
) -> Dict[str, float]:
    """
    Evaluate quality of a single generated play.
    
    Args:
        trajectory: [T, P, F] - trajectory where F includes [x, y, s]
        field_bounds: [x_min, x_max, y_min, y_max]
        speed_cap: Maximum allowed speed in yd/s
        
    Returns:
        Dict with quality metrics
    """
    # Convert to torch tensor for metrics
    traj_tensor = torch.FloatTensor(trajectory).unsqueeze(0)  # [1, T, P, F]
    
    # Extract positions and speeds
    T, P, F = trajectory.shape
    positions = trajectory[:, :, :2]  # [T, P, 2]
    speeds = trajectory[:, :, 2] if F >= 3 else None  # [T, P]
    
    # Validity checks
    x_min, x_max, y_min, y_max = field_bounds
    in_bounds = (
        (positions[:, :, 0] >= x_min) & (positions[:, :, 0] <= x_max) &
        (positions[:, :, 1] >= y_min) & (positions[:, :, 1] <= y_max)
    )
    bounds_validity = in_bounds.mean()
    
    # Speed validity
    speed_validity = 1.0
    if speeds is not None:
        under_speed_cap = (speeds <= speed_cap) & (speeds >= 0)
        speed_validity = under_speed_cap.mean()
    
    # Overall validity
    overall_validity = bounds_validity * speed_validity
    
    # Smoothness: check acceleration (change in speed)
    smoothness = 1.0
    if speeds is not None and T > 1:
        acceleration = np.abs(np.diff(speeds, axis=0))  # [T-1, P]
        # High acceleration = less smooth (penalize)
        max_accel = 10.0  # yd/s² (reasonable max)
        smoothness = 1.0 - min(1.0, acceleration.mean() / max_accel)
    
    # Movement: check if players actually moved
    movement = 0.0
    if T > 1:
        total_displacement = np.linalg.norm(
            positions[-1] - positions[0], axis=-1
        )  # [P]
        movement = total_displacement.mean()  # Average displacement per player
    
    # Play direction consistency (offense should move right on average)
    # Assuming first 11 players are offense
    forward_progress = 0.0
    if P >= 11:
        offense_x_delta = positions[-1, :11, 0] - positions[0, :11, 0]
        forward_progress = offense_x_delta.mean()  # Positive = forward
        # Normalize by expected forward progress (should be positive for most plays)
        direction_consistency = max(0, min(1, (forward_progress + 20) / 40))
    else:
        direction_consistency = 0.5
    
    return {
        'bounds_validity': float(bounds_validity),
        'speed_validity': float(speed_validity),
        'overall_validity': float(overall_validity),
        'smoothness': float(smoothness),
        'movement': float(movement),
        'forward_progress': float(forward_progress) if P >= 11 else 0.0,
        'direction_consistency': float(direction_consistency),
        'quality_score': float(overall_validity * smoothness * direction_consistency)
    }


def print_play_quality_report(quality_metrics: Dict[str, float]):
    """
    Print a formatted quality report for a play.
    
    Args:
        quality_metrics: Dict from evaluate_play_quality()
    """
    print("=" * 50)
    print("PLAY QUALITY REPORT")
    print("=" * 50)
    print(f"Overall Quality Score: {quality_metrics['quality_score']:.3f}")
    print()
    print("Validity Checks:")
    print(f"  ✓ Field Bounds:      {quality_metrics['bounds_validity']*100:.1f}% valid")
    print(f"  ✓ Speed Cap:         {quality_metrics['speed_validity']*100:.1f}% valid")
    print(f"  ✓ Overall Validity:  {quality_metrics['overall_validity']*100:.1f}% valid")
    print()
    print("Realism Metrics:")
    print(f"  ✓ Smoothness:        {quality_metrics['smoothness']:.3f} (1.0 = perfectly smooth)")
    print(f"  ✓ Movement:          {quality_metrics['movement']:.2f} yards avg displacement")
    print(f"  ✓ Forward Progress:  {quality_metrics['forward_progress']:.2f} yards")
    print(f"  ✓ Direction:         {quality_metrics['direction_consistency']:.3f} (1.0 = consistently forward)")
    print()
    
    # Quality assessment
    score = quality_metrics['quality_score']
    if score >= 0.8:
        assessment = "🟢 EXCELLENT"
    elif score >= 0.6:
        assessment = "🟡 GOOD"
    elif score >= 0.4:
        assessment = "🟠 FAIR"
    else:
        assessment = "🔴 POOR"
    
    print(f"Assessment: {assessment}")
    print("=" * 50)

