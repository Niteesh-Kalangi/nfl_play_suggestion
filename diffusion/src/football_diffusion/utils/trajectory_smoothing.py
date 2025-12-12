"""
Post-processing utilities for smoothing generated trajectories.

These functions work immediately without retraining to make movements more gradual and realistic.
"""
import numpy as np
import torch
from typing import Union, Optional


def smooth_trajectory(
    trajectory: Union[np.ndarray, torch.Tensor],
    max_velocity_yards_per_frame: float = 1.0,
    smooth_frames: int = 5,
    preserve_t0: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Smooth a trajectory by constraining velocity and applying temporal smoothing.
    
    This ensures gradual, realistic movement (~1 yard per frame) and prevents
    erratic direction changes.
    
    Args:
        trajectory: [T, P, F] array where F >= 2 (x, y, ...)
        max_velocity_yards_per_frame: Maximum allowed velocity (yards per frame)
        smooth_frames: Number of frames to apply smoothing window to
        preserve_t0: If True, don't modify t=0 (formation snap position)
        
    Returns:
        Smoothed trajectory with same shape and type as input
    """
    is_torch = isinstance(trajectory, torch.Tensor)
    if is_torch:
        device = trajectory.device
        traj = trajectory.detach().cpu().numpy().copy()
    else:
        traj = trajectory.copy()
    
    T, P, F = traj.shape
    if F < 2:
        return trajectory  # Need at least x, y
    
    # Extract positions (x, y)
    positions = traj[:, :, :2]  # [T, P, 2]
    
    # Start from t=0 (preserve formation) or t=1
    start_idx = 1 if preserve_t0 else 0
    
    # Smooth trajectory frame by frame
    for t in range(start_idx, T):
        for p in range(P):
            # Current position
            current_pos = positions[t, p, :]
            
            # Previous position
            prev_pos = positions[t-1, p, :]
            
            # Compute velocity
            velocity = current_pos - prev_pos
            velocity_magnitude = np.linalg.norm(velocity)
            
            # Constrain velocity to max (yards per frame)
            if velocity_magnitude > max_velocity_yards_per_frame:
                # Scale down velocity to max
                velocity_unit = velocity / (velocity_magnitude + 1e-8)
                velocity_clamped = velocity_unit * max_velocity_yards_per_frame
                # Update position
                positions[t, p, :] = prev_pos + velocity_clamped
            
            # Apply temporal smoothing (moving average) to reduce jitter
            if t >= smooth_frames and smooth_frames > 1:
                # Use moving average of recent positions
                window_start = max(0, t - smooth_frames)
                recent_positions = positions[window_start:t+1, p, :]
                
                # Weighted average (more weight to recent frames)
                weights = np.linspace(0.5, 1.0, len(recent_positions))
                weights = weights / weights.sum()
                smoothed_pos = np.average(recent_positions, axis=0, weights=weights)
                
                # Blend: 70% smoothed, 30% original (preserve some variation)
                positions[t, p, :] = 0.7 * smoothed_pos + 0.3 * positions[t, p, :]
    
    # Update trajectory with smoothed positions
    traj[:, :, :2] = positions
    
    # Recompute speed (3rd feature) if available
    if F >= 3:
        for t in range(start_idx, T):
            for p in range(P):
                if t > 0:
                    # Compute speed from smoothed positions
                    vel = positions[t, p, :] - positions[t-1, p, :]
                    speed = np.linalg.norm(vel)
                    traj[t, p, 2] = speed
                else:
                    traj[t, p, 2] = 0.0  # Speed = 0 at snap
    
    if is_torch:
        return torch.from_numpy(traj).to(device)
    return traj


def enforce_direction_consistency(
    trajectory: Union[np.ndarray, torch.Tensor],
    min_angle_change: float = 45.0,  # degrees
    preserve_t0: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Enforce direction consistency to prevent sudden reversals.
    
    Prevents players from turning around suddenly or making erratic direction changes.
    
    Args:
        trajectory: [T, P, F] array
        min_angle_change: Minimum angle change threshold (degrees) to consider reversal
        preserve_t0: If True, don't modify t=0
        
    Returns:
        Trajectory with consistent direction
    """
    is_torch = isinstance(trajectory, torch.Tensor)
    if is_torch:
        device = trajectory.device
        traj = trajectory.detach().cpu().numpy().copy()
    else:
        traj = trajectory.copy()
    
    T, P, F = traj.shape
    if F < 2 or T < 3:
        return trajectory
    
    positions = traj[:, :, :2]
    start_idx = 2 if preserve_t0 else 1  # Need at least 2 previous frames
    
    for t in range(start_idx, T):
        for p in range(P):
            # Get last 3 positions
            pos_prev2 = positions[t-2, p, :]
            pos_prev1 = positions[t-1, p, :]
            pos_current = positions[t, p, :]
            
            # Compute velocities
            vel1 = pos_prev1 - pos_prev2  # Velocity from t-2 to t-1
            vel2 = pos_current - pos_prev1  # Velocity from t-1 to t
            
            # Compute angle between velocities
            if np.linalg.norm(vel1) > 1e-6 and np.linalg.norm(vel2) > 1e-6:
                # Normalize
                vel1_unit = vel1 / np.linalg.norm(vel1)
                vel2_unit = vel2 / np.linalg.norm(vel2)
                
                # Dot product gives cos(angle)
                cos_angle = np.clip(np.dot(vel1_unit, vel2_unit), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                
                # If angle change is too large (sudden reversal), smooth it
                if angle_deg > min_angle_change:
                    # Blend velocity: 80% previous direction, 20% new direction
                    vel2_smoothed = 0.8 * vel1_unit * np.linalg.norm(vel2) + 0.2 * vel2
                    positions[t, p, :] = pos_prev1 + vel2_smoothed
    
    traj[:, :, :2] = positions
    
    if is_torch:
        return torch.from_numpy(traj).to(device)
    return traj


def apply_comprehensive_smoothing(
    trajectory: Union[np.ndarray, torch.Tensor],
    max_velocity: float = 1.0,
    preserve_t0: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply comprehensive smoothing to make trajectory gradual and realistic.
    
    Combines velocity constraints and direction consistency.
    
    Args:
        trajectory: [T, P, F] trajectory
        max_velocity: Max yards per frame (~1.0 for gradual movement)
        preserve_t0: Preserve t=0 positions (formation anchors)
        
    Returns:
        Smoothed trajectory
    """
    # Step 1: Constrain velocity (prevent drastic movements)
    traj = smooth_trajectory(
        trajectory,
        max_velocity_yards_per_frame=max_velocity,
        smooth_frames=3,
        preserve_t0=preserve_t0
    )
    
    # Step 2: Enforce direction consistency (prevent reversals)
    traj = enforce_direction_consistency(
        traj,
        min_angle_change=60.0,  # Prevent sharp turns > 60 degrees
        preserve_t0=preserve_t0
    )
    
    # Step 3: Final smoothing pass
    traj = smooth_trajectory(
        traj,
        max_velocity_yards_per_frame=max_velocity,
        smooth_frames=2,
        preserve_t0=preserve_t0
    )
    
    return traj

