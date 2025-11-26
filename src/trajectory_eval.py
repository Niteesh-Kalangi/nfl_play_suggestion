"""
Evaluation metrics for trajectory generation models.

Includes:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- Physical validity metrics (speed, acceleration bounds)
- Context adherence metrics
"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List


def compute_ade(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Average Displacement Error.
    
    ADE = mean over all frames and players of L2 distance between
    predicted and ground truth positions.
    
    Args:
        pred: Predicted trajectories (n_samples, n_frames, n_players, 2)
        target: Ground truth trajectories (n_samples, n_frames, n_players, 2)
        mask: Optional mask for valid frames (n_samples, n_frames)
        
    Returns:
        ADE value in yards
    """
    # Compute L2 distance per player per frame
    diff = pred - target  # (n_samples, n_frames, n_players, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (n_samples, n_frames, n_players)
    
    if mask is not None:
        # Expand mask to player dimension
        mask_expanded = mask[:, :, np.newaxis]  # (n_samples, n_frames, 1)
        dist = dist * mask_expanded
        ade = np.sum(dist) / np.sum(mask_expanded * np.ones_like(dist))
    else:
        ade = np.mean(dist)
    
    return float(ade)


def compute_fde(
    pred: np.ndarray,
    target: np.ndarray,
    final_frame_idx: Optional[np.ndarray] = None
) -> float:
    """
    Compute Final Displacement Error.
    
    FDE = mean L2 distance at the final frame.
    
    Args:
        pred: Predicted trajectories (n_samples, n_frames, n_players, 2)
        target: Ground truth trajectories (n_samples, n_frames, n_players, 2)
        final_frame_idx: Optional per-sample final frame index (n_samples,)
        
    Returns:
        FDE value in yards
    """
    n_samples = pred.shape[0]
    
    if final_frame_idx is not None:
        # Use specified final frames
        fde_values = []
        for i in range(n_samples):
            idx = int(final_frame_idx[i])
            diff = pred[i, idx] - target[i, idx]  # (n_players, 2)
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (n_players,)
            fde_values.append(np.mean(dist))
        fde = np.mean(fde_values)
    else:
        # Use last frame
        diff = pred[:, -1] - target[:, -1]  # (n_samples, n_players, 2)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (n_samples, n_players)
        fde = np.mean(dist)
    
    return float(fde)


def compute_speed(trajectories: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Compute instantaneous speed from trajectories.
    
    Args:
        trajectories: (n_samples, n_frames, n_players, 2)
        dt: Time step between frames (seconds)
        
    Returns:
        Speed array (n_samples, n_frames-1, n_players) in yards/second
    """
    # Compute displacement between consecutive frames
    displacement = np.diff(trajectories, axis=1)  # (n_samples, n_frames-1, n_players, 2)
    
    # Compute speed magnitude
    speed = np.sqrt(np.sum(displacement ** 2, axis=-1)) / dt  # yards/second
    
    return speed


def compute_acceleration(trajectories: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Compute instantaneous acceleration from trajectories.
    
    Args:
        trajectories: (n_samples, n_frames, n_players, 2)
        dt: Time step between frames (seconds)
        
    Returns:
        Acceleration array (n_samples, n_frames-2, n_players) in yards/second^2
    """
    # Compute velocity
    velocity = np.diff(trajectories, axis=1) / dt  # (n_samples, n_frames-1, n_players, 2)
    
    # Compute acceleration
    acceleration = np.diff(velocity, axis=1) / dt  # (n_samples, n_frames-2, n_players, 2)
    
    # Compute magnitude
    accel_mag = np.sqrt(np.sum(acceleration ** 2, axis=-1))
    
    return accel_mag


def compute_physical_validity(
    trajectories: np.ndarray,
    dt: float = 0.1,
    max_speed: float = 12.0,  # ~27 mph, elite NFL speed
    max_accel: float = 10.0,  # yards/s^2
    field_bounds: Tuple[float, float, float, float] = (0, 120, 0, 53.3)
) -> Dict[str, float]:
    """
    Compute physical validity metrics for generated trajectories.
    
    Args:
        trajectories: (n_samples, n_frames, n_players, 2)
        dt: Time step between frames
        max_speed: Maximum realistic speed (yards/second)
        max_accel: Maximum realistic acceleration (yards/second^2)
        field_bounds: (x_min, x_max, y_min, y_max) field boundaries
        
    Returns:
        Dict with validity metrics:
        - speed_violation_rate: Fraction of frames with unrealistic speed
        - accel_violation_rate: Fraction of frames with unrealistic acceleration
        - out_of_bounds_rate: Fraction of frames with players out of bounds
        - mean_speed: Average speed across all players/frames
        - max_observed_speed: Maximum observed speed
    """
    x_min, x_max, y_min, y_max = field_bounds
    
    # Compute speed
    speed = compute_speed(trajectories, dt)
    speed_violations = speed > max_speed
    speed_violation_rate = np.mean(speed_violations)
    
    # Compute acceleration
    accel = compute_acceleration(trajectories, dt)
    accel_violations = accel > max_accel
    accel_violation_rate = np.mean(accel_violations)
    
    # Check bounds
    x_coords = trajectories[..., 0]
    y_coords = trajectories[..., 1]
    out_of_bounds = (
        (x_coords < x_min) | (x_coords > x_max) |
        (y_coords < y_min) | (y_coords > y_max)
    )
    out_of_bounds_rate = np.mean(out_of_bounds)
    
    return {
        'speed_violation_rate': float(speed_violation_rate),
        'accel_violation_rate': float(accel_violation_rate),
        'out_of_bounds_rate': float(out_of_bounds_rate),
        'mean_speed': float(np.mean(speed)),
        'max_observed_speed': float(np.max(speed)),
        'mean_accel': float(np.mean(accel)),
        'max_observed_accel': float(np.max(accel))
    }


def compute_formation_coherence(
    trajectories: np.ndarray,
    initial_frame: int = 0
) -> Dict[str, float]:
    """
    Measure how well the formation structure is preserved.
    
    Args:
        trajectories: (n_samples, n_frames, n_players, 2)
        initial_frame: Frame to use as reference formation
        
    Returns:
        Dict with coherence metrics:
        - mean_spread_change: Average change in team spread
        - mean_centroid_displacement: Average centroid movement
    """
    n_samples, n_frames, n_players, _ = trajectories.shape
    
    # Compute team centroid per frame
    centroids = np.mean(trajectories, axis=2)  # (n_samples, n_frames, 2)
    
    # Compute spread (std of positions) per frame
    spread = np.std(trajectories, axis=2)  # (n_samples, n_frames, 2)
    spread_mag = np.sqrt(np.sum(spread ** 2, axis=-1))  # (n_samples, n_frames)
    
    # Initial values
    initial_spread = spread_mag[:, initial_frame]
    initial_centroid = centroids[:, initial_frame]
    
    # Compute changes
    spread_change = np.abs(spread_mag - initial_spread[:, np.newaxis])
    centroid_disp = np.sqrt(np.sum((centroids - initial_centroid[:, np.newaxis]) ** 2, axis=-1))
    
    return {
        'mean_spread_change': float(np.mean(spread_change)),
        'max_spread_change': float(np.max(spread_change)),
        'mean_centroid_displacement': float(np.mean(centroid_disp)),
        'final_centroid_displacement': float(np.mean(centroid_disp[:, -1]))
    }


def compute_accuracy_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    thresholds: list = [1.0, 2.0, 5.0]
) -> Dict[str, float]:
    """
    Compute accuracy-style metrics for trajectory prediction.
    
    Args:
        pred: Predicted trajectories (n_samples, n_frames, n_players, 2)
        target: Ground truth trajectories
        thresholds: Distance thresholds in yards for accuracy computation
        
    Returns:
        Dict with accuracy metrics at various thresholds
    """
    # Compute per-player, per-frame displacement
    diff = pred - target
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (n_samples, n_frames, n_players)
    
    metrics = {}
    
    for thresh in thresholds:
        # Player-level accuracy: % of player-frames within threshold
        player_acc = np.mean(dist < thresh)
        metrics[f'accuracy_within_{thresh}yd'] = float(player_acc)
        
        # Frame-level accuracy: % of frames where ALL players within threshold
        frame_within = np.all(dist < thresh, axis=-1)  # (n_samples, n_frames)
        frame_acc = np.mean(frame_within)
        metrics[f'frame_accuracy_within_{thresh}yd'] = float(frame_acc)
    
    # Direction accuracy: is predicted movement in same direction as ground truth?
    pred_velocity = np.diff(pred, axis=1)  # (n_samples, n_frames-1, n_players, 2)
    target_velocity = np.diff(target, axis=1)
    
    # Dot product > 0 means same general direction
    dot_product = np.sum(pred_velocity * target_velocity, axis=-1)
    direction_correct = dot_product > 0
    metrics['direction_accuracy'] = float(np.mean(direction_correct))
    
    return metrics


def compute_collision_rate(
    trajectories: np.ndarray,
    min_distance: float = 1.0  # Minimum distance between players (yards)
) -> float:
    """
    Compute rate of player collisions (unrealistically close positions).
    
    Args:
        trajectories: (n_samples, n_frames, n_players, 2)
        min_distance: Minimum realistic distance between players
        
    Returns:
        Collision rate (fraction of frame-pairs with collision)
    """
    n_samples, n_frames, n_players, _ = trajectories.shape
    
    collision_count = 0
    total_pairs = 0
    
    for sample_idx in range(n_samples):
        for frame_idx in range(n_frames):
            positions = trajectories[sample_idx, frame_idx]  # (n_players, 2)
            
            # Compute pairwise distances
            for i in range(n_players):
                for j in range(i + 1, n_players):
                    dist = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
                    total_pairs += 1
                    if dist < min_distance:
                        collision_count += 1
    
    return collision_count / total_pairs if total_pairs > 0 else 0.0


def evaluate_trajectory_generation(
    pred_trajectories: np.ndarray,
    target_trajectories: np.ndarray,
    dt: float = 0.1,
    denorm_stats: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of generated trajectories.
    
    Args:
        pred_trajectories: Generated trajectories (n_samples, n_frames, n_players, 2)
        target_trajectories: Ground truth trajectories
        dt: Time step between frames
        denorm_stats: Optional normalization stats for denormalization
        
    Returns:
        Dict with all evaluation metrics
    """
    # Denormalize if needed
    if denorm_stats is not None:
        pred = pred_trajectories * denorm_stats['traj_std'] + denorm_stats['traj_mean']
        target = target_trajectories * denorm_stats['traj_std'] + denorm_stats['traj_mean']
    else:
        pred = pred_trajectories
        target = target_trajectories
    
    metrics = {}
    
    # Displacement errors
    metrics['ade'] = compute_ade(pred, target)
    metrics['fde'] = compute_fde(pred, target)
    
    # Accuracy metrics
    acc_metrics = compute_accuracy_metrics(pred, target)
    metrics.update(acc_metrics)
    
    # Physical validity
    phys_metrics = compute_physical_validity(pred, dt)
    metrics.update({f'phys_{k}': v for k, v in phys_metrics.items()})
    
    # Formation coherence
    coherence_metrics = compute_formation_coherence(pred)
    metrics.update({f'coherence_{k}': v for k, v in coherence_metrics.items()})
    
    # Collision rate
    metrics['collision_rate'] = compute_collision_rate(pred)
    
    # Also compute metrics on ground truth for reference
    gt_phys = compute_physical_validity(target, dt)
    metrics['gt_mean_speed'] = gt_phys['mean_speed']
    metrics['gt_max_speed'] = gt_phys['max_observed_speed']
    
    return metrics


def print_evaluation_report(metrics: Dict[str, float], model_name: str = "Model"):
    """Print formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"Trajectory Evaluation Report: {model_name}")
    print(f"{'='*60}")
    
    print("\n📏 Displacement Errors:")
    print(f"  ADE: {metrics['ade']:.3f} yards")
    print(f"  FDE: {metrics['fde']:.3f} yards")
    
    print("\n🎯 Accuracy Metrics:")
    print(f"  Within 1 yard:  {metrics.get('accuracy_within_1.0yd', 0)*100:.2f}%")
    print(f"  Within 2 yards: {metrics.get('accuracy_within_2.0yd', 0)*100:.2f}%")
    print(f"  Within 5 yards: {metrics.get('accuracy_within_5.0yd', 0)*100:.2f}%")
    print(f"  Direction accuracy: {metrics.get('direction_accuracy', 0)*100:.2f}%")
    print(f"  Frame accuracy (all players within 2yd): {metrics.get('frame_accuracy_within_2.0yd', 0)*100:.2f}%")
    
    print("\n🏃 Physical Validity:")
    print(f"  Speed violation rate: {metrics['phys_speed_violation_rate']*100:.2f}%")
    print(f"  Acceleration violation rate: {metrics['phys_accel_violation_rate']*100:.2f}%")
    print(f"  Out of bounds rate: {metrics['phys_out_of_bounds_rate']*100:.2f}%")
    print(f"  Mean speed: {metrics['phys_mean_speed']:.2f} yards/s")
    print(f"  Max speed: {metrics['phys_max_observed_speed']:.2f} yards/s")
    
    print("\n🏈 Formation Coherence:")
    print(f"  Mean spread change: {metrics['coherence_mean_spread_change']:.3f} yards")
    print(f"  Final centroid displacement: {metrics['coherence_final_centroid_displacement']:.3f} yards")
    
    print("\n💥 Collision Rate:")
    print(f"  Player collision rate: {metrics['collision_rate']*100:.2f}%")
    
    print(f"\n{'='*60}\n")
