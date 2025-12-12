"""
Evaluation metrics for play prediction and trajectory generation models.

Includes:
- Play-level metrics (yards, success prediction)
- Trajectory metrics (ADE, FDE, collision rate, speed distribution)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from scipy.stats import spearmanr, wasserstein_distance
from typing import Dict, Tuple, Optional, Union


# ============================================================================
# TRAJECTORY GENERATION METRICS
# ============================================================================

def compute_ade(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> float:
    """
    Compute Average Displacement Error (ADE).
    
    ADE measures the average L2 distance between predicted and ground truth
    positions across all timesteps.
    
    Args:
        pred: Predicted trajectories [batch, T, D] or [batch, T, P, 2]
        target: Ground truth trajectories, same shape as pred
        mask: Optional mask [batch, T] where 1 = valid timestep
        num_players: Number of players (if D is flattened)
        features_per_player: Features per player (default 4: x, y, s, a)
        
    Returns:
        Scalar ADE value
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    
    # Handle 4D input (already in [batch, T, P, 2] format)
    if pred.ndim == 4:
        # Already in player format, just compute position differences
        pos_diff = pred - target
        # Compute L2 distance per player per timestep
        distances = np.sqrt(np.sum(pos_diff ** 2, axis=-1))  # [batch, T, P]
        
        if mask is not None:
            mask = np.asarray(mask)
            # Expand mask for players
            mask_expanded = mask[:, :, np.newaxis]
            distances = distances * mask_expanded
            total = mask_expanded.sum()
        else:
            total = distances.size
        
        return float(distances.sum() / (total + 1e-8))
    
    # Handle 3D input (flattened format [batch, T, D])
    if pred.ndim == 3:
        batch_size, T, D = pred.shape
        
        # Infer num_players if not provided
        if num_players is None:
            num_players = D // features_per_player
        
        # Extract position features (x, y) for each player
        # Assuming format: [x1, y1, s1, a1, x2, y2, s2, a2, ...]
        pred_positions = []
        target_positions = []
        
        for p in range(num_players):
            base_idx = p * features_per_player
            if base_idx + 1 < D:
                pred_positions.append(pred[:, :, base_idx:base_idx+2])
                target_positions.append(target[:, :, base_idx:base_idx+2])
        
        if len(pred_positions) == 0:
            # Fallback: treat entire output as positions (pairs)
            pred_positions = [pred[:, :, i:i+2] for i in range(0, D-1, 2)]
            target_positions = [target[:, :, i:i+2] for i in range(0, D-1, 2)]
        
        # Stack and compute distances
        pred_pos = np.stack(pred_positions, axis=2)  # [batch, T, P, 2]
        target_pos = np.stack(target_positions, axis=2)
        
        pos_diff = pred_pos - target_pos
        distances = np.sqrt(np.sum(pos_diff ** 2, axis=-1))  # [batch, T, P]
        
        if mask is not None:
            mask = np.asarray(mask)
            mask_expanded = mask[:, :, np.newaxis]
            distances = distances * mask_expanded
            total = mask_expanded.sum() * num_players
        else:
            total = distances.size
        
        return float(distances.sum() / (total + 1e-8))
    
    raise ValueError(f"Expected 3D or 4D array, got shape {pred.shape}")


def compute_fde(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> float:
    """
    Compute Final Displacement Error (FDE).
    
    FDE measures the L2 distance between predicted and ground truth positions
    at the final timestep of each sequence.
    
    Args:
        pred: Predicted trajectories [batch, T, D] or [batch, T, P, 2]
        target: Ground truth trajectories, same shape as pred
        mask: Optional mask [batch, T] where 1 = valid timestep
        num_players: Number of players (if D is flattened)
        features_per_player: Features per player (default 4: x, y, s, a)
        
    Returns:
        Scalar FDE value
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    
    batch_size = pred.shape[0]
    
    # Determine final timestep for each sequence
    if mask is not None:
        mask = np.asarray(mask)
        # Find last valid timestep per sequence
        final_indices = []
        for b in range(batch_size):
            valid_steps = np.where(mask[b] > 0)[0]
            if len(valid_steps) > 0:
                final_indices.append(valid_steps[-1])
            else:
                final_indices.append(0)
        final_indices = np.array(final_indices)
    else:
        # Use last timestep
        final_indices = np.full(batch_size, pred.shape[1] - 1)
    
    # Handle 4D input
    if pred.ndim == 4:
        fde_sum = 0.0
        for b in range(batch_size):
            t = final_indices[b]
            pos_diff = pred[b, t] - target[b, t]  # [P, 2]
            distances = np.sqrt(np.sum(pos_diff ** 2, axis=-1))
            fde_sum += distances.mean()
        return float(fde_sum / batch_size)
    
    # Handle 3D input
    if pred.ndim == 3:
        D = pred.shape[-1]
        
        if num_players is None:
            num_players = D // features_per_player
        
        fde_sum = 0.0
        for b in range(batch_size):
            t = final_indices[b]
            
            # Extract positions
            distances = []
            for p in range(num_players):
                base_idx = p * features_per_player
                if base_idx + 1 < D:
                    pred_pos = pred[b, t, base_idx:base_idx+2]
                    target_pos = target[b, t, base_idx:base_idx+2]
                    dist = np.sqrt(np.sum((pred_pos - target_pos) ** 2))
                    distances.append(dist)
            
            if len(distances) > 0:
                fde_sum += np.mean(distances)
        
        return float(fde_sum / batch_size)
    
    raise ValueError(f"Expected 3D or 4D array, got shape {pred.shape}")


def compute_collision_rate(
    trajectories: np.ndarray,
    threshold: float = 1.0,
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> float:
    """
    Compute collision rate between players.
    
    A collision is defined as two players being within threshold distance.
    
    Args:
        trajectories: Generated trajectories [batch, T, D] or [batch, T, P, 2]
        threshold: Distance threshold for collision (in yards)
        num_players: Number of players
        features_per_player: Features per player
        
    Returns:
        Collision rate (fraction of timesteps with at least one collision)
    """
    trajectories = np.asarray(trajectories)
    
    # Convert to [batch, T, P, 2] format
    if trajectories.ndim == 3:
        batch_size, T, D = trajectories.shape
        if num_players is None:
            num_players = D // features_per_player
        
        positions = []
        for p in range(num_players):
            base_idx = p * features_per_player
            if base_idx + 1 < D:
                positions.append(trajectories[:, :, base_idx:base_idx+2])
        
        if len(positions) == 0:
            return 0.0
        
        trajectories = np.stack(positions, axis=2)  # [batch, T, P, 2]
    
    batch_size, T, P, _ = trajectories.shape
    
    collision_count = 0
    total_timesteps = 0
    
    for b in range(batch_size):
        for t in range(T):
            # Check all pairs of players
            has_collision = False
            for i in range(P):
                for j in range(i + 1, P):
                    pos_i = trajectories[b, t, i]
                    pos_j = trajectories[b, t, j]
                    dist = np.sqrt(np.sum((pos_i - pos_j) ** 2))
                    if dist < threshold:
                        has_collision = True
                        break
                if has_collision:
                    break
            
            if has_collision:
                collision_count += 1
            total_timesteps += 1
    
    return float(collision_count / (total_timesteps + 1e-8))


def compute_speed_distribution_distance(
    pred_trajectories: np.ndarray,
    target_trajectories: np.ndarray,
    fps: float = 10.0,
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> float:
    """
    Compute Wasserstein distance between speed distributions.
    
    Measures how well the generated trajectories match the speed distribution
    of real trajectories.
    
    Args:
        pred_trajectories: Generated trajectories
        target_trajectories: Ground truth trajectories
        fps: Frames per second of tracking data
        num_players: Number of players
        features_per_player: Features per player
        
    Returns:
        Wasserstein distance between speed distributions
    """
    def extract_speeds(trajectories):
        trajectories = np.asarray(trajectories)
        
        if trajectories.ndim == 3:
            batch_size, T, D = trajectories.shape
            if num_players is None:
                n_players = D // features_per_player
            else:
                n_players = num_players
            
            # Check if speed is directly available (index 2 for each player)
            speeds = []
            for p in range(n_players):
                speed_idx = p * features_per_player + 2  # s is at index 2
                if speed_idx < D:
                    speeds.append(trajectories[:, :, speed_idx].flatten())
            
            if len(speeds) > 0:
                return np.concatenate(speeds)
            
            # Fallback: compute from positions
            positions = []
            for p in range(n_players):
                base_idx = p * features_per_player
                if base_idx + 1 < D:
                    positions.append(trajectories[:, :, base_idx:base_idx+2])
            
            if len(positions) == 0:
                return np.array([])
            
            positions = np.stack(positions, axis=2)  # [batch, T, P, 2]
        else:
            positions = trajectories
        
        # Compute speeds from position differences
        pos_diff = np.diff(positions, axis=1)  # [batch, T-1, P, 2]
        speeds = np.sqrt(np.sum(pos_diff ** 2, axis=-1)) * fps  # yards per second
        return speeds.flatten()
    
    pred_speeds = extract_speeds(pred_trajectories)
    target_speeds = extract_speeds(target_trajectories)
    
    if len(pred_speeds) == 0 or len(target_speeds) == 0:
        return float('nan')
    
    return float(wasserstein_distance(pred_speeds, target_speeds))


def compute_accuracy_thresholds(
    pred: np.ndarray,
    target: np.ndarray,
    thresholds: list = [1.0, 2.0, 5.0],
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> Dict[str, float]:
    """
    Compute percentage of predictions within various distance thresholds.
    
    Args:
        pred: Predicted trajectories [batch, T, D]
        target: Ground truth trajectories
        thresholds: List of distance thresholds in yards
        num_players: Number of players
        features_per_player: Features per player
        
    Returns:
        Dictionary with accuracy at each threshold
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    
    if pred.ndim == 3:
        batch_size, T, D = pred.shape
        if num_players is None:
            num_players = D // features_per_player
        
        # Extract positions
        all_distances = []
        for p in range(num_players):
            base_idx = p * features_per_player
            if base_idx + 1 < D:
                pred_pos = pred[:, :, base_idx:base_idx+2]
                target_pos = target[:, :, base_idx:base_idx+2]
                dist = np.sqrt(np.sum((pred_pos - target_pos) ** 2, axis=-1))
                all_distances.append(dist.flatten())
        
        if len(all_distances) == 0:
            return {f'within_{t}yd': 0.0 for t in thresholds}
        
        distances = np.concatenate(all_distances)
    else:
        # Assume 4D: [batch, T, P, 2]
        pos_diff = pred - target
        distances = np.sqrt(np.sum(pos_diff ** 2, axis=-1)).flatten()
    
    results = {}
    for threshold in thresholds:
        results[f'within_{int(threshold)}yd'] = float(np.mean(distances <= threshold))
    
    return results


def compute_direction_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> float:
    """
    Compute direction accuracy - whether predicted movement direction matches ground truth.
    
    Args:
        pred: Predicted trajectories [batch, T, D]
        target: Ground truth trajectories
        num_players: Number of players
        features_per_player: Features per player
        
    Returns:
        Fraction of timesteps where movement direction is correct (within 90 degrees)
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    
    if pred.ndim != 3:
        return 0.0
    
    batch_size, T, D = pred.shape
    if num_players is None:
        num_players = D // features_per_player
    
    correct_directions = 0
    total_directions = 0
    
    for p in range(num_players):
        base_idx = p * features_per_player
        if base_idx + 1 < D:
            pred_pos = pred[:, :, base_idx:base_idx+2]
            target_pos = target[:, :, base_idx:base_idx+2]
            
            # Compute velocity vectors
            pred_vel = np.diff(pred_pos, axis=1)  # [batch, T-1, 2]
            target_vel = np.diff(target_pos, axis=1)
            
            # Compute dot product to check direction similarity
            for b in range(batch_size):
                for t in range(T - 1):
                    pv = pred_vel[b, t]
                    tv = target_vel[b, t]
                    
                    pv_norm = np.linalg.norm(pv)
                    tv_norm = np.linalg.norm(tv)
                    
                    if pv_norm > 0.1 and tv_norm > 0.1:  # Only count if moving
                        cos_angle = np.dot(pv, tv) / (pv_norm * tv_norm)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        if cos_angle > 0:  # Within 90 degrees
                            correct_directions += 1
                        total_directions += 1
    
    if total_directions == 0:
        return 0.0
    
    return float(correct_directions / total_directions)


def compute_frame_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 2.0,
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> float:
    """
    Compute frame accuracy - percentage of frames where ALL players are within threshold.
    
    Args:
        pred: Predicted trajectories [batch, T, D]
        target: Ground truth trajectories
        threshold: Distance threshold in yards
        num_players: Number of players
        features_per_player: Features per player
        
    Returns:
        Fraction of frames where all players are within threshold
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    
    if pred.ndim != 3:
        return 0.0
    
    batch_size, T, D = pred.shape
    if num_players is None:
        num_players = D // features_per_player
    
    accurate_frames = 0
    total_frames = batch_size * T
    
    for b in range(batch_size):
        for t in range(T):
            all_within = True
            for p in range(num_players):
                base_idx = p * features_per_player
                if base_idx + 1 < D:
                    pred_pos = pred[b, t, base_idx:base_idx+2]
                    target_pos = target[b, t, base_idx:base_idx+2]
                    dist = np.sqrt(np.sum((pred_pos - target_pos) ** 2))
                    if dist > threshold:
                        all_within = False
                        break
            if all_within:
                accurate_frames += 1
    
    return float(accurate_frames / total_frames)


def compute_physical_validity(
    pred: np.ndarray,
    fps: float = 10.0,
    max_speed: float = 12.0,  # ~12 yards/s is elite NFL speed
    max_accel: float = 10.0,  # yards/s^2
    field_length: float = 120.0,
    field_width: float = 53.3,
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> Dict[str, float]:
    """
    Compute physical validity metrics.
    
    Args:
        pred: Predicted trajectories [batch, T, D]
        fps: Frames per second
        max_speed: Maximum realistic speed in yards/second
        max_accel: Maximum realistic acceleration in yards/s^2
        field_length: Field length in yards
        field_width: Field width in yards
        num_players: Number of players
        features_per_player: Features per player
        
    Returns:
        Dictionary with physical validity metrics
    """
    pred = np.asarray(pred)
    
    if pred.ndim != 3:
        return {
            'speed_violation_rate': 0.0,
            'accel_violation_rate': 0.0,
            'out_of_bounds_rate': 0.0,
            'mean_speed': 0.0,
            'max_speed': 0.0
        }
    
    batch_size, T, D = pred.shape
    if num_players is None:
        num_players = D // features_per_player
    
    all_speeds = []
    speed_violations = 0
    accel_violations = 0
    out_of_bounds = 0
    total_positions = 0
    total_speed_samples = 0
    total_accel_samples = 0
    
    for p in range(num_players):
        base_idx = p * features_per_player
        if base_idx + 1 < D:
            positions = pred[:, :, base_idx:base_idx+2]  # [batch, T, 2]
            
            # Check out of bounds
            x = positions[:, :, 0]
            y = positions[:, :, 1]
            oob = (x < 0) | (x > field_length) | (y < 0) | (y > field_width)
            out_of_bounds += oob.sum()
            total_positions += x.size
            
            # Compute velocities
            vel = np.diff(positions, axis=1) * fps  # [batch, T-1, 2]
            speeds = np.sqrt(np.sum(vel ** 2, axis=-1))  # [batch, T-1]
            all_speeds.append(speeds.flatten())
            
            # Check speed violations
            speed_violations += (speeds > max_speed).sum()
            total_speed_samples += speeds.size
            
            # Compute accelerations
            accel = np.diff(vel, axis=1) * fps  # [batch, T-2, 2]
            accel_mag = np.sqrt(np.sum(accel ** 2, axis=-1))  # [batch, T-2]
            
            accel_violations += (accel_mag > max_accel).sum()
            total_accel_samples += accel_mag.size
    
    all_speeds = np.concatenate(all_speeds) if all_speeds else np.array([0.0])
    
    return {
        'speed_violation_rate': float(speed_violations / (total_speed_samples + 1e-8)),
        'accel_violation_rate': float(accel_violations / (total_accel_samples + 1e-8)),
        'out_of_bounds_rate': float(out_of_bounds / (total_positions + 1e-8)),
        'mean_speed': float(all_speeds.mean()),
        'max_speed': float(all_speeds.max())
    }


def compute_formation_coherence(
    pred: np.ndarray,
    target: np.ndarray,
    num_players: Optional[int] = None,
    features_per_player: int = 4
) -> Dict[str, float]:
    """
    Compute formation coherence metrics.
    
    Args:
        pred: Predicted trajectories [batch, T, D]
        target: Ground truth trajectories
        num_players: Number of players
        features_per_player: Features per player
        
    Returns:
        Dictionary with formation coherence metrics
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    
    if pred.ndim != 3:
        return {
            'mean_spread_change': 0.0,
            'final_centroid_displacement': 0.0
        }
    
    batch_size, T, D = pred.shape
    if num_players is None:
        num_players = D // features_per_player
    
    def compute_spread(positions):
        """Compute average pairwise distance between players."""
        if len(positions) < 2:
            return 0.0
        
        total_dist = 0.0
        count = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total_dist += np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
                count += 1
        return total_dist / count if count > 0 else 0.0
    
    def get_positions_at_t(traj, t):
        """Extract all player positions at timestep t."""
        positions = []
        for p in range(num_players):
            base_idx = p * features_per_player
            if base_idx + 1 < D:
                positions.append(traj[t, base_idx:base_idx+2])
        return np.array(positions)
    
    spread_changes = []
    centroid_displacements = []
    
    for b in range(batch_size):
        # Compare initial and final spread
        pred_pos_init = get_positions_at_t(pred[b], 0)
        pred_pos_final = get_positions_at_t(pred[b], T-1)
        target_pos_final = get_positions_at_t(target[b], T-1)
        
        pred_spread_init = compute_spread(pred_pos_init)
        pred_spread_final = compute_spread(pred_pos_final)
        target_spread_final = compute_spread(target_pos_final)
        
        # Compare spread change with target
        pred_spread_change = abs(pred_spread_final - pred_spread_init)
        target_spread_change = abs(target_spread_final - compute_spread(get_positions_at_t(target[b], 0)))
        spread_changes.append(abs(pred_spread_change - target_spread_change))
        
        # Compare final centroids
        pred_centroid = pred_pos_final.mean(axis=0)
        target_centroid = target_pos_final.mean(axis=0)
        centroid_displacements.append(np.sqrt(np.sum((pred_centroid - target_centroid) ** 2)))
    
    return {
        'mean_spread_change': float(np.mean(spread_changes)),
        'final_centroid_displacement': float(np.mean(centroid_displacements))
    }


def evaluate_trajectory_model(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_players: int = 11,
    features_per_player: int = 4,
    collision_threshold: float = 1.0,
    fps: float = 10.0
) -> Dict[str, float]:
    """
    Comprehensive trajectory model evaluation.
    
    Args:
        pred: Predicted trajectories
        target: Ground truth trajectories
        mask: Valid timestep mask
        num_players: Number of players
        features_per_player: Features per player
        collision_threshold: Distance threshold for collisions
        fps: Frames per second
        
    Returns:
        Dictionary of all trajectory metrics
    """
    metrics = {}
    
    # Basic displacement errors
    metrics['ade'] = compute_ade(pred, target, mask, num_players, features_per_player)
    metrics['fde'] = compute_fde(pred, target, mask, num_players, features_per_player)
    
    # Accuracy thresholds
    accuracy_metrics = compute_accuracy_thresholds(pred, target, [1.0, 2.0, 5.0], num_players, features_per_player)
    metrics.update(accuracy_metrics)
    
    # Direction accuracy
    metrics['direction_accuracy'] = compute_direction_accuracy(pred, target, num_players, features_per_player)
    
    # Frame accuracy
    metrics['frame_accuracy'] = compute_frame_accuracy(pred, target, 2.0, num_players, features_per_player)
    
    # Physical validity
    phys_metrics = compute_physical_validity(pred, fps, num_players=num_players, features_per_player=features_per_player)
    for k, v in phys_metrics.items():
        metrics[f'phys_{k}'] = v
    
    # Formation coherence
    form_metrics = compute_formation_coherence(pred, target, num_players, features_per_player)
    for k, v in form_metrics.items():
        metrics[f'form_{k}'] = v
    
    # Collision rate
    metrics['collision_rate'] = compute_collision_rate(
        pred, collision_threshold, num_players, features_per_player
    )
    
    # Speed distribution distance
    metrics['speed_dist'] = compute_speed_distribution_distance(
        pred, target, fps, num_players, features_per_player
    )
    
    return metrics


def print_trajectory_report(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print a nicely formatted trajectory evaluation report.
    
    Args:
        metrics: Dictionary of metrics from evaluate_trajectory_model
        model_name: Name of the model for the header
    """
    print("\n" + "=" * 60)
    print(f"Trajectory Evaluation Report: {model_name}")
    print("=" * 60)
    
    print("\n📏 Displacement Errors:")
    print(f"  ADE: {metrics.get('ade', 0):.3f} yards")
    print(f"  FDE: {metrics.get('fde', 0):.3f} yards")
    
    print("\n🎯 Accuracy Metrics:")
    print(f"  Within 1 yard:  {metrics.get('within_1yd', 0)*100:.2f}%")
    print(f"  Within 2 yards: {metrics.get('within_2yd', 0)*100:.2f}%")
    print(f"  Within 5 yards: {metrics.get('within_5yd', 0)*100:.2f}%")
    print(f"  Direction accuracy: {metrics.get('direction_accuracy', 0)*100:.2f}%")
    print(f"  Frame accuracy (all players within 2yd): {metrics.get('frame_accuracy', 0)*100:.2f}%")
    
    print("\n🏃 Physical Validity:")
    print(f"  Speed violation rate: {metrics.get('phys_speed_violation_rate', 0)*100:.2f}%")
    print(f"  Acceleration violation rate: {metrics.get('phys_accel_violation_rate', 0)*100:.2f}%")
    print(f"  Out of bounds rate: {metrics.get('phys_out_of_bounds_rate', 0)*100:.2f}%")
    print(f"  Mean speed: {metrics.get('phys_mean_speed', 0):.2f} yards/s")
    print(f"  Max speed: {metrics.get('phys_max_speed', 0):.2f} yards/s")
    
    print("\n🏈 Formation Coherence:")
    print(f"  Mean spread change: {metrics.get('form_mean_spread_change', 0):.3f} yards")
    print(f"  Final centroid displacement: {metrics.get('form_final_centroid_displacement', 0):.3f} yards")
    
    print("\n💥 Collision Rate:")
    print(f"  Player collision rate: {metrics.get('collision_rate', 0)*100:.2f}%")
    
    print("\n" + "=" * 60)


# ============================================================================
# PLAY-LEVEL PREDICTION METRICS (Auxiliary models)
# ============================================================================


def evaluate_yards(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate yards prediction metrics.
    
    Args:
        y_true: True yards gained
        y_pred: Predicted yards gained
        
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Spearman rank correlation
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p
    }


def evaluate_success(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate success classification metrics.
    
    Args:
        y_true: True success labels (0/1)
        y_pred_prob: Predicted success probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    brier = brier_score_loss(y_true, y_pred_prob)
    
    # AUC (only if both classes present)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_prob)
    else:
        auc = np.nan
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    return {
        'brier_score': brier,
        'auc': auc,
        'accuracy': accuracy,
        'threshold': threshold
    }


def evaluate_model(
    y_true_yards: np.ndarray,
    y_pred_yards: np.ndarray,
    y_true_success: Optional[np.ndarray] = None,
    y_pred_success: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true_yards: True yards
        y_pred_yards: Predicted yards
        y_true_success: True success labels (optional)
        y_pred_success: Predicted success probabilities (optional)
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Yards metrics
    yards_metrics = evaluate_yards(y_true_yards, y_pred_yards)
    metrics.update({f'yards_{k}': v for k, v in yards_metrics.items()})
    
    # Success metrics (if provided)
    if y_true_success is not None and y_pred_success is not None:
        success_metrics = evaluate_success(y_true_success, y_pred_success)
        metrics.update({f'success_{k}': v for k, v in success_metrics.items()})
    
    return metrics


def top_k_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    by_group: Optional[pd.Series] = None
) -> float:
    """
    Top-K precision: Are the top-K predictions actually good outcomes?
    
    Args:
        y_true: True outcomes
        y_pred: Predicted outcomes (used for ranking)
        k: Number of top predictions to consider
        by_group: Optional grouping (e.g., by situation bucket)
        
    Returns:
        Average true outcome for top-K predictions
    """
    if by_group is not None:
        # Compute per-group top-K
        df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'group': by_group})
        top_k_values = []
        for group, group_df in df.groupby('group'):
            top_k_indices = group_df.nlargest(min(k, len(group_df)), 'pred').index
            top_k_values.append(group_df.loc[top_k_indices, 'true'].mean())
        return np.mean(top_k_values)
    else:
        # Global top-K
        top_k_indices = np.argsort(y_pred)[-k:]
        return np.mean(y_true[top_k_indices])


def counterfactual_policy_evaluation(
    plays_eval: pd.DataFrame,
    behavior_policy,
    target_policy_yards: np.ndarray,
    behavior_policy_yards: np.ndarray
) -> Dict[str, float]:
    """
    Simple Counterfactual Policy Evaluation using Self-Normalized IPS.
    
    Args:
        plays_eval: Evaluation plays DataFrame
        behavior_policy: Fitted behavior policy (for propensity estimation)
        target_policy_yards: Yards predicted by target policy
        behavior_policy_yards: Yards predicted by behavior policy
        
    Returns:
        Dictionary with CPE metrics
    """
    # Use uniform propensity as simple baseline (can be improved with bucket frequencies)
    # For simplicity, assume behavior policy is uniform over play types
    # In practice, estimate from bucket frequencies
    
    # Simple implementation: assume behavior propensities from bucket frequencies
    plays_eval = plays_eval.copy()
    plays_eval['bucket_key'] = plays_eval.apply(behavior_policy._bucket_key, axis=1)
    
    # Get bucket counts for propensity estimation
    if behavior_policy.bucket_table is not None:
        bucket_counts = behavior_policy.bucket_table.set_index('bucket_key')['count']
        total_plays = bucket_counts.sum()
        props = plays_eval['bucket_key'].map(bucket_counts).fillna(1) / total_plays
        props = np.clip(props, 1e-6, 1.0)  # Avoid zero/overflow
    else:
        props = np.ones(len(plays_eval)) / len(plays_eval)
    
    # Self-Normalized IPS
    weights = 1.0 / props
    weights = weights / weights.sum() * len(weights)  # Normalize
    
    # Estimated value
    snips_value = np.sum(weights * target_policy_yards) / len(plays_eval)
    
    # Doubly-robust (simplified - would need Q-function estimates)
    # For now, just return SNIPS
    return {
        'snips_value': snips_value,
        'behavior_value': np.mean(behavior_policy_yards),
        'target_value': np.mean(target_policy_yards),
        'uplift': snips_value - np.mean(behavior_policy_yards)
    }

