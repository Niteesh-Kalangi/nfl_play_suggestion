"""
Utilities for parsing personnel and assigning position labels.
"""
import re
from typing import List, Optional
import numpy as np


def parse_personnel(personnel_str: str) -> dict:
    """
    Parse personnel string like "1 RB, 1 TE, 3 WR" into counts.
    
    Args:
        personnel_str: String like "1 RB, 1 TE, 3 WR"
        
    Returns:
        Dict with counts: {'RB': 1, 'TE': 1, 'WR': 3, 'OL': 5}
    """
    personnel_str = str(personnel_str).upper()
    
    # Initialize counts
    counts = {'QB': 1, 'RB': 0, 'WR': 0, 'TE': 0, 'OL': 5}  # Default: QB + 5 OL
    
    # Parse pattern: "X RB", "X TE", "X WR", etc.
    patterns = {
        r'(\d+)\s*RB': 'RB',
        r'(\d+)\s*WR': 'WR',
        r'(\d+)\s*TE': 'TE',
        r'(\d+)\s*HB': 'RB',  # Halfback = RB
        r'(\d+)\s*FB': 'RB',  # Fullback = RB
    }
    
    for pattern, pos in patterns.items():
        match = re.search(pattern, personnel_str)
        if match:
            counts[pos] = int(match.group(1))
    
    # Calculate OL count: 11 total - QB - skill positions
    skill_count = counts['RB'] + counts['WR'] + counts['TE']
    counts['OL'] = 11 - 1 - skill_count  # 11 total - 1 QB - skill players
    
    return counts


def generate_position_labels(personnel_str: str, sort_by_x: Optional[List[float]] = None) -> List[str]:
    """
    Generate position labels for 11 offensive players based on personnel.
    
    Args:
        personnel_str: String like "1 RB, 1 TE, 3 WR"
        sort_by_x: Optional list of x-coordinates to sort players (left to right)
        
    Returns:
        List of 11 position labels like ['QB', 'WR', 'WR', 'WR', 'RB', 'TE', 'OL', 'OL', 'OL', 'OL', 'OL']
    """
    counts = parse_personnel(personnel_str)
    
    # Build list of positions
    labels = ['QB']  # QB always first
    
    # Add WRs (typically split out wide)
    labels.extend(['WR'] * counts['WR'])
    
    # Add RB
    labels.extend(['RB'] * counts['RB'])
    
    # Add TE
    labels.extend(['TE'] * counts['TE'])
    
    # Add OL (offensive linemen)
    labels.extend(['OL'] * counts['OL'])
    
    # If we have x-coordinates, we can sort players to match typical formations
    # For now, assume personnel order reflects formation order
    
    # Ensure we have exactly 11 positions
    while len(labels) < 11:
        labels.append('OL')
    labels = labels[:11]
    
    return labels


def assign_positions_by_motion(
    personnel_str: str,
    trajectory: 'np.ndarray'  # [T, P, 2] or [T, P, F]
) -> List[str]:
    """
    Assign positions based on motion patterns and initial formation.
    
    Uses motion characteristics:
    - QB: Stays in pocket (low initial movement, moves backwards/sideways)
    - RB: Fast acceleration, runs forward routes
    - WR: Already identified by width, fast movement
    - TE: Moderate movement, often starts near OL then moves out
    - OL: Minimal movement, stays on line
    
    Args:
        personnel_str: String like "1 RB, 1 TE, 3 WR"
        trajectory: Full trajectory [T, P, 2] or [T, P, F] (positions over time)
    """
    import numpy as np
    
    counts = parse_personnel(personnel_str)
    
    # Extract x, y positions
    if trajectory.shape[-1] >= 2:
        positions = trajectory[:, :, :2]  # [T, P, 2]
    else:
        positions = trajectory
    
    T, P, _ = positions.shape
    if P < 11:
        return generate_position_labels(personnel_str)
    
    # Only analyze first 11 players (offense)
    off_positions = positions[:, :11, :]  # [T, 11, 2]
    
    # Get initial positions (first few frames average)
    initial_frames = min(5, T)
    initial_x = np.mean(off_positions[:initial_frames, :, 0], axis=0)  # [11]
    initial_y = np.mean(off_positions[:initial_frames, :, 1], axis=0)  # [11]
    
    # Calculate motion metrics for each player
    motion_metrics = []
    center_y = 26.65
    
    for p in range(11):
        player_pos = off_positions[:, p, :]  # [T, 2]
        
        # 1. Total displacement (how far they moved)
        displacement = np.linalg.norm(player_pos[-1] - player_pos[0])
        
        # 2. Forward progress (x-direction movement)
        forward_progress = player_pos[-1, 0] - player_pos[0, 0]
        
        # 3. Lateral movement (y-direction variation)
        lateral_movement = np.std(player_pos[:, 1])
        
        # 4. Average speed
        if T > 1:
            velocities = np.diff(player_pos, axis=0)  # [T-1, 2]
            speeds = np.linalg.norm(velocities, axis=1)
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
        else:
            avg_speed = 0
            max_speed = 0
        
        # 5. Initial x position (back to front)
        initial_x_pos = initial_x[p]
        
        # 6. Distance from center line
        dist_from_center = abs(initial_y[p] - center_y)
        
        motion_metrics.append({
            'idx': p,
            'displacement': displacement,
            'forward_progress': forward_progress,
            'lateral_movement': lateral_movement,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'initial_x': initial_x_pos,
            'dist_from_center': dist_from_center
        })
    
    labels = [''] * 11
    
    # 1. QB: Furthest back (smallest x) AND minimal forward movement initially
    qb_candidates = sorted(motion_metrics, key=lambda m: (m['initial_x'], m['forward_progress']))
    qb_idx = qb_candidates[0]['idx']
    labels[qb_idx] = 'QB'
    
    # 2. WRs: Widest players (furthest from center) - already correctly identified
    wr_candidates = [(m['idx'], m['dist_from_center']) 
                     for m in motion_metrics if labels[m['idx']] == '']
    wr_candidates.sort(key=lambda x: x[1], reverse=True)
    
    wr_count = counts['WR']
    for i in range(min(wr_count, len(wr_candidates))):
        labels[wr_candidates[i][0]] = 'WR'
    
    # 3. OL: On line (largest x), minimal movement, near center
    ol_candidates = [(m['idx'], m['initial_x'], m['dist_from_center'], m['displacement'])
                     for m in motion_metrics if labels[m['idx']] == '']
    # Sort by: front (large x), centered (small y dist), minimal movement
    ol_candidates.sort(key=lambda x: (-x[1], x[2], x[3]))
    
    ol_count = counts['OL']
    for i in range(min(ol_count, len(ol_candidates))):
        labels[ol_candidates[i][0]] = 'OL'
    
    # 4. RB: High speed/acceleration, good forward progress, behind OL
    rb_candidates = [(m['idx'], m['max_speed'], m['forward_progress'], -m['initial_x'])
                     for m in motion_metrics if labels[m['idx']] == '']
    # Sort by: high speed, good forward progress, back position
    rb_candidates.sort(key=lambda x: (-x[1], -x[2], x[3]))
    
    rb_count = counts['RB']
    for i in range(min(rb_count, len(rb_candidates))):
        labels[rb_candidates[i][0]] = 'RB'
    
    # 5. TE: Remaining players (moderate movement, near OL but moves)
    te_count = counts['TE']
    te_candidates = [m['idx'] for m in motion_metrics if labels[m['idx']] == '']
    for i in range(min(te_count, len(te_candidates))):
        labels[te_candidates[i]] = 'TE'
    
    # Fill any remaining with OL
    for i in range(11):
        if labels[i] == '':
            labels[i] = 'OL'
    
    return labels[:11]


def assign_positions_by_formation(
    personnel_str: str,
    initial_x: List[float],
    initial_y: List[float],
    trajectory: Optional[np.ndarray] = None
) -> List[str]:
    """
    Assign positions based on personnel and initial formation.
    If trajectory is provided, uses motion-based assignment instead.
    """
    # If trajectory is available, use motion-based assignment (more accurate)
    if trajectory is not None:
        return assign_positions_by_motion(personnel_str, trajectory)
    
    # Otherwise, use formation-based assignment
    counts = parse_personnel(personnel_str)
    
    if len(initial_x) < 11 or len(initial_y) < 11:
        # Fallback to simple generation
        return generate_position_labels(personnel_str)
    
    # Only use first 11 players (offense)
    x_pos = initial_x[:11]
    y_pos = initial_y[:11]
    
    labels = [''] * 11
    center_y = 26.65
    
    # 1. QB is typically the furthest back (smallest x)
    qb_idx = min(range(11), key=lambda i: x_pos[i])
    labels[qb_idx] = 'QB'
    
    # 2. WRs are typically the widest (furthest from center line)
    # Calculate y-distance from center for all non-QB players
    y_distances = [abs(y_pos[i] - center_y) for i in range(11)]
    wr_candidates = [(i, y_distances[i]) for i in range(11) if labels[i] == '']
    wr_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Assign WRs (widest players)
    wr_count = counts['WR']
    for i in range(min(wr_count, len(wr_candidates))):
        labels[wr_candidates[i][0]] = 'WR'
    
    # 3. OL are typically on the line of scrimmage (largest x, near center)
    # and form a group in the middle
    ol_candidates = [(i, x_pos[i], abs(y_pos[i] - center_y)) 
                     for i in range(11) if labels[i] == '']
    # Sort by x (front) then by y-distance from center (most centered first)
    ol_candidates.sort(key=lambda x: (-x[1], x[2]))
    
    ol_count = counts['OL']
    for i in range(min(ol_count, len(ol_candidates))):
        labels[ol_candidates[i][0]] = 'OL'
    
    # 4. TE is typically near OL but slightly wider
    te_candidates = [(i, abs(y_pos[i] - center_y)) 
                     for i in range(11) if labels[i] == '']
    te_candidates.sort(key=lambda x: x[1])  # Closest to center first
    
    te_count = counts['TE']
    for i in range(min(te_count, len(te_candidates))):
        labels[te_candidates[i][0]] = 'TE'
    
    # 5. RB is whatever is left (typically behind QB)
    rb_count = counts['RB']
    rb_candidates = [i for i in range(11) if labels[i] == '']
    for i in range(min(rb_count, len(rb_candidates))):
        labels[rb_candidates[i]] = 'RB'
    
    # Fill any remaining with OL
    for i in range(11):
        if labels[i] == '':
            labels[i] = 'OL'
    
    return labels[:11]

