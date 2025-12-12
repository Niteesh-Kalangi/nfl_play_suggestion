"""
Role mapping and player ordering for consistent tensor representation.

Maintains stable order of 22 players: 11 offense + 11 defense.
"""
from typing import List, Dict, Optional
import numpy as np


# Stable role order for offense (first 11 players)
OFFENSE_ROLE_ORDER = [
    'QB', 'RB', 'WR1', 'WR2', 'WR3', 'TE1', 'LT', 'LG', 'C', 'RG', 'RT'
]

# Defense roles (next 11 players) - generic positions
DEFENSE_ROLE_ORDER = [
    'CB1', 'CB2', 'S1', 'S2', 'LB1', 'LB2', 'LB3', 'DE1', 'DT', 'DE2', 'NB'
]

# Full role order (22 players)
FULL_ROLE_ORDER = OFFENSE_ROLE_ORDER + DEFENSE_ROLE_ORDER


def get_role_mapping_from_anchors(
    anchors: Dict[str, tuple],
    actual_positions: np.ndarray,
    formation: str,
    personnel: str
) -> Dict[int, str]:
    """
    Map actual player positions to roles by matching to anchors.
    
    Args:
        anchors: Dict of role -> (x, y) from formation_anchors
        actual_positions: Array [P, 2] of actual (x, y) positions at t=0
        formation: Formation name
        personnel: Personnel string
        
    Returns:
        Dict mapping player index -> role name
    """
    if len(actual_positions) < 11:
        # Not enough players, use distance-to-ball ordering
        return get_role_mapping_by_distance(actual_positions)
    
    # Match offense players (first 11) to anchor roles
    mapping = {}
    offense_positions = actual_positions[:11]  # First 11 are offense
    
    # Get anchor positions for offense roles
    anchor_positions = []
    anchor_roles = []
    for role in OFFENSE_ROLE_ORDER:
        if role in anchors:
            anchor_positions.append(anchors[role])
            anchor_roles.append(role)
    
    if len(anchor_positions) == 0:
        # No anchors, use distance ordering
        return get_role_mapping_by_distance(actual_positions)
    
    anchor_positions = np.array(anchor_positions)
    
    # Greedy matching: assign each actual position to nearest unmatched anchor
    used_anchors = set()
    for i in range(len(offense_positions)):
        best_match = None
        best_dist = float('inf')
        
        for j, anchor_pos in enumerate(anchor_positions):
            if j in used_anchors:
                continue
            
            dist = np.linalg.norm(offense_positions[i] - anchor_pos)
            if dist < best_dist:
                best_dist = dist
                best_match = j
        
        if best_match is not None and best_dist < 10.0:  # Max 10 yards away
            mapping[i] = anchor_roles[best_match]
            used_anchors.add(best_match)
        else:
            # Fallback: use distance ordering
            mapping[i] = OFFENSE_ROLE_ORDER[i] if i < len(OFFENSE_ROLE_ORDER) else f'UNKNOWN{i}'
    
    # Defense players keep default order
    for i in range(11, min(22, len(actual_positions))):
        if i - 11 < len(DEFENSE_ROLE_ORDER):
            mapping[i] = DEFENSE_ROLE_ORDER[i - 11]
        else:
            mapping[i] = f'DEF{i-11}'
    
    return mapping


def get_role_mapping_by_distance(positions: np.ndarray) -> Dict[int, str]:
    """
    Fallback: assign roles based on distance from ball (assumed at LOS center).
    
    Args:
        positions: Array [P, 2] of (x, y) positions
        
    Returns:
        Dict mapping player index -> role name
    """
    if len(positions) < 11:
        # Not enough players
        mapping = {}
        for i in range(min(len(positions), len(OFFENSE_ROLE_ORDER))):
            mapping[i] = OFFENSE_ROLE_ORDER[i]
        return mapping
    
    # Find ball position (approximate: average of first few frames' center)
    # For now, use center of field at yardline 50
    ball_pos = np.array([50.0, 26.65])
    
    # Calculate distances
    distances = np.linalg.norm(positions - ball_pos, axis=1)
    
    # Sort by distance (closest first)
    sorted_indices = np.argsort(distances)
    
    mapping = {}
    # Assign offense roles to closest 11
    for idx, player_idx in enumerate(sorted_indices[:11]):
        if idx < len(OFFENSE_ROLE_ORDER):
            mapping[player_idx] = OFFENSE_ROLE_ORDER[idx]
        else:
            mapping[player_idx] = f'OFF{idx}'
    
    # Assign defense roles to remaining
    for idx, player_idx in enumerate(sorted_indices[11:22]):
        def_idx = idx
        if def_idx < len(DEFENSE_ROLE_ORDER):
            mapping[player_idx] = DEFENSE_ROLE_ORDER[def_idx]
        else:
            mapping[player_idx] = f'DEF{def_idx}'
    
    return mapping


def get_anchor_mask_for_offense(num_players: int = 22) -> np.ndarray:
    """
    Get boolean mask indicating which players are anchored at t=0.
    
    Args:
        num_players: Total number of players (default 22)
        
    Returns:
        Boolean array [P] where True means anchored (offense players)
    """
    mask = np.zeros(num_players, dtype=bool)
    mask[:11] = True  # First 11 are offense, anchored
    return mask

