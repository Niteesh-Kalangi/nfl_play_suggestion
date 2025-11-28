"""
Utilities for parsing personnel and assigning position labels.
"""
import re
from typing import List, Optional


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


def assign_positions_by_formation(
    personnel_str: str,
    initial_x: List[float],
    initial_y: List[float]
) -> List[str]:
    """
    Assign positions based on personnel and initial formation.
    
    Uses position on field to infer roles:
    - QB: typically furthest back (smallest x)
    - WR: typically widest (furthest from center line y=26.65)
    - OL: typically on line of scrimmage (largest x, near center)
    - RB: typically behind QB but ahead of some players
    - TE: typically next to OL but not as wide as WR
    """
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

