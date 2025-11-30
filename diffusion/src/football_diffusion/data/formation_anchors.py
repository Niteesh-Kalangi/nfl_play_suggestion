"""
Formation anchor templates for initial frame (t=0) positioning.

Anchors are specified in field coordinates (yards) relative to the line of scrimmage
and hash mark position.
"""
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


def get_hash_y_position(hash_mark: str) -> float:
    """
    Get Y coordinate for hash mark position.
    
    Args:
        hash_mark: 'LEFT', 'MIDDLE', or 'RIGHT'
        
    Returns:
        Y coordinate in yards (field width is 53.3 yards)
    """
    hash_map = {
        'LEFT': 18.0,      # Left hash
        'MIDDLE': 26.65,   # Middle of field (53.3 / 2)
        'RIGHT': 35.3,     # Right hash
        'left': 18.0,
        'middle': 26.65,
        'right': 35.3
    }
    return hash_map.get(hash_mark.upper(), 26.65)  # Default to middle


def parse_personnel(personnel_str: str) -> Dict[str, int]:
    """
    Parse personnel string to get counts.
    
    Args:
        personnel_str: e.g., "1 RB, 1 TE, 3 WR"
        
    Returns:
        Dict with counts for RB, TE, WR
    """
    counts = {'RB': 0, 'TE': 0, 'WR': 0}
    if not personnel_str or pd.isna(personnel_str):
        return counts
    
    parts = str(personnel_str).upper().split(',')
    for part in parts:
        part = part.strip()
        for role in ['RB', 'TE', 'WR']:
            if role in part:
                try:
                    num = int(''.join(filter(str.isdigit, part)))
                    counts[role] = num
                except:
                    counts[role] = 1
                break
    
    return counts


def get_shotgun_anchors(personnel: Dict[str, int], yardline: float, hash_y: float) -> Dict[str, Tuple[float, float]]:
    """
    SHOTGUN formation anchors.
    
    QB: 4.5 yards behind LOS
    OL: Across LOS, centered at hash
    RB: Next to QB (slightly offset)
    WR/TE: Spread wide based on personnel
    """
    anchors = {}
    los_x = yardline
    
    # Offensive line (5 players across LOS)
    ol_spacing = 2.0  # yards between centers
    ol_y_start = hash_y - 2 * ol_spacing  # Center C at hash_y
    
    anchors['C'] = (los_x, hash_y)
    anchors['LG'] = (los_x, hash_y - ol_spacing)
    anchors['LT'] = (los_x, hash_y - 2 * ol_spacing)
    anchors['RG'] = (los_x, hash_y + ol_spacing)
    anchors['RT'] = (los_x, hash_y + 2 * ol_spacing)
    
    # QB in shotgun (4.5 yards behind LOS)
    anchors['QB'] = (los_x - 4.5, hash_y)
    
    # RB (next to QB, slightly offset) - typically 4-6 yards from QB
    if personnel['RB'] > 0:
        anchors['RB'] = (los_x - 4.5, hash_y - 4.0)  # Left of QB, typical shotgun spacing
    
    # WRs and TE based on personnel
    wr_count = personnel.get('WR', 0)
    te_count = personnel.get('TE', 0)
    
    # Wide receivers - spread outside
    if wr_count >= 1:
        anchors['WR1'] = (los_x, 2.0)  # Left sideline
    if wr_count >= 2:
        anchors['WR2'] = (los_x, 51.3)  # Right sideline
    if wr_count >= 3:
        anchors['WR3'] = (los_x, hash_y - 8.0)  # Slot left
    if wr_count >= 4:
        anchors['WR4'] = (los_x, hash_y + 8.0)  # Slot right
    
    # Tight ends - tight to LOS
    if te_count >= 1:
        anchors['TE1'] = (los_x, hash_y - 4.0)  # Left side
    if te_count >= 2:
        anchors['TE2'] = (los_x, hash_y + 4.0)  # Right side
    
    return anchors


def get_singleback_anchors(personnel: Dict[str, int], yardline: float, hash_y: float) -> Dict[str, Tuple[float, float]]:
    """SINGLEBACK formation - RB directly behind QB."""
    anchors = {}
    los_x = yardline
    
    # OL
    ol_spacing = 2.0
    anchors['C'] = (los_x, hash_y)
    anchors['LG'] = (los_x, hash_y - ol_spacing)
    anchors['LT'] = (los_x, hash_y - 2 * ol_spacing)
    anchors['RG'] = (los_x, hash_y + ol_spacing)
    anchors['RT'] = (los_x, hash_y + 2 * ol_spacing)
    
    # QB under center
    anchors['QB'] = (los_x - 1.5, hash_y)
    
    # RB directly behind QB
    if personnel['RB'] > 0:
        anchors['RB'] = (los_x - 6.0, hash_y)
    
    # WRs and TE
    wr_count = personnel.get('WR', 0)
    te_count = personnel.get('TE', 0)
    
    if wr_count >= 1:
        anchors['WR1'] = (los_x, 5.0)
    if wr_count >= 2:
        anchors['WR2'] = (los_x, 48.3)
    if wr_count >= 3:
        anchors['WR3'] = (los_x, hash_y - 7.0)
    
    if te_count >= 1:
        anchors['TE1'] = (los_x, hash_y - 3.5)
    if te_count >= 2:
        anchors['TE2'] = (los_x, hash_y + 3.5)
    
    return anchors


def get_i_form_anchors(personnel: Dict[str, int], yardline: float, hash_y: float) -> Dict[str, Tuple[float, float]]:
    """I_FORM (21 personnel) - RB in I-formation behind QB."""
    anchors = {}
    los_x = yardline
    
    # OL
    ol_spacing = 2.0
    anchors['C'] = (los_x, hash_y)
    anchors['LG'] = (los_x, hash_y - ol_spacing)
    anchors['LT'] = (los_x, hash_y - 2 * ol_spacing)
    anchors['RG'] = (los_x, hash_y + ol_spacing)
    anchors['RT'] = (los_x, hash_y + 2 * ol_spacing)
    
    # QB under center
    anchors['QB'] = (los_x - 1.5, hash_y)
    
    # RB in I-formation
    if personnel['RB'] >= 1:
        anchors['RB'] = (los_x - 7.0, hash_y)
    if personnel['RB'] >= 2:
        anchors['FB'] = (los_x - 5.0, hash_y)  # Fullback
    
    # WRs and TE
    wr_count = personnel.get('WR', 0)
    te_count = personnel.get('TE', 0)
    
    if wr_count >= 1:
        anchors['WR1'] = (los_x, 8.0)
    if wr_count >= 2:
        anchors['WR2'] = (los_x, 45.3)
    
    if te_count >= 1:
        anchors['TE1'] = (los_x, hash_y - 3.5)
    if te_count >= 2:
        anchors['TE2'] = (los_x, hash_y + 3.5)
    
    return anchors


def get_empty_anchors(personnel: Dict[str, int], yardline: float, hash_y: float) -> Dict[str, Tuple[float, float]]:
    """EMPTY formation - QB in shotgun, no RB."""
    anchors = {}
    los_x = yardline
    
    # OL
    ol_spacing = 2.0
    anchors['C'] = (los_x, hash_y)
    anchors['LG'] = (los_x, hash_y - ol_spacing)
    anchors['LT'] = (los_x, hash_y - 2 * ol_spacing)
    anchors['RG'] = (los_x, hash_y + ol_spacing)
    anchors['RT'] = (los_x, hash_y + 2 * ol_spacing)
    
    # QB in shotgun
    anchors['QB'] = (los_x - 4.5, hash_y)
    
    # All WRs/TE spread wide
    wr_count = personnel.get('WR', 0)
    te_count = personnel.get('TE', 0)
    
    # Spread receivers wide
    positions = []
    if wr_count >= 1:
        positions.append((los_x, 3.0))  # Left
    if wr_count >= 2:
        positions.append((los_x, 50.3))  # Right
    if wr_count >= 3:
        positions.append((los_x, hash_y - 9.0))  # Slot left
    if wr_count >= 4:
        positions.append((los_x, hash_y + 9.0))  # Slot right
    if wr_count >= 5:
        positions.append((los_x, hash_y - 5.0))  # Inside left
    
    if te_count >= 1:
        positions.append((los_x, hash_y - 4.5))
    if te_count >= 2:
        positions.append((los_x, hash_y + 4.5))
    
    for i, (x, y) in enumerate(positions[:5]):  # Max 5 skill positions
        if i < wr_count:
            anchors[f'WR{i+1}'] = (x, y)
        else:
            anchors[f'TE{i-wr_count+1}'] = (x, y)
    
    return anchors


def get_trips_anchors(personnel: Dict[str, int], yardline: float, hash_y: float) -> Dict[str, Tuple[float, float]]:
    """TRIPS formation - 3 receivers to one side."""
    anchors = {}
    los_x = yardline
    
    # OL
    ol_spacing = 2.0
    anchors['C'] = (los_x, hash_y)
    anchors['LG'] = (los_x, hash_y - ol_spacing)
    anchors['LT'] = (los_x, hash_y - 2 * ol_spacing)
    anchors['RG'] = (los_x, hash_y + ol_spacing)
    anchors['RT'] = (los_x, hash_y + 2 * ol_spacing)
    
    # QB in shotgun
    anchors['QB'] = (los_x - 4.5, hash_y)
    
    # RB
    if personnel['RB'] > 0:
        anchors['RB'] = (los_x - 4.5, hash_y - 2.0)
    
    # Trips to right side
    wr_count = personnel.get('WR', 0)
    te_count = personnel.get('TE', 0)
    
    if wr_count >= 1:
        anchors['WR1'] = (los_x, 5.0)  # Left single
    if wr_count >= 2:
        anchors['WR2'] = (los_x, 45.0)  # Right outside
    if wr_count >= 3:
        anchors['WR3'] = (los_x, 38.0)  # Right middle
    if wr_count >= 4:
        anchors['WR4'] = (los_x, 31.0)  # Right inside
    
    if te_count >= 1:
        anchors['TE1'] = (los_x, hash_y - 3.5)
    
    return anchors


def get_bunch_anchors(personnel: Dict[str, int], yardline: float, hash_y: float) -> Dict[str, Tuple[float, float]]:
    """BUNCH formation - receivers bunched together."""
    anchors = {}
    los_x = yardline
    
    # OL
    ol_spacing = 2.0
    anchors['C'] = (los_x, hash_y)
    anchors['LG'] = (los_x, hash_y - ol_spacing)
    anchors['LT'] = (los_x, hash_y - 2 * ol_spacing)
    anchors['RG'] = (los_x, hash_y + ol_spacing)
    anchors['RT'] = (los_x, hash_y + 2 * ol_spacing)
    
    # QB in shotgun
    anchors['QB'] = (los_x - 4.5, hash_y)
    
    # RB
    if personnel['RB'] > 0:
        anchors['RB'] = (los_x - 4.5, hash_y - 2.0)
    
    # Bunch to right side
    wr_count = personnel.get('WR', 0)
    te_count = personnel.get('TE', 0)
    
    if wr_count >= 1:
        anchors['WR1'] = (los_x, 10.0)  # Left
    if wr_count >= 2:
        anchors['WR2'] = (los_x, 40.0)  # Bunch outside
    if wr_count >= 3:
        anchors['WR3'] = (los_x, 36.0)  # Bunch middle
    if wr_count >= 4:
        anchors['WR4'] = (los_x, 32.0)  # Bunch inside
    
    if te_count >= 1:
        anchors['TE1'] = (los_x, hash_y - 3.5)
    
    return anchors


def get_12_personnel_anchors(personnel: Dict[str, int], yardline: float, hash_y: float) -> Dict[str, Tuple[float, float]]:
    """12 PERSONNEL - 1 RB, 2 TE."""
    anchors = {}
    los_x = yardline
    
    # OL
    ol_spacing = 2.0
    anchors['C'] = (los_x, hash_y)
    anchors['LG'] = (los_x, hash_y - ol_spacing)
    anchors['LT'] = (los_x, hash_y - 2 * ol_spacing)
    anchors['RG'] = (los_x, hash_y + ol_spacing)
    anchors['RT'] = (los_x, hash_y + 2 * ol_spacing)
    
    # QB under center
    anchors['QB'] = (los_x - 1.5, hash_y)
    
    # RB
    if personnel['RB'] > 0:
        anchors['RB'] = (los_x - 6.0, hash_y)
    
    # 2 TEs
    if personnel['TE'] >= 1:
        anchors['TE1'] = (los_x, hash_y - 3.5)
    if personnel['TE'] >= 2:
        anchors['TE2'] = (los_x, hash_y + 3.5)
    
    # 2 WRs
    wr_count = personnel.get('WR', 0)
    if wr_count >= 1:
        anchors['WR1'] = (los_x, 8.0)
    if wr_count >= 2:
        anchors['WR2'] = (los_x, 45.3)
    
    return anchors


def get_anchors(
    formation: str,
    personnel: str,
    yardline: float,
    hash_mark: str = "MIDDLE",
    direction: str = "right"
) -> Dict[str, Tuple[float, float]]:
    """
    Get formation anchors for t=0 frame.
    
    Args:
        formation: Formation name (SHOTGUN, SINGLEBACK, I_FORM, EMPTY, etc.)
        personnel: Personnel string (e.g., "1 RB, 1 TE, 3 WR")
        yardline: Yardline in field coordinates (0-100)
        hash_mark: 'LEFT', 'MIDDLE', or 'RIGHT'
        direction: Offense direction ('right' or 'left') - always right after normalization
        
    Returns:
        Dict mapping role names to (x, y) coordinates in field space
    """
    # Get hash Y position
    hash_y = get_hash_y_position(hash_mark)
    
    # Parse personnel
    personnel_dict = parse_personnel(personnel)
    
    # Select formation
    formation_upper = formation.upper()
    
    if 'SHOTGUN' in formation_upper or 'SHOT' in formation_upper:
        anchors = get_shotgun_anchors(personnel_dict, yardline, hash_y)
    elif 'SINGLEBACK' in formation_upper or 'SINGLE' in formation_upper:
        anchors = get_singleback_anchors(personnel_dict, yardline, hash_y)
    elif 'I_FORM' in formation_upper or 'I_FORMATION' in formation_upper or formation_upper == 'I':
        anchors = get_i_form_anchors(personnel_dict, yardline, hash_y)
    elif 'EMPTY' in formation_upper:
        anchors = get_empty_anchors(personnel_dict, yardline, hash_y)
    elif 'TRIPS' in formation_upper:
        anchors = get_trips_anchors(personnel_dict, yardline, hash_y)
    elif 'BUNCH' in formation_upper:
        anchors = get_bunch_anchors(personnel_dict, yardline, hash_y)
    elif '12' in formation_upper or '21' in formation_upper:
        if '21' in personnel.upper() or personnel_dict['RB'] == 2:
            anchors = get_i_form_anchors(personnel_dict, yardline, hash_y)
        else:
            anchors = get_12_personnel_anchors(personnel_dict, yardline, hash_y)
    else:
        # Default to SHOTGUN
        anchors = get_shotgun_anchors(personnel_dict, yardline, hash_y)
    
    # Validate anchors are within field bounds
    for role, (x, y) in anchors.items():
        if x < 0 or x > 120 or y < 0 or y > 53.3:
            # Clamp to field bounds
            x = max(0, min(120, x))
            y = max(0, min(53.3, y))
            anchors[role] = (x, y)
    
    return anchors


def anchors_to_tensor(
    anchors: Dict[str, Tuple[float, float]],
    role_order: List[str],
    num_features: int = 3
) -> np.ndarray:
    """
    Convert anchor dict to tensor [P, F] where P matches role_order.
    
    Args:
        anchors: Dict of role -> (x, y)
        role_order: List of role names in order [QB, RB, WR1, ...]
        num_features: Number of features (x, y, s)
        
    Returns:
        Array [P, F] with coordinates, padded with zeros if role missing
    """
    tensor = np.zeros((len(role_order), num_features), dtype=np.float32)
    
    for i, role in enumerate(role_order):
        if role in anchors:
            x, y = anchors[role]
            tensor[i, 0] = x
            tensor[i, 1] = y
            tensor[i, 2] = 0.0  # Speed = 0 at t=0
    
    return tensor


# Default role order for offense (first 11 players)
OFFENSE_ROLE_ORDER = [
    'QB', 'RB', 'WR1', 'WR2', 'WR3', 'TE1', 'LT', 'LG', 'C', 'RG', 'RT'
]

# Defense roles (next 11 players) - generic for now
DEFENSE_ROLE_ORDER = [
    'CB1', 'CB2', 'S1', 'S2', 'LB1', 'LB2', 'LB3', 'DE1', 'DT', 'DE2', 'NB'
]

