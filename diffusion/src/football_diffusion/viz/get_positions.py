"""
Load actual player positions from players.csv for accurate labeling.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np


# Cache players_df to avoid reloading
_players_df_cache = None


def load_players_df(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Load players.csv with caching."""
    global _players_df_cache
    if _players_df_cache is not None:
        return _players_df_cache
    
    players_file = raw_dir / 'players.csv'
    if players_file.exists():
        _players_df_cache = pd.read_csv(players_file)
        return _players_df_cache
    return None


def get_player_positions_from_tracking(
    tracking_data: pd.DataFrame,
    players_df: pd.DataFrame,
    possession_team: str
) -> List[str]:
    """
    Get actual player positions from tracking data by looking up in players.csv.
    
    Args:
        tracking_data: Tracking data for the play (with nflId, team)
        players_df: DataFrame from players.csv with nflId and officialPosition
        possession_team: The offensive team
        
    Returns:
        List of 22 position labels (11 offense + 11 defense)
    """
    # Get unique player IDs
    offense_ids = tracking_data[
        (tracking_data['team'] == possession_team) & 
        (tracking_data['team'] != 'football')
    ]['nflId'].dropna().unique()[:11]
    
    defense_ids = tracking_data[
        (tracking_data['team'] != possession_team) & 
        (tracking_data['team'] != 'football')
    ]['nflId'].dropna().unique()[:11]
    
    all_ids = list(offense_ids) + list(defense_ids)
    
    # Look up positions
    positions = []
    for player_id in all_ids:
        if pd.notna(player_id):
            player_row = players_df[players_df['nflId'] == player_id]
            if len(player_row) > 0:
                pos = str(player_row.iloc[0].get('officialPosition', 'UNKNOWN')).strip().upper()
                # Map to standard positions
                pos = normalize_position(pos)
                positions.append(pos)
            else:
                positions.append('UNKNOWN')
        else:
            positions.append('UNKNOWN')
    
    # Pad to 22 if needed
    while len(positions) < 22:
        positions.append('UNKNOWN')
    
    return positions[:22]


def normalize_position(pos: str) -> str:
    """Normalize position labels to standard set."""
    pos = str(pos).strip().upper()
    
    # Offensive positions
    if pos in ['QB']:
        return 'QB'
    elif pos in ['RB', 'HB', 'FB']:
        return 'RB'
    elif pos in ['WR']:
        return 'WR'
    elif pos in ['TE']:
        return 'TE'
    elif pos in ['T', 'G', 'C', 'OL', 'OT', 'OG']:
        return 'OL'
    
    # Defensive positions
    elif pos in ['CB', 'S', 'FS', 'SS', 'DB']:
        return 'DB'
    elif pos in ['DT', 'DE', 'NT', 'DL']:
        return 'DL'
    elif pos in ['OLB', 'ILB', 'MLB', 'LB']:
        return 'LB'
    else:
        return 'UNKNOWN'

