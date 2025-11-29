"""
Load actual player positions from players.csv for a specific play.
This is the most accurate way to label players.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Cache players_df
_players_df = None


def get_real_player_positions(
    raw_dir: Path,
    game_id: int,
    play_id: int
) -> Optional[List[str]]:
    """
    Load actual player positions for a specific play from players.csv.
    
    This joins tracking data (which has nflId) with players.csv (which has officialPosition)
    to get the true position labels.
    
    Args:
        raw_dir: Directory containing players.csv and week*.csv files
        game_id: Game ID
        play_id: Play ID
        
    Returns:
        List of 22 position labels [QB, RB, WR, ...] or None if not found
    """
    global _players_df
    
    # Load players.csv once
    if _players_df is None:
        players_file = raw_dir / 'players.csv'
        if not players_file.exists():
            return None
        _players_df = pd.read_csv(players_file)
    
    # Find the week file that contains this play
    for week_file in sorted(raw_dir.glob('week*.csv')):
        tracking = pd.read_csv(week_file)
        play_tracking = tracking[
            (tracking['gameId'] == game_id) & 
            (tracking['playId'] == play_id)
        ]
        
        if len(play_tracking) == 0:
            continue
        
        # Get possession team (need to join with plays.csv or infer)
        # Try to get from tracking data - offense is usually the team with the ball initially
        # Get unique teams (excluding 'football')
        teams = [t for t in play_tracking['team'].unique() if t != 'football']
        if len(teams) < 2:
            continue
        
        # Load plays.csv to get possession team
        plays_file = raw_dir / 'plays.csv'
        if plays_file.exists():
            plays_df = pd.read_csv(plays_file)
            play_info = plays_df[
                (plays_df['gameId'] == game_id) & 
                (plays_df['playId'] == play_id)
            ]
            if len(play_info) > 0:
                possession_team = play_info.iloc[0].get('possessionTeam', teams[0])
            else:
                possession_team = teams[0]
        else:
            possession_team = teams[0]
        
        # Get player IDs for offense and defense
        offense_ids = play_tracking[
            (play_tracking['team'] == possession_team) & 
            (play_tracking['team'] != 'football')
        ]['nflId'].dropna().unique()[:11]
        
        defense_ids = play_tracking[
            (play_tracking['team'] != possession_team) & 
            (play_tracking['team'] != 'football')
        ]['nflId'].dropna().unique()[:11]
        
        # Look up positions
        positions = []
        for player_id in list(offense_ids) + list(defense_ids):
            if pd.notna(player_id):
                player_row = _players_df[_players_df['nflId'] == player_id]
                if len(player_row) > 0:
                    pos = str(player_row.iloc[0].get('officialPosition', 'UNKNOWN')).strip().upper()
                    # Normalize
                    if pos in ['HB', 'FB']:
                        pos = 'RB'
                    elif pos in ['CB', 'S', 'FS', 'SS', 'DB']:
                        pos = 'DB'
                    positions.append(pos)
                else:
                    positions.append('UNKNOWN')
            else:
                positions.append('UNKNOWN')
        
        # Pad to 22
        while len(positions) < 22:
            positions.append('UNKNOWN')
        
        return positions[:22]
    
    return None

