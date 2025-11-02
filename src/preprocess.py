"""
Preprocessing utilities for standardizing field coordinates and joining context.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def parse_game_clock(clock_str: str) -> int:
    """
    Convert game clock string (MM:SS) to total seconds.
    
    Args:
        clock_str: Clock string like "13:33"
        
    Returns:
        Total seconds remaining in quarter
    """
    try:
        parts = clock_str.split(':')
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        return 900  # Default to 15:00 if parsing fails
    except:
        return 900


def compute_yardline_100(plays: pd.DataFrame) -> pd.Series:
    """
    Compute distance to end zone (yardline_100) from yardlineNumber and yardlineSide.
    
    Args:
        plays: Plays DataFrame with yardlineNumber and yardlineSide
        
    Returns:
        Series with yardline_100 values (0-100)
    """
    yardline_100 = pd.Series(index=plays.index, dtype=float)
    
    for idx, row in plays.iterrows():
        yardline_num = row['yardlineNumber']
        side = row['yardlineSide']
        poss_team = row['possessionTeam']
        
        if pd.isna(yardline_num):
            yardline_100[idx] = 50.0  # Default to midfield
            continue
        
        # If on own side, yardline_100 = yardlineNumber
        # If on opponent side, yardline_100 = 100 - yardlineNumber
        if side == poss_team:
            yardline_100[idx] = float(yardline_num)
        else:
            yardline_100[idx] = 100.0 - float(yardline_num)
    
    return yardline_100


def standardize_and_join(
    plays: pd.DataFrame,
    games: pd.DataFrame,
    tracking: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Standardize field coordinates, join games context, and prepare tracking data.
    
    Args:
        plays: Filtered plays DataFrame
        games: Games DataFrame
        tracking: Optional tracking DataFrame
        
    Returns:
        Tuple of (plays_std, tracking_std)
        - plays_std: Plays with standardized features and game context
        - tracking_std: Tracking data with standardized coordinates (if provided)
    """
    plays_std = plays.copy()
    
    # Join game context
    plays_std = plays_std.merge(
        games[['gameId', 'week', 'homeTeamAbbr', 'visitorTeamAbbr']],
        on='gameId',
        how='left'
    )
    
    # Add yardline_100
    plays_std['yardline_100'] = compute_yardline_100(plays_std)
    
    # Parse clock to seconds
    plays_std['clock_seconds'] = plays_std['gameClock'].apply(parse_game_clock)
    
    # Compute score differential (offense - defense)
    # Need to determine if offense is home or away
    plays_std['is_home'] = plays_std['possessionTeam'] == plays_std['homeTeamAbbr']
    plays_std['score_diff'] = np.where(
        plays_std['is_home'],
        plays_std['preSnapHomeScore'] - plays_std['preSnapVisitorScore'],
        plays_std['preSnapVisitorScore'] - plays_std['preSnapHomeScore']
    )
    
    # Standardize tracking if provided
    tracking_std = None
    if tracking is not None:
        tracking_std = tracking.copy()
        
        # Join possession team from plays
        tracking_std = tracking_std.merge(
            plays_std[['gameId', 'playId', 'possessionTeam', 'homeTeamAbbr', 'visitorTeamAbbr']],
            on=['gameId', 'playId'],
            how='left'
        )
        
        # Standardize coordinates: always make offense move left→right
        # If playDirection is 'left', flip x and y, and adjust angles
        mask_left = tracking_std['playDirection'] == 'left'
        
        # Flip x coordinate
        tracking_std.loc[mask_left, 'x'] = 120 - tracking_std.loc[mask_left, 'x']
        
        # Flip y coordinate (field is 53.3 yards wide)
        tracking_std.loc[mask_left, 'y'] = 53.3 - tracking_std.loc[mask_left, 'y']
        
        # Flip angles (rotate 180 degrees)
        tracking_std.loc[mask_left, 'o'] = (tracking_std.loc[mask_left, 'o'] + 180) % 360
        tracking_std.loc[mask_left, 'dir'] = (tracking_std.loc[mask_left, 'dir'] + 180) % 360
        
        # Mark offense/defense
        tracking_std['is_offense'] = (
            (tracking_std['team'] == tracking_std['possessionTeam']) |
            (tracking_std['team'] == 'football')
        )
        
        print(f"Standardized {len(tracking_std)} tracking frames")
    
    print(f"Preprocessed {len(plays_std)} plays")
    
    return plays_std, tracking_std


def extract_presnap_frame(
    tracking_std: pd.DataFrame,
    plays_std: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract the pre-snap frame (ball_snap event) for each play.
    
    Args:
        tracking_std: Standardized tracking DataFrame
        plays_std: Standardized plays DataFrame
        
    Returns:
        DataFrame with one row per play containing pre-snap positions
    """
    # Find ball_snap frames
    snap_frames = tracking_std[tracking_std['event'] == 'ball_snap'].copy()
    
    if len(snap_frames) == 0:
        # Fallback: use frame 1 for each play
        print("Warning: No ball_snap events found, using frame 1")
        first_frames = tracking_std.groupby(['gameId', 'playId']).first().reset_index()
        snap_frames = first_frames
    
    # Get just the snap frame for each play
    presnap = snap_frames.groupby(['gameId', 'playId']).first().reset_index()
    
    return presnap

