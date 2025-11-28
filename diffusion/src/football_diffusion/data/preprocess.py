"""
Preprocessing pipeline to convert raw NFL tracking CSVs to cached pickle files.

Output: Each play → tensor X [T, P, F] where T=60 frames, P=22 players, F=[x, y, s]
        Context vector c with categorical and continuous features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm


def parse_game_clock(clock_str: str) -> float:
    """Parse game clock string (MM:SS) to total seconds."""
    if pd.isna(clock_str):
        return 0.0
    try:
        parts = str(clock_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0.0
    except:
        return 0.0


def compute_yardline_100(plays_df: pd.DataFrame) -> pd.Series:
    """Convert yardline to distance from endzone (0-100 scale)."""
    yardline_100 = plays_df['absoluteYardlineNumber'].copy()
    # If absoluteYardlineNumber is NaN, compute from yardlineSide and yardlineNumber
    mask_na = yardline_100.isna()
    if mask_na.any():
        # For plays on offense side, yardline_100 = 100 - yardlineNumber
        # For plays on defense side, yardline_100 = yardlineNumber
        yardline_100[mask_na] = plays_df.loc[mask_na, 'yardlineNumber']
    return yardline_100.clip(0, 100)


def derive_situation(down: int, yards_to_go: float) -> str:
    """Derive situation category: 'short', 'medium', 'long'."""
    if yards_to_go <= 3:
        return 'short'
    elif yards_to_go <= 7:
        return 'medium'
    else:
        return 'long'


def standardize_coordinates(
    tracking: pd.DataFrame,
    plays: pd.DataFrame
) -> pd.DataFrame:
    """
    Standardize tracking coordinates so offense always moves right.
    
    Args:
        tracking: Tracking DataFrame with x, y columns
        plays: Plays DataFrame with possessionTeam
        
    Returns:
        Standardized tracking DataFrame
    """
    tracking_std = tracking.copy()
    
    # Join with plays to get possession team
    tracking_std = tracking_std.merge(
        plays[['gameId', 'playId', 'possessionTeam']],
        on=['gameId', 'playId'],
        how='left'
    )
    
    # Determine play direction (need to check actual data)
    if 'playDirection' not in tracking_std.columns:
        # Default: assume right, will be corrected by checking first frame
        tracking_std['playDirection'] = 'right'
    
    # Flip coordinates for left-moving plays
    mask_left = tracking_std['playDirection'] == 'left'
    
    # Flip x (field is 120 yards)
    tracking_std.loc[mask_left, 'x'] = 120 - tracking_std.loc[mask_left, 'x']
    
    # Flip y (field is 53.3 yards)
    tracking_std.loc[mask_left, 'y'] = 53.3 - tracking_std.loc[mask_left, 'y']
    
    # Flip angles if present
    if 'o' in tracking_std.columns:
        tracking_std.loc[mask_left, 'o'] = (tracking_std.loc[mask_left, 'o'] + 180) % 360
    if 'dir' in tracking_std.columns:
        tracking_std.loc[mask_left, 'dir'] = (tracking_std.loc[mask_left, 'dir'] + 180) % 360
    
    return tracking_std


def extract_play_frames(
    tracking: pd.DataFrame,
    game_id: int,
    play_id: int,
    max_frames: int = 60
) -> pd.DataFrame:
    """
    Extract frames from ball_snap to play_end, pad/truncate to max_frames.
    
    Args:
        tracking: Tracking DataFrame filtered to one play
        game_id: Game ID
        play_id: Play ID
        max_frames: Maximum number of frames to return
        
    Returns:
        DataFrame with frames from ball_snap to play_end (padded/truncated)
    """
    play_tracking = tracking[
        (tracking['gameId'] == game_id) &
        (tracking['playId'] == play_id)
    ].copy()
    
    if len(play_tracking) == 0:
        return pd.DataFrame()
    
    # Find snap frame
    snap_frames = play_tracking[play_tracking['event'].str.contains('snap', case=False, na=False)]
    if len(snap_frames) == 0:
        # Use first frame if no snap event
        start_frame = play_tracking['frameId'].min()
    else:
        start_frame = snap_frames['frameId'].min()
    
    # Find end frame
    end_events = ['tackle', 'touchdown', 'out_of_bounds', 'fumble', 'pass_outcome']
    end_frames = play_tracking[
        play_tracking['event'].str.contains('|'.join(end_events), case=False, na=False)
    ]
    if len(end_frames) == 0:
        # Use last frame if no end event
        end_frame = play_tracking['frameId'].max()
    else:
        end_frame = end_frames['frameId'].min()
    
    # Extract frames in range
    play_tracking = play_tracking[
        (play_tracking['frameId'] >= start_frame) &
        (play_tracking['frameId'] <= end_frame)
    ].copy()
    
    # Get unique frames and sort
    frames = sorted(play_tracking['frameId'].unique())
    
    if len(frames) == 0:
        return pd.DataFrame()
    
    # Truncate or pad to max_frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        # Pad by repeating last frame
        last_frame = frames[-1]
        frames.extend([last_frame] * (max_frames - len(frames)))
    
    # Filter to selected frames
    play_tracking = play_tracking[play_tracking['frameId'].isin(frames)].copy()
    
    return play_tracking


def extract_player_tensor(
    play_tracking: pd.DataFrame,
    play_info: pd.Series,
    num_players: int = 22,
    features: List[str] = ['x', 'y', 's']
) -> np.ndarray:
    """
    Extract player positions as tensor [T, P, F].
    
    Args:
        play_tracking: Tracking DataFrame for one play (standardized, filtered frames)
        play_info: Play metadata Series
        num_players: Number of players (22 = 11 offense + 11 defense)
        features: Features to extract [x, y, s]
        
    Returns:
        Tensor of shape [T, P, F] where T=num_frames, P=num_players, F=len(features)
    """
    frames = sorted(play_tracking['frameId'].unique())
    T = len(frames)
    F = len(features)
    
    # Initialize tensor
    tensor = np.zeros((T, num_players, F), dtype=np.float32)
    
    # Get possession team
    possession_team = play_info.get('possessionTeam', '')
    
    # Separate offense and defense
    play_tracking = play_tracking.copy()
    play_tracking['is_offense'] = (
        (play_tracking['team'] == possession_team) & 
        (play_tracking['team'] != 'football')
    )
    play_tracking['is_defense'] = (
        (play_tracking['team'] != possession_team) & 
        (play_tracking['team'] != 'football')
    )
    
    # Get player IDs: first 11 offense, then 11 defense
    offense_players = play_tracking[
        play_tracking['is_offense']
    ]['nflId'].dropna().unique()[:11]
    defense_players = play_tracking[
        play_tracking['is_defense']
    ]['nflId'].dropna().unique()[:11]
    
    all_players = list(offense_players) + list(defense_players)
    
    # Pad if needed
    while len(all_players) < num_players:
        all_players.append(None)
    all_players = all_players[:num_players]
    
    # Fill tensor
    for t_idx, frame_id in enumerate(frames):
        frame_data = play_tracking[play_tracking['frameId'] == frame_id]
        
        for p_idx, player_id in enumerate(all_players):
            if player_id is None:
                continue
            
            player_frame = frame_data[frame_data['nflId'] == player_id]
            if len(player_frame) > 0:
                row = player_frame.iloc[0]
                for f_idx, feat in enumerate(features):
                    if feat in row:
                        val = row[feat]
                        tensor[t_idx, p_idx, f_idx] = float(val) if pd.notna(val) else 0.0
    
    return tensor


def build_context_vector(
    play_info: pd.Series
) -> Dict[str, np.ndarray]:
    """
    Build context vector with categorical and continuous features.
    
    Returns:
        Dict with 'categorical' and 'continuous' arrays
    """
    # Categorical features
    down = int(play_info.get('down', 1))
    offensive_formation = str(play_info.get('offenseFormation', 'UNKNOWN'))
    personnel_o = str(play_info.get('personnelO', 'UNKNOWN'))
    def_team = str(play_info.get('defensiveTeam', 'UNKNOWN'))
    yards_to_go = float(play_info.get('yardsToGo', 10))
    situation = derive_situation(down, yards_to_go)
    
    categorical = {
        'down': down,
        'offensiveFormation': offensive_formation,
        'personnelO': personnel_o,
        'defTeam': def_team,
        'situation': situation
    }
    
    # Continuous features
    yards_to_go_val = yards_to_go
    yardline_100 = compute_yardline_100(play_info.to_frame().T).iloc[0]
    yardline_norm = yardline_100 / 100.0  # Normalize to [0, 1]
    
    continuous = np.array([yards_to_go_val, yardline_norm], dtype=np.float32)
    
    return {
        'categorical': categorical,
        'continuous': continuous
    }


def preprocess_week(
    week_file: Path,
    plays_df: pd.DataFrame,
    games_df: pd.DataFrame,
    output_dir: Path,
    config: Dict
) -> List[Dict]:
    """
    Preprocess tracking data for one week.
    
    Returns:
        List of dicts with 'gameId', 'playId', 'tensor', 'context', 'frame_count'
    """
    print(f"Processing {week_file.name}...")
    
    # Load tracking data
    tracking = pd.read_csv(week_file)
    
    # Standardize coordinates
    if config.get('flip_to_right', True):
        tracking = standardize_coordinates(tracking, plays_df)
    
    # Join with games to get week
    tracking = tracking.merge(
        games_df[['gameId', 'week']],
        on='gameId',
        how='left'
    )
    
    week_num = tracking['week'].iloc[0] if 'week' in tracking.columns else None
    
    # Get plays for this week
    week_plays = plays_df[
        plays_df['gameId'].isin(tracking['gameId'].unique())
    ]
    
    max_frames = config.get('frames', 60)
    num_players = config.get('min_players', 22)
    features = config.get('features', ['x', 'y', 's'])
    
    processed_plays = []
    
    for _, play_info in tqdm(week_plays.iterrows(), total=len(week_plays), desc=f"Week {week_num}"):
        game_id = play_info['gameId']
        play_id = play_info['playId']
        
        # Extract frames
        play_tracking = extract_play_frames(
            tracking, game_id, play_id, max_frames=max_frames
        )
        
        if len(play_tracking) == 0:
            continue
        
        # Extract tensor
        tensor = extract_player_tensor(
            play_tracking, play_info,
            num_players=num_players,
            features=features
        )
        
        # Ensure tensor is exactly max_frames (pad or truncate)
        T_current, P_current, F_current = tensor.shape
        if T_current < max_frames:
            # Pad by repeating last frame
            last_frame = tensor[-1:]  # [1, P, F]
            padding = np.repeat(last_frame, max_frames - T_current, axis=0)
            tensor = np.concatenate([tensor, padding], axis=0)
        elif T_current > max_frames:
            # Truncate
            tensor = tensor[:max_frames]
        
        # Build context
        context = build_context_vector(play_info)
        
        # Check if we have enough valid data (before padding)
        original_valid_frames = T_current if T_current <= max_frames else max_frames
        non_zero_frames = (tensor[:original_valid_frames] != 0).any(axis=(1, 2)).sum()
        if non_zero_frames < original_valid_frames * 0.5:  # Need at least 50% valid frames
            continue
        
        processed_plays.append({
            'gameId': int(game_id),
            'playId': int(play_id),
            'week': week_num,
            'tensor': tensor,  # Now guaranteed to be [max_frames, P, F]
            'context': context,
            'frame_count': non_zero_frames
        })
    
    return processed_plays


def preprocess_all(
    raw_dir: Path,
    cache_dir: Path,
    config: Dict
):
    """
    Preprocess all tracking weeks and save to Parquet cache.
    
    Args:
        raw_dir: Directory with raw CSV files
        cache_dir: Directory to save cached Parquet files
        config: Configuration dict
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    plays_df = pd.read_csv(raw_dir / 'plays.csv')
    games_df = pd.read_csv(raw_dir / 'games.csv')
    
    # Add derived fields
    plays_df['clock_seconds'] = plays_df['gameClock'].apply(parse_game_clock)
    plays_df['yardline_100'] = compute_yardline_100(plays_df)
    
    # Join with games
    plays_df = plays_df.merge(
        games_df[['gameId', 'week']],
        on='gameId',
        how='left'
    )
    
    all_processed = []
    
    # Process each week
    for week_file in sorted(raw_dir.glob('week*.csv')):
        processed = preprocess_week(
            week_file, plays_df, games_df, cache_dir, config
        )
        all_processed.extend(processed)
    
    print(f"\nProcessed {len(all_processed)} plays total")
    
    # Save data - use pickle to avoid PyArrow compatibility issues
    # Save as pickle which is simpler and more reliable
    output_file = cache_dir / 'processed_plays.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_processed, f)
    
    # Also save a simplified DataFrame for easy inspection (without binary data)
    records_summary = []
    for play in all_processed:
        records_summary.append({
            'gameId': play['gameId'],
            'playId': play['playId'],
            'week': play['week'],
            'frame_count': play['frame_count']
        })
    
    df_summary = pd.DataFrame(records_summary)
    df_summary.to_csv(cache_dir / 'processed_plays_summary.csv', index=False)
    
    # Save metadata separately
    metadata = {
        'num_plays': len(all_processed),
        'tensor_shape': list(all_processed[0]['tensor'].shape) if all_processed else None,
        'features': config.get('features', ['x', 'y', 's']),
        'frames': config.get('frames', 60),
        'players': config.get('min_players', 22)
    }
    
    # Save metadata
    import json
    with open(cache_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved to {output_file}")
    print(f"Summary CSV saved to {cache_dir / 'processed_plays_summary.csv'}")
    print(f"Metadata: {metadata}")


if __name__ == '__main__':
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--raw_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data', {})
    raw_dir = Path(args.raw_dir or data_config.get('raw_dir', '../../../data/nfl-big-data-bowl-2023'))
    cache_dir = Path(args.cache_dir or data_config.get('cache_dir', '../../../data/cache'))
    
    preprocess_all(raw_dir, cache_dir, data_config)

