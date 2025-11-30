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
from .formation_anchors import get_anchors, anchors_to_tensor, OFFENSE_ROLE_ORDER
from .role_mapping import get_anchor_mask_for_offense


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
    # For passing plays, we want to stop when the route ends (pass caught/incomplete),
    # NOT when the ball carrier is tackled or play ends
    # Priority: pass_outcome events first (route ends), then other end events
    pass_outcome_events = ['pass_outcome_caught', 'pass_outcome_incomplete', 'pass_arrived']
    pass_outcome_frames = play_tracking[
        play_tracking['event'].str.contains('|'.join(pass_outcome_events), case=False, na=False)
    ]
    
    if len(pass_outcome_frames) > 0:
        # For passing plays: stop when pass is caught/incomplete (route is done)
        end_frame = pass_outcome_frames['frameId'].min()
    else:
        # Fallback to other end events if no pass outcome found
        # (for sacks, fumbles, etc.)
        end_events = ['qb_sack', 'fumble', 'autoevent_passinterrupted']
        end_frames = play_tracking[
            play_tracking['event'].str.contains('|'.join(end_events), case=False, na=False)
        ]
        if len(end_frames) > 0:
            end_frame = end_frames['frameId'].min()
        else:
            # Last resort: use last frame
            end_frame = play_tracking['frameId'].max()
    
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


def normalize_position(pos_str: str, side: str = 'offense') -> str:
    """
    Normalize position string to standard abbreviations.
    
    Args:
        pos_str: Raw position string (e.g., 'HB', 'FS', 'WR')
        side: 'offense' or 'defense'
        
    Returns:
        Normalized position string
    """
    if pd.isna(pos_str):
        return 'UNKNOWN'
    
    pos = str(pos_str).strip().upper()
    
    if side == 'offense':
        # Offensive positions
        if pos in ['QB']:
            return 'QB'
        elif pos in ['RB', 'HB', 'FB']:
            return 'RB'
        elif pos in ['WR']:
            return 'WR'
        elif pos in ['TE']:
            return 'TE'
        elif pos in ['C', 'G', 'T', 'OL', 'LG', 'RG', 'LT', 'RT']:
            return 'OL'
        else:
            return 'UNKNOWN'
    else:
        # Defensive positions
        if pos in ['CB', 'S', 'FS', 'SS', 'DB']:
            return 'DB'
        elif pos in ['DT', 'DE', 'NT', 'DL']:
            return 'DL'
        elif pos in ['OLB', 'ILB', 'MLB', 'LB']:
            return 'LB'
        else:
            return 'UNKNOWN'


def get_position_priority(pos: str, side: str = 'offense') -> int:
    """
    Get sorting priority for positions (lower = earlier in ordering).
    
    Offense: QB (0), RB (1), WR (2), TE (3), OL (4)
    Defense: DL (0), LB (1), DB (2)
    """
    if side == 'offense':
        priorities = {
            'QB': 0,
            'RB': 1,
            'WR': 2,
            'TE': 3,
            'OL': 4,
            'UNKNOWN': 99
        }
    else:
        priorities = {
            'DL': 0,
            'LB': 1,
            'DB': 2,
            'UNKNOWN': 99
        }
    
    return priorities.get(pos, 99)


def extract_player_tensor(
    play_tracking: pd.DataFrame,
    play_info: pd.Series,
    num_players: int = 22,
    features: List[str] = ['x', 'y', 's'],
    players_df: Optional[pd.DataFrame] = None
) -> tuple:
    """
    Extract player positions as tensor [T, P, F] with FIXED ORDERING by position.
    
    Fixed ordering ensures consistent player positions:
    - Offense (indices 0-10): QB, then RB(s), WR(s), TE(s), OL(s)
    - Defense (indices 11-21): DL, LB, DB
    
    Args:
        play_tracking: Tracking DataFrame for one play (standardized, filtered frames)
        play_info: Play metadata Series
        num_players: Number of players (22 = 11 offense + 11 defense)
        features: Features to extract [x, y, s]
        players_df: Required DataFrame with player info (nflId, officialPosition)
        
    Returns:
        Tuple of (tensor [T, P, F], player_positions [P] list of position strings)
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
    
    # Get player IDs and their positions
    offense_data = play_tracking[play_tracking['is_offense']].copy()
    defense_data = play_tracking[play_tracking['is_defense']].copy()
    
    # Build player-position mapping
    def get_player_with_position(player_data, side):
        """Get list of (player_id, position, x_avg) tuples sorted by position priority."""
        players_list = []
        
        for player_id in player_data['nflId'].dropna().unique():
            if player_id is None:
                continue
            
            # Get position from players_df
            position = 'UNKNOWN'
            if players_df is not None and 'officialPosition' in players_df.columns:
                player_row = players_df[players_df['nflId'] == player_id]
                if len(player_row) > 0:
                    raw_pos = str(player_row.iloc[0]['officialPosition']).strip().upper()
                    position = normalize_position(raw_pos, side)
            
            # Get average x position (for tie-breaking within same position)
            player_frames = player_data[player_data['nflId'] == player_id]
            if len(player_frames) > 0 and 'x' in player_frames.columns:
                x_avg = player_frames['x'].mean()
            else:
                x_avg = 0.0
            
            players_list.append((player_id, position, x_avg))
        
        # Sort by position priority, then by x position (for consistent ordering within position)
        players_list.sort(key=lambda x: (get_position_priority(x[1], side), x[2]))
        
        return players_list
    
    # Get sorted offense and defense players
    offense_players_sorted = get_player_with_position(offense_data, 'offense')[:11]
    defense_players_sorted = get_player_with_position(defense_data, 'defense')[:11]
    
    # Combine: offense first, then defense
    all_players_sorted = offense_players_sorted + defense_players_sorted
    
    # Extract player IDs and positions
    all_players = [p[0] for p in all_players_sorted]
    player_positions = [p[1] for p in all_players_sorted]
    
    # Pad if needed
    while len(all_players) < num_players:
        all_players.append(None)
        player_positions.append('UNKNOWN')
    all_players = all_players[:num_players]
    player_positions = player_positions[:num_players]
    
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
    
    return tensor, player_positions


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
    
    # Hash mark encoding: LEFT = 0.0, MIDDLE = 0.5, RIGHT = 1.0
    # Default to MIDDLE (will be overridden in preprocess_week with actual hash)
    hash_mark_str = play_info.get('hash_mark', 'MIDDLE')
    if isinstance(hash_mark_str, str):
        hash_map = {'LEFT': 0.0, 'MIDDLE': 0.5, 'RIGHT': 1.0, 'left': 0.0, 'middle': 0.5, 'right': 1.0}
        hash_encoded = hash_map.get(hash_mark_str.upper(), 0.5)
    else:
        hash_encoded = float(hash_mark_str) if not pd.isna(hash_mark_str) else 0.5
    
    continuous = np.array([yards_to_go_val, yardline_norm, hash_encoded], dtype=np.float32)
    
    return {
        'categorical': categorical,
        'continuous': continuous
    }


def preprocess_week(
    week_file: Path,
    plays_df: pd.DataFrame,
    games_df: pd.DataFrame,
    players_df: pd.DataFrame,
    output_dir: Path,
    config: Dict
) -> List[Dict]:
    """
    Preprocess tracking data for one week.
    
    Args:
        week_file: Path to tracking CSV file for this week
        plays_df: DataFrame with play metadata
        games_df: DataFrame with game metadata
        players_df: DataFrame with player info (nflId, officialPosition)
        output_dir: Output directory (not used, kept for compatibility)
        config: Configuration dict
        
    Returns:
        List of dicts with 'gameId', 'playId', 'tensor', 'player_positions', 'context', 'frame_count'
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
        
        # FILTER: Skip rush plays (R) - we only want passing routes
        # Also skip QB runs/scrambles if we want to focus on designed routes
        if 'passResult' in play_info and play_info['passResult'] == 'R':
            continue  # Skip rush plays
        
        # Get hash mark from original tracking data BEFORE extraction
        # This ensures we have all columns available (club/team)
        play_tracking_raw = tracking[
            (tracking['gameId'] == game_id) &
            (tracking['playId'] == play_id)
        ].copy()
        
        hash_mark = "MIDDLE"  # Default
        if len(play_tracking_raw) > 0:
            # Find snap frame
            snap_frames = play_tracking_raw[play_tracking_raw['event'].str.contains('snap', case=False, na=False)]
            if len(snap_frames) > 0:
                start_frame = snap_frames['frameId'].min()
            else:
                start_frame = play_tracking_raw['frameId'].min()
            
            snap_frame = play_tracking_raw[play_tracking_raw['frameId'] == start_frame]
            
            # Try both 'club' and 'team' columns (different datasets use different names)
            ball_col = None
            if 'club' in snap_frame.columns:
                ball_col = 'club'
            elif 'team' in snap_frame.columns:
                ball_col = 'team'
            
            if ball_col is not None:
                ball_snap_pos = snap_frame[snap_frame[ball_col] == 'football']
                if len(ball_snap_pos) > 0:
                    ball_y = ball_snap_pos['y'].iloc[0]
                    if ball_y < 22.0:
                        hash_mark = "LEFT"
                    elif ball_y > 31.0:
                        hash_mark = "RIGHT"
                    else:
                        hash_mark = "MIDDLE"
        
        # Extract frames (will stop at pass outcome, not at tackle)
        play_tracking = extract_play_frames(
            tracking, game_id, play_id, max_frames=max_frames
        )
        
        if len(play_tracking) == 0:
            continue
        
        # Extract tensor with fixed player ordering
        tensor, player_positions = extract_player_tensor(
            play_tracking, play_info,
            num_players=num_players,
            features=features,
            players_df=players_df
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
        
        # Store hash_mark in play_info for build_context_vector
        play_info = play_info.copy()
        play_info['hash_mark'] = hash_mark
        
        # Build context (now includes hash_mark)
        context = build_context_vector(play_info)
        
        # Get yardline for anchors
        yardline_100 = compute_yardline_100(play_info.to_frame().T).iloc[0]
        
        # Get formation and personnel for anchor computation
        formation = str(play_info.get('offenseFormation', 'SHOTGUN'))
        personnel = str(play_info.get('personnelO', '1 RB, 1 TE, 3 WR'))
        
        # Compute anchors using hash_mark detected above
        anchors_dict = get_anchors(
            formation=formation,
            personnel=personnel,
            yardline=yardline_100,
            hash_mark=hash_mark,
            direction="right"  # Already normalized/flipped
        )
        
        # Convert anchors to tensor [P, 2] for offense players
        anchors_t0 = np.zeros((num_players, 2), dtype=np.float32)
        for i, role in enumerate(OFFENSE_ROLE_ORDER):
            if i < num_players and role in anchors_dict:
                x, y = anchors_dict[role]
                anchors_t0[i, 0] = x
                anchors_t0[i, 1] = y
        
        # Create anchor mask (offense players only)
        anchor_mask = get_anchor_mask_for_offense(num_players)
        
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
            'player_positions': player_positions,  # List of position strings [P]
            'context': context,
            'frame_count': non_zero_frames,
            'anchors_t0': anchors_t0,  # [P, 2] anchor positions for t=0
            'anchor_mask': anchor_mask,  # [P] boolean mask for anchored players
            'hash_mark': hash_mark,  # Hash position for this play
            'yardline': yardline_100  # Yardline for this play
        })
    
    return processed_plays


def preprocess_all(
    raw_dir: Path,
    cache_dir: Path,
    config: Dict
):
    """
    Preprocess all tracking weeks and save to pickle cache with fixed player ordering.
    
    Players are sorted by position for consistent ordering:
    - Offense: QB, RB, WR, TE, OL
    - Defense: DL, LB, DB
    
    Args:
        raw_dir: Directory with raw CSV files (must contain players.csv)
        cache_dir: Directory to save cached pickle files
        config: Configuration dict
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    plays_df = pd.read_csv(raw_dir / 'plays.csv')
    games_df = pd.read_csv(raw_dir / 'games.csv')
    
    # Load player positions - REQUIRED for fixed ordering
    players_file = raw_dir / 'players.csv'
    if not players_file.exists():
        raise FileNotFoundError(
            f"players.csv not found at {players_file}. "
            "This file is required for fixed player ordering."
        )
    players_df = pd.read_csv(players_file)
    print(f"Loaded {len(players_df)} players from players.csv")
    
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
    
    # Process each week - handle both 'week*.csv' and 'tracking_week_*.csv' patterns
    week_files = list(raw_dir.glob('week*.csv')) + list(raw_dir.glob('tracking_week_*.csv'))
    week_files = sorted(set(week_files))  # Remove duplicates
    
    if not week_files:
        raise FileNotFoundError(
            f"No tracking files found in {raw_dir}. "
            "Expected files named 'week*.csv' or 'tracking_week_*.csv'"
        )
    
    for week_file in week_files:
        processed = preprocess_week(
            week_file, plays_df, games_df, players_df, cache_dir, config
        )
        all_processed.extend(processed)
    
    print(f"\nProcessed {len(all_processed)} plays total")
    
    # COMPUTE NORMALIZATION STATISTICS (CRITICAL FOR DIFFUSION MODELS)
    print("\nComputing normalization statistics...")
    features = config.get('features', ['x', 'y', 's'])
    num_features = len(features)
    
    # Collect all feature values (excluding padding zeros)
    feature_means = []
    feature_stds = []
    
    for f_idx in range(num_features):
        # Get all values for this feature across all plays
        feature_values = []
        for play in all_processed:
            tensor = play['tensor']
            feature_data = tensor[:, :, f_idx].flatten()
            # Filter out padding (near-zero values)
            non_padding = feature_data[np.abs(feature_data) > 1e-6]
            if len(non_padding) > 0:
                feature_values.extend(non_padding)
        
        if len(feature_values) > 0:
            feature_values = np.array(feature_values)
            mean = float(np.mean(feature_values))
            std = float(np.std(feature_values))
            # Avoid division by zero
            if std < 1e-6:
                std = 1.0
        else:
            mean, std = 0.0, 1.0
        
        feature_means.append(mean)
        feature_stds.append(std)
        print(f"  {features[f_idx]}: mean={mean:.4f}, std={std:.4f}")
    
    # NORMALIZE ALL TENSORS
    print("\nNormalizing coordinates (mean=0, std=1 per feature)...")
    for play in all_processed:
        tensor = play['tensor']
        for f_idx in range(num_features):
            # Normalize: (x - mean) / std
            tensor[:, :, f_idx] = (tensor[:, :, f_idx] - feature_means[f_idx]) / feature_stds[f_idx]
    
    print("✅ Normalization complete")
    
    # Save data - use pickle to avoid PyArrow compatibility issues
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
            'frame_count': play['frame_count'],
            'player_positions': ','.join(play.get('player_positions', ['UNKNOWN'] * 22))
        })
    
    df_summary = pd.DataFrame(records_summary)
    df_summary.to_csv(cache_dir / 'processed_plays_summary.csv', index=False)
    
    # Save metadata WITH NORMALIZATION STATS
    metadata = {
        'num_plays': len(all_processed),
        'tensor_shape': list(all_processed[0]['tensor'].shape) if all_processed else None,
        'features': features,
        'frames': config.get('frames', 60),
        'players': config.get('min_players', 22),
        'fixed_ordering': True,
        'ordering_info': {
            'offense': 'QB, RB, WR, TE, OL (sorted by position priority)',
            'defense': 'DL, LB, DB (sorted by position priority)'
        },
        'normalization': {
            'means': feature_means,
            'stds': feature_stds,
            'feature_names': features
        }
    }
    
    # Save metadata
    import json
    with open(cache_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved to {output_file}")
    print(f"Summary CSV saved to {cache_dir / 'processed_plays_summary.csv'}")
    print(f"Metadata saved with normalization stats for denormalization")


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

