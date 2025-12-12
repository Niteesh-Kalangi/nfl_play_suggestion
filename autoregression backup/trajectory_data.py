"""
Trajectory dataset module for autoregressive and diffusion models.

Transforms raw tracking CSVs into sequences suitable for trajectory generation:
- Sequences: [T, feature_dim] or [T, num_players, 2]
- Labels/targets: next-step positions
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.preprocess import parse_game_clock


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for player trajectory sequences.
    
    Each item contains:
    - play_id: unique play identifier
    - X: conditioning features [T, D_in] (state + optional formation)
    - Y: player positions/velocities per timestep [T, D_out]
    - mask: valid timestep mask [T]
    """
    
    def __init__(
        self,
        sequences: List[Dict],
        max_timesteps: int = 50,
        input_dim: int = None,
        output_dim: int = None
    ):
        """
        Args:
            sequences: List of sequence dicts with 'play_id', 'X', 'Y' keys
            max_timesteps: Maximum sequence length (pad/truncate to this)
            input_dim: Input feature dimension (for padding)
            output_dim: Output feature dimension (for padding)
        """
        self.sequences = sequences
        self.max_timesteps = max_timesteps
        
        # Infer dimensions from first sequence if not provided
        if len(sequences) > 0:
            self.input_dim = input_dim or sequences[0]['X'].shape[-1]
            self.output_dim = output_dim or sequences[0]['Y'].shape[-1]
        else:
            self.input_dim = input_dim or 1
            self.output_dim = output_dim or 1
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        X = seq['X']  # [T, D_in]
        Y = seq['Y']  # [T, D_out]
        T_actual = X.shape[0]
        
        # Create mask for valid timesteps
        mask = np.ones(min(T_actual, self.max_timesteps), dtype=np.float32)
        
        # Pad or truncate
        if T_actual < self.max_timesteps:
            # Pad with zeros
            X_padded = np.zeros((self.max_timesteps, self.input_dim), dtype=np.float32)
            Y_padded = np.zeros((self.max_timesteps, self.output_dim), dtype=np.float32)
            mask_padded = np.zeros(self.max_timesteps, dtype=np.float32)
            
            X_padded[:T_actual] = X
            Y_padded[:T_actual] = Y
            mask_padded[:T_actual] = mask
            
            X, Y, mask = X_padded, Y_padded, mask_padded
        else:
            # Truncate
            X = X[:self.max_timesteps]
            Y = Y[:self.max_timesteps]
            mask = mask[:self.max_timesteps]
        
        return {
            'play_id': seq['play_id'],
            'X': torch.FloatTensor(X),
            'Y': torch.FloatTensor(Y),
            'mask': torch.FloatTensor(mask),
            'seq_len': min(T_actual, self.max_timesteps)
        }


def standardize_tracking_coordinates(
    tracking: pd.DataFrame,
    plays: pd.DataFrame
) -> pd.DataFrame:
    """
    Standardize tracking coordinates so offense always moves left-to-right.
    
    Args:
        tracking: Tracking DataFrame with x, y, o, dir columns
        plays: Plays DataFrame with possessionTeam
        
    Returns:
        Standardized tracking DataFrame
    """
    tracking_std = tracking.copy()
    
    # Join play direction info
    if 'playDirection' not in tracking_std.columns:
        # Infer from first frame if not present
        tracking_std['playDirection'] = 'right'
    
    # Standardize coordinates: always make offense move left→right
    mask_left = tracking_std['playDirection'] == 'left'
    
    # Flip x coordinate (field is 120 yards including end zones)
    tracking_std.loc[mask_left, 'x'] = 120 - tracking_std.loc[mask_left, 'x']
    
    # Flip y coordinate (field is 53.3 yards wide)
    tracking_std.loc[mask_left, 'y'] = 53.3 - tracking_std.loc[mask_left, 'y']
    
    # Flip angles (rotate 180 degrees)
    if 'o' in tracking_std.columns:
        tracking_std.loc[mask_left, 'o'] = (tracking_std.loc[mask_left, 'o'] + 180) % 360
    if 'dir' in tracking_std.columns:
        tracking_std.loc[mask_left, 'dir'] = (tracking_std.loc[mask_left, 'dir'] + 180) % 360
    
    return tracking_std


def extract_player_positions(
    tracking: pd.DataFrame,
    plays: pd.DataFrame,
    players: Optional[pd.DataFrame] = None,
    players_mode: str = 'offense_only',
    include_ball: bool = True
) -> pd.DataFrame:
    """
    Extract player position data organized by play and frame.
    
    Args:
        tracking: Tracking DataFrame
        plays: Plays DataFrame
        players: Players DataFrame (optional, for position filtering)
        players_mode: 'offense_only', 'defense_only', 'all', or 'ball_carrier'
        include_ball: Whether to include football position
        
    Returns:
        DataFrame with columns: gameId, playId, frameId, nflId, x, y, s, a, dis, o, dir, is_offense
    """
    # Join possession team info
    tracking_with_poss = tracking.merge(
        plays[['gameId', 'playId', 'possessionTeam']],
        on=['gameId', 'playId'],
        how='left'
    )
    
    # Mark offense/defense
    tracking_with_poss['is_offense'] = (
        (tracking_with_poss['team'] == tracking_with_poss['possessionTeam'])
    )
    tracking_with_poss['is_ball'] = tracking_with_poss['team'] == 'football'
    
    # Filter based on mode
    if players_mode == 'offense_only':
        mask = tracking_with_poss['is_offense'] | (include_ball & tracking_with_poss['is_ball'])
    elif players_mode == 'defense_only':
        mask = ~tracking_with_poss['is_offense'] | (include_ball & tracking_with_poss['is_ball'])
    elif players_mode == 'ball_carrier':
        # Only keep ball carrier (would need additional logic to identify)
        mask = tracking_with_poss['is_ball']  # Fallback to ball only
    else:  # 'all'
        mask = pd.Series([True] * len(tracking_with_poss))
        if not include_ball:
            mask = ~tracking_with_poss['is_ball']
    
    return tracking_with_poss[mask].copy()


def build_sequence_for_play(
    play_tracking: pd.DataFrame,
    play_info: pd.Series,
    num_players: int = 11,
    include_velocity: bool = True,
    condition_on_state: bool = True
) -> Dict:
    """
    Build a single trajectory sequence for one play.
    
    Args:
        play_tracking: Tracking data for this play (all frames, filtered players)
        play_info: Series with play metadata (down, yardsToGo, etc.)
        num_players: Number of players to include (pad/truncate)
        include_velocity: Include speed/acceleration features
        condition_on_state: Include game state as conditioning
        
    Returns:
        Dict with 'play_id', 'X', 'Y' arrays
    """
    game_id = int(play_info['gameId'])
    play_id = int(play_info['playId'])
    
    # Sort by frame
    play_tracking = play_tracking.sort_values('frameId')
    frames = sorted(play_tracking['frameId'].unique())
    
    if len(frames) == 0:
        return None
    
    # Build conditioning features (game state)
    if condition_on_state:
        state_features = np.array([
            play_info.get('down', 1),
            play_info.get('yardsToGo', 10),
            play_info.get('yardline_100', 50),
            play_info.get('clock_seconds', 900) if pd.notna(play_info.get('clock_seconds')) else 900,
            play_info.get('score_diff', 0) if pd.notna(play_info.get('score_diff')) else 0,
            play_info.get('quarter', 1)
        ], dtype=np.float32)
    else:
        state_features = np.array([], dtype=np.float32)
    
    # Build position sequences
    T = len(frames)
    
    # Get unique player IDs (excluding ball)
    player_ids = play_tracking[play_tracking['team'] != 'football']['nflId'].dropna().unique()
    player_ids = sorted(player_ids)[:num_players]  # Take first N players
    
    # Position features: [x, y] per player, flattened
    # With velocity: [x, y, s, a] per player
    features_per_player = 4 if include_velocity else 2
    output_dim = num_players * features_per_player
    
    Y = np.zeros((T, output_dim), dtype=np.float32)
    
    for t_idx, frame_id in enumerate(frames):
        frame_data = play_tracking[play_tracking['frameId'] == frame_id]
        
        for p_idx, player_id in enumerate(player_ids):
            if p_idx >= num_players:
                break
            
            player_frame = frame_data[frame_data['nflId'] == player_id]
            if len(player_frame) > 0:
                row = player_frame.iloc[0]
                base_idx = p_idx * features_per_player
                Y[t_idx, base_idx] = row['x']
                Y[t_idx, base_idx + 1] = row['y']
                if include_velocity:
                    Y[t_idx, base_idx + 2] = row.get('s', 0)  # speed
                    Y[t_idx, base_idx + 3] = row.get('a', 0)  # acceleration
    
    # Conditioning input X: state features + previous positions
    # For autoregressive: X[t] = [state, Y[t-1]] to predict Y[t]
    state_dim = len(state_features)
    input_dim = state_dim + output_dim
    
    X = np.zeros((T, input_dim), dtype=np.float32)
    
    # Broadcast state features across all timesteps
    if state_dim > 0:
        X[:, :state_dim] = state_features
    
    # Shift Y by 1 to create input (previous positions)
    # X[t, state_dim:] = Y[t-1]
    X[1:, state_dim:] = Y[:-1]
    # First timestep uses initial positions (could be zeros or first frame)
    X[0, state_dim:] = Y[0]  # Use first frame as initial
    
    return {
        'play_id': f"{game_id}_{play_id}",
        'game_id': game_id,
        'play_id_int': play_id,
        'X': X,
        'Y': Y,
        'num_frames': T
    }


def build_trajectory_dataset(
    data_dir: str,
    weeks: List[int],
    config: Dict,
    plays_df: Optional[pd.DataFrame] = None,
    games_df: Optional[pd.DataFrame] = None
) -> TrajectoryDataset:
    """
    Build trajectory dataset from tracking CSVs.
    
    Args:
        data_dir: Path to data directory
        weeks: List of week numbers to include
        config: Configuration dict with trajectory settings
        plays_df: Optional pre-loaded plays DataFrame
        games_df: Optional pre-loaded games DataFrame
        
    Returns:
        TrajectoryDataset ready for DataLoader
    """
    data_path = Path(data_dir)
    
    # Load plays if not provided
    if plays_df is None:
        plays_df = pd.read_csv(data_path / 'plays.csv')
    
    # Load games if not provided
    if games_df is None:
        games_df = pd.read_csv(data_path / 'games.csv')
    
    # Join game info to plays
    plays_df = plays_df.merge(
        games_df[['gameId', 'week', 'homeTeamAbbr', 'visitorTeamAbbr']],
        on='gameId',
        how='left'
    )
    
    # Add derived fields
    plays_df['is_home'] = plays_df['possessionTeam'] == plays_df['homeTeamAbbr']
    plays_df['score_diff'] = np.where(
        plays_df['is_home'],
        plays_df['preSnapHomeScore'] - plays_df['preSnapVisitorScore'],
        plays_df['preSnapVisitorScore'] - plays_df['preSnapHomeScore']
    )
    
    # Parse clock
    if 'gameClock' in plays_df.columns:
        plays_df['clock_seconds'] = plays_df['gameClock'].apply(parse_game_clock)
    
    # Compute yardline_100
    from src.preprocess import compute_yardline_100
    plays_df['yardline_100'] = compute_yardline_100(plays_df)
    
    # Filter to specified weeks
    plays_df = plays_df[plays_df['week'].isin(weeks)]
    
    # Get config options
    traj_config = config.get('trajectories', {})
    max_timesteps = traj_config.get('max_timesteps', 50)
    players_mode = traj_config.get('players', 'offense_only')
    condition_on_state = traj_config.get('condition_on_state', True)
    num_players = traj_config.get('num_players', 11)
    include_velocity = traj_config.get('include_velocity', True)
    
    sequences = []
    
    # Process each week
    for week in weeks:
        week_file = data_path / f'week{week}.csv'
        if not week_file.exists():
            print(f"Warning: {week_file} not found, skipping")
            continue
        
        print(f"Processing week {week}...")
        tracking = pd.read_csv(week_file)
        
        # Standardize coordinates
        tracking = standardize_tracking_coordinates(tracking, plays_df)
        
        # Extract player positions
        tracking_filtered = extract_player_positions(
            tracking, plays_df, 
            players_mode=players_mode,
            include_ball=False
        )
        
        # Get plays for this week
        week_plays = plays_df[plays_df['week'] == week]
        
        # Build sequences for each play
        for _, play_info in week_plays.iterrows():
            game_id = play_info['gameId']
            play_id = play_info['playId']
            
            play_tracking = tracking_filtered[
                (tracking_filtered['gameId'] == game_id) &
                (tracking_filtered['playId'] == play_id)
            ]
            
            if len(play_tracking) == 0:
                continue
            
            seq = build_sequence_for_play(
                play_tracking,
                play_info,
                num_players=num_players,
                include_velocity=include_velocity,
                condition_on_state=condition_on_state
            )
            
            if seq is not None and seq['num_frames'] >= 5:  # Min sequence length
                sequences.append(seq)
    
    print(f"Built {len(sequences)} trajectory sequences")
    
    # Infer dimensions from first sequence
    if len(sequences) > 0:
        input_dim = sequences[0]['X'].shape[-1]
        output_dim = sequences[0]['Y'].shape[-1]
    else:
        input_dim = 50  # Default
        output_dim = 44  # 11 players * 4 features
    
    return TrajectoryDataset(
        sequences,
        max_timesteps=max_timesteps,
        input_dim=input_dim,
        output_dim=output_dim
    )


def make_trajectory_splits(
    data_dir: str,
    config: Dict
) -> Dict[str, TrajectoryDataset]:
    """
    Create train/val/test trajectory datasets using config splits.
    
    Args:
        data_dir: Path to data directory
        config: Configuration dict with 'splits' and 'trajectories' sections
        
    Returns:
        Dict with 'train', 'val', 'test' TrajectoryDataset objects
    """
    split_config = config.get('splits', {})
    train_weeks = split_config.get('train', [1, 2, 3, 4, 5, 6])
    val_weeks = split_config.get('val', [7])
    test_weeks = split_config.get('test', [8])
    
    # Load shared data
    data_path = Path(data_dir)
    plays_df = pd.read_csv(data_path / 'plays.csv')
    games_df = pd.read_csv(data_path / 'games.csv')
    
    print(f"Building train dataset (weeks {train_weeks})...")
    train_dataset = build_trajectory_dataset(
        data_dir, train_weeks, config, plays_df, games_df
    )
    
    print(f"Building val dataset (weeks {val_weeks})...")
    val_dataset = build_trajectory_dataset(
        data_dir, val_weeks, config, plays_df, games_df
    )
    
    print(f"Building test dataset (weeks {test_weeks})...")
    test_dataset = build_trajectory_dataset(
        data_dir, test_weeks, config, plays_df, games_df
    )
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


def create_dataloaders(
    datasets: Dict[str, TrajectoryDataset],
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders from trajectory datasets.
    
    Args:
        datasets: Dict of TrajectoryDataset objects
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        
    Returns:
        Dict of DataLoader objects
    """
    loaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders

