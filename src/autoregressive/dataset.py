"""
Dataset module for autoregressive trajectory generation with [T, P, F] format.

Matches diffusion model structure: T=60 frames, P=22 players (11 offense + 11 defense), F=3 features (x, y, s).
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.preprocess import parse_game_clock, compute_yardline_100


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
    
    # Determine play direction
    if 'playDirection' not in tracking_std.columns:
        tracking_std['playDirection'] = 'right'
    
    # Flip coordinates for left-moving plays
    mask_left = tracking_std['playDirection'] == 'left'
    
    # Flip x (field is 120 yards including end zones, but we use 0-100 for playable field)
    tracking_std.loc[mask_left, 'x'] = 120 - tracking_std.loc[mask_left, 'x']
    
    # Flip y (field is 53.3 yards wide)
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
        start_frame = play_tracking['frameId'].min()
    else:
        start_frame = snap_frames['frameId'].min()
    
    # Find end frame - prioritize pass_outcome for route-focused data
    pass_outcome_events = ['pass_outcome_caught', 'pass_outcome_incomplete', 'pass_arrived']
    pass_outcome_frames = play_tracking[
        play_tracking['event'].str.contains('|'.join(pass_outcome_events), case=False, na=False)
    ]
    
    if len(pass_outcome_frames) > 0:
        end_frame = pass_outcome_frames['frameId'].min()
    else:
        end_events = ['qb_sack', 'fumble', 'autoevent_passinterrupted']
        end_frames = play_tracking[
            play_tracking['event'].str.contains('|'.join(end_events), case=False, na=False)
        ]
        if len(end_frames) > 0:
            end_frame = end_frames['frameId'].min()
        else:
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


def extract_player_tensor(
    play_tracking: pd.DataFrame,
    play_info: pd.Series,
    num_players: int = 22,
    features: List[str] = ['x', 'y', 's'],
    max_frames: int = 60
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract player trajectory tensor [T, P, F] from tracking data.
    
    Args:
        play_tracking: Tracking DataFrame for one play (all frames)
        play_info: Series with play metadata
        num_players: Number of players (22 = 11 offense + 11 defense)
        features: Features to extract ['x', 'y', 's']
        max_frames: Maximum number of frames (will pad/truncate to this)
        
    Returns:
        Tuple of (tensor [max_frames, P, F], player_positions list)
    """
    frames = sorted(play_tracking['frameId'].unique())
    T_actual = len(frames)
    F = len(features)
    
    # Ensure we have exactly max_frames
    if T_actual > max_frames:
        frames = frames[:max_frames]
        T_actual = max_frames
    
    # Initialize tensor with max_frames
    tensor = np.zeros((max_frames, num_players, F), dtype=np.float32)
    player_positions = ['UNKNOWN'] * num_players
    
    # Get offense and defense player IDs
    possession_team = play_info.get('possessionTeam', '')
    
    # Separate offense and defense
    offense_players = play_tracking[
        (play_tracking['team'] == possession_team) &
        (play_tracking['team'] != 'football')
    ]['nflId'].dropna().unique()
    
    defense_players = play_tracking[
        (play_tracking['team'] != possession_team) &
        (play_tracking['team'] != 'football')
    ]['nflId'].dropna().unique()
    
    # Take first 11 offense and 11 defense players
    offense_players = sorted(offense_players)[:11]
    defense_players = sorted(defense_players)[:11]
    
    # Fill offense positions (indices 0-10)
    for p_idx, player_id in enumerate(offense_players):
        if p_idx >= 11:
            break
        for t_idx, frame_id in enumerate(frames):
            if t_idx >= max_frames:
                break
            frame_data = play_tracking[play_tracking['frameId'] == frame_id]
            player_frame = frame_data[frame_data['nflId'] == player_id]
            if len(player_frame) > 0:
                row = player_frame.iloc[0]
                for f_idx, feat in enumerate(features):
                    if feat in row:
                        tensor[t_idx, p_idx, f_idx] = float(row[feat])
                    elif feat == 's' and 'speed' in row:
                        tensor[t_idx, p_idx, f_idx] = float(row['speed'])
    
    # Fill defense positions (indices 11-21)
    for p_idx, player_id in enumerate(defense_players):
        if p_idx >= 11:
            break
        def_idx = 11 + p_idx
        for t_idx, frame_id in enumerate(frames):
            if t_idx >= max_frames:
                break
            frame_data = play_tracking[play_tracking['frameId'] == frame_id]
            player_frame = frame_data[frame_data['nflId'] == player_id]
            if len(player_frame) > 0:
                row = player_frame.iloc[0]
                for f_idx, feat in enumerate(features):
                    if feat in row:
                        tensor[t_idx, def_idx, f_idx] = float(row[feat])
                    elif feat == 's' and 'speed' in row:
                        tensor[t_idx, def_idx, f_idx] = float(row['speed'])
    
    # Pad remaining frames with last frame if needed
    if T_actual < max_frames and T_actual > 0:
        last_frame = tensor[T_actual - 1:T_actual]  # [1, P, F]
        for t_idx in range(T_actual, max_frames):
            tensor[t_idx] = last_frame[0]
    
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


class AutoregressivePlayDataset(Dataset):
    """
    PyTorch Dataset for football play trajectories with [T, P, F] format.
    
    Matches diffusion model structure:
    - X: [T, P, F] - player trajectories (T=60 frames, P=22 players, F=3 features: x, y, s)
    - context_categorical: Dict with down, offensiveFormation, personnelO, defTeam, situation
    - context_continuous: [3] - yardsToGo, yardlineNorm, hash_mark
    """
    
    def __init__(
        self,
        data_dir: str,
        weeks: List[int],
        config: Dict,
        plays_df: Optional[pd.DataFrame] = None,
        games_df: Optional[pd.DataFrame] = None
    ):
        """
        Args:
            data_dir: Path to data directory
            weeks: List of week numbers to include
            config: Configuration dict
            plays_df: Optional pre-loaded plays DataFrame
            games_df: Optional pre-loaded games DataFrame
        """
        self.data_dir = Path(data_dir)
        self.weeks = weeks
        self.config = config
        
        traj_config = config.get('trajectories', {})
        self.max_frames = traj_config.get('max_timesteps', 60)
        self.num_players = traj_config.get('num_players', 22)
        self.features = ['x', 'y', 's']  # Match diffusion model
        
        # Load plays and games if not provided
        if plays_df is None:
            plays_df = pd.read_csv(self.data_dir / 'plays.csv')
        if games_df is None:
            games_df = pd.read_csv(self.data_dir / 'games.csv')
        
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
        
        if 'gameClock' in plays_df.columns:
            plays_df['clock_seconds'] = plays_df['gameClock'].apply(parse_game_clock)
        
        plays_df['yardline_100'] = compute_yardline_100(plays_df)
        
        # Filter to specified weeks
        plays_df = plays_df[plays_df['week'].isin(weeks)]
        
        # Build dataset
        self.plays = []
        
        for week in weeks:
            week_file = self.data_dir / f'week{week}.csv'
            if not week_file.exists():
                print(f"Warning: {week_file} not found, skipping")
                continue
            
            print(f"Processing week {week}...")
            tracking = pd.read_csv(week_file)
            
            # Standardize coordinates
            tracking = standardize_coordinates(tracking, plays_df)
            
            # Get plays for this week
            week_plays = plays_df[plays_df['week'] == week]
            
            # Process each play
            for _, play_info in week_plays.iterrows():
                game_id = play_info['gameId']
                play_id = play_info['playId']
                
                # Extract frames
                play_tracking = extract_play_frames(
                    tracking, game_id, play_id, max_frames=self.max_frames
                )
                
                if len(play_tracking) == 0:
                    continue
                
                # Extract tensor
                tensor, player_positions = extract_player_tensor(
                    play_tracking, play_info, 
                    num_players=self.num_players, 
                    features=self.features,
                    max_frames=self.max_frames
                )
                
                # Build context
                context = build_context_vector(play_info)
                
                # Store play data
                self.plays.append({
                    'gameId': int(game_id),
                    'playId': int(play_id),
                    'week': int(week),
                    'tensor': tensor,  # [T, P, F]
                    'context_categorical': context['categorical'],
                    'context_continuous': context['continuous'],
                    'player_positions': player_positions
                })
        
        print(f"Built {len(self.plays)} plays")
        
        # Store dimensions
        if len(self.plays) > 0:
            self.T, self.P, self.F = self.plays[0]['tensor'].shape
        else:
            self.T, self.P, self.F = 60, 22, 3
    
    def __len__(self) -> int:
        return len(self.plays)
    
    def __getitem__(self, idx: int) -> Dict:
        play = self.plays[idx]
        tensor = play['tensor']  # [T, P, F] - may have variable T
        
        # Pad or truncate to max_frames
        T_actual = tensor.shape[0]
        if T_actual < self.max_frames:
            # Pad with last frame
            padding = np.repeat(tensor[-1:], self.max_frames - T_actual, axis=0)
            tensor = np.concatenate([tensor, padding], axis=0)
        elif T_actual > self.max_frames:
            # Truncate
            tensor = tensor[:self.max_frames]
        
        return {
            'X': torch.FloatTensor(tensor),  # [max_frames, P, F]
            'context_categorical': play['context_categorical'],
            'context_continuous': torch.FloatTensor(play['context_continuous']),  # [3]
            'gameId': play['gameId'],
            'playId': play['playId'],
            'week': play['week'],
            'player_positions': play['player_positions']
        }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Stacks tensors and preserves context dictionaries.
    """
    X = torch.stack([item['X'] for item in batch])  # [B, T, P, F]
    context_continuous = torch.stack([item['context_continuous'] for item in batch])  # [B, 3]
    
    # Keep categorical as list of dicts
    context_categorical = [item['context_categorical'] for item in batch]
    
    return {
        'X': X,
        'context_categorical': context_categorical,
        'context_continuous': context_continuous,
        'gameIds': [item['gameId'] for item in batch],
        'playIds': [item['playId'] for item in batch],
        'weeks': [item['week'] for item in batch]
    }


def make_autoregressive_splits(
    data_dir: str,
    config: Dict
) -> Dict[str, AutoregressivePlayDataset]:
    """
    Create train/val/test autoregressive datasets using config splits.
    
    Args:
        data_dir: Path to data directory
        config: Configuration dict with 'splits' section
        
    Returns:
        Dict with 'train', 'val', 'test' AutoregressivePlayDataset objects
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
    train_dataset = AutoregressivePlayDataset(
        data_dir, train_weeks, config, plays_df, games_df
    )
    
    print(f"Building val dataset (weeks {val_weeks})...")
    val_dataset = AutoregressivePlayDataset(
        data_dir, val_weeks, config, plays_df, games_df
    )
    
    print(f"Building test dataset (weeks {test_weeks})...")
    test_dataset = AutoregressivePlayDataset(
        data_dir, test_weeks, config, plays_df, games_df
    )
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


def create_autoregressive_dataloaders(
    datasets: Dict[str, AutoregressivePlayDataset],
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders from autoregressive datasets.
    
    Args:
        datasets: Dict of AutoregressivePlayDataset objects
        batch_size: Batch size for training
        num_workers: Number of data loading workers (0 for MPS compatibility)
        
    Returns:
        Dict of DataLoader objects
    """
    loaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        # Disable pin_memory for MPS (not supported)
        pin_memory = torch.cuda.is_available()  # Only pin memory for CUDA
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
    
    return loaders

