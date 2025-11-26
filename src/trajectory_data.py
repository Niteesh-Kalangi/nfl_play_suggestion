"""
Trajectory data utilities for extracting multi-player trajectories from tracking data.
Used for training autoregressive trajectory generation models.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset


# Constants
NUM_OFFENSE_PLAYERS = 11  # Standard offensive players
PLAYER_FEATURES = 2  # x, y coordinates per player
CONTEXT_FEATURES = [
    'down', 'yardsToGo', 'yardline_100', 'clock_seconds', 'score_diff', 'quarter'
]


def extract_play_trajectories(
    tracking_std: pd.DataFrame,
    plays_std: pd.DataFrame,
    max_frames: int = 50,
    offense_only: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract multi-player trajectories for each play.
    
    Args:
        tracking_std: Standardized tracking DataFrame with columns:
            gameId, playId, frameId, nflId, x, y, is_offense, team
        plays_std: Standardized plays DataFrame with context features
        max_frames: Maximum number of frames per trajectory
        offense_only: If True, only extract offensive player trajectories
        
    Returns:
        Tuple of:
        - trajectories: Array of shape (n_plays, max_frames, num_players, 2)
        - context: Array of shape (n_plays, context_dim)
        - meta: DataFrame with gameId, playId for each trajectory
    """
    trajectories = []
    contexts = []
    meta_rows = []
    
    # Get unique plays
    play_keys = tracking_std.groupby(['gameId', 'playId']).ngroups
    
    for (game_id, play_id), play_tracking in tracking_std.groupby(['gameId', 'playId']):
        # Filter to offensive players only
        if offense_only:
            play_tracking = play_tracking[
                (play_tracking['is_offense']) & 
                (play_tracking['team'] != 'football')
            ]
        
        if len(play_tracking) == 0:
            continue
            
        # Get unique players and frames
        players = play_tracking['nflId'].unique()
        frames = sorted(play_tracking['frameId'].unique())
        
        # Skip if not enough players
        if len(players) < NUM_OFFENSE_PLAYERS:
            continue
        
        # Take first NUM_OFFENSE_PLAYERS (consistent ordering)
        players = sorted(players)[:NUM_OFFENSE_PLAYERS]
        
        # Build trajectory tensor: (frames, players, 2)
        n_frames = min(len(frames), max_frames)
        traj = np.zeros((max_frames, NUM_OFFENSE_PLAYERS, PLAYER_FEATURES))
        
        for f_idx, frame_id in enumerate(frames[:n_frames]):
            frame_data = play_tracking[play_tracking['frameId'] == frame_id]
            for p_idx, player_id in enumerate(players):
                player_frame = frame_data[frame_data['nflId'] == player_id]
                if len(player_frame) > 0:
                    traj[f_idx, p_idx, 0] = player_frame['x'].values[0]
                    traj[f_idx, p_idx, 1] = player_frame['y'].values[0]
        
        # Pad remaining frames with last known position
        if n_frames < max_frames:
            for f_idx in range(n_frames, max_frames):
                traj[f_idx] = traj[n_frames - 1]
        
        trajectories.append(traj)
        
        # Get context features from plays
        play_info = plays_std[
            (plays_std['gameId'] == game_id) & 
            (plays_std['playId'] == play_id)
        ]
        
        if len(play_info) > 0:
            ctx = []
            for feat in CONTEXT_FEATURES:
                if feat in play_info.columns:
                    val = play_info[feat].values[0]
                    ctx.append(float(val) if not pd.isna(val) else 0.0)
                else:
                    ctx.append(0.0)
            contexts.append(ctx)
        else:
            contexts.append([0.0] * len(CONTEXT_FEATURES))
        
        meta_rows.append({'gameId': game_id, 'playId': play_id, 'n_frames': n_frames})
    
    trajectories = np.array(trajectories, dtype=np.float32)
    contexts = np.array(contexts, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    
    print(f"Extracted {len(trajectories)} play trajectories")
    print(f"  Shape: {trajectories.shape}")
    print(f"  Context shape: {contexts.shape}")
    
    return trajectories, contexts, meta


def normalize_trajectories(
    trajectories: np.ndarray,
    contexts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Normalize trajectories and context features.
    
    Args:
        trajectories: Shape (n_plays, max_frames, num_players, 2)
        contexts: Shape (n_plays, context_dim)
        
    Returns:
        Tuple of:
        - normalized trajectories
        - normalized contexts
        - normalization stats dict
    """
    # Trajectory normalization (field coordinates)
    # x: 0-120 yards, y: 0-53.3 yards
    traj_mean = np.array([60.0, 26.65])  # Field center
    traj_std = np.array([30.0, 15.0])    # Approximate std
    
    traj_norm = (trajectories - traj_mean) / traj_std
    
    # Context normalization (z-score)
    ctx_mean = contexts.mean(axis=0)
    ctx_std = contexts.std(axis=0) + 1e-8
    ctx_norm = (contexts - ctx_mean) / ctx_std
    
    stats = {
        'traj_mean': traj_mean,
        'traj_std': traj_std,
        'ctx_mean': ctx_mean,
        'ctx_std': ctx_std
    }
    
    return traj_norm, ctx_norm, stats


def denormalize_trajectories(
    trajectories: np.ndarray,
    stats: Dict
) -> np.ndarray:
    """
    Denormalize trajectories back to field coordinates.
    """
    return trajectories * stats['traj_std'] + stats['traj_mean']


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory generation.
    
    For autoregressive training:
    - Input: trajectory frames 0 to T-1
    - Target: trajectory frames 1 to T
    """
    
    def __init__(
        self,
        trajectories: np.ndarray,
        contexts: np.ndarray,
        normalize: bool = True
    ):
        """
        Args:
            trajectories: Shape (n_plays, max_frames, num_players, 2)
            contexts: Shape (n_plays, context_dim)
            normalize: Whether to normalize data
        """
        self.raw_trajectories = trajectories
        self.raw_contexts = contexts
        
        if normalize:
            self.trajectories, self.contexts, self.stats = normalize_trajectories(
                trajectories, contexts
            )
        else:
            self.trajectories = trajectories
            self.contexts = contexts
            self.stats = None
        
        self.n_samples = len(trajectories)
        self.max_frames = trajectories.shape[1]
        self.n_players = trajectories.shape[2]
        self.player_dim = trajectories.shape[3]
        self.context_dim = contexts.shape[1]
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
            - 'input_traj': (max_frames-1, n_players * player_dim) - flattened input
            - 'target_traj': (max_frames-1, n_players * player_dim) - flattened target
            - 'context': (context_dim,) - context features
            - 'full_traj': (max_frames, n_players, player_dim) - full trajectory
        """
        traj = self.trajectories[idx]  # (max_frames, n_players, 2)
        ctx = self.contexts[idx]       # (context_dim,)
        
        # Flatten player dimension for sequence modeling
        # Shape: (max_frames, n_players * 2)
        traj_flat = traj.reshape(self.max_frames, -1)
        
        # Input: frames 0 to T-1, Target: frames 1 to T
        input_traj = traj_flat[:-1]   # (max_frames-1, n_players * 2)
        target_traj = traj_flat[1:]   # (max_frames-1, n_players * 2)
        
        return {
            'input_traj': torch.tensor(input_traj, dtype=torch.float32),
            'target_traj': torch.tensor(target_traj, dtype=torch.float32),
            'context': torch.tensor(ctx, dtype=torch.float32),
            'full_traj': torch.tensor(traj, dtype=torch.float32)
        }


def create_trajectory_splits(
    trajectories: np.ndarray,
    contexts: np.ndarray,
    meta: pd.DataFrame,
    plays_std: pd.DataFrame,
    train_weeks: List[int],
    val_weeks: List[int],
    test_weeks: List[int]
) -> Dict[str, TrajectoryDataset]:
    """
    Create train/val/test splits for trajectory data.
    
    Args:
        trajectories: Full trajectory array
        contexts: Full context array
        meta: Metadata with gameId, playId
        plays_std: Plays DataFrame with week info
        train_weeks, val_weeks, test_weeks: Week numbers for each split
        
    Returns:
        Dict with 'train', 'val', 'test' TrajectoryDataset objects
    """
    # Join week info to meta
    meta_with_week = meta.merge(
        plays_std[['gameId', 'playId', 'week']].drop_duplicates(),
        on=['gameId', 'playId'],
        how='left'
    )
    
    splits = {}
    
    for split_name, weeks in [('train', train_weeks), ('val', val_weeks), ('test', test_weeks)]:
        mask = meta_with_week['week'].isin(weeks)
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            split_traj = trajectories[indices]
            split_ctx = contexts[indices]
            
            # Only normalize based on training data
            if split_name == 'train':
                splits[split_name] = TrajectoryDataset(split_traj, split_ctx, normalize=True)
                train_stats = splits[split_name].stats
            else:
                # Use training stats for val/test
                traj_norm = (split_traj - train_stats['traj_mean']) / train_stats['traj_std']
                ctx_norm = (split_ctx - train_stats['ctx_mean']) / train_stats['ctx_std']
                
                dataset = TrajectoryDataset(split_traj, split_ctx, normalize=False)
                dataset.trajectories = traj_norm
                dataset.contexts = ctx_norm
                dataset.stats = train_stats
                splits[split_name] = dataset
        
        print(f"{split_name}: {len(indices)} trajectories")
    
    return splits
