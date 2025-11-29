"""
PyTorch Dataset for loading preprocessed football play tensors.
"""
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple
import json


class FootballPlayDataset(Dataset):
    """
    Dataset for football play trajectories with conditioning.
    
    Returns:
        X: Tensor [T, P, F] - player trajectories
        context: Dict with 'categorical' and 'continuous' features
        mask: Tensor [T] - valid frame mask
    """
    
    def __init__(
        self,
        cache_file: Path,
        metadata_file: Optional[Path] = None,
        split: Optional[str] = None,
        weeks: Optional[list] = None
    ):
        """
        Args:
            cache_file: Path to processed_plays.pkl (or .parquet for backwards compat)
            metadata_file: Path to metadata.json (optional)
            split: 'train', 'val', 'test' (optional, filters by week)
            weeks: List of week numbers to include (optional)
        """
        self.cache_file = cache_file
        
        # Auto-detect file format if no extension provided
        if cache_file.suffix == '':
            # Try pickle first (preferred), then parquet
            pkl_file = cache_file.parent / (cache_file.name + '.pkl')
            parquet_file = cache_file.parent / (cache_file.name + '.parquet')
            if pkl_file.exists():
                cache_file = pkl_file
            elif parquet_file.exists():
                cache_file = parquet_file
            else:
                raise FileNotFoundError(f"Cache file not found. Tried {pkl_file} and {parquet_file}")
        
        # Load data - support both pickle and parquet formats
        if cache_file.suffix == '.pkl':
            import pickle
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
        elif cache_file.suffix == '.parquet':
            # Try parquet format (for backwards compatibility)
            try:
                df = pd.read_parquet(cache_file)
                # Convert parquet format to list of dicts
                self.data = []
                for _, row in df.iterrows():
                    # Reconstruct from parquet format
                    tensor_bytes = row['tensor']
                    tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
                    # Parse shape string - handle both tuple and string formats
                    shape_str = row['tensor_shape']
                    if isinstance(shape_str, str):
                        import ast
                        shape = ast.literal_eval(shape_str)
                    else:
                        shape = shape_str
                    tensor = tensor.reshape(shape)
                    self.data.append({
                        'gameId': row['gameId'],
                        'playId': row['playId'],
                        'week': row['week'],
                        'tensor': tensor,
                        'context': {
                            'categorical': pickle.loads(row['context_cat']),
                            'continuous': np.frombuffer(row['context_cont'], dtype=np.float32)
                        },
                        'frame_count': row['frame_count']
                    })
            except Exception as e:
                raise ValueError(f"Failed to load parquet file: {e}. Please use pickle format.")
        else:
            raise ValueError(f"Unsupported file format: {cache_file.suffix}. Use .pkl or .parquet")
        
        # Load metadata
        if metadata_file is None:
            metadata_file = cache_file.parent / 'metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Filter by split/weeks if specified
        if split is not None or weeks is not None:
            if weeks is None:
                # Use default splits by week
                if split == 'train':
                    weeks = [1, 2, 3, 4, 5, 6]
                elif split == 'val':
                    weeks = [7]
                elif split == 'test':
                    weeks = [8]
            
            if weeks:
                self.data = [d for d in self.data if d.get('week') in weeks]
        
        # Store tensor shape info
        if len(self.data) > 0:
            tensor_shape = self.data[0]['tensor'].shape
            self.T, self.P, self.F = tensor_shape
        else:
            self.T, self.P, self.F = 60, 22, 3
    
    def __len__(self) -> int:
        return len(self.data)
    
    def denormalize_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        Denormalize tensor from normalized space (mean=0, std=1) back to original coordinates.
        
        Args:
            tensor: Normalized tensor [T, P, F]
            
        Returns:
            Denormalized tensor [T, P, F] in original coordinate space
        """
        if 'normalization' not in self.metadata:
            # No normalization was applied - return as is
            return tensor
        
        norm_stats = self.metadata['normalization']
        means = np.array(norm_stats['means'])
        stds = np.array(norm_stats['stds'])
        
        # Denormalize: x = x_norm * std + mean
        denormalized = tensor.copy()
        for f_idx in range(tensor.shape[-1]):
            denormalized[:, :, f_idx] = tensor[:, :, f_idx] * stds[f_idx] + means[f_idx]
        
        return denormalized
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        play = self.data[idx]
        
        # Get tensor directly (already normalized during preprocessing)
        tensor = play['tensor']
        
        # Get context
        context_cat = play['context']['categorical']
        context_cont = play['context']['continuous']
        
        # Create mask (all frames are valid for now)
        # Could be improved to track actual frame counts
        mask = torch.ones(self.T, dtype=torch.float32)
        
        return {
            'X': torch.FloatTensor(tensor),  # [T, P, F]
            'context_categorical': context_cat,
            'context_continuous': torch.FloatTensor(context_cont),  # [2]
            'mask': mask,
            'player_positions': play.get('player_positions', None),  # List of position labels [P]
            'gameId': int(play['gameId']),
            'playId': int(play['playId']),
            'week': int(play.get('week', 0))
        }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Stacks tensors and preserves context dictionaries.
    """
    X = torch.stack([item['X'] for item in batch])  # [B, T, P, F]
    context_continuous = torch.stack([item['context_continuous'] for item in batch])  # [B, 2]
    mask = torch.stack([item['mask'] for item in batch])  # [B, T]
    
    # Keep categorical as list of dicts
    context_categorical = [item['context_categorical'] for item in batch]
    
    return {
        'X': X,
        'context_categorical': context_categorical,
        'context_continuous': context_continuous,
        'mask': mask,
        'gameIds': [item['gameId'] for item in batch],
        'playIds': [item['playId'] for item in batch],
        'weeks': [item['week'] for item in batch]
    }

