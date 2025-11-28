"""
Data splitting utilities for train/val/test splits.
"""
from pathlib import Path
from typing import Dict, List
import pandas as pd
import json


def create_splits_by_week(
    cache_file: Path,
    train_weeks: List[int] = [1, 2, 3, 4, 5, 6],
    val_weeks: List[int] = [7],
    test_weeks: List[int] = [8]
) -> Dict[str, List[int]]:
    """
    Create train/val/test splits by week.
    
    Returns:
        Dict mapping split name to list of gameIds/playIds
    """
    # Load from pickle or parquet
    if cache_file.suffix == '.pkl':
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        splits = {
            'train': [{'gameId': d['gameId'], 'playId': d['playId']} 
                     for d in data if d.get('week') in train_weeks],
            'val': [{'gameId': d['gameId'], 'playId': d['playId']} 
                   for d in data if d.get('week') in val_weeks],
            'test': [{'gameId': d['gameId'], 'playId': d['playId']} 
                    for d in data if d.get('week') in test_weeks]
        }
    else:
        # Fallback to parquet
        df = pd.read_parquet(cache_file)
        splits = {
            'train': df[df['week'].isin(train_weeks)][['gameId', 'playId']].to_dict('records'),
            'val': df[df['week'].isin(val_weeks)][['gameId', 'playId']].to_dict('records'),
            'test': df[df['week'].isin(test_weeks)][['gameId', 'playId']].to_dict('records')
        }
    
    return splits


def save_splits(splits: Dict[str, List[int]], output_file: Path):
    """Save split indices to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)


def load_splits(split_file: Path) -> Dict[str, List[int]]:
    """Load split indices from JSON file."""
    with open(split_file) as f:
        return json.load(f)

