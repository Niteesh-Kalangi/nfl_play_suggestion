"""
Train/validation/test splitting utilities.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def make_splits(
    plays_labeled: pd.DataFrame,
    by: str = 'week',
    train: List[int] = [1, 2, 3, 4, 5, 6],
    val: List[int] = [7, 8],
    test: List[int] = [8]  # Default to week 8 if only 8 weeks available
) -> Dict[str, pd.DataFrame]:
    """
    Split plays into train/val/test sets by week or game.
    
    Args:
        plays_labeled: Labeled plays DataFrame with 'week' column
        by: 'week' or 'game' - how to split
        train: List of week numbers (or game IDs) for training
        val: List of week numbers (or game IDs) for validation
        test: List of week numbers (or game IDs) for testing
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to DataFrames
    """
    if by == 'week':
        split_col = 'week'
    elif by == 'game':
        split_col = 'gameId'
    else:
        raise ValueError(f"Unknown split method: {by}")
    
    # Get available values
    available_values = sorted(plays_labeled[split_col].unique())
    
    # Adjust defaults if needed
    if by == 'week':
        # Use available weeks
        max_week = int(plays_labeled['week'].max())
        if max_week < max(train):
            train = [w for w in train if w <= max_week]
        if max_week < max(val):
            val = [w for w in val if w <= max_week]
        if max_week < max(test):
            test = [w for w in test if w <= max_week]
        # If no test weeks specified or available, use last week
        if not test or max(test) > max_week:
            test = [max_week]
    
    train_df = plays_labeled[plays_labeled[split_col].isin(train)].copy()
    val_df = plays_labeled[plays_labeled[split_col].isin(val)].copy()
    test_df = plays_labeled[plays_labeled[split_col].isin(test)].copy()
    
    print(f"Split by {by}:")
    print(f"  Train: {len(train_df)} plays (weeks {sorted(train_df[split_col].unique())})")
    print(f"  Val:   {len(val_df)} plays (weeks {sorted(val_df[split_col].unique())})")
    print(f"  Test:  {len(test_df)} plays (weeks {sorted(test_df[split_col].unique())})")
    
    return {
        'train': train_df.reset_index(drop=True),
        'val': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True)
    }

