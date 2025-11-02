"""
Reward computation utilities for labeling plays.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def add_labels(plays_std: pd.DataFrame) -> pd.DataFrame:
    """
    Add reward labels to plays: yards gained, success, and derived metrics.
    
    Args:
        plays_std: Standardized plays DataFrame
        
    Returns:
        Plays DataFrame with added columns:
        - reward_yards: yardsGained (from playResult)
        - reward_success: 1 if play was successful, 0 otherwise
        - reward_td: 1 if touchdown, 0 otherwise
    """
    plays_labeled = plays_std.copy()
    
    # Use playResult as yards gained
    plays_labeled['reward_yards'] = plays_labeled['playResult'].fillna(0)
    
    # Compute success based on down and yards to go
    # 1st/2nd down: gain >= 50% (1st) or >= 70% (2nd) of yards-to-go
    # 3rd/4th: gain >= yards-to-go
    def compute_success(row):
        down = row['down']
        ytg = row['yardsToGo']
        yards = row['reward_yards']
        
        if pd.isna(down) or pd.isna(ytg) or pd.isna(yards):
            return 0
        
        down = int(down)
        ytg = float(ytg)
        yards = float(yards)
        
        if down == 1:
            threshold = 0.5 * ytg
        elif down == 2:
            threshold = 0.7 * ytg
        else:  # 3rd or 4th down
            threshold = ytg
        
        return 1 if yards >= threshold else 0
    
    plays_labeled['reward_success'] = plays_labeled.apply(compute_success, axis=1)
    
    # Touchdown indicator (check play description or if playResult resulted in TD)
    # Simple heuristic: very high yardage gain or TD mentioned in description
    plays_labeled['reward_td'] = (
        (plays_labeled['playDescription'].str.contains('TOUCHDOWN', case=False, na=False)) |
        (plays_labeled['reward_yards'] >= plays_labeled['yardline_100'])
    ).astype(int)
    
    # Ensure reward_yards is numeric
    plays_labeled['reward_yards'] = pd.to_numeric(plays_labeled['reward_yards'], errors='coerce').fillna(0)
    
    print(f"Added labels: avg_yards={plays_labeled['reward_yards'].mean():.2f}, "
          f"success_rate={plays_labeled['reward_success'].mean():.3f}, "
          f"td_rate={plays_labeled['reward_td'].mean():.3f}")
    
    return plays_labeled

