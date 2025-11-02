"""
Data loading utilities for NFL play data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_raw(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load all raw CSV files from the data directory.
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        Dictionary mapping table names to DataFrames:
        {
            'plays': plays DataFrame,
            'games': games DataFrame,
            'players': players DataFrame,
            'tracking': concatenated tracking DataFrames from all weeks
        }
    """
    data_path = Path(data_dir)
    
    # Load core tables
    plays = pd.read_csv(data_path / "plays.csv")
    games = pd.read_csv(data_path / "games.csv")
    players = pd.read_csv(data_path / "players.csv")
    
    # Load tracking data from all weeks
    tracking_files = sorted(data_path.glob("week*.csv"))
    if not tracking_files:
        raise FileNotFoundError(f"No tracking files found in {data_dir}")
    
    tracking_dfs = []
    for file in tracking_files:
        week_df = pd.read_csv(file)
        tracking_dfs.append(week_df)
    
    tracking = pd.concat(tracking_dfs, ignore_index=True)
    
    print(f"Loaded {len(plays)} plays, {len(games)} games, {len(players)} players")
    print(f"Loaded {len(tracking)} tracking frames from {len(tracking_files)} weeks")
    
    return {
        'plays': plays,
        'games': games,
        'players': players,
        'tracking': tracking
    }


def filter_normal_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to normal offensive plays, dropping penalties, kneel-downs, spikes.
    
    Args:
        plays: Raw plays DataFrame
        
    Returns:
        Filtered plays DataFrame
    """
    # Drop plays with penalties (penaltyYards not NA)
    plays_clean = plays[plays['penaltyYards'].isna()].copy()
    
    # Drop special play types (kneel-downs, spikes, etc.)
    # These often have specific play descriptions or very negative yardage
    plays_clean = plays_clean[
        ~plays_clean['playDescription'].str.contains(
            'kneel|spike|victory formation', case=False, na=False
        )
    ].copy()
    
    # Drop plays with missing critical fields
    plays_clean = plays_clean.dropna(subset=['down', 'yardsToGo', 'playResult'])
    
    print(f"Filtered to {len(plays_clean)} normal offensive plays (from {len(plays)})")
    
    return plays_clean.reset_index(drop=True)

