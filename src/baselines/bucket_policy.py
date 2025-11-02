"""
Bucketed frequency policy - interpretable lookup table.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class BucketPolicy:
    """
    Policy that buckets situations and looks up best historical play.
    """
    
    def __init__(self):
        self.bucket_table = None
        self.meta = None  # Maps bucket keys to play IDs
    
    def _bucket_key(self, row: pd.Series) -> str:
        """
        Create bucket key from play row.
        
        Args:
            row: Play row with down, yardsToGo, yardline_100, clock_seconds
            
        Returns:
            String bucket key
        """
        down = int(row['down'])
        
        # Yards to go bins
        ytg = row['yardsToGo']
        if ytg <= 2:
            ytg_bin = '1-2'
        elif ytg <= 5:
            ytg_bin = '3-5'
        elif ytg <= 9:
            ytg_bin = '6-9'
        elif ytg <= 15:
            ytg_bin = '10-15'
        else:
            ytg_bin = '16+'
        
        # Yardline bins
        yl = row['yardline_100']
        if yl <= 20:
            yl_bin = 'RZ'  # Red zone
        elif yl <= 50:
            yl_bin = '21-50'
        elif yl <= 80:
            yl_bin = '51-80'
        else:
            yl_bin = 'Own1-20'
        
        # Clock bins (optional)
        clock = row.get('clock_seconds', 900)
        if clock > 600:
            clock_bin = '>600'
        elif clock > 120:
            clock_bin = '600-120'
        else:
            clock_bin = '<=120'
        
        return f"{down}_{ytg_bin}_{yl_bin}_{clock_bin}"
    
    def fit(self, plays_labeled: pd.DataFrame):
        """
        Build bucket lookup table from labeled plays.
        
        Args:
            plays_labeled: Labeled plays DataFrame
        """
        plays = plays_labeled.copy()
        plays['bucket_key'] = plays.apply(self._bucket_key, axis=1)
        
        # Aggregate by bucket
        bucket_stats = plays.groupby('bucket_key').agg({
            'reward_yards': ['mean', 'std', 'count'],
            'reward_success': 'mean',
            'reward_td': 'mean'
        }).reset_index()
        
        bucket_stats.columns = ['bucket_key', 'avg_yards', 'std_yards', 'count', 'success_rate', 'td_rate']
        
        # Find best play per bucket (highest yards)
        best_plays = plays.loc[
            plays.groupby('bucket_key')['reward_yards'].idxmax()
        ][['bucket_key', 'gameId', 'playId', 'reward_yards', 'reward_success']].copy()
        best_plays.columns = ['bucket_key', 'best_gameId', 'best_playId', 'best_yards', 'best_success']
        
        # Merge
        self.bucket_table = bucket_stats.merge(best_plays, on='bucket_key', how='left')
        
        # Store metadata for all plays in each bucket
        bucket_meta = {}
        for key, group in plays.groupby('bucket_key'):
            bucket_meta[key] = group[['gameId', 'playId', 'reward_yards']].to_dict('records')
        self.meta = bucket_meta
        
        print(f"Built bucket table with {len(self.bucket_table)} unique buckets")
        print(f"  Avg plays per bucket: {plays.groupby('bucket_key').size().mean():.1f}")
    
    def predict(self, plays: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict expected yards and success for plays.
        
        Args:
            plays: Plays DataFrame with same columns as training
            
        Returns:
            Tuple of (pred_yards, pred_success_prob)
        """
        plays = plays.copy()
        plays['bucket_key'] = plays.apply(self._bucket_key, axis=1)
        
        # Lookup bucket stats
        pred = plays.merge(
            self.bucket_table[['bucket_key', 'avg_yards', 'success_rate']],
            on='bucket_key',
            how='left'
        )
        
        # Fill missing with global means
        pred_yards = pred['avg_yards'].fillna(self.bucket_table['avg_yards'].mean()).values
        pred_success = pred['success_rate'].fillna(self.bucket_table['success_rate'].mean()).values
        
        return pred_yards, pred_success
    
    def suggest_play(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Suggest best play for each situation.
        
        Args:
            plays: Plays DataFrame
            
        Returns:
            DataFrame with suggestions
        """
        plays = plays.copy()
        plays['bucket_key'] = plays.apply(self._bucket_key, axis=1)
        
        # Lookup best play from bucket table
        suggestions = plays.merge(
            self.bucket_table[['bucket_key', 'best_gameId', 'best_playId', 'avg_yards', 'success_rate']],
            on='bucket_key',
            how='left'
        )
        
        suggestions = suggestions.rename(columns={
            'best_gameId': 'gameId',
            'best_playId': 'playId',
            'avg_yards': 'expected_yards',
            'success_rate': 'success_prob'
        })
        
        return suggestions[['gameId', 'playId', 'expected_yards', 'success_prob']]
    
    def get_table(self) -> pd.DataFrame:
        """Return the bucket lookup table."""
        return self.bucket_table.copy()

