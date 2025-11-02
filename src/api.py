"""
API for play suggestion inference.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from src.baselines.knn_policy import KNNPolicy
from src.baselines.bucket_policy import BucketPolicy
from src.baselines.linear_heads import LinearHeads
from src.features import build_state_matrix
import joblib
from pathlib import Path


class BaselineSuite:
    """
    Unified API for all baseline models.
    """
    
    def __init__(self, artifacts: Optional[Dict] = None):
        """
        Initialize baseline suite from artifacts.
        
        Args:
            artifacts: Dictionary containing:
                - 'knn': KNNPolicy instance
                - 'bucket': BucketPolicy instance
                - 'ridge': RidgeYardsHead instance
                - 'logit': LogitSuccessHead instance
                - 'linear': LinearHeads instance (alternative)
                - 'scaler': StandardScaler for features
                - 'meta': Metadata mappings
                - 'feature_template': Optional DataFrame template for feature building
        """
        if artifacts is None:
            artifacts = {}
        
        self.knn = artifacts.get('knn')
        self.bucket = artifacts.get('bucket')
        self.ridge = artifacts.get('ridge')
        self.logit = artifacts.get('logit')
        self.linear = artifacts.get('linear')
        self.scaler = artifacts.get('scaler')
        self.meta = artifacts.get('meta', {})
        self.feature_template = artifacts.get('feature_template')
        
    def suggest_play(
        self,
        situation: Dict,
        k: int = 50,
        mode: str = 'knn'
    ) -> Dict:
        """
        Suggest a play for a given situation.
        
        Args:
            situation: Dictionary with keys:
                - down: int (1-4)
                - ydstogo: float (yards to go)
                - yardline_100: float (distance to end zone, 0-100)
                - clock_seconds: int (seconds remaining in quarter)
                - score_diff: int (offense - defense score)
                - quarter: int (1-5, optional)
            k: Number of neighbors for kNN (if mode='knn')
            mode: 'knn', 'bucket', or 'linear'
            
        Returns:
            Dictionary with:
                - suggestion: Dict with 'gameId', 'playId' of exemplar play
                - expected_yards: float
                - success_prob: float
                - debug: Dict with additional info
        """
        # Convert situation to feature vector
        # This is simplified - in practice, use the same feature pipeline
        X = self._situation_to_features(situation)
        
        result = {
            'mode': mode,
            'suggestion': None,
            'expected_yards': None,
            'success_prob': None,
            'debug': {}
        }
        
        if mode == 'knn' and self.knn is not None:
            # Use kNN
            pred_yards, pred_success = self.knn.predict(X.reshape(1, -1))
            suggestion_df = self.knn.suggest_play(X.reshape(1, -1))
            
            result['suggestion'] = {
                'gameId': int(suggestion_df.iloc[0]['gameId']),
                'playId': int(suggestion_df.iloc[0]['playId'])
            }
            result['expected_yards'] = float(pred_yards[0])
            result['success_prob'] = float(pred_success[0])
            result['debug'] = {
                'exemplar_yards': float(suggestion_df.iloc[0]['exemplar_yards'])
            }
            
        elif mode == 'bucket' and self.bucket is not None:
            # Use bucket policy
            situation_df = pd.DataFrame([situation])
            pred_yards, pred_success = self.bucket.predict(situation_df)
            suggestion_df = self.bucket.suggest_play(situation_df)
            
            result['suggestion'] = {
                'gameId': int(suggestion_df.iloc[0]['gameId']),
                'playId': int(suggestion_df.iloc[0]['playId'])
            }
            result['expected_yards'] = float(pred_yards[0])
            result['success_prob'] = float(pred_success[0])
            
        elif mode == 'linear' and self.linear is not None:
            # Use linear models
            pred_yards, pred_success = self.linear.predict(X.reshape(1, -1))
            
            result['suggestion'] = None  # Linear models don't suggest specific plays
            result['expected_yards'] = float(pred_yards[0])
            result['success_prob'] = float(pred_success[0])
            result['debug'] = {'note': 'Linear model does not provide exemplar plays'}
            
        else:
            raise ValueError(f"Mode '{mode}' not available or model not fitted")
        
        return result
    
    def _situation_to_features(self, situation: Dict) -> np.ndarray:
        """
        Convert situation dict to feature vector using the exact same logic as build_state_matrix.
        Uses scaler's feature count to ensure correct dimensions.
        """
        import pandas as pd
        import numpy as np
        from src.features import build_state_matrix
        
        # If we have a template with all feature values, use it
        if self.feature_template is not None:
            # Create a copy and update with actual values for first row
            df = self.feature_template.copy()
            df.iloc[0] = {
                'yardsToGo': situation.get('ydstogo', 10.0),
                'yardline_100': situation.get('yardline_100', 50.0),
                'clock_seconds': situation.get('clock_seconds', 900),
                'score_diff': situation.get('score_diff', 0),
                'down': int(situation.get('down', 1)),
                'quarter': int(situation.get('quarter', 1)),
                'reward_yards': 0,
                'reward_success': 0,
                'gameId': 0,
                'playId': 0
            }
            X, _, _, _ = build_state_matrix(df)
            features = X[0]
            
            # Ensure correct dimensions using scaler
            if self.scaler is not None:
                expected_dim = self.scaler.n_features_in_
                if len(features) != expected_dim:
                    if len(features) < expected_dim:
                        # Pad with zeros
                        features = np.pad(features, (0, expected_dim - len(features)), 'constant')
                    else:
                        features = features[:expected_dim]
            
            return features
        
        # Otherwise, create a DataFrame with all possible categorical values
        # to ensure pd.get_dummies creates all columns
        all_downs = [1, 2, 3, 4]
        all_quarters = [1, 2, 3, 4, 5]
        
        rows = []
        # Create rows with all combinations, but use actual values for numerical features
        actual_down = int(situation.get('down', 1))
        actual_quarter = int(situation.get('quarter', 1))
        
        # Put the actual combination first, then add others
        rows.append({
            'yardsToGo': situation.get('ydstogo', 10.0),
            'yardline_100': situation.get('yardline_100', 50.0),
            'clock_seconds': situation.get('clock_seconds', 900),
            'score_diff': situation.get('score_diff', 0),
            'down': actual_down,
            'quarter': actual_quarter,
            'reward_yards': 0,
            'reward_success': 0,
            'gameId': 0,
            'playId': 0
        })
        
        # Add a few more combinations to ensure all columns are created
        for down in all_downs:
            for quarter in all_quarters:
                if (down, quarter) != (actual_down, actual_quarter):
                    rows.append({
                        'yardsToGo': 10.0,
                        'yardline_100': 50.0,
                        'clock_seconds': 900,
                        'score_diff': 0,
                        'down': down,
                        'quarter': quarter,
                        'reward_yards': 0,
                        'reward_success': 0,
                        'gameId': 0,
                        'playId': 0
                    })
                    if len(rows) >= 20:  # Limit to avoid too many rows
                        break
            if len(rows) >= 20:
                break
        
        df = pd.DataFrame(rows)
        X, _, _, _ = build_state_matrix(df)
        
        # Use the first row which has the actual values
        features = X[0]
        
        # Ensure correct dimensions using scaler
        if self.scaler is not None:
            expected_dim = self.scaler.n_features_in_
            if len(features) != expected_dim:
                if len(features) < expected_dim:
                    features = np.pad(features, (0, expected_dim - len(features)), 'constant')
                else:
                    features = features[:expected_dim]
        
        return features
    
    def save(self, path: str):
        """Save artifacts to disk."""
        artifacts = {
            'knn': self.knn,
            'bucket': self.bucket,
            'ridge': self.ridge,
            'logit': self.logit,
            'linear': self.linear,
            'scaler': self.scaler,
            'meta': self.meta
        }
        joblib.dump(artifacts, path)
        print(f"Saved artifacts to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load artifacts from disk."""
        artifacts = joblib.load(path)
        return cls(artifacts)

