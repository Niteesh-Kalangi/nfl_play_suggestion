"""
API for play suggestion and trajectory generation inference.

This module provides two main API classes:

1. BaselineSuite - Auxiliary play suggestion models (kNN, bucket, linear)
   These are NON-GENERATIVE models for play-level decisions.

2. TrajectoryBaselines - Autoregressive trajectory generators (LSTM, Transformer)
   These are the PRIMARY BASELINES for comparing against diffusion models.
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Union
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


# ============================================================================
# TRAJECTORY GENERATION API (Primary Baselines for Diffusion Comparison)
# ============================================================================

class TrajectoryBaselines:
    """
    API for autoregressive trajectory generation models.
    
    These are the PRIMARY BASELINES for comparing against diffusion models
    in the trajectory generation task. They support:
    - LSTM-based autoregressive generation
    - Transformer-based autoregressive generation
    
    Note: The play suggestion models (kNN, bucket, linear) in BaselineSuite
    are AUXILIARY models for play-level decisions, not trajectory generation.
    """
    
    def __init__(
        self,
        lstm_model: Optional[torch.nn.Module] = None,
        transformer_model: Optional[torch.nn.Module] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trajectory baselines.
        
        Args:
            lstm_model: Trained LSTM trajectory generator
            transformer_model: Trained Transformer trajectory generator
            config: Configuration dict
            device: Device for inference
        """
        self.lstm = lstm_model
        self.transformer = transformer_model
        self.config = config or {}
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Move models to device
        if self.lstm is not None:
            self.lstm = self.lstm.to(self.device)
            self.lstm.eval()
        if self.transformer is not None:
            self.transformer = self.transformer.to(self.device)
            self.transformer.eval()
    
    @classmethod
    def load(cls, checkpoint_dir: str, device: Optional[torch.device] = None):
        """
        Load trained models from checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing lstm.pt and transformer.pt
            device: Device for inference
            
        Returns:
            TrajectoryBaselines instance
        """
        from autoreg.models.autoregressive_lstm import LSTMTrajectoryGenerator
        from autoreg.models.autoregressive_transformer import TransformerTrajectoryGenerator
        
        checkpoint_dir = Path(checkpoint_dir)
        
        lstm_model = None
        transformer_model = None
        config = None
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        
        # Load LSTM
        lstm_path = checkpoint_dir / 'lstm.pt'
        if lstm_path.exists():
            checkpoint = torch.load(lstm_path, map_location=device)
            config = checkpoint.get('config', {})
            ar_config = config.get('autoregressive', {})
            lstm_config = ar_config.get('lstm', {})
            
            lstm_model = LSTMTrajectoryGenerator(
                input_dim=checkpoint['input_dim'],
                output_dim=checkpoint['output_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                num_layers=lstm_config.get('num_layers', 2),
                dropout=lstm_config.get('dropout', 0.1)
            )
            lstm_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded LSTM model from {lstm_path}")
        
        # Load Transformer
        transformer_path = checkpoint_dir / 'transformer.pt'
        if transformer_path.exists():
            checkpoint = torch.load(transformer_path, map_location=device)
            config = checkpoint.get('config', config or {})
            ar_config = config.get('autoregressive', {})
            tf_config = ar_config.get('transformer', {})
            
            transformer_model = TransformerTrajectoryGenerator(
                input_dim=checkpoint['input_dim'],
                output_dim=checkpoint['output_dim'],
                d_model=checkpoint['hidden_dim'],
                nhead=tf_config.get('nhead', 8),
                num_layers=tf_config.get('num_layers', 4),
                dim_feedforward=tf_config.get('dim_feedforward', 512),
                dropout=tf_config.get('dropout', 0.1)
            )
            transformer_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded Transformer model from {transformer_path}")
        
        return cls(lstm_model, transformer_model, config, device)
    
    @torch.no_grad()
    def generate_with_lstm(
        self,
        context: Union[Dict, torch.Tensor, np.ndarray],
        horizon: int,
        init_positions: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generate trajectory using LSTM model.
        
        Args:
            context: Game state context (dict, tensor, or array)
            horizon: Number of timesteps to generate
            init_positions: Optional initial player positions
            
        Returns:
            Generated trajectory [horizon, output_dim] or [batch, horizon, output_dim]
        """
        if self.lstm is None:
            raise ValueError("LSTM model not loaded")
        
        # Convert context to tensor
        context_tensor = self._prepare_context(context)
        
        # Convert init_positions if provided
        if init_positions is not None:
            if isinstance(init_positions, np.ndarray):
                init_positions = torch.FloatTensor(init_positions).to(self.device)
            elif isinstance(init_positions, torch.Tensor):
                init_positions = init_positions.to(self.device)
        
        # Generate
        trajectory = self.lstm.rollout(context_tensor, horizon, init_positions)
        
        return trajectory.cpu().numpy()
    
    @torch.no_grad()
    def generate_with_transformer(
        self,
        context: Union[Dict, torch.Tensor, np.ndarray],
        horizon: int,
        init_positions: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generate trajectory using Transformer model.
        
        Args:
            context: Game state context (dict, tensor, or array)
            horizon: Number of timesteps to generate
            init_positions: Optional initial player positions
            
        Returns:
            Generated trajectory [horizon, output_dim] or [batch, horizon, output_dim]
        """
        if self.transformer is None:
            raise ValueError("Transformer model not loaded")
        
        # Convert context to tensor
        context_tensor = self._prepare_context(context)
        
        # Convert init_positions if provided
        if init_positions is not None:
            if isinstance(init_positions, np.ndarray):
                init_positions = torch.FloatTensor(init_positions).to(self.device)
            elif isinstance(init_positions, torch.Tensor):
                init_positions = init_positions.to(self.device)
        
        # Generate
        trajectory = self.transformer.rollout(context_tensor, horizon, init_positions)
        
        return trajectory.cpu().numpy()
    
    def generate(
        self,
        context: Union[Dict, torch.Tensor, np.ndarray],
        horizon: int,
        model: str = 'lstm',
        init_positions: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generate trajectory using specified model.
        
        Args:
            context: Game state context
            horizon: Number of timesteps to generate
            model: 'lstm' or 'transformer'
            init_positions: Optional initial player positions
            
        Returns:
            Generated trajectory
        """
        if model == 'lstm':
            return self.generate_with_lstm(context, horizon, init_positions)
        elif model == 'transformer':
            return self.generate_with_transformer(context, horizon, init_positions)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'lstm' or 'transformer'.")
    
    def _prepare_context(self, context: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Prepare context for model input.
        
        Args:
            context: Context in various formats
            
        Returns:
            Context tensor [batch, context_dim]
        """
        if isinstance(context, dict):
            # Convert dict to tensor
            context_list = [
                context.get('down', 1),
                context.get('yardsToGo', context.get('ydstogo', 10)),
                context.get('yardline_100', 50),
                context.get('clock_seconds', 900),
                context.get('score_diff', 0),
                context.get('quarter', 1)
            ]
            context_tensor = torch.FloatTensor([context_list]).to(self.device)
        elif isinstance(context, np.ndarray):
            context_tensor = torch.FloatTensor(context).to(self.device)
            if context_tensor.dim() == 1:
                context_tensor = context_tensor.unsqueeze(0)
        elif isinstance(context, torch.Tensor):
            context_tensor = context.to(self.device)
            if context_tensor.dim() == 1:
                context_tensor = context_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
        
        return context_tensor
    
    def available_models(self) -> list:
        """Return list of available model types."""
        models = []
        if self.lstm is not None:
            models.append('lstm')
        if self.transformer is not None:
            models.append('transformer')
        return models

