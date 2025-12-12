"""
Classifier to measure context adherence of generated plays.
"""
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List
import numpy as np


class ContextAdherenceClassifier:
    """
    Classifier to infer game context (down, distance, formation) from generated plays.
    
    High accuracy indicates that the model successfully encodes context in generated plays.
    """
    
    def __init__(self):
        self.models = {}
        self.feature_extractor = None
    
    def extract_features(self, trajectories: torch.Tensor) -> np.ndarray:
        """
        Extract features from trajectories for classification.
        
        Args:
            trajectories: [N, T, P, F] - generated trajectories
            
        Returns:
            [N, feature_dim] - extracted features
        """
        N, T, P, F = trajectories.shape
        
        features = []
        
        for traj in trajectories:
            # Extract statistics
            if F >= 2:
                # Position features
                x_pos = traj[:, :, 0].cpu().numpy()
                y_pos = traj[:, :, 1].cpu().numpy()
                
                # Spatial statistics
                x_mean = np.mean(x_pos)
                x_std = np.std(x_pos)
                y_mean = np.mean(y_pos)
                y_std = np.std(y_pos)
                
                # Spread
                x_range = np.max(x_pos) - np.min(x_pos)
                y_range = np.max(y_pos) - np.min(y_pos)
                
                # Speed features (if available)
                if F >= 3:
                    speeds = traj[:, :, 2].cpu().numpy()
                    speed_mean = np.mean(speeds)
                    speed_std = np.std(speeds)
                    speed_max = np.max(speeds)
                else:
                    speed_mean = speed_std = speed_max = 0.0
                
                # Initial and final positions
                x_init = x_pos[0].mean()
                y_init = y_pos[0].mean()
                x_final = x_pos[-1].mean()
                y_final = y_pos[-1].mean()
                
                # Displacement
                displacement = np.sqrt((x_final - x_init)**2 + (y_final - y_init)**2)
                
                feature_vec = np.array([
                    x_mean, x_std, y_mean, y_std,
                    x_range, y_range,
                    speed_mean, speed_std, speed_max,
                    x_init, y_init, x_final, y_final,
                    displacement
                ])
            else:
                # Fallback: just use raw values
                feature_vec = traj.cpu().numpy().flatten()[:14]  # Take first 14 values
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def train(
        self,
        trajectories: torch.Tensor,
        contexts: List[Dict]
    ):
        """
        Train classifiers to predict context from trajectories.
        
        Args:
            trajectories: [N, T, P, F] - training trajectories
            contexts: List of context dicts with categorical features
        """
        # Extract features
        X = self.extract_features(trajectories)
        
        # Extract labels for each categorical feature
        labels = {
            'down': [ctx.get('down', 1) for ctx in contexts],
            'offensiveFormation': [ctx.get('offensiveFormation', 'UNKNOWN') for ctx in contexts],
            'personnelO': [ctx.get('personnelO', 'UNKNOWN') for ctx in contexts],
            'defTeam': [ctx.get('defTeam', 'UNKNOWN') for ctx in contexts],
            'situation': [ctx.get('situation', 'medium') for ctx in contexts]
        }
        
        # Train a classifier for each feature
        for feat_name, y in labels.items():
            # Convert to numeric labels
            if feat_name in ['down', 'situation']:
                if feat_name == 'down':
                    y_numeric = [min(int(v), 4) for v in y]
                else:
                    # Map situation strings
                    situation_map = {'short': 0, 'medium': 1, 'long': 2, 'UNKNOWN': 3}
                    y_numeric = [situation_map.get(str(v), 3) for v in y]
            else:
                # Hash-based encoding for string features
                unique_vals = list(set(y))
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                y_numeric = [val_to_idx.get(v, 0) for v in y]
            
            # Train Random Forest
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y_numeric)
            
            self.models[feat_name] = {
                'classifier': clf,
                'label_map': y_numeric if feat_name in ['down', 'situation'] else val_to_idx
            }
    
    def evaluate(
        self,
        trajectories: torch.Tensor,
        contexts: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate context adherence.
        
        Returns:
            Dict with accuracy for each context feature
        """
        if len(self.models) == 0:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        X = self.extract_features(trajectories)
        
        accuracies = {}
        
        # Evaluate each classifier
        for feat_name, model_info in self.models.items():
            clf = model_info['classifier']
            label_map = model_info['label_map']
            
            # Get true labels
            if feat_name in ['down', 'situation']:
                if feat_name == 'down':
                    y_true = [min(int(ctx.get('down', 1)), 4) for ctx in contexts]
                else:
                    situation_map = {'short': 0, 'medium': 1, 'long': 2, 'UNKNOWN': 3}
                    y_true = [situation_map.get(str(ctx.get('situation', 'medium')), 3) for ctx in contexts]
            else:
                y_true = [label_map.get(ctx.get(feat_name, 'UNKNOWN'), 0) for ctx in contexts]
            
            # Predict
            y_pred = clf.predict(X)
            
            # Compute accuracy
            accuracy = accuracy_score(y_true, y_pred)
            accuracies[feat_name] = accuracy
        
        # Average accuracy
        accuracies['mean'] = np.mean(list(accuracies.values()))
        
        return accuracies

