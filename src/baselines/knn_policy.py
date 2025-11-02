"""
kNN-based policy for play suggestion.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, Optional


class KNNPolicy:
    """
    k-Nearest Neighbors policy for play suggestion.
    
    Given a situation, finds similar historical situations and:
    - Predicts expected yards from mean of neighbors
    - Suggests exemplar play (neighbor with highest yards)
    """
    
    def __init__(self, k: int = 50, metric: str = 'euclidean', algorithm: str = 'ball_tree'):
        """
        Args:
            k: Number of neighbors to consider
            metric: Distance metric ('euclidean', 'manhattan', etc.)
            algorithm: Algorithm for nearest neighbors ('ball_tree', 'kd_tree', 'brute')
        """
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.knn = None
        self.X_train = None
        self.y_train_yards = None
        self.y_train_success = None
        self.meta_train = None
        
    def fit(self, X_train: np.ndarray, y_train_yards: np.ndarray, 
            y_train_success: np.ndarray, meta_train: pd.DataFrame):
        """
        Fit the kNN model on training data.
        
        Args:
            X_train: Training feature matrix
            y_train_yards: Training yards gained
            y_train_success: Training success labels
            meta_train: Training metadata (gameId, playId)
        """
        self.X_train = X_train
        self.y_train_yards = y_train_yards
        self.y_train_success = y_train_success
        self.meta_train = meta_train.copy()
        
        self.knn = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.metric,
            algorithm=self.algorithm
        )
        self.knn.fit(X_train)
        
        print(f"Fitted kNN with k={self.k} on {len(X_train)} training samples")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict expected yards and success probability.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (pred_yards, pred_success_prob)
        """
        distances, indices = self.knn.kneighbors(X, return_distance=True)
        
        pred_yards = []
        pred_success = []
        
        for i, neighbor_indices in enumerate(indices):
            # Average yards from neighbors
            yards = np.mean(self.y_train_yards[neighbor_indices])
            pred_yards.append(yards)
            
            # Average success probability
            success = np.mean(self.y_train_success[neighbor_indices])
            pred_success.append(success)
        
        return np.array(pred_yards), np.array(pred_success)
    
    def suggest_play(self, X: np.ndarray) -> pd.DataFrame:
        """
        Suggest exemplar plays for each query.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            DataFrame with columns: gameId, playId, expected_yards, success_prob
        """
        distances, indices = self.knn.kneighbors(X, return_distance=True)
        
        suggestions = []
        
        for i, neighbor_indices in enumerate(indices):
            # Find neighbor with highest yards
            neighbor_yards = self.y_train_yards[neighbor_indices]
            best_idx = neighbor_indices[np.argmax(neighbor_yards)]
            
            # Get exemplar play info
            exemplar = self.meta_train.iloc[best_idx]
            
            suggestions.append({
                'gameId': exemplar['gameId'],
                'playId': exemplar['playId'],
                'expected_yards': np.mean(neighbor_yards),
                'success_prob': np.mean(self.y_train_success[neighbor_indices]),
                'exemplar_yards': self.y_train_yards[best_idx]
            })
        
        return pd.DataFrame(suggestions)
    
    def get_neighbors(self, X: np.ndarray, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get neighbor indices and distances for queries.
        
        Args:
            X: Feature matrix
            top_k: Number of neighbors to return (defaults to self.k)
            
        Returns:
            Tuple of (indices, distances) arrays
        """
        k = top_k if top_k else self.k
        distances, indices = self.knn.kneighbors(X, n_neighbors=k, return_distance=True)
        return indices, distances

