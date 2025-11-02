"""
Linear models for yards prediction and success classification.
"""
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class RidgeYardsHead:
    """Ridge regression for yards prediction."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: L2 regularization strength
        """
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit ridge regression.
        
        Args:
            X_train: Training features
            y_train: Training yards gained
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print(f"Fitted Ridge (α={self.alpha}) - R²={self.model.score(X_train_scaled, y_train):.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict yards.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted yards
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LogitSuccessHead:
    """Logistic regression for success classification."""
    
    def __init__(self, C: float = 1.0, class_weight: str = 'balanced'):
        """
        Args:
            C: Inverse regularization strength
            class_weight: Class weight handling ('balanced' or None)
        """
        self.C = C
        self.class_weight = class_weight
        self.model = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)
        self.scaler = StandardScaler()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit logistic regression.
        
        Args:
            X_train: Training features
            y_train: Training success labels (0/1)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        train_acc = self.model.score(X_train_scaled, y_train)
        print(f"Fitted Logistic Regression (C={self.C}) - Train Acc={train_acc:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict success probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array (n_samples, 2) - [P(fail), P(success)]
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict success binary labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LinearHeads:
    """
    Combined linear models for both yards and success.
    """
    
    def __init__(self, ridge_alpha: float = 1.0, logit_C: float = 1.0):
        """
        Args:
            ridge_alpha: Ridge regularization strength
            logit_C: Logistic regression inverse regularization
        """
        self.yards_head = RidgeYardsHead(alpha=ridge_alpha)
        self.success_head = LogitSuccessHead(C=logit_C)
        
    def fit(self, X_train: np.ndarray, y_train_yards: np.ndarray, y_train_success: np.ndarray):
        """
        Fit both models.
        
        Args:
            X_train: Training features
            y_train_yards: Training yards
            y_train_success: Training success labels
        """
        self.yards_head.fit(X_train, y_train_yards)
        self.success_head.fit(X_train, y_train_success)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both yards and success probability.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (pred_yards, pred_success_prob)
        """
        pred_yards = self.yards_head.predict(X)
        pred_success = self.success_head.predict_proba(X)[:, 1]  # Probability of success class
        return pred_yards, pred_success

