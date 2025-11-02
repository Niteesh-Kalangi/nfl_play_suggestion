"""
Evaluation metrics for play prediction models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from scipy.stats import spearmanr
from typing import Dict, Tuple, Optional


def evaluate_yards(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate yards prediction metrics.
    
    Args:
        y_true: True yards gained
        y_pred: Predicted yards gained
        
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Spearman rank correlation
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p
    }


def evaluate_success(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate success classification metrics.
    
    Args:
        y_true: True success labels (0/1)
        y_pred_prob: Predicted success probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    brier = brier_score_loss(y_true, y_pred_prob)
    
    # AUC (only if both classes present)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_prob)
    else:
        auc = np.nan
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    return {
        'brier_score': brier,
        'auc': auc,
        'accuracy': accuracy,
        'threshold': threshold
    }


def evaluate_model(
    y_true_yards: np.ndarray,
    y_pred_yards: np.ndarray,
    y_true_success: Optional[np.ndarray] = None,
    y_pred_success: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true_yards: True yards
        y_pred_yards: Predicted yards
        y_true_success: True success labels (optional)
        y_pred_success: Predicted success probabilities (optional)
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Yards metrics
    yards_metrics = evaluate_yards(y_true_yards, y_pred_yards)
    metrics.update({f'yards_{k}': v for k, v in yards_metrics.items()})
    
    # Success metrics (if provided)
    if y_true_success is not None and y_pred_success is not None:
        success_metrics = evaluate_success(y_true_success, y_pred_success)
        metrics.update({f'success_{k}': v for k, v in success_metrics.items()})
    
    return metrics


def top_k_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    by_group: Optional[pd.Series] = None
) -> float:
    """
    Top-K precision: Are the top-K predictions actually good outcomes?
    
    Args:
        y_true: True outcomes
        y_pred: Predicted outcomes (used for ranking)
        k: Number of top predictions to consider
        by_group: Optional grouping (e.g., by situation bucket)
        
    Returns:
        Average true outcome for top-K predictions
    """
    if by_group is not None:
        # Compute per-group top-K
        df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'group': by_group})
        top_k_values = []
        for group, group_df in df.groupby('group'):
            top_k_indices = group_df.nlargest(min(k, len(group_df)), 'pred').index
            top_k_values.append(group_df.loc[top_k_indices, 'true'].mean())
        return np.mean(top_k_values)
    else:
        # Global top-K
        top_k_indices = np.argsort(y_pred)[-k:]
        return np.mean(y_true[top_k_indices])


def counterfactual_policy_evaluation(
    plays_eval: pd.DataFrame,
    behavior_policy,
    target_policy_yards: np.ndarray,
    behavior_policy_yards: np.ndarray
) -> Dict[str, float]:
    """
    Simple Counterfactual Policy Evaluation using Self-Normalized IPS.
    
    Args:
        plays_eval: Evaluation plays DataFrame
        behavior_policy: Fitted behavior policy (for propensity estimation)
        target_policy_yards: Yards predicted by target policy
        behavior_policy_yards: Yards predicted by behavior policy
        
    Returns:
        Dictionary with CPE metrics
    """
    # Use uniform propensity as simple baseline (can be improved with bucket frequencies)
    # For simplicity, assume behavior policy is uniform over play types
    # In practice, estimate from bucket frequencies
    
    # Simple implementation: assume behavior propensities from bucket frequencies
    plays_eval = plays_eval.copy()
    plays_eval['bucket_key'] = plays_eval.apply(behavior_policy._bucket_key, axis=1)
    
    # Get bucket counts for propensity estimation
    if behavior_policy.bucket_table is not None:
        bucket_counts = behavior_policy.bucket_table.set_index('bucket_key')['count']
        total_plays = bucket_counts.sum()
        props = plays_eval['bucket_key'].map(bucket_counts).fillna(1) / total_plays
        props = np.clip(props, 1e-6, 1.0)  # Avoid zero/overflow
    else:
        props = np.ones(len(plays_eval)) / len(plays_eval)
    
    # Self-Normalized IPS
    weights = 1.0 / props
    weights = weights / weights.sum() * len(weights)  # Normalize
    
    # Estimated value
    snips_value = np.sum(weights * target_policy_yards) / len(plays_eval)
    
    # Doubly-robust (simplified - would need Q-function estimates)
    # For now, just return SNIPS
    return {
        'snips_value': snips_value,
        'behavior_value': np.mean(behavior_policy_yards),
        'target_value': np.mean(target_policy_yards),
        'uplift': snips_value - np.mean(behavior_policy_yards)
    }

