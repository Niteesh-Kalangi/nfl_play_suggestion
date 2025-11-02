"""
Feature engineering for state (situation) and pre-snap shape features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Optional, Dict


def build_state_matrix(plays_labeled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build state (situation) feature matrix from plays.
    
    Args:
        plays_labeled: Plays DataFrame with labels and context
        
    Returns:
        Tuple of:
        - X: Feature matrix (n_samples, n_features)
        - y_yards: Target yards gained
        - y_success: Target success binary
        - meta: DataFrame with gameId, playId for tracking
    """
    df = plays_labeled.copy()
    
    # Numerical features (will be z-scored)
    numerical_cols = [
        'yardsToGo',
        'yardline_100',
        'clock_seconds',
        'score_diff'
    ]
    
    # Categorical features (will be one-hot encoded)
    categorical_cols = [
        'down',
        'quarter'
    ]
    
    # Extract numerical features
    X_num = df[numerical_cols].fillna(0).values
    
    # Extract and one-hot encode categorical features
    X_cat_list = []
    for col in categorical_cols:
        # Fill missing with most common value
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 1)
        # One-hot encode
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        X_cat_list.append(dummies.values)
    
    X_cat = np.hstack(X_cat_list) if X_cat_list else np.array([]).reshape(len(df), 0)
    
    # Derived flags
    df['red_zone'] = (df['yardline_100'] <= 20).astype(int)
    df['short_yards'] = (df['yardsToGo'] <= 2).astype(int)
    df['two_minute'] = (df['clock_seconds'] <= 120).astype(int)
    
    derived_cols = ['red_zone', 'short_yards', 'two_minute']
    X_derived = df[derived_cols].values
    
    # Combine all features
    if X_cat.size > 0:
        X = np.hstack([X_num, X_cat, X_derived])
    else:
        X = np.hstack([X_num, X_derived])
    
    # Extract targets
    y_yards = df['reward_yards'].values
    y_success = df['reward_success'].values
    
    # Meta information
    meta = df[['gameId', 'playId']].copy()
    
    print(f"Built state features: {X.shape[1]} features from {len(df)} plays")
    
    return X, y_yards, y_success, meta


def build_presnap_shape(
    tracking_std: pd.DataFrame,
    plays_std: pd.DataFrame,
    players: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build pre-snap formation shape features from tracking data.
    
    Args:
        tracking_std: Standardized tracking DataFrame
        plays_std: Standardized plays DataFrame with possessionTeam
        players: Players DataFrame with positions
        
    Returns:
        Tuple of:
        - X_shape: Feature matrix (n_samples, n_shape_features)
        - meta: DataFrame with gameId, playId matching plays_std
    """
    # Get pre-snap frame
    from src.preprocess import extract_presnap_frame
    presnap = extract_presnap_frame(tracking_std, plays_std)
    
    # Join player positions
    presnap = presnap.merge(
        players[['nflId', 'officialPosition']],
        on='nflId',
        how='left'
    )
    
    # Initialize feature arrays
    shape_features = []
    meta_rows = []
    
    for (game_id, play_id), play_data in plays_std.groupby(['gameId', 'playId']):
        play_presnap = presnap[
            (presnap['gameId'] == game_id) &
            (presnap['playId'] == play_id) &
            (presnap['is_offense']) &
            (presnap['team'] != 'football')
        ].copy()
        
        if len(play_presnap) == 0:
            # Default features if no players found
            features = np.zeros(8)
        else:
            # Split field into left/right halves (x < 60 is left, x >= 60 is right)
            midline = 60.0
            
            # Count WRs on each side
            wr_positions = ['WR', 'TE']  # Include TEs as eligible receivers
            wrs = play_presnap[play_presnap['officialPosition'].isin(wr_positions)]
            wr_count_left = len(wrs[wrs['x'] < midline])
            wr_count_right = len(wrs[wrs['x'] >= midline])
            
            # Average depth of receivers
            if len(wrs) > 0:
                slot_depth_avg = wrs['y'].mean()  # Average y-position (width-wise)
            else:
                slot_depth_avg = 26.65  # Middle of field
            
            # Count inline TEs (close to line)
            tes = play_presnap[play_presnap['officialPosition'] == 'TE']
            te_inline_count = len(tes[tes['x'] < 5])  # Within 5 yards of LOS
            
            # RB depth (average x for RBs)
            rbs = play_presnap[play_presnap['officialPosition'] == 'RB']
            if len(rbs) > 0:
                rb_depth = rbs['x'].mean()
            else:
                rb_depth = 0.0
            
            # QB depth
            qbs = play_presnap[play_presnap['officialPosition'] == 'QB']
            if len(qbs) > 0:
                qb_depth = qbs['x'].mean()
            else:
                qb_depth = 0.0
            
            # Hash side (assume right hash if average y > 26.65)
            hash_side = 1 if play_presnap['y'].mean() > 26.65 else 0
            
            features = np.array([
                wr_count_left,
                wr_count_right,
                slot_depth_avg,
                te_inline_count,
                rb_depth,
                qb_depth,
                hash_side,
                len(play_presnap)  # Total offensive players
            ])
        
        shape_features.append(features)
        meta_rows.append({'gameId': game_id, 'playId': play_id})
    
    X_shape = np.array(shape_features)
    meta_shape = pd.DataFrame(meta_rows)
    
    print(f"Built pre-snap shape features: {X_shape.shape[1]} features from {len(plays_std)} plays")
    
    return X_shape, meta_shape


def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                   feature_weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler fitted on training data.
    
    Args:
        X_train: Training feature matrix
        X_val: Validation feature matrix
        X_test: Test feature matrix
        feature_weights: Optional dict mapping feature names to weights
        
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply feature weights if provided
    if feature_weights:
        # This is a simplified version - would need feature names mapping
        # For now, we'll skip this or implement a basic version
        pass
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

