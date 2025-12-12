"""
Main script to train auxiliary play suggestion models.

NOTE: These are NON-GENERATIVE models for play-level decision quality.
For trajectory generation baselines, use train_autoregressive.py instead.

Models trained here:
- kNN Policy (state-based similarity search)
- Bucket Policy (interpretable lookup table)
- Linear Models (Ridge/Logistic regression)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_io import load_raw, filter_normal_plays
from src.preprocess import standardize_and_join
from src.rewards import add_labels
from src.features import build_state_matrix, scale_features
from src.splits import make_splits
from src.baselines.knn_policy import KNNPolicy
from src.baselines.bucket_policy import BucketPolicy
from src.baselines.linear_heads import LinearHeads
from src.eval import evaluate_model, top_k_precision
from src.api import BaselineSuite


def main():
    """Train and evaluate all baseline models."""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("NFL Play Suggestion - Baseline Training")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/8] Loading raw data...")
    data = load_raw(config['data_dir'])
    
    # Step 2: Filter normal plays
    print("\n[2/8] Filtering normal plays...")
    plays_clean = filter_normal_plays(data['plays'])
    
    # Step 3: Preprocess
    print("\n[3/8] Preprocessing and standardizing...")
    plays_std, tracking_std = standardize_and_join(
        plays_clean,
        data['games'],
        data['tracking']
    )
    
    # Step 4: Add labels
    print("\n[4/8] Computing rewards and labels...")
    plays_labeled = add_labels(plays_std)
    
    # Step 5: Split data
    print("\n[5/8] Creating train/val/test splits...")
    splits = make_splits(
        plays_labeled,
        by=config['splits']['by'],
        train=config['splits']['train'],
        val=config['splits']['val'],
        test=config['splits']['test']
    )
    
    # Step 6: Build features
    print("\n[6/8] Building state features...")
    X_train, y_train_yards, y_train_success, meta_train = build_state_matrix(splits['train'])
    X_val, y_val_yards, y_val_success, meta_val = build_state_matrix(splits['val'])
    X_test, y_test_yards, y_test_success, meta_test = build_state_matrix(splits['test'])
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    # Step 7: Train baselines
    print("\n[7/8] Training baseline models...")
    
    # Baseline 1: kNN
    print("  Training kNN policy...")
    knn = KNNPolicy(
        k=config['baselines']['knn']['k'],
        metric=config['baselines']['knn']['metric'],
        algorithm=config['baselines']['knn']['algorithm']
    )
    knn.fit(X_train_scaled, y_train_yards, y_train_success, meta_train)
    
    # Baseline 2: Bucket Policy
    print("  Training bucket policy...")
    bucket = BucketPolicy()
    bucket.fit(splits['train'])
    
    # Baseline 3: Linear Models
    print("  Training linear models...")
    linear = LinearHeads(
        ridge_alpha=config['baselines']['linear']['ridge_alpha'],
        logit_C=config['baselines']['linear']['logit_C']
    )
    linear.fit(X_train_scaled, y_train_yards, y_train_success)
    
    # Step 8: Evaluate
    print("\n[8/8] Evaluating models...")
    
    results = {}
    
    # Evaluate kNN
    print("\n  kNN Policy:")
    pred_yards_knn, pred_success_knn = knn.predict(X_val_scaled)
    metrics_knn = evaluate_model(y_val_yards, pred_yards_knn, y_val_success, pred_success_knn)
    results['knn'] = metrics_knn
    print(f"    MAE: {metrics_knn['yards_mae']:.2f}, RMSE: {metrics_knn['yards_rmse']:.2f}")
    print(f"    R²: {metrics_knn['yards_r2']:.4f}")
    print(f"    Success AUC: {metrics_knn.get('success_auc', np.nan):.4f}")
    
    # Evaluate Bucket
    print("\n  Bucket Policy:")
    pred_yards_bucket, pred_success_bucket = bucket.predict(splits['val'])
    metrics_bucket = evaluate_model(y_val_yards, pred_yards_bucket, y_val_success, pred_success_bucket)
    results['bucket'] = metrics_bucket
    print(f"    MAE: {metrics_bucket['yards_mae']:.2f}, RMSE: {metrics_bucket['yards_rmse']:.2f}")
    print(f"    R²: {metrics_bucket['yards_r2']:.4f}")
    print(f"    Success AUC: {metrics_bucket.get('success_auc', np.nan):.4f}")
    
    # Evaluate Linear
    print("\n  Linear Models:")
    pred_yards_linear, pred_success_linear = linear.predict(X_val_scaled)
    metrics_linear = evaluate_model(y_val_yards, pred_yards_linear, y_val_success, pred_success_linear)
    results['linear'] = metrics_linear
    print(f"    MAE: {metrics_linear['yards_mae']:.2f}, RMSE: {metrics_linear['yards_rmse']:.2f}")
    print(f"    R²: {metrics_linear['yards_r2']:.4f}")
    print(f"    Success AUC: {metrics_linear.get('success_auc', np.nan):.4f}")
    
    # Save artifacts
    if config['output']['save_models']:
        artifacts_dir = Path(config['output']['artifacts_dir'])
        artifacts_dir.mkdir(exist_ok=True)
        
        # Create a feature template for API inference
        # This ensures pd.get_dummies creates all columns
        all_downs = [1, 2, 3, 4]
        all_quarters = [1, 2, 3, 4, 5]
        template_rows = []
        for down in all_downs:
            for quarter in all_quarters:
                template_rows.append({
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
        feature_template = pd.DataFrame(template_rows)
        
        artifacts = {
            'knn': knn,
            'bucket': bucket,
            'linear': linear,
            'scaler': scaler,
            'feature_template': feature_template,
            'meta': {
                'feature_dim': X_train_scaled.shape[1],
                'train_size': len(X_train_scaled),
                'val_size': len(X_val_scaled)
            }
        }
        
        artifacts_path = artifacts_dir / 'baselines.pkl'
        joblib.dump(artifacts, artifacts_path)
        print(f"\n✓ Saved artifacts to {artifacts_path}")
        
        # Save evaluation results
        results_path = artifacts_dir / 'results.json'
        import json
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON
            results_json = {}
            for model, metrics in results.items():
                results_json[model] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in metrics.items()
                }
            json.dump(results_json, f, indent=2)
        print(f"✓ Saved evaluation results to {results_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Example usage
    print("\nExample: Suggest a play")
    print("-" * 60)
    suite = BaselineSuite(artifacts)
    situation = {
        'down': 3,
        'ydstogo': 7,
        'yardline_100': 45,
        'clock_seconds': 300,
        'score_diff': -3,
        'quarter': 2
    }
    result = suite.suggest_play(situation, mode='knn')
    print(f"Situation: {situation}")
    print(f"Suggested play: gameId={result['suggestion']['gameId']}, playId={result['suggestion']['playId']}")
    print(f"Expected yards: {result['expected_yards']:.2f}")
    print(f"Success probability: {result['success_prob']:.3f}")


if __name__ == '__main__':
    main()

