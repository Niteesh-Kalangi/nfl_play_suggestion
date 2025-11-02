# NFL Play Suggestion Baseline Models

Baseline models for suggesting NFL plays based on game situation (down, distance, field position, etc.) using historical play data from the NFL Big Data Bowl 2023.

## Overview

This project implements three progressively stronger baseline models for play suggestion:

1. **State-only kNN** - Finds similar historical situations and suggests the best exemplar play
2. **Bucketed Frequency Policy** - Interpretable lookup table by situation buckets
3. **Light Linear Models** - Ridge regression for yards, Logistic regression for success

All models share a clean data pipeline and can be extended to Baseline 1.5 (State + Pre-snap Shape kNN).

## Setup

### 1. Create Conda Environment

```bash
conda create -n nfl_baselines python=3.9
conda activate nfl_baselines
pip install pandas numpy scikit-learn scipy pyyaml joblib
```

### 2. Download Data

Place the NFL Big Data Bowl 2023 data files in the `data/` directory. See `data/README.md` for details on where to download the dataset.

### 3. Train Models

```bash
python train_baselines.py
```

## Project Structure

```
dl_project/
├── data/                    # Raw CSV files
│   ├── plays.csv
│   ├── games.csv
│   ├── players.csv
│   └── week*.csv           # Tracking data
├── src/
│   ├── data_io.py          # Data loading
│   ├── preprocess.py        # Standardization and joining
│   ├── rewards.py           # Label computation
│   ├── features.py          # Feature engineering
│   ├── splits.py            # Train/val/test splitting
│   ├── baselines/
│   │   ├── knn_policy.py    # kNN baseline
│   │   ├── bucket_policy.py # Bucket policy
│   │   └── linear_heads.py  # Linear models
│   ├── eval.py              # Evaluation metrics
│   └── api.py               # Inference API
├── notebooks/
│   └── 00_explore.ipynb     # Data exploration
├── config.yaml              # Configuration
└── train_baselines.py       # Main training script
```

## Quick Start

### 1. Train All Baselines

```bash
python train_baselines.py
```

This will:
- Load and preprocess the data
- Build state features
- Split into train/val/test
- Fit all three baseline models
- Evaluate on validation set
- Save artifacts

### 2. Use the API

```python
from src.api import BaselineSuite

# Load saved models
suite = BaselineSuite.load('artifacts/baselines.pkl')

# Suggest a play
situation = {
    'down': 3,
    'ydstogo': 7,
    'yardline_100': 45,
    'clock_seconds': 300,
    'score_diff': -3,
    'quarter': 2
}

result = suite.suggest_play(situation, mode='knn')
print(f"Suggested play: {result['suggestion']}")
print(f"Expected yards: {result['expected_yards']:.2f}")
print(f"Success probability: {result['success_prob']:.3f}")
```

## Baseline Models

### Baseline 1: State-only kNN

Finds the k nearest neighbors in situation space and:
- Predicts yards from mean of neighbors
- Suggests the exemplar play (neighbor with highest yards)

**Hyperparameters:**
- `k=50` (number of neighbors)
- `metric='euclidean'`
- Optionally weight `yardsToGo` and `yardline_100` higher

### Baseline 2: Bucketed Frequency Policy

Interpretable lookup table that buckets situations by:
- Down (1-4)
- Yards to go bins (1-2, 3-5, 6-9, 10-15, 16+)
- Yardline bins (Own 1-20, 21-50, 51-80, Red Zone)
- Clock bins (>600s, 600-120s, ≤120s)

Returns the historically best play for each bucket.

### Baseline 3: Light Linear Models

- **Ridge Regression** for yards prediction (α=1.0)
- **Logistic Regression** for success classification (C=1.0, class_weight='balanced')

Provides calibrated probabilities and ranking baselines.

## Evaluation Metrics

- **Yards prediction:** MAE, RMSE, R², Spearman correlation
- **Success classification:** Brier score, AUC, Accuracy
- **Policy quality:** Top-K Precision
- **Counterfactual Policy Eval:** Self-Normalized IPS, Doubly-Robust

## Data Processing

The pipeline:

1. **Load raw CSVs** → `data_io.load_raw()`
2. **Filter normal plays** → Drop penalties, kneel-downs, spikes
3. **Standardize coordinates** → Always offense left→right
4. **Join game context** → Add week, score differential, clock
5. **Compute rewards** → Yards gained, success, TD indicators
6. **Build features** → State features (+ optional pre-snap shape)
7. **Split by week** → Train on weeks 1-6, val on 7-8, test on 8

## Configuration

Edit `config.yaml` to adjust:
- Train/val/test splits
- Feature settings (including pre-snap shape)
- Model hyperparameters
- Evaluation options

## Next Steps

- Add Baseline 1.5 (State + Pre-snap Shape kNN)
- Integrate play type classification (pass/run)
- Add EPA (Expected Points Added) as reward
- Visualize suggested plays using tracking data
- Build Streamlit demo interface

## License

Data from NFL Big Data Bowl 2023.

