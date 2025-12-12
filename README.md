# NFL Player Trajectory Generation

This project implements **autoregressive trajectory generators (LSTM + Transformer)** for NFL player movement prediction, serving as the primary baselines for comparing against diffusion-based models. It also includes auxiliary play suggestion models for game-level decision making.

## Overview

The project has two main components:

### 1. Autoregressive Trajectory Baselines (Primary)

For the player trajectory generation task, the main baselines are:

- **LSTM-based autoregressive trajectory generator** - Uses recurrent architecture for sequential prediction
- **Transformer-based autoregressive trajectory generator** - Uses attention mechanism with causal masking

These are used as the **primary comparison points** for the diffusion model in the NeurIPS-style report.

### 2. Play Suggestion Auxiliary Models

Auxiliary, non-generative models for play-level decisions:

- **State-only kNN** - Finds similar historical situations and suggests the best exemplar play
- **Bucketed Frequency Policy** - Interpretable lookup table by situation buckets
- **Light Linear Models** - Ridge regression for yards, Logistic regression for success

> **Note:** These auxiliary models are for play-level decision quality, not for trajectory generation.

## Setup

### 1. Create Conda Environment

```bash
conda create -n nfl_trajectory python=3.9
conda activate nfl_trajectory
pip install pandas numpy scikit-learn scipy pyyaml joblib torch tqdm tabulate
```

### 2. Download Data

Place the NFL Big Data Bowl 2023 data files in the `data/` directory. See `data/README.md` for details.

### 3. Train Autoregressive Models

```bash
# Train both LSTM and Transformer
python train_autoregressive.py --model all

# Or train individually
python train_autoregressive.py --model lstm
python train_autoregressive.py --model transformer
```

### 4. Evaluate Trajectory Models

```bash
# Evaluate all models on test set
python eval_trajectories.py --models all --split test

# Evaluate specific model
python eval_trajectories.py --models lstm transformer --split test
```

## Project Structure

```
nfl_play_suggestion/
├── data/                    # Raw CSV files (NFL Big Data Bowl 2023)
│   ├── plays.csv
│   ├── games.csv
│   ├── players.csv
│   └── week*.csv            # Tracking data
├── src/
│   ├── trajectory_data.py   # Trajectory dataset building
│   ├── models/              # Trajectory generation models
│   │   ├── base_autoregressive.py     # Abstract base class
│   │   ├── autoregressive_lstm.py     # LSTM generator
│   │   └── autoregressive_transformer.py  # Transformer generator
│   ├── data_io.py           # Data loading
│   ├── preprocess.py        # Standardization and joining
│   ├── rewards.py           # Label computation
│   ├── features.py          # Feature engineering
│   ├── splits.py            # Train/val/test splitting
│   ├── baselines/           # Auxiliary play suggestion models
│   │   ├── knn_policy.py
│   │   ├── bucket_policy.py
│   │   └── linear_heads.py
│   ├── eval.py              # Evaluation metrics (ADE, FDE, etc.)
│   └── api.py               # Inference APIs
├── train_autoregressive.py  # Train LSTM/Transformer baselines
├── eval_trajectories.py     # Evaluate trajectory models
├── train_baselines.py       # Train auxiliary play suggestion models
├── config.yaml              # Configuration
└── artifacts/
    └── autoregressive/      # Saved model checkpoints
        ├── lstm.pt
        └── transformer.pt
```

## Autoregressive Trajectory Baselines

### LSTM Trajectory Generator

```python
from src.api import TrajectoryBaselines

# Load trained models
baselines = TrajectoryBaselines.load('artifacts/autoregressive')

# Generate trajectory for a game situation
context = {
    'down': 2,
    'yardsToGo': 7,
    'yardline_100': 45,
    'clock_seconds': 300,
    'score_diff': -3,
    'quarter': 3
}

# Generate 50-frame trajectory using LSTM
trajectory = baselines.generate_with_lstm(context, horizon=50)
print(f"Trajectory shape: {trajectory.shape}")  # [1, 50, output_dim]
```

### Transformer Trajectory Generator

```python
# Generate using Transformer
trajectory = baselines.generate_with_transformer(context, horizon=50)

# Or use the unified interface
trajectory = baselines.generate(context, horizon=50, model='transformer')
```

### Model Architecture

**LSTM Generator:**
- Input projection layer
- Multi-layer LSTM with hidden_dim=256
- Output projection to player positions
- Teacher forcing during training, autoregressive rollout during inference

**Transformer Generator:**
- Input projection with positional encoding
- Causal-masked Transformer decoder (4 layers, 8 heads)
- Learned memory for task-specific context
- Output projection to player positions

## Evaluation Metrics

For trajectory generation, we use:

| Metric | Description |
|--------|-------------|
| **ADE** | Average Displacement Error - mean L2 distance across all timesteps |
| **FDE** | Final Displacement Error - L2 distance at the last timestep |
| **Collision Rate** | Fraction of timesteps with player collisions |
| **Speed Distribution** | Wasserstein distance between predicted and real speed distributions |

### Running Evaluation

```bash
python eval_trajectories.py --models lstm transformer --split test
```

Output:
```
+-------------+--------+--------+------------------+-------------+
| Model       | ADE    | FDE    | Collision Rate   | Speed Dist  |
+=============+========+========+==================+=============+
| LSTM        | X.XXXX | X.XXXX | X.XXXX           | X.XXXX      |
| TRANSFORMER | X.XXXX | X.XXXX | X.XXXX           | X.XXXX      |
+-------------+--------+--------+------------------+-------------+
```

## Configuration

Edit `config.yaml` to adjust:

```yaml
# Trajectory generation settings
trajectories:
  max_timesteps: 50
  players: "offense_only"
  num_players: 11
  include_velocity: true
  condition_on_state: true

# Autoregressive model hyperparameters
autoregressive:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  
  lstm:
    hidden_dim: 256
    num_layers: 2
    dropout: 0.1
  
  transformer:
    d_model: 256
    nhead: 8
    num_layers: 4
```

## Auxiliary Play Suggestion Models

> These are **non-generative** models for play-level decision quality, kept for completeness.

### Train Auxiliary Models

```bash
python train_baselines.py
```

### Use Auxiliary Models

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
print(f"Expected yards: {result['expected_yards']:.2f}")
print(f"Success probability: {result['success_prob']:.3f}")
```

## Paper/Report Notes

For the NeurIPS-style writeup:

> **Baselines Section:**
> "Our primary baselines are autoregressive trajectory generators implemented with LSTM and Transformer architectures. These models predict player positions autoregressively, conditioned on game state. kNN, bucket, and linear models are only used as auxiliary, non-generative baselines for play-level decision quality, not for trajectory generation."

## Data Processing

The trajectory pipeline:

1. **Load tracking CSVs** → Merge with plays and games
2. **Standardize coordinates** → Always offense left→right
3. **Extract player positions** → Filter by offense/defense
4. **Build sequences** → [T, num_players * features_per_player]
5. **Split by week** → Train on weeks 1-6, val on 7, test on 8

## License

Data from NFL Big Data Bowl 2023.
