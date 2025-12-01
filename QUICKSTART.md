# Quick Start Guide - Autoregressive Trajectory Generation

This guide shows you how to train and generate plays with the updated autoregressive models.

## Prerequisites

Make sure you have:
- Data files in `data/` directory (plays.csv, games.csv, week*.csv)
- PyTorch installed
- Required packages: `pandas`, `numpy`, `matplotlib`, `tqdm`, `pyyaml`

## Step 1: Train Models

Train LSTM and/or Transformer models:

```bash
# Train both models
python train_autoregressive.py --model all

# Train only LSTM
python train_autoregressive.py --model lstm

# Train only Transformer
python train_autoregressive.py --model transformer
```

**Options:**
- `--config`: Path to config file (default: `config.yaml`)
- `--device`: Device to use (`cuda`, `cpu`, `mps`) - auto-detects if not specified
- `--resume`: Path to checkpoint to resume from (optional)

**Output:**
- Models saved to `artifacts/autoregressive/lstm.pt` and `transformer.pt`
- Training history saved to `artifacts/autoregressive/lstm_history.json` and `transformer_history.json`
- Results saved to `artifacts/autoregressive/results.json`

## Step 2: Generate Plays

Generate plays with custom conditioning:

```bash
# Generate with LSTM model
python generate_play.py \
    --model lstm \
    --checkpoint artifacts/autoregressive/lstm.pt \
    --down 1 \
    --yards_to_go 10 \
    --formation SHOTGUN \
    --personnel "1 RB, 1 TE, 3 WR" \
    --def_team BEN \
    --yardline 50 \
    --hash_mark MIDDLE

# Generate with Transformer model
python generate_play.py \
    --model transformer \
    --checkpoint artifacts/autoregressive/transformer.pt \
    --down 2 \
    --yards_to_go 7 \
    --formation UNDER_CENTER \
    --personnel "2 RB, 1 TE, 2 WR" \
    --def_team ATL \
    --yardline 35 \
    --hash_mark LEFT
```

**Options:**
- `--model`: Model type (`lstm` or `transformer`) - **required**
- `--checkpoint`: Path to model checkpoint - **required**
- `--down`: Down (1-4), default: 1
- `--yards_to_go`: Yards to go, default: 10.0
- `--formation`: Offensive formation (e.g., `SHOTGUN`, `UNDER_CENTER`, `EMPTY`), default: `SHOTGUN`
- `--personnel`: Personnel (e.g., `"1 RB, 1 TE, 3 WR"`), default: `"1 RB, 1 TE, 3 WR"`
- `--def_team`: Defensive team abbreviation, default: `BEN`
- `--yardline`: Yardline (0-100), default: 50
- `--hash_mark`: Hash mark (`LEFT`, `MIDDLE`, `RIGHT`), default: `MIDDLE`
- `--horizon`: Number of timesteps to generate, default: 60
- `--animate`: Create animated visualization (default: static plot)
- `--save`: Save visualization to file (e.g., `play.mp4` or `play.png`)

**Examples:**

```bash
# Generate and animate a play
python generate_play.py \
    --model lstm \
    --checkpoint artifacts/autoregressive/lstm.pt \
    --down 3 \
    --yards_to_go 5 \
    --animate \
    --save play_animation.mp4

# Generate and save static plot
python generate_play.py \
    --model transformer \
    --checkpoint artifacts/autoregressive/transformer.pt \
    --down 1 \
    --yards_to_go 10 \
    --save play_static.png
```

## Step 3: Use in Python Code

You can also generate plays programmatically:

```python
import torch
from src.autoregressive import (
    LSTMTrajectoryGenerator,
    generate_play_with_context
)
from src.autoregressive.viz import animate_trajectory
import matplotlib.pyplot as plt

# Load model
checkpoint = torch.load('artifacts/autoregressive/lstm.pt')
model = LSTMTrajectoryGenerator(
    num_players=22,
    num_features=3,
    hidden_dim=256,
    num_layers=2,
    context_dim=256
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate play
trajectory = generate_play_with_context(
    model=model,
    down=1,
    yards_to_go=10,
    offensive_formation="SHOTGUN",
    personnel_o="1 RB, 1 TE, 3 WR",
    def_team="BEN",
    yardline=50,
    hash_mark="MIDDLE",
    horizon=60
)

# Visualize
anim, fig = animate_trajectory(trajectory, animate_skill_only=True)
plt.show()
```

## Configuration

Edit `config.yaml` to adjust training settings:

```yaml
autoregressive:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 50
  early_stopping_patience: 10
  scheduler: "cosine"  # 'cosine', 'step', 'none'
  
  lstm:
    hidden_dim: 256
    num_layers: 2
    dropout: 0.1
  
  transformer:
    d_model: 256
    nhead: 8
    num_layers: 4
    dim_feedforward: 512
    dropout: 0.1
    max_len: 100
```

## Troubleshooting

**Issue: Out of memory**
- Reduce `batch_size` in `config.yaml`
- Use CPU instead of GPU: `--device cpu`

**Issue: Model checkpoint not found**
- Make sure you've trained the model first
- Check that the checkpoint path is correct

**Issue: Animation not working**
- Install ffmpeg: `brew install ffmpeg` (macOS) or `sudo apt-get install ffmpeg` (Linux)
- Or use static plots instead: remove `--animate` flag

## Next Steps

- Compare with diffusion model results
- Experiment with different formations and personnel
- Adjust hyperparameters in `config.yaml`
- Evaluate on test set using evaluation scripts

