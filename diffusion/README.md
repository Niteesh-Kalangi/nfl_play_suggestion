# Football Diffusion: Conditional Diffusion Model for NFL Play Generation

This directory contains a complete implementation of a conditional diffusion model for generating multi-agent football play trajectories from NFL Big Data Bowl 2023 tracking data.

## Overview

The diffusion model generates player trajectories (22 players: 11 offense + 11 defense) conditioned on game context:
- **Categorical**: down, offensive formation, personnel, defensive team, situation (short/medium/long)
- **Continuous**: yards to go, normalized yardline

## Project Structure

```
diffusion/
├── src/football_diffusion/
│   ├── config/              # YAML configuration files
│   ├── data/                # Data preprocessing and loading
│   │   ├── preprocess.py    # CSV → Parquet conversion
│   │   ├── dataset.py       # PyTorch Dataset
│   │   └── splits.py        # Train/val/test splitting
│   ├── models/              # Model components
│   │   ├── diffusion_unet.py        # Temporal U-Net backbone
│   │   ├── diffusion_wrapper.py     # DDPM/DDIM sampling
│   │   └── context_encoder.py       # Context feature encoding
│   ├── training/            # Training utilities
│   │   └── train_diffusion.py       # Lightning module
│   ├── eval/                # Evaluation
│   │   ├── metrics.py               # ADE, FDE, validity, diversity
│   │   ├── eval_diffusion.py        # Evaluation script
│   │   └── adherence_classifier.py  # Context adherence metric
│   ├── viz/                 # Visualization
│   │   ├── field.py         # Field drawing
│   │   └── animate.py       # Trajectory animations
│   └── utils/               # Utilities
│       ├── tensor_ops.py    # Diffusion tensor operations
│       └── seed.py          # Reproducibility
├── notebooks/
│   └── 00_end_to_end.ipynb  # Orchestration notebook
├── scripts/                 # Shell scripts
│   ├── preprocess.sh
│   ├── train_diffusion.sh
│   └── eval_diffusion.sh
├── tests/                   # Unit tests
├── train_main.py            # Main training entry point
└── README.md
```

## Quick Start

### 1. Preprocess Data

```bash
cd diffusion
bash scripts/preprocess.sh \
    ../../data/nfl-big-data-bowl-2023 \
    ../../data/cache \
    src/football_diffusion/config/default.yaml
```

This converts raw CSV tracking files into cached Parquet format with:
- Standardized coordinates (offense always moves right)
- Extracted frames from snap to play end
- Player positions as tensors [T, P, F] where T=60 frames, P=22 players, F=3 (x, y, s)
- Context vectors with categorical and continuous features

### 2. Train Model

```bash
python train_main.py \
    --config src/football_diffusion/config/train.yaml \
    --cache_dir ../../data/cache \
    --output_dir ../../artifacts/diffusion \
    --gpus 1 \
    --max_epochs 50
```

Or use the script:
```bash
bash scripts/train_diffusion.sh
```

### 3. Evaluate

```bash
python -m src.football_diffusion.eval.eval_diffusion \
    --checkpoint ../../artifacts/diffusion/last.ckpt \
    --cache_dir ../../data/cache \
    --config src/football_diffusion/config/eval.yaml \
    --split test \
    --num_samples 8 \
    --sample_steps 20 50 100
```

Or use the script:
```bash
bash scripts/eval_diffusion.sh
```

## Model Architecture

### Temporal U-Net
- **Input**: [B, P*F, T] where B=batch, P=22 players, F=3 features (x,y,s), T=60 frames
- **Backbone**: 1D temporal convolutions with grouped channels (player-wise)
- **Conditioning**: FiLM (Feature-wise Linear Modulation) using context embeddings
- **Depth**: 4-level encoder-decoder with skip connections

### Diffusion Process
- **Schedule**: Cosine or linear beta schedule (1000 timesteps)
- **Sampling**: DDPM or DDIM (configurable steps: 20, 50, 100)
- **Guidance**: Classifier-free guidance (scale=2.0, drop_prob=0.1)

### Context Encoding
- **Categorical**: Embedding layers for down, formation, personnel, team, situation
- **Continuous**: MLP for yardsToGo and yardlineNorm
- **Output**: 256-dim context vector

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ADE** | Average Displacement Error - mean L2 distance across all timesteps |
| **FDE** | Final Displacement Error - L2 distance at last timestep |
| **Validity Rate** | % of points within field bounds [0,120]×[0,53.3] and speed < 12 yd/s |
| **Diversity** | Pairwise endpoint distance for samples under same context |
| **Context Adherence** | Accuracy of classifier trained to infer context from generated plays |
| **Collision Rate** | Fraction of timesteps with player collisions (< 1 yard apart) |

## Configuration

See `src/football_diffusion/config/default.yaml` for all configuration options:

```yaml
data:
  frames: 60
  features: [x, y, s]
  flip_to_right: true
  min_players: 22

conditioning:
  cat: [down, offensiveFormation, personnelO, defTeam, situation]
  cont: [yardsToGo, yardlineNorm]
  drop_prob: 0.1

diffusion:
  beta_schedule: cosine
  steps: 1000
  sample_steps: 50
  guidance_scale: 2.0
  model:
    channels: 256
    depth: 4
    temporal_kernel: 3

train:
  optimizer: adamw
  lr: 1e-4
  wd: 1e-2
  batch_size: 32
  max_epochs: 50
```

## Running Tests

```bash
pytest tests/
```

## Notebook

See `notebooks/00_end_to_end.ipynb` for a complete workflow:
1. Data preprocessing
2. Model training
3. Evaluation
4. Visualization

## Outputs

- **Checkpoints**: Saved to `artifacts/diffusion/`
- **Logs**: TensorBoard logs in `artifacts/diffusion/logs/`
- **Evaluation**: Results saved as `eval_results.json`
- **Animations**: Visualizations in `reports/videos/`

## Dependencies

- PyTorch >= 1.10
- PyTorch Lightning >= 1.5
- NumPy, Pandas
- Matplotlib (for visualization)
- PyArrow (for Parquet)
- scikit-learn (for evaluation)
- tqdm, yaml

Install with:
```bash
pip install torch pytorch-lightning numpy pandas matplotlib pyarrow scikit-learn tqdm pyyaml
```

