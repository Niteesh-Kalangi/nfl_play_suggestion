# Autoregressive Trajectory Generation - Updated Structure

This module has been refactored to match the diffusion model's structure for direct comparison.

## Key Changes

### 1. Data Structure
- **Old**: `[T, D_out]` where D_out = 11 players × 4 features = 44 (flattened, offense only)
- **New**: `[T, P, F]` where T=60 frames, P=22 players (11 offense + 11 defense), F=3 features (x, y, s)

### 2. Conditioning Structure
- **Old**: Concatenated features `[state_features, Y[t-1]]` where state_features = 6 dims
- **New**: Separate categorical and continuous context:
  - **Categorical**: `down`, `offensiveFormation`, `personnelO`, `defTeam`, `situation`
  - **Continuous**: `[yardsToGo, yardlineNorm, hash_mark]` (3 features)
  - Encoded via `ContextEncoder` (matches diffusion model)

### 3. Models
- Updated `BaseAutoregressiveModel` to use context encoder
- Updated `LSTMTrajectoryGenerator` to accept separate context
- Updated `TransformerTrajectoryGenerator` to accept separate context
- Both models now work with `[T, P*F]` flattened format internally

### 4. Visualization
- Added `draw_field()` - Draws football field
- Added `plot_trajectory()` - Static trajectory plot
- Added `animate_trajectory()` - Animated play visualization
- Matches diffusion model visualization structure

### 5. Custom Generation API
- Added `generate_play_with_context()` - Generate plays with custom conditioning
- Added `generate_play_from_formation_anchors()` - Generate from formation anchors
- Matches diffusion model's custom generation API

## Usage

### Loading Dataset

```python
from src.autoregressive import (
    make_autoregressive_splits,
    create_autoregressive_dataloaders
)
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create datasets
datasets = make_autoregressive_splits('data', config)

# Create dataloaders
dataloaders = create_autoregressive_dataloaders(
    datasets, 
    batch_size=32
)
```

### Training a Model

```python
from src.autoregressive import LSTMTrajectoryGenerator
import torch

# Create model
model = LSTMTrajectoryGenerator(
    num_players=22,
    num_features=3,
    hidden_dim=256,
    num_layers=2,
    context_dim=256
)

# Training loop
for batch in dataloader:
    X = batch['X']  # [B, T, P, F]
    context_cat = batch['context_categorical']  # List of dicts
    context_cont = batch['context_continuous']  # [B, 3]
    
    # Flatten positions for input: [B, T, P*F]
    X_flat = X.reshape(X.shape[0], X.shape[1], -1)
    
    # Shift by 1 for teacher forcing
    X_prev = torch.cat([X_flat[:, :1, :], X_flat[:, :-1, :]], dim=1)
    
    # Forward pass
    output, _ = model(X_prev, context_cat, context_cont)
    
    # Compute loss
    loss = model.compute_loss(output, X_flat)
    loss.backward()
```

### Generating Custom Plays

```python
from src.autoregressive import generate_play_with_context
from src.autoregressive.viz import animate_trajectory
import torch

# Load trained model
model = torch.load('artifacts/autoregressive/lstm.pt')
model.eval()

# Generate play with custom conditioning
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
anim, fig = animate_trajectory(trajectory)
plt.show()
```

### Visualization

```python
from src.autoregressive.viz import draw_field, plot_trajectory, animate_trajectory
import matplotlib.pyplot as plt

# Static plot
fig, ax = plt.subplots(figsize=(14, 7))
draw_field(ax)
plot_trajectory(trajectory, ax=ax, highlight_skill_only=True)
plt.show()

# Animation
anim, fig = animate_trajectory(
    trajectory,
    animate_skill_only=True,
    show_trails=True
)
plt.show()
```

## Structure

```
src/autoregressive/
├── __init__.py                 # Main exports
├── dataset.py                  # Dataset with [T, P, F] format
├── context_encoder.py          # Context encoder (matches diffusion)
├── generate.py                 # Custom generation API
├── models/
│   ├── __init__.py
│   ├── base_autoregressive.py  # Base class with context encoder
│   ├── autoregressive_lstm.py  # LSTM model
│   └── autoregressive_transformer.py  # Transformer model
└── viz/
    ├── __init__.py
    ├── field.py                # Field drawing utilities
    └── animate.py              # Animation utilities
```

## Comparison with Diffusion Model

| Feature | Autoregressive | Diffusion |
|---------|---------------|-----------|
| Data Format | [T, P, F] | [T, P, F] |
| Players | 22 (11 offense + 11 defense) | 22 (11 offense + 11 defense) |
| Features | 3 (x, y, s) | 3 (x, y, s) |
| Context | Categorical + Continuous | Categorical + Continuous |
| Context Encoder | ContextEncoder | ContextEncoder |
| Visualization | draw_field, plot_trajectory, animate | draw_field, plot_trajectory, animate |
| Custom Generation | generate_play_with_context | sample_with_setup |

The structures are now aligned for direct comparison!

