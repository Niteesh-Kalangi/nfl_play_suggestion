"""
Autoregressive trajectory generation models with diffusion-compatible structure.
"""
from .dataset import (
    AutoregressivePlayDataset, 
    collate_fn,
    make_autoregressive_splits,
    create_autoregressive_dataloaders
)
from .context_encoder import ContextEncoder
from .models.base_autoregressive import BaseAutoregressiveModel
from .models.autoregressive_lstm import LSTMTrajectoryGenerator
from .models.autoregressive_transformer import TransformerTrajectoryGenerator
from .generate import generate_play_with_context, generate_play_from_formation_anchors
from .viz import draw_field, plot_trajectory, animate_trajectory

__all__ = [
    'AutoregressivePlayDataset',
    'collate_fn',
    'make_autoregressive_splits',
    'create_autoregressive_dataloaders',
    'ContextEncoder',
    'BaseAutoregressiveModel',
    'LSTMTrajectoryGenerator',
    'TransformerTrajectoryGenerator',
    'generate_play_with_context',
    'generate_play_from_formation_anchors',
    'draw_field',
    'plot_trajectory',
    'animate_trajectory',
]

