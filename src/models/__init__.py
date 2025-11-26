"""
Autoregressive trajectory generation models.
"""
from .lstm_generator import (
    LSTMTrajectoryGenerator,
    LSTMTrajectoryGeneratorConfig,
    create_lstm_generator
)
from .transformer_generator import (
    TransformerTrajectoryGenerator,
    TransformerTrajectoryGeneratorConfig,
    create_transformer_generator,
    WarmupCosineScheduler
)

__all__ = [
    'LSTMTrajectoryGenerator',
    'LSTMTrajectoryGeneratorConfig',
    'create_lstm_generator',
    'TransformerTrajectoryGenerator',
    'TransformerTrajectoryGeneratorConfig',
    'create_transformer_generator',
    'WarmupCosineScheduler'
]
