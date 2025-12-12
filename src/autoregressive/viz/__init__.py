"""
Visualization utilities for autoregressive trajectory generation.

Matches diffusion model visualization structure for direct comparison.
"""
from .field import draw_field, plot_trajectory
from .animate import animate_trajectory

__all__ = [
    'draw_field',
    'plot_trajectory',
    'animate_trajectory',
]

