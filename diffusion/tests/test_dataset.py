"""
Unit tests for dataset module.
"""
import pytest
import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import tempfile
import json

from src.football_diffusion.data.dataset import FootballPlayDataset, collate_fn
from src.football_diffusion.data.preprocess import (
    standardize_coordinates,
    extract_play_frames,
    extract_player_tensor,
    build_context_vector
)


def test_tensor_shapes():
    """Test that tensor shapes are correct."""
    # Create dummy tensor
    T, P, F = 60, 22, 3
    tensor = np.random.randn(T, P, F).astype(np.float32)
    
    assert tensor.shape == (T, P, F)
    assert tensor.dtype == np.float32


def test_coordinate_flipping():
    """Test coordinate standardization (flip to right)."""
    # Create dummy tracking data
    tracking = pd.DataFrame({
        'gameId': [1] * 10,
        'playId': [1] * 10,
        'playDirection': ['left'] * 5 + ['right'] * 5,
        'x': np.linspace(10, 100, 10),
        'y': np.linspace(10, 40, 10)
    })
    
    plays = pd.DataFrame({
        'gameId': [1],
        'playId': [1],
        'possessionTeam': ['A']
    })
    
    # Standardize
    tracking_std = standardize_coordinates(tracking, plays)
    
    # Check that left-moving plays are flipped
    left_mask = tracking_std['playDirection'] == 'left'
    original_x = tracking.loc[left_mask, 'x'].values[0]
    flipped_x = tracking_std.loc[left_mask, 'x'].values[0]
    
    # x should be flipped: new_x = 120 - old_x
    assert abs(flipped_x - (120 - original_x)) < 0.01


def test_field_bounds():
    """Test that positions are within field bounds."""
    # Field dimensions: x ∈ [0, 120], y ∈ [0, 53.3]
    x_pos = np.random.uniform(0, 120, size=(60, 22))
    y_pos = np.random.uniform(0, 53.3, size=(60, 22))
    
    assert np.all(x_pos >= 0) and np.all(x_pos <= 120)
    assert np.all(y_pos >= 0) and np.all(y_pos <= 53.3)


def test_padding():
    """Test sequence padding to fixed length."""
    T_short = 30
    T_target = 60
    P, F = 22, 3
    
    tensor = np.random.randn(T_short, P, F)
    
    # Pad
    padded = np.zeros((T_target, P, F))
    padded[:T_short] = tensor
    
    assert padded.shape == (T_target, P, F)
    assert np.allclose(padded[T_short:], 0)


if __name__ == '__main__':
    pytest.main([__file__])

