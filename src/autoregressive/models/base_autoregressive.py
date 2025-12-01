"""
Base class for autoregressive trajectory generation models with context encoder.

Updated to match diffusion model structure with separate categorical/continuous context.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict
from ..context_encoder import ContextEncoder


class BaseAutoregressiveModel(nn.Module, ABC):
    """
    Abstract base class for autoregressive trajectory generators.
    
    All autoregressive models should implement:
    - forward(): Teacher-forced training pass
    - rollout(): Autoregressive generation at inference
    
    Updated to use context encoder matching diffusion model structure.
    """
    
    def __init__(
        self,
        num_players: int = 22,
        num_features: int = 3,  # x, y, s
        hidden_dim: int = 256,
        context_dim: int = 256
    ):
        """
        Args:
            num_players: Number of players (22 = 11 offense + 11 defense)
            num_features: Features per player (3 = x, y, s)
            hidden_dim: Hidden state dimension
            context_dim: Context embedding dimension
        """
        super().__init__()
        self.num_players = num_players
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        
        # Context encoder (matches diffusion model)
        self.context_encoder = ContextEncoder(output_dim=context_dim)
        
        # Output dimension: flattened player positions
        self.output_dim = num_players * num_features
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        context_categorical: List[Dict],
        context_continuous: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            x: Input tensor [batch, T, P*F] containing previous positions (flattened)
            context_categorical: List of categorical context dicts
            context_continuous: Continuous context [batch, 3] (yardsToGo, yardlineNorm, hash_mark)
            mask: Optional mask [batch, T] for valid timesteps
            
        Returns:
            Output tensor [batch, T, P*F] of predicted positions (flattened)
        """
        pass
    
    @abstractmethod
    def rollout(
        self,
        context_categorical: List[Dict],
        context_continuous: torch.Tensor,
        horizon: int,
        init_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive rollout for trajectory generation.
        
        Args:
            context_categorical: List of categorical context dicts
            context_continuous: Continuous context [batch, 3]
            horizon: Number of timesteps to generate
            init_positions: Optional initial positions [batch, P*F] (flattened)
            
        Returns:
            Generated trajectory [batch, horizon, P*F] (flattened)
        """
        pass
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            pred: Predicted positions [batch, T, P*F] (flattened)
            target: Target positions [batch, T, P*F] (flattened)
            mask: Optional mask [batch, T] for valid timesteps
            
        Returns:
            Scalar loss tensor
        """
        # Compute squared error
        sq_error = (pred - target) ** 2
        
        if mask is not None:
            # Expand mask to match output dimensions
            mask = mask.unsqueeze(-1).expand_as(sq_error)
            # Masked mean
            loss = (sq_error * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = sq_error.mean()
        
        return loss
    
    def encode_context(
        self,
        context_categorical: List[Dict],
        context_continuous: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode context using context encoder.
        
        Args:
            context_categorical: List of categorical context dicts
            context_continuous: Continuous context [batch, 3]
            
        Returns:
            Context embedding [batch, context_dim]
        """
        return self.context_encoder(context_categorical, context_continuous)

