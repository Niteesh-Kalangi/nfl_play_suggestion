"""
Base class for autoregressive trajectory generation models.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseAutoregressiveModel(nn.Module, ABC):
    """
    Abstract base class for autoregressive trajectory generators.
    
    All autoregressive models should implement:
    - forward(): Teacher-forced training pass
    - rollout(): Autoregressive generation at inference
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: Dimension of input features (conditioning + previous output)
            output_dim: Dimension of output (player positions per timestep)
            hidden_dim: Hidden state dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            x: Input tensor [batch, T, input_dim] containing conditioning + ground truth previous positions
            mask: Optional mask [batch, T] for valid timesteps
            
        Returns:
            Output tensor [batch, T, output_dim] of predicted positions
        """
        pass
    
    @abstractmethod
    def rollout(
        self,
        init_context: torch.Tensor,
        horizon: int,
        init_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive rollout for trajectory generation.
        
        Args:
            init_context: Initial conditioning features [batch, context_dim] (game state, etc.)
            horizon: Number of timesteps to generate
            init_positions: Optional initial positions [batch, output_dim]
            
        Returns:
            Generated trajectory [batch, horizon, output_dim]
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
            pred: Predicted positions [batch, T, output_dim]
            target: Target positions [batch, T, output_dim]
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
    
    def get_context_dim(self) -> int:
        """
        Get dimension of conditioning context (input_dim - output_dim).
        
        Returns:
            Context dimension (game state features)
        """
        return self.input_dim - self.output_dim

