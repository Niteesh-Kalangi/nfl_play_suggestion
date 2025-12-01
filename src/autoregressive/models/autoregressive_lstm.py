"""
LSTM-based autoregressive trajectory generator with context encoder.

Updated to match diffusion model structure with separate categorical/continuous context.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
from .base_autoregressive import BaseAutoregressiveModel


class LSTMTrajectoryGenerator(BaseAutoregressiveModel):
    """
    LSTM-based autoregressive model for player trajectory generation.
    
    Architecture:
    - Context encoder (categorical + continuous)
    - Input projection layer (previous positions + context)
    - Multi-layer LSTM
    - Output projection layer
    
    Training uses teacher forcing (ground truth previous positions as input).
    Inference uses autoregressive rollout (model's own predictions as input).
    """
    
    def __init__(
        self,
        num_players: int = 22,
        num_features: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        context_dim: int = 256
    ):
        """
        Args:
            num_players: Number of players (22 = 11 offense + 11 defense)
            num_features: Features per player (3 = x, y, s)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM (training only)
            context_dim: Context embedding dimension
        """
        super().__init__(num_players, num_features, hidden_dim, context_dim)
        
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection: previous positions (P*F) + context (context_dim)
        input_dim = self.output_dim + context_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * self.num_directions
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        context_categorical: List[Dict],
        context_continuous: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with teacher forcing.
        
        Args:
            x: Input tensor [batch, T, P*F] containing previous positions (flattened)
            context_categorical: List of categorical context dicts
            context_continuous: Continuous context [batch, 3]
            mask: Optional mask [batch, T] for valid timesteps
            hidden: Optional initial hidden state (h_0, c_0)
            
        Returns:
            Tuple of:
            - Output tensor [batch, T, P*F] (flattened)
            - Final hidden state (h_n, c_n)
        """
        batch_size, T, _ = x.shape
        
        # Encode context
        context_emb = self.encode_context(context_categorical, context_continuous)  # [batch, context_dim]
        
        # Expand context to match sequence length
        context_expanded = context_emb.unsqueeze(1).expand(-1, T, -1)  # [batch, T, context_dim]
        
        # Concatenate previous positions with context
        x_with_context = torch.cat([x, context_expanded], dim=-1)  # [batch, T, P*F + context_dim]
        
        # Project input
        x_proj = self.input_proj(x_with_context)  # [batch, T, hidden_dim]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # LSTM forward
        lstm_out, hidden_out = self.lstm(x_proj, hidden)  # [batch, T, hidden_dim * num_directions]
        
        # Project to output
        output = self.output_proj(lstm_out)  # [batch, T, P*F]
        
        return output, hidden_out
    
    def _init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (h_0, c_0) tensors
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        return h_0, c_0
    
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
            init_positions: Initial positions [batch, P*F] (flattened, optional)
            
        Returns:
            Generated trajectory [batch, horizon, P*F] (flattened)
        """
        batch_size = context_continuous.shape[0]
        device = context_continuous.device
        
        # Encode context once
        context_emb = self.encode_context(context_categorical, context_continuous)  # [batch, context_dim]
        
        # Initialize positions
        if init_positions is None:
            prev_pos = torch.zeros(batch_size, self.output_dim, device=device)
        else:
            prev_pos = init_positions
        
        # Initialize hidden state
        hidden = self._init_hidden(batch_size, device)
        
        # Storage for generated trajectory
        trajectory = []
        
        for t in range(horizon):
            # Build input: [previous_positions, context]
            x_t = torch.cat([prev_pos, context_emb], dim=-1)  # [batch, P*F + context_dim]
            x_t = x_t.unsqueeze(1)  # [batch, 1, P*F + context_dim]
            
            # Project input
            x_proj = self.input_proj(x_t)  # [batch, 1, hidden_dim]
            
            # Forward one step
            lstm_out, hidden = self.lstm(x_proj, hidden)
            pred_pos = self.output_proj(lstm_out).squeeze(1)  # [batch, P*F]
            
            trajectory.append(pred_pos)
            
            # Update previous positions for next step
            prev_pos = pred_pos.detach()
        
        # Stack trajectory
        trajectory = torch.stack(trajectory, dim=1)  # [batch, horizon, P*F]
        
        return trajectory

