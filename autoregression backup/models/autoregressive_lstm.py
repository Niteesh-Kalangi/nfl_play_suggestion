"""
LSTM-based autoregressive trajectory generator.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base_autoregressive import BaseAutoregressiveModel


class LSTMTrajectoryGenerator(BaseAutoregressiveModel):
    """
    LSTM-based autoregressive model for player trajectory generation.
    
    Architecture:
    - Input projection layer
    - Multi-layer LSTM
    - Output projection layer
    
    Training uses teacher forcing (ground truth previous positions as input).
    Inference uses autoregressive rollout (model's own predictions as input).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Args:
            input_dim: Dimension of input (conditioning + previous output)
            output_dim: Dimension of output (player positions)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM (training only)
        """
        super().__init__(input_dim, output_dim, hidden_dim)
        
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
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
            nn.Linear(hidden_dim, output_dim)
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
        mask: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with teacher forcing.
        
        Args:
            x: Input tensor [batch, T, input_dim]
            mask: Optional mask [batch, T] for valid timesteps
            hidden: Optional initial hidden state (h_0, c_0)
            
        Returns:
            Tuple of:
            - Output tensor [batch, T, output_dim]
            - Final hidden state (h_n, c_n)
        """
        batch_size, T, _ = x.shape
        
        # Project input
        x_proj = self.input_proj(x)  # [batch, T, hidden_dim]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # LSTM forward
        lstm_out, hidden_out = self.lstm(x_proj, hidden)  # [batch, T, hidden_dim * num_directions]
        
        # Project to output
        output = self.output_proj(lstm_out)  # [batch, T, output_dim]
        
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
        init_context: torch.Tensor,
        horizon: int,
        init_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive rollout for trajectory generation.
        
        Args:
            init_context: Conditioning features [batch, context_dim]
            horizon: Number of timesteps to generate
            init_positions: Initial positions [batch, output_dim] (optional)
            
        Returns:
            Generated trajectory [batch, horizon, output_dim]
        """
        batch_size = init_context.shape[0]
        device = init_context.device
        context_dim = init_context.shape[-1]
        
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
            # Build input: [context, previous_positions]
            x_t = torch.cat([init_context, prev_pos], dim=-1)  # [batch, input_dim]
            x_t = x_t.unsqueeze(1)  # [batch, 1, input_dim]
            
            # Forward one step
            output, hidden = self.forward(x_t, hidden=hidden)
            pred_pos = output.squeeze(1)  # [batch, output_dim]
            
            trajectory.append(pred_pos)
            
            # Update previous positions for next step
            prev_pos = pred_pos.detach()
        
        # Stack trajectory
        trajectory = torch.stack(trajectory, dim=1)  # [batch, horizon, output_dim]
        
        return trajectory
    
    def step(
        self,
        x_t: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single step forward for incremental generation.
        
        Args:
            x_t: Input at current timestep [batch, input_dim]
            hidden: Current hidden state (h, c)
            
        Returns:
            Tuple of:
            - Output [batch, output_dim]
            - Updated hidden state
        """
        x_t = x_t.unsqueeze(1)  # [batch, 1, input_dim]
        output, hidden = self.forward(x_t, hidden=hidden)
        return output.squeeze(1), hidden

