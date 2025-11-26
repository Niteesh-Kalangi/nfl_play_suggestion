"""
LSTM Autoregressive Generator for NFL offensive trajectory synthesis.

Generates multi-player trajectories step-by-step, conditioned on context features.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class ContextEncoder(nn.Module):
    """
    Encodes game context (down, distance, formation, etc.) into a fixed-size embedding.
    """
    
    def __init__(self, context_dim: int, embed_dim: int):
        """
        Args:
            context_dim: Number of context features
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (batch_size, context_dim)
            
        Returns:
            (batch_size, embed_dim)
        """
        return self.encoder(context)


class LSTMTrajectoryGenerator(nn.Module):
    """
    LSTM-based autoregressive generator for multi-player trajectories.
    
    At each time step:
    - Input: previous frame positions (all players) + context embedding
    - Output: predicted next frame positions (all players)
    """
    
    def __init__(
        self,
        n_players: int = 11,
        player_dim: int = 2,
        context_dim: int = 6,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        context_embed_dim: int = 64
    ):
        """
        Args:
            n_players: Number of players per frame
            player_dim: Features per player (x, y)
            context_dim: Number of context features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            context_embed_dim: Context embedding dimension
        """
        super().__init__()
        
        self.n_players = n_players
        self.player_dim = player_dim
        self.input_dim = n_players * player_dim  # Flattened player positions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Context encoder
        self.context_encoder = ContextEncoder(context_dim, context_embed_dim)
        
        # Input projection: positions + context -> LSTM input
        self.input_proj = nn.Linear(self.input_dim + context_embed_dim, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection: LSTM hidden -> next frame positions
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.input_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        input_traj: torch.Tensor,
        context: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            input_traj: (batch_size, seq_len, n_players * player_dim)
            context: (batch_size, context_dim)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of:
            - output: (batch_size, seq_len, n_players * player_dim)
            - hidden: Final hidden state tuple (h_n, c_n)
        """
        batch_size, seq_len, _ = input_traj.shape
        
        # Encode context
        ctx_embed = self.context_encoder(context)  # (batch_size, context_embed_dim)
        
        # Expand context to all time steps
        ctx_expand = ctx_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, context_embed_dim)
        
        # Concatenate input with context
        x = torch.cat([input_traj, ctx_expand], dim=-1)  # (batch_size, seq_len, input_dim + context_embed_dim)
        
        # Project to LSTM input size
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_size)
        
        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Project to output positions
        output = self.output_proj(lstm_out)  # (batch_size, seq_len, input_dim)
        
        return output, hidden
    
    def generate(
        self,
        initial_frame: torch.Tensor,
        context: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressively generate trajectory from initial frame.
        
        Args:
            initial_frame: (batch_size, n_players * player_dim) - first frame positions
            context: (batch_size, context_dim)
            n_steps: Number of frames to generate
            temperature: Sampling temperature (1.0 = deterministic)
            
        Returns:
            Generated trajectory: (batch_size, n_steps + 1, n_players * player_dim)
        """
        self.eval()
        batch_size = initial_frame.shape[0]
        device = initial_frame.device
        
        # Initialize trajectory with first frame
        trajectory = [initial_frame]
        current_frame = initial_frame.unsqueeze(1)  # (batch_size, 1, input_dim)
        hidden = None
        
        # Encode context once
        ctx_embed = self.context_encoder(context)  # (batch_size, context_embed_dim)
        
        with torch.no_grad():
            for step in range(n_steps):
                # Concatenate current frame with context
                ctx_expand = ctx_embed.unsqueeze(1)  # (batch_size, 1, context_embed_dim)
                x = torch.cat([current_frame, ctx_expand], dim=-1)
                
                # Project and run LSTM
                x = self.input_proj(x)
                lstm_out, hidden = self.lstm(x, hidden)
                
                # Predict next frame
                next_frame = self.output_proj(lstm_out)  # (batch_size, 1, input_dim)
                
                # Add noise if temperature > 1
                if temperature > 1.0:
                    noise = torch.randn_like(next_frame) * (temperature - 1.0) * 0.1
                    next_frame = next_frame + noise
                
                trajectory.append(next_frame.squeeze(1))
                current_frame = next_frame
        
        # Stack trajectory: (batch_size, n_steps + 1, input_dim)
        trajectory = torch.stack(trajectory, dim=1)
        
        return trajectory
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target trajectories.
        
        Args:
            pred: (batch_size, seq_len, n_players * player_dim)
            target: (batch_size, seq_len, n_players * player_dim)
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            Loss tensor
        """
        loss_fn = nn.MSELoss(reduction=reduction)
        return loss_fn(pred, target)


class LSTMTrajectoryGeneratorConfig:
    """Configuration for LSTM trajectory generator."""
    
    def __init__(
        self,
        n_players: int = 11,
        player_dim: int = 2,
        context_dim: int = 6,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        context_embed_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_frames: int = 50
    ):
        self.n_players = n_players
        self.player_dim = player_dim
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.context_embed_dim = context_embed_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_frames = max_frames
    
    def to_dict(self) -> Dict:
        return vars(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'LSTMTrajectoryGeneratorConfig':
        return cls(**d)


def create_lstm_generator(config: LSTMTrajectoryGeneratorConfig) -> LSTMTrajectoryGenerator:
    """Factory function to create LSTM generator from config."""
    return LSTMTrajectoryGenerator(
        n_players=config.n_players,
        player_dim=config.player_dim,
        context_dim=config.context_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        context_embed_dim=config.context_embed_dim
    )
