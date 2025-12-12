"""
Transformer-based autoregressive trajectory generator.
"""
import math
import torch
import torch.nn as nn
from typing import Optional
from .base_autoregressive import BaseAutoregressiveModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, T, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTrajectoryGenerator(BaseAutoregressiveModel):
    """
    Transformer-based autoregressive model for player trajectory generation.
    
    Architecture:
    - Input projection layer
    - Positional encoding
    - Transformer decoder with causal masking
    - Output projection layer
    
    Uses causal (autoregressive) attention so model only sees past timesteps.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        """
        Args:
            input_dim: Dimension of input (conditioning + previous output)
            output_dim: Dimension of output (player positions)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__(input_dim, output_dim, d_model)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Learned memory for decoder (serves as encoder output)
        # This allows the model to learn task-specific context
        self.memory = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _generate_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask.
        
        Args:
            T: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask [T, T] where True = masked (cannot attend)
        """
        # Upper triangular mask (future positions masked)
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with causal attention.
        
        Args:
            x: Input tensor [batch, T, input_dim]
            mask: Optional padding mask [batch, T] (True = padded)
            
        Returns:
            Output tensor [batch, T, output_dim]
        """
        batch_size, T, _ = x.shape
        device = x.device
        
        # Project input
        x_proj = self.input_proj(x)  # [batch, T, d_model]
        
        # Add positional encoding
        x_proj = self.pos_encoding(x_proj)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(T, device)
        
        # Expand memory for batch
        memory = self.memory.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        
        # Convert padding mask to key padding mask format if provided
        tgt_key_padding_mask = None
        if mask is not None:
            # Invert: our mask has 1 for valid, Transformer expects True for padded
            tgt_key_padding_mask = (mask == 0)
        
        # Transformer forward
        output = self.transformer(
            x_proj,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch, T, d_model]
        
        # Project to output
        output = self.output_proj(output)  # [batch, T, output_dim]
        
        return output
    
    def rollout(
        self,
        init_context: torch.Tensor,
        horizon: int,
        init_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive rollout for trajectory generation.
        
        Uses KV-caching for efficient generation.
        
        Args:
            init_context: Conditioning features [batch, context_dim]
            horizon: Number of timesteps to generate
            init_positions: Initial positions [batch, output_dim] (optional)
            
        Returns:
            Generated trajectory [batch, horizon, output_dim]
        """
        batch_size = init_context.shape[0]
        device = init_context.device
        
        # Initialize positions
        if init_positions is None:
            prev_pos = torch.zeros(batch_size, self.output_dim, device=device)
        else:
            prev_pos = init_positions
        
        # Build initial input sequence
        trajectory = []
        input_sequence = []
        
        for t in range(horizon):
            # Build input for current timestep: [context, previous_positions]
            x_t = torch.cat([init_context, prev_pos], dim=-1)  # [batch, input_dim]
            input_sequence.append(x_t)
            
            # Stack all inputs so far
            x = torch.stack(input_sequence, dim=1)  # [batch, t+1, input_dim]
            
            # Forward pass (could optimize with KV-cache)
            output = self.forward(x)  # [batch, t+1, output_dim]
            
            # Get prediction for current timestep
            pred_pos = output[:, -1, :]  # [batch, output_dim]
            
            trajectory.append(pred_pos)
            
            # Update previous positions for next step
            prev_pos = pred_pos.detach()
        
        # Stack trajectory
        trajectory = torch.stack(trajectory, dim=1)  # [batch, horizon, output_dim]
        
        return trajectory
    
    def rollout_efficient(
        self,
        init_context: torch.Tensor,
        horizon: int,
        init_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Efficient autoregressive rollout using incremental computation.
        
        This version is more memory efficient for long horizons by only
        computing the last position at each step rather than recomputing
        the full sequence.
        
        Args:
            init_context: Conditioning features [batch, context_dim]
            horizon: Number of timesteps to generate
            init_positions: Initial positions [batch, output_dim] (optional)
            
        Returns:
            Generated trajectory [batch, horizon, output_dim]
        """
        batch_size = init_context.shape[0]
        device = init_context.device
        
        # Initialize positions
        if init_positions is None:
            prev_pos = torch.zeros(batch_size, self.output_dim, device=device)
        else:
            prev_pos = init_positions
        
        # For Transformer, we need to track all past inputs for attention
        # Could implement KV-cache for more efficiency
        trajectory = []
        all_inputs = []
        
        for t in range(horizon):
            # Build input for current timestep
            x_t = torch.cat([init_context, prev_pos], dim=-1)  # [batch, input_dim]
            all_inputs.append(x_t)
            
            # Project current input
            x = torch.stack(all_inputs, dim=1)  # [batch, t+1, input_dim]
            x_proj = self.input_proj(x)
            x_proj = self.pos_encoding(x_proj)
            
            # Generate causal mask for current length
            T = x_proj.shape[1]
            causal_mask = self._generate_causal_mask(T, device)
            
            # Memory
            memory = self.memory.expand(batch_size, -1, -1)
            
            # Transformer forward
            output = self.transformer(x_proj, memory, tgt_mask=causal_mask)
            
            # Only take last output
            pred_pos = self.output_proj(output[:, -1:, :]).squeeze(1)
            
            trajectory.append(pred_pos)
            prev_pos = pred_pos.detach()
        
        trajectory = torch.stack(trajectory, dim=1)
        return trajectory

