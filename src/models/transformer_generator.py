"""
Transformer Autoregressive Generator for NFL offensive trajectory synthesis.

Uses a Transformer decoder with causal masking to generate multi-player 
trajectories step-by-step, conditioned on context features.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence positions.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ContextEncoder(nn.Module):
    """
    Encodes game context into embeddings that condition the transformer.
    """
    
    def __init__(self, context_dim: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.encoder(context)


class TransformerTrajectoryGenerator(nn.Module):
    """
    Transformer decoder-based autoregressive generator for multi-player trajectories.
    
    Each time step is represented as a token embedding that includes all players' features.
    Uses causal masking so the model only attends to previous steps.
    Context embeddings are added to each token.
    """
    
    def __init__(
        self,
        n_players: int = 11,
        player_dim: int = 2,
        context_dim: int = 6,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Args:
            n_players: Number of players per frame
            player_dim: Features per player (x, y)
            context_dim: Number of context features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer decoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.n_players = n_players
        self.player_dim = player_dim
        self.input_dim = n_players * player_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input embedding: project flattened positions to d_model
        self.input_embed = nn.Linear(self.input_dim, d_model)
        
        # Context encoder
        self.context_encoder = ContextEncoder(context_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.input_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask.
        
        Returns:
            Mask tensor of shape (seq_len, seq_len) with -inf for masked positions
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_traj: torch.Tensor,
        context: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            input_traj: (batch_size, seq_len, n_players * player_dim)
            context: (batch_size, context_dim)
            memory: Optional memory tensor (not used in decoder-only mode)
            
        Returns:
            output: (batch_size, seq_len, n_players * player_dim)
        """
        batch_size, seq_len, _ = input_traj.shape
        device = input_traj.device
        
        # Embed input positions
        x = self.input_embed(input_traj)  # (batch_size, seq_len, d_model)
        
        # Add context embedding to each position
        ctx_embed = self.context_encoder(context)  # (batch_size, d_model)
        x = x + ctx_embed.unsqueeze(1)  # Broadcast context to all positions
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Create dummy memory for decoder-only mode
        # Using context as memory allows cross-attention to condition on context
        memory = ctx_embed.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Transformer decoder forward
        output = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask
        )
        
        # Project to output positions
        output = self.output_proj(output)  # (batch_size, seq_len, input_dim)
        
        return output
    
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
        
        # Encode context once
        ctx_embed = self.context_encoder(context)  # (batch_size, d_model)
        memory = ctx_embed.unsqueeze(1)  # (batch_size, 1, d_model)
        
        with torch.no_grad():
            for step in range(n_steps):
                # Current sequence
                current_seq = torch.stack(trajectory, dim=1)  # (batch_size, current_len, input_dim)
                seq_len = current_seq.shape[1]
                
                # Embed and add context
                x = self.input_embed(current_seq)
                x = x + ctx_embed.unsqueeze(1)
                x = self.pos_encoder(x)
                
                # Causal mask
                causal_mask = self._generate_causal_mask(seq_len, device)
                
                # Transformer forward
                output = self.transformer_decoder(
                    tgt=x,
                    memory=memory,
                    tgt_mask=causal_mask
                )
                
                # Get prediction for next frame (last position)
                next_frame = self.output_proj(output[:, -1, :])  # (batch_size, input_dim)
                
                # Add noise if temperature > 1
                if temperature > 1.0:
                    noise = torch.randn_like(next_frame) * (temperature - 1.0) * 0.1
                    next_frame = next_frame + noise
                
                trajectory.append(next_frame)
        
        # Stack trajectory: (batch_size, n_steps + 1, input_dim)
        trajectory = torch.stack(trajectory, dim=1)
        
        return trajectory
    
    def generate_efficient(
        self,
        initial_frame: torch.Tensor,
        context: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Efficient autoregressive generation using KV caching.
        
        Note: This is a simplified version. Full KV caching would require
        modifying the transformer decoder internals.
        
        Args:
            initial_frame: (batch_size, n_players * player_dim)
            context: (batch_size, context_dim)
            n_steps: Number of frames to generate
            temperature: Sampling temperature
            
        Returns:
            Generated trajectory: (batch_size, n_steps + 1, n_players * player_dim)
        """
        # For now, use the standard generate method
        # KV caching optimization can be added later
        return self.generate(initial_frame, context, n_steps, temperature)
    
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


class TransformerTrajectoryGeneratorConfig:
    """Configuration for Transformer trajectory generator."""
    
    def __init__(
        self,
        n_players: int = 11,
        player_dim: int = 2,
        context_dim: int = 6,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_frames: int = 50
    ):
        self.n_players = n_players
        self.player_dim = player_dim
        self.context_dim = context_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_frames = max_frames
    
    def to_dict(self) -> Dict:
        return vars(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TransformerTrajectoryGeneratorConfig':
        return cls(**d)


def create_transformer_generator(config: TransformerTrajectoryGeneratorConfig) -> TransformerTrajectoryGenerator:
    """Factory function to create Transformer generator from config."""
    return TransformerTrajectoryGenerator(
        n_players=config.n_players,
        player_dim=config.player_dim,
        context_dim=config.context_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    )


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _compute_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
