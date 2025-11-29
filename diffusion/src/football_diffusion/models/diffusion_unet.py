"""
Temporal U-Net backbone for diffusion model.

Uses grouped 1D convolutions over temporal dimension with player-wise channels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalConvBlock(nn.Module):
    """1D temporal convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups
        )
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] where T is temporal dimension
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class TemporalDownBlock(nn.Module):
    """Downsampling block for temporal U-Net."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1
    ):
        super().__init__()
        self.conv1 = TemporalConvBlock(in_channels, out_channels, kernel_size, groups=groups)
        self.conv2 = TemporalConvBlock(out_channels, out_channels, kernel_size, groups=groups)
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, T]
            
        Returns:
            (output, skip_connection)
        """
        skip = self.conv2(self.conv1(x))
        out = self.downsample(skip)
        return out, skip


class TemporalUpBlock(nn.Module):
    """Upsampling block for temporal U-Net."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
        skip_channels: Optional[int] = None
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        # Skip connection channels may differ from out_channels, account for that
        skip_ch = skip_channels if skip_channels is not None else out_channels
        concat_channels = out_channels + skip_ch
        self.conv1 = TemporalConvBlock(concat_channels, out_channels, kernel_size, groups=groups)
        self.conv2 = TemporalConvBlock(out_channels, out_channels, kernel_size, groups=groups)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] - input from previous layer
            skip: [B, C, T_skip] - skip connection from encoder
            
        Returns:
            [B, C, T_skip] - upsampled output
        """
        x = self.upsample(x)
        # Handle size mismatch
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation for conditioning.
    FiLM(x) = scale * x + shift
    """
    
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_dim)
        self.shift = nn.Linear(condition_dim, feature_dim)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] - features to modulate
            condition: [B, condition_dim] - conditioning vector
            
        Returns:
            [B, C, T] - modulated features
        """
        scale = self.scale(condition).unsqueeze(-1)  # [B, C, 1]
        shift = self.shift(condition).unsqueeze(-1)  # [B, C, 1]
        return scale * x + shift


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timestep."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B] - timestep indices
            
        Returns:
            [B, dim] - positional encodings
        """
        device = time.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb


class TemporalUNet(nn.Module):
    """
    Temporal U-Net for denoising player trajectories.
    
    Architecture:
    - Input: [B, P, F, T] -> reshape to [B, P*F, T]
    - Temporal convolutions with grouped channels (player-wise)
    - FiLM conditioning at each level
    - U-Net encoder-decoder structure
    """
    
    def __init__(
        self,
        in_channels: int,  # P * F (players * features)
        condition_dim: int = 256,
        channels: int = 256,
        depth: int = 4,
        temporal_kernel: int = 3,
        player_heads: int = 4
    ):
        """
        Args:
            in_channels: Number of input channels (P * F, e.g., 22 * 3 = 66)
            condition_dim: Dimension of condition embedding
            channels: Base number of channels
            depth: Number of down/up blocks
            temporal_kernel: Kernel size for temporal convolutions
            player_heads: Number of player groups for grouped convolutions
        """
        super().__init__()
        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.depth = depth
        
        # Timestep embedding
        time_dim = channels
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEncoding(channels),
            nn.Linear(channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial projection
        self.input_proj = TemporalConvBlock(in_channels, channels, temporal_kernel)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_films = nn.ModuleList()
        curr_channels = channels
        for i in range(depth):
            next_channels = min(channels * (2 ** (i + 1)), 512)
            groups = max(1, curr_channels // player_heads)
            self.down_blocks.append(
                TemporalDownBlock(curr_channels, next_channels, temporal_kernel, groups=groups)
            )
            self.down_films.append(FiLM(condition_dim + time_dim, next_channels))
            curr_channels = next_channels
        
        # Bottleneck
        self.bottleneck_conv1 = TemporalConvBlock(curr_channels, curr_channels, temporal_kernel)
        self.bottleneck_conv2 = TemporalConvBlock(curr_channels, curr_channels, temporal_kernel)
        self.bottleneck_film = FiLM(condition_dim + time_dim, curr_channels)
        
        # Decoder - track skip connection channels to match encoder output
        self.up_blocks = nn.ModuleList()
        self.up_films = nn.ModuleList()
        
        # Calculate skip channel sizes (same as encoder output channels, in encoder order)
        skip_channel_list = []
        temp_curr = channels
        for i in range(depth):
            temp_next = min(channels * (2 ** (i + 1)), 512)
            skip_channel_list.append(temp_next)
            temp_curr = temp_next
        
        # Build up blocks - reverse through depths
        curr_up_channels = curr_channels  # Start from bottleneck (512)
        for i in range(depth - 1, -1, -1):
            # Skip connection from encoder level i
            skip_ch = skip_channel_list[i]
            # Output channels for this decoder level
            next_channels = min(channels * (2 ** i), 512) if i > 0 else channels
            groups = max(1, next_channels // player_heads)
            
            self.up_blocks.append(
                TemporalUpBlock(curr_up_channels, next_channels, temporal_kernel, groups=groups, skip_channels=skip_ch)
            )
            self.up_films.append(FiLM(condition_dim + time_dim, next_channels))
            curr_up_channels = next_channels
        
        # Output projection
        self.output_proj = nn.Conv1d(channels, in_channels, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, P, F, T] or [B, P*F, T] - noisy input
            timestep: [B] - timestep indices
            condition: [B, condition_dim] - context embedding
            
        Returns:
            [B, P, F, T] or [B, P*F, T] - predicted noise
        """
        # Reshape if needed: [B, P, F, T] -> [B, P*F, T]
        original_shape = x.shape
        if len(original_shape) == 4:
            B, P, F, T = original_shape
            x = x.reshape(B, P * F, T)
        else:
            B, C, T = x.shape
            P, F = None, None  # Will reshape back at end
        
        # Timestep embedding
        time_emb = self.time_embed(timestep)  # [B, time_dim]
        combined_cond = torch.cat([condition, time_emb], dim=1)  # [B, condition_dim + time_dim]
        
        # Input projection
        x = self.input_proj(x)  # [B, channels, T]
        
        # Encoder
        skips = []
        for down_block, film in zip(self.down_blocks, self.down_films):
            x, skip = down_block(x)
            x = film(x, combined_cond)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_film(x, combined_cond)
        x = self.bottleneck_conv2(x)
        
        # Decoder
        for up_block, film, skip in zip(self.up_blocks, self.up_films, reversed(skips)):
            x = up_block(x, skip)
            x = film(x, combined_cond)
        
        # Output projection
        x = self.output_proj(x)  # [B, in_channels, T]
        
        # Reshape back if needed
        if len(original_shape) == 4:
            x = x.reshape(original_shape)
        
        return x

