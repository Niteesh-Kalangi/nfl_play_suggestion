"""
PyTorch Lightning module for training the diffusion model.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional
import numpy as np

from ..models.diffusion_wrapper import FootballDiffusion


class DiffusionLightningModule(pl.LightningModule):
    """
    Lightning module for training the diffusion model.
    """
    
    def __init__(
        self,
        config: Dict,
        num_players: int = 22,
        num_features: int = 3
    ):
        """
        Args:
            config: Configuration dictionary
            num_players: Number of players
            num_features: Features per player
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Build model
        diffusion_config = config.get('diffusion', {})
        model_config = diffusion_config.get('model', {})
        
        self.model = FootballDiffusion(
            num_players=num_players,
            num_features=num_features,
            num_timesteps=diffusion_config.get('steps', 1000),
            beta_schedule=diffusion_config.get('beta_schedule', 'cosine'),
            condition_dim=model_config.get('channels', 256),
            model_channels=model_config.get('channels', 256),
            model_depth=model_config.get('depth', 4),
            temporal_kernel=model_config.get('temporal_kernel', 3),
            player_heads=model_config.get('player_heads', 4),
            guidance_scale=diffusion_config.get('guidance_scale', 2.0),
            context_drop_prob=config.get('conditioning', {}).get('drop_prob', 0.1)
        )
        
        # Loss weights
        self.noise_loss_weight = 1.0
        self.velocity_loss_weight = 0.1
        self.boundary_loss_weight = 0.05
    
    def forward(
        self,
        x: torch.Tensor,
        context_categorical: list,
        context_continuous: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        predicted_noise, _ = self.model(
            x, context_categorical, context_continuous
        )
        return predicted_noise
    
    def compute_loss(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss: noise MSE + velocity smoothness + boundary penalty.
        
        Args:
            predicted_noise: [B, T, P, F] - predicted noise
            target_noise: [B, T, P, F] - target noise
            x: [B, T, P, F] - original trajectories (for velocity computation)
            
        Returns:
            Dict with loss components
        """
        # Noise MSE loss
        noise_loss = F.mse_loss(predicted_noise, target_noise)
        
        # Velocity smoothness loss (penalize large accelerations)
        # Compute velocity from positions (x, y)
        if x.shape[-1] >= 2:
            velocity = x[:, 1:, :, :2] - x[:, :-1, :, :2]  # [B, T-1, P, 2]
            acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, P, 2]
            velocity_loss = torch.mean(acceleration ** 2)
        else:
            velocity_loss = torch.tensor(0.0, device=x.device)
        
        # Boundary penalty (penalize positions outside field bounds)
        field_bounds = self.config.get('eval', {}).get('field_bounds', [0, 120, 0, 53.3])
        x_min, x_max, y_min, y_max = field_bounds
        
        if x.shape[-1] >= 2:
            x_pos = x[:, :, :, 0]
            y_pos = x[:, :, :, 1]
            
            # Penalize positions outside bounds
            x_violation = torch.clamp(x_min - x_pos, 0) + torch.clamp(x_pos - x_max, 0)
            y_violation = torch.clamp(y_min - y_pos, 0) + torch.clamp(y_pos - y_max, 0)
            boundary_loss = torch.mean(x_violation ** 2 + y_violation ** 2)
        else:
            boundary_loss = torch.tensor(0.0, device=x.device)
        
        # Combined loss
        total_loss = (
            self.noise_loss_weight * noise_loss +
            self.velocity_loss_weight * velocity_loss +
            self.boundary_loss_weight * boundary_loss
        )
        
        return {
            'loss': total_loss,
            'noise_loss': noise_loss,
            'velocity_loss': velocity_loss,
            'boundary_loss': boundary_loss
        }
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch['X']  # [B, T, P, F]
        context_cat = batch['context_categorical']
        context_cont = batch['context_continuous']
        
        # Forward pass
        predicted_noise, target_noise = self.model(
            x, context_cat, context_cont
        )
        
        # Compute loss
        losses = self.compute_loss(predicted_noise, target_noise, x)
        
        # Log metrics
        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/noise_loss', losses['noise_loss'], on_step=True, on_epoch=True)
        self.log('train/velocity_loss', losses['velocity_loss'], on_step=True, on_epoch=True)
        self.log('train/boundary_loss', losses['boundary_loss'], on_step=True, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step."""
        x = batch['X']
        context_cat = batch['context_categorical']
        context_cont = batch['context_continuous']
        
        # Forward pass
        predicted_noise, target_noise = self.model(
            x, context_cat, context_cont, drop_context=False
        )
        
        # Compute loss
        losses = self.compute_loss(predicted_noise, target_noise, x)
        
        # Log metrics
        self.log('val/loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/noise_loss', losses['noise_loss'], on_step=False, on_epoch=True)
        
        return losses
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        train_config = self.config.get('train', {})
        
        # Ensure lr and wd are floats (YAML might load them as strings)
        lr = float(train_config.get('lr', 1e-4))
        wd = float(train_config.get('wd', 1e-2))
        
        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=wd
        )
        
        max_epochs = train_config.get('max_epochs', 50)
        if isinstance(max_epochs, str):
            max_epochs = int(float(max_epochs)) if '.' in max_epochs else int(max_epochs)
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(max_epochs),
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

