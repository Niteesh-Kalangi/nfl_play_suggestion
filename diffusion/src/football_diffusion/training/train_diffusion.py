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
        loss_config = config.get('loss', {})
        self.velocity_loss_weight = float(loss_config.get('lambda_smooth', 0.5))  # Increased for smoother trajectories
        self.boundary_loss_weight = 0.0  # Disabled: data is normalized, boundary loss not applicable
        
        # Anchor loss weight
        self.anchor_loss_weight = config.get('loss', {}).get('lambda_anchor', 1.0)
        self.anchor_delta = config.get('generation', {}).get('anchor_delta', 0.25)  # Huber loss delta in yards
    
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
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        anchors_t0: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss: noise MSE + velocity smoothness + boundary penalty.
        
        Args:
            predicted_noise: [B, T, P, F] - predicted noise
            target_noise: [B, T, P, F] - target noise
            x: [B, T, P, F] - original trajectories (for velocity computation)
            mask: Optional [B, T] mask (1 for valid, 0 for padded)
            
        Returns:
            Dict with loss components
        """
        # Noise MSE loss (with optional masking)
        if mask is not None:
            # Expand mask: [B, T] -> [B, T, 1, 1] for broadcasting
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
            # Mask out padded elements (multiply by mask, then normalize by valid elements)
            masked_pred = predicted_noise * mask_expanded
            masked_target = target_noise * mask_expanded
            squared_error = (masked_pred - masked_target) ** 2
            valid_elements = mask_expanded.sum()
            if valid_elements > 0:
                noise_loss = squared_error.sum() / valid_elements
            else:
                noise_loss = torch.tensor(0.0, device=predicted_noise.device)
        else:
            # No mask - compute standard MSE (backward compatible)
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
        # NOTE: If training on normalized data, boundary loss doesn't apply to normalized coordinates
        # Disable boundary loss during training (will be checked after denormalization during eval)
        # For normalized data, coordinates are roughly in [-3, 3] range, so raw bounds [0, 120] don't apply
        boundary_loss = torch.tensor(0.0, device=x.device)
        
        # Anchor loss: penalize deviation from anchors at t=0 for anchored players
        anchor_loss = torch.tensor(0.0, device=x.device)
        if anchors_t0 is not None and anchor_mask is not None and timestep is not None:
            # Reconstruct x0 from noisy x_t and predicted noise for timestep 0
            # Only compute anchor loss when timestep is 0
            timestep_zero_mask = (timestep == 0)  # [B]
            
            if timestep_zero_mask.any():
                # For timestep 0, we can directly compare x with anchors
                # Get x0 positions (first 2 features are x, y)
                x0_positions = x[:, 0, :, :2]  # [B, P, 2]
                anchors_expanded = anchors_t0.unsqueeze(0).expand_as(x0_positions)  # [B, P, 2]
                anchor_mask_expanded = anchor_mask.unsqueeze(0).unsqueeze(-1)  # [B, P, 1]
                
                # Compute Huber loss (smooth L1) only for anchored players
                diff = x0_positions - anchors_expanded  # [B, P, 2]
                
                # Apply anchor mask
                diff_masked = diff * anchor_mask_expanded.float()  # [B, P, 2]
                
                # Huber loss (smooth L1)
                abs_diff = torch.abs(diff_masked)
                quadratic = 0.5 * (self.anchor_delta ** 2)
                linear = self.anchor_delta * (abs_diff - 0.5 * self.anchor_delta)
                anchor_loss_t = torch.where(abs_diff < self.anchor_delta, 0.5 * (abs_diff ** 2), linear)
                
                # Average over spatial dimensions and anchored players
                anchor_loss = anchor_loss_t.mean()
        
        # Combined loss
        total_loss = (
            self.noise_loss_weight * noise_loss +
            self.velocity_loss_weight * velocity_loss +
            self.boundary_loss_weight * boundary_loss +
            self.anchor_loss_weight * anchor_loss
        )
        
        return {
            'loss': total_loss,
            'noise_loss': noise_loss,
            'velocity_loss': velocity_loss,
            'boundary_loss': boundary_loss,
            'anchor_loss': anchor_loss
        }
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch['X']  # [B, T, P, F]
        context_cat = batch['context_categorical']
        context_cont = batch['context_continuous']
        mask = batch.get('mask', None)  # [B, T] - optional mask for padding
        anchors_t0 = batch.get('anchors_t0', None)  # [P, 2]
        anchor_mask = batch.get('anchor_mask', None)  # [P]
        
        # Forward pass
        predicted_noise, target_noise, timestep = self.model(
            x, context_cat, context_cont, return_timestep=True
        )
        
        # Compute loss (with optional mask and anchors)
        losses = self.compute_loss(
            predicted_noise, target_noise, x, mask,
            anchors_t0=anchors_t0, anchor_mask=anchor_mask, timestep=timestep
        )
        
        # Log metrics
        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/noise_loss', losses['noise_loss'], on_step=True, on_epoch=True)
        self.log('train/velocity_loss', losses['velocity_loss'], on_step=True, on_epoch=True)
        self.log('train/boundary_loss', losses['boundary_loss'], on_step=True, on_epoch=True)
        self.log('train/anchor_loss', losses['anchor_loss'], on_step=True, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step."""
        x = batch['X']
        context_cat = batch['context_categorical']
        context_cont = batch['context_continuous']
        mask = batch.get('mask', None)  # [B, T] - optional mask for padding
        anchors_t0 = batch.get('anchors_t0', None)  # [P, 2]
        anchor_mask = batch.get('anchor_mask', None)  # [P]
        
        # Forward pass
        predicted_noise, target_noise, timestep = self.model(
            x, context_cat, context_cont, drop_context=False, return_timestep=True
        )
        
        # Compute loss (with optional mask and anchors)
        losses = self.compute_loss(
            predicted_noise, target_noise, x, mask,
            anchors_t0=anchors_t0, anchor_mask=anchor_mask, timestep=timestep
        )
        
        # Log metrics
        self.log('val/loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/noise_loss', losses['noise_loss'], on_step=False, on_epoch=True)
        self.log('val/anchor_loss', losses['anchor_loss'], on_step=False, on_epoch=True)
        
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

