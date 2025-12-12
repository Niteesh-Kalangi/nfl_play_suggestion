"""
Training script for autoregressive trajectory generators (LSTM + Transformer).

Updated to use new structure matching diffusion model.

Usage:
    python train_autoregressive.py --model lstm
    python train_autoregressive.py --model transformer
    python train_autoregressive.py --model all
"""
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from src.autoregressive import (
    make_autoregressive_splits,
    create_autoregressive_dataloaders,
    LSTMTrajectoryGenerator,
    TransformerTrajectoryGenerator
)
from src.eval import compute_ade, compute_fde


def parse_args():
    parser = argparse.ArgumentParser(description='Train autoregressive trajectory generators')
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['lstm', 'transformer', 'all'],
        default='all',
        help='Which model(s) to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu/mps). Auto-detects if not specified.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    return parser.parse_args()


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get the best available device."""
    if device_str:
        return torch.device(device_str)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def build_model(
    model_type: str,
    config: Dict
) -> nn.Module:
    """
    Build a trajectory generator model.
    
    Args:
        model_type: 'lstm' or 'transformer'
        config: Configuration dict
        
    Returns:
        Initialized model
    """
    ar_config = config.get('autoregressive', {})
    
    # Fixed dimensions for new structure
    num_players = 22  # 11 offense + 11 defense
    num_features = 3  # x, y, s
    context_dim = 256
    
    if model_type == 'lstm':
        lstm_config = ar_config.get('lstm', {})
        model = LSTMTrajectoryGenerator(
            num_players=num_players,
            num_features=num_features,
            hidden_dim=lstm_config.get('hidden_dim', 256),
            num_layers=lstm_config.get('num_layers', 2),
            dropout=lstm_config.get('dropout', 0.1),
            bidirectional=lstm_config.get('bidirectional', False),
            context_dim=context_dim
        )
    elif model_type == 'transformer':
        tf_config = ar_config.get('transformer', {})
        model = TransformerTrajectoryGenerator(
            num_players=num_players,
            num_features=num_features,
            d_model=tf_config.get('d_model', 256),
            nhead=tf_config.get('nhead', 8),
            num_layers=tf_config.get('num_layers', 4),
            dim_feedforward=tf_config.get('dim_feedforward', 512),
            dropout=tf_config.get('dropout', 0.1),
            max_len=tf_config.get('max_len', 100),
            context_dim=context_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict,
    num_training_steps: int
) -> Tuple[optim.Optimizer, Optional[object]]:
    """
    Build optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        config: Configuration dict
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    ar_config = config.get('autoregressive', {})
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ar_config.get('learning_rate', 0.001),
        weight_decay=ar_config.get('weight_decay', 0.0001)
    )
    
    scheduler_type = ar_config.get('scheduler', 'cosine')
    epochs = ar_config.get('epochs', 50)
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int = 100
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to use
        log_interval: How often to log progress
        
    Returns:
        Dict of training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        X = batch['X'].to(device)  # [B, T, P, F]
        context_cat = batch['context_categorical']
        context_cont = batch['context_continuous'].to(device)  # [B, 3]
        
        # Flatten positions: [B, T, P, F] -> [B, T, P*F]
        B, T, P, F = X.shape
        X_flat = X.reshape(B, T, P * F)
        
        # Create input: previous positions (shift by 1 for teacher forcing)
        X_prev = torch.cat([X_flat[:, :1, :], X_flat[:, :-1, :]], dim=1)
        
        optimizer.zero_grad()
        
        # Forward pass with teacher forcing
        if isinstance(model, LSTMTrajectoryGenerator):
            output, _ = model(X_prev, context_cat, context_cont)
        else:
            output = model(X_prev, context_cat, context_cont)
        
        # Compute loss
        loss = model.compute_loss(output, X_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({'loss': total_loss / num_batches})
    
    return {
        'loss': total_loss / num_batches
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    use_rollout: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to use
        use_rollout: Whether to use autoregressive rollout (True) or teacher forcing (False)
        
    Returns:
        Dict of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        X = batch['X'].to(device)  # [B, T, P, F]
        context_cat = batch['context_categorical']
        context_cont = batch['context_continuous'].to(device)  # [B, 3]
        
        B, T, P, F = X.shape
        X_flat = X.reshape(B, T, P * F)
        
        if use_rollout:
            # Get initial positions from first timestep
            init_positions = X_flat[:, 0, :]  # [B, P*F]
            
            # Rollout
            pred = model.rollout(
                context_categorical=context_cat,
                context_continuous=context_cont,
                horizon=T,
                init_positions=init_positions
            )  # [B, T, P*F]
        else:
            # Teacher forcing
            X_prev = torch.cat([X_flat[:, :1, :], X_flat[:, :-1, :]], dim=1)
            if isinstance(model, LSTMTrajectoryGenerator):
                output, _ = model(X_prev, context_cat, context_cont)
            else:
                output = model(X_prev, context_cat, context_cont)
            pred = output
        
        # Compute loss
        loss = model.compute_loss(pred, X_flat)
        total_loss += loss.item()
        
        # Reshape for metrics: [B, T, P*F] -> [B, T, P, F]
        pred_reshaped = pred.reshape(B, T, P, F)
        target_reshaped = X.reshape(B, T, P, F)
        
        # Compute ADE/FDE on positions (x, y)
        ade = compute_ade(pred_reshaped.cpu().numpy(), target_reshaped.cpu().numpy())
        fde = compute_fde(
            pred_reshaped.cpu().numpy(), 
            target_reshaped.cpu().numpy()
        )
        
        total_ade += ade
        total_fde += fde
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'ade': total_ade / num_batches,
        'fde': total_fde / num_batches
    }


def train_model(
    model_type: str,
    config: Dict,
    device: torch.device,
    dataloaders: Dict,
) -> Tuple[nn.Module, Dict]:
    """
    Train a single model.
    
    Args:
        model_type: 'lstm' or 'transformer'
        config: Configuration dict
        device: Device to use
        dataloaders: Dict of dataloaders
        
    Returns:
        Tuple of (trained model, training history)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Trajectory Generator")
    print(f"{'='*60}")
    
    ar_config = config.get('autoregressive', {})
    epochs = ar_config.get('epochs', 50)
    patience = ar_config.get('early_stopping_patience', 10)
    # Disable early stopping if patience is very large (>= epochs)
    if patience >= epochs:
        patience = epochs  # Will never trigger
        print(f"  Early stopping disabled - training for all {epochs} epochs")
    log_interval = config.get('output', {}).get('log_interval', 100)
    
    # Build model
    model = build_model(model_type, config)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build optimizer and scheduler
    num_training_steps = epochs * len(dataloaders['train'])
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, num_training_steps)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ade': [],
        'val_fde': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, device, log_interval
        )
        history['train_loss'].append(train_metrics['loss'])
        
        # Evaluate on validation set
        val_metrics = evaluate(model, dataloaders['val'], device, use_rollout=True)
        history['val_loss'].append(val_metrics['loss'])
        history['val_ade'].append(val_metrics['ade'])
        history['val_fde'].append(val_metrics['fde'])
        
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping (disabled if patience >= epochs)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience and patience < epochs:
                print(f"  Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def save_checkpoint(
    model: nn.Module,
    model_type: str,
    config: Dict,
    history: Dict,
    output_dir: Path
):
    """Save model checkpoint and training history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f'{model_type}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'num_players': model.num_players,
        'num_features': model.num_features,
        'hidden_dim': model.hidden_dim,
        'config': config
    }, model_path)
    print(f"✓ Saved model to {model_path}")
    
    # Save history
    history_path = output_dir / f'{model_type}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved history to {history_path}")


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Autoregressive Trajectory Generator Training")
    print("=" * 60)
    
    # Get device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")
    
    # Build datasets
    print("\n[1/4] Building trajectory datasets...")
    datasets = make_autoregressive_splits(config['data_dir'], config)
    
    print(f"  Train: {len(datasets['train'])} plays")
    print(f"  Val: {len(datasets['val'])} plays")
    print(f"  Test: {len(datasets['test'])} plays")
    
    # Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    ar_config = config.get('autoregressive', {})
    batch_size = ar_config.get('batch_size', 32)
    dataloaders = create_autoregressive_dataloaders(datasets, batch_size=batch_size)
    
    # Determine which models to train
    models_to_train = ['lstm', 'transformer'] if args.model == 'all' else [args.model]
    
    # Output directory
    output_dir = Path(config.get('output', {}).get('autoregressive_dir', 'artifacts/autoregressive'))
    
    # Train models
    print(f"\n[3/4] Training models: {models_to_train}")
    
    trained_models = {}
    all_results = {}
    
    for model_type in models_to_train:
        model, history = train_model(
            model_type, config, device, dataloaders
        )
        trained_models[model_type] = model
        
        # Final evaluation on test set
        print(f"\n[4/4] Final evaluation on test set ({model_type})...")
        test_metrics = evaluate(model, dataloaders['test'], device, use_rollout=True)
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test ADE: {test_metrics['ade']:.4f}")
        print(f"  Test FDE: {test_metrics['fde']:.4f}")
        
        all_results[model_type] = {
            'history': history,
            'test_metrics': test_metrics
        }
        
        # Save checkpoint
        save_checkpoint(model, model_type, config, history, output_dir)
    
    # Save combined results
    results_path = output_dir / 'results.json'
    results_json = {}
    for model_type, result in all_results.items():
        results_json[model_type] = {
            'final_train_loss': result['history']['train_loss'][-1],
            'final_val_loss': result['history']['val_loss'][-1],
            'final_val_ade': result['history']['val_ade'][-1],
            'final_val_fde': result['history']['val_fde'][-1],
            'test_loss': result['test_metrics']['loss'],
            'test_ade': result['test_metrics']['ade'],
            'test_fde': result['test_metrics']['fde']
        }
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModel Performance Summary:")
    print("-" * 40)
    print(f"{'Model':<15} {'Test ADE':<12} {'Test FDE':<12}")
    print("-" * 40)
    for model_type, result in all_results.items():
        print(f"{model_type.upper():<15} {result['test_metrics']['ade']:<12.4f} {result['test_metrics']['fde']:<12.4f}")
    print("-" * 40)


if __name__ == '__main__':
    main()

