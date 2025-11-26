"""
Training script for autoregressive trajectory generation baselines.

Trains LSTM and Transformer models for NFL offensive trajectory synthesis.
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_io import load_raw, filter_normal_plays
from src.preprocess import standardize_and_join
from src.trajectory_data import (
    extract_play_trajectories,
    TrajectoryDataset,
    create_trajectory_splits,
    denormalize_trajectories
)
from src.models import (
    LSTMTrajectoryGenerator,
    LSTMTrajectoryGeneratorConfig,
    create_lstm_generator,
    TransformerTrajectoryGenerator,
    TransformerTrajectoryGeneratorConfig,
    create_transformer_generator,
    WarmupCosineScheduler
)
from src.trajectory_eval import (
    evaluate_trajectory_generation,
    print_evaluation_report
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_traj = batch['input_traj'].to(device)
        target_traj = batch['target_traj'].to(device)
        context = batch['context'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, LSTMTrajectoryGenerator):
            output, _ = model(input_traj, context)
        else:
            output = model(input_traj, context)
        
        # Compute loss
        loss = model.compute_loss(output, target_traj)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_traj = batch['input_traj'].to(device)
            target_traj = batch['target_traj'].to(device)
            context = batch['context'].to(device)
            
            if isinstance(model, LSTMTrajectoryGenerator):
                output, _ = model(input_traj, context)
            else:
                output = model(input_traj, context)
            
            loss = model.compute_loss(output, target_traj)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def generate_and_evaluate(
    model: nn.Module,
    dataset: TrajectoryDataset,
    device: torch.device,
    n_samples: int = 100,
    n_steps: int = 49
) -> dict:
    """Generate trajectories and evaluate."""
    model.eval()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    pred_trajectories = []
    target_trajectories = []
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            full_traj = sample['full_traj'].unsqueeze(0).to(device)  # (1, n_frames, n_players, 2)
            context = sample['context'].unsqueeze(0).to(device)
            
            # Get initial frame (flattened)
            n_players = full_traj.shape[2]
            initial_frame = full_traj[:, 0].reshape(1, -1)  # (1, n_players * 2)
            
            # Generate
            generated = model.generate(initial_frame, context, n_steps)  # (1, n_steps+1, n_players*2)
            
            # Reshape to (1, n_frames, n_players, 2)
            generated = generated.reshape(1, -1, n_players, 2)
            
            pred_trajectories.append(generated.cpu().numpy())
            target_trajectories.append(full_traj.cpu().numpy())
    
    pred_trajectories = np.concatenate(pred_trajectories, axis=0)
    target_trajectories = np.concatenate(target_trajectories, axis=0)
    
    # Evaluate
    metrics = evaluate_trajectory_generation(
        pred_trajectories,
        target_trajectories,
        dt=0.1,
        denorm_stats=dataset.stats
    )
    
    return metrics


def train_model(
    model: nn.Module,
    train_dataset: TrajectoryDataset,
    val_dataset: TrajectoryDataset,
    config: dict,
    device: torch.device,
    model_name: str
) -> nn.Module:
    """Full training loop."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    total_steps = len(train_loader) * config['epochs']
    if 'warmup_steps' in config:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=config['warmup_steps'],
            total_steps=total_steps
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train trajectory generation baselines')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer', 'both'], 
                        default='both', help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--save_dir', type=str, default='artifacts', help='Save directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NFL Trajectory Generation - Autoregressive Baselines")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/6] Loading raw data...")
    data = load_raw(config['data_dir'])
    
    # Step 2: Filter normal plays
    print("\n[2/6] Filtering normal plays...")
    plays_clean = filter_normal_plays(data['plays'])
    
    # Step 3: Preprocess
    print("\n[3/6] Preprocessing and standardizing...")
    plays_std, tracking_std = standardize_and_join(
        plays_clean,
        data['games'],
        data['tracking']
    )
    
    # Step 4: Extract trajectories
    print("\n[4/6] Extracting play trajectories...")
    max_frames = config.get('trajectory', {}).get('max_frames', 50)
    trajectories, contexts, meta = extract_play_trajectories(
        tracking_std,
        plays_std,
        max_frames=max_frames,
        offense_only=True
    )
    
    # Step 5: Create splits
    print("\n[5/6] Creating train/val/test splits...")
    splits = create_trajectory_splits(
        trajectories,
        contexts,
        meta,
        plays_std,
        train_weeks=config['splits']['train'],
        val_weeks=config['splits']['val'],
        test_weeks=config['splits']['test']
    )
    
    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits.get('test', val_dataset)
    
    print(f"  Train: {len(train_dataset)} trajectories")
    print(f"  Val: {len(val_dataset)} trajectories")
    print(f"  Test: {len(test_dataset)} trajectories")
    
    # Training config
    train_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'patience': 10,
        'warmup_steps': 500
    }
    
    # Get dimensions from dataset
    n_players = train_dataset.n_players
    player_dim = train_dataset.player_dim
    context_dim = train_dataset.context_dim
    
    results = {}
    
    # Step 6: Train models
    print("\n[6/6] Training models...")
    
    if args.model in ['lstm', 'both']:
        # LSTM Generator
        lstm_config = LSTMTrajectoryGeneratorConfig(
            n_players=n_players,
            player_dim=player_dim,
            context_dim=context_dim,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
            context_embed_dim=64
        )
        lstm_model = create_lstm_generator(lstm_config).to(device)
        
        print(f"\nLSTM Model Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
        
        lstm_model = train_model(
            lstm_model,
            train_dataset,
            val_dataset,
            train_config,
            device,
            "LSTM Trajectory Generator"
        )
        
        # Evaluate LSTM
        print("\nEvaluating LSTM on test set...")
        lstm_metrics = generate_and_evaluate(lstm_model, test_dataset, device)
        print_evaluation_report(lstm_metrics, "LSTM Generator")
        results['lstm'] = lstm_metrics
        
        # Save LSTM model
        torch.save({
            'model_state_dict': lstm_model.state_dict(),
            'config': lstm_config.to_dict(),
            'metrics': lstm_metrics,
            'norm_stats': train_dataset.stats
        }, save_dir / 'lstm_trajectory_generator.pt')
        print(f"Saved LSTM model to {save_dir / 'lstm_trajectory_generator.pt'}")
    
    if args.model in ['transformer', 'both']:
        # Transformer Generator
        transformer_config = TransformerTrajectoryGeneratorConfig(
            n_players=n_players,
            player_dim=player_dim,
            context_dim=context_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )
        transformer_model = create_transformer_generator(transformer_config).to(device)
        
        print(f"\nTransformer Model Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
        
        # Use lower learning rate for transformer
        train_config_transformer = train_config.copy()
        train_config_transformer['learning_rate'] = 1e-4
        
        transformer_model = train_model(
            transformer_model,
            train_dataset,
            val_dataset,
            train_config_transformer,
            device,
            "Transformer Trajectory Generator"
        )
        
        # Evaluate Transformer
        print("\nEvaluating Transformer on test set...")
        transformer_metrics = generate_and_evaluate(transformer_model, test_dataset, device)
        print_evaluation_report(transformer_metrics, "Transformer Generator")
        results['transformer'] = transformer_metrics
        
        # Save Transformer model
        torch.save({
            'model_state_dict': transformer_model.state_dict(),
            'config': transformer_config.to_dict(),
            'metrics': transformer_metrics,
            'norm_stats': train_dataset.stats
        }, save_dir / 'transformer_trajectory_generator.pt')
        print(f"Saved Transformer model to {save_dir / 'transformer_trajectory_generator.pt'}")
    
    # Save comparison results
    with open(save_dir / 'trajectory_baseline_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        results_json = {}
        for model_name, metrics in results.items():
            results_json[model_name] = {k: float(v) for k, v in metrics.items()}
        json.dump(results_json, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Print comparison table
    if len(results) > 1:
        print("\nModel Comparison:")
        print("-" * 50)
        print(f"{'Metric':<30} {'LSTM':<10} {'Transformer':<10}")
        print("-" * 50)
        for metric in ['ade', 'fde', 'phys_speed_violation_rate', 'collision_rate']:
            lstm_val = results.get('lstm', {}).get(metric, 'N/A')
            trans_val = results.get('transformer', {}).get(metric, 'N/A')
            if isinstance(lstm_val, float):
                lstm_str = f"{lstm_val:.4f}"
            else:
                lstm_str = str(lstm_val)
            if isinstance(trans_val, float):
                trans_str = f"{trans_val:.4f}"
            else:
                trans_str = str(trans_val)
            print(f"{metric:<30} {lstm_str:<10} {trans_str:<10}")
        print("-" * 50)


if __name__ == '__main__':
    main()
