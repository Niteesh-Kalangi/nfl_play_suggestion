"""
Unified trajectory evaluation script for autoregressive and diffusion models.

Computes ADE, FDE, and other metrics for direct comparison between models.

Usage:
    python eval_trajectories.py --models lstm transformer
    python eval_trajectories.py --models all
    python eval_trajectories.py --models lstm --checkpoint artifacts/autoregressive/lstm.pt
"""
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Dict, List, Optional
from tabulate import tabulate

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trajectory_data import make_trajectory_splits, create_dataloaders
from autoreg.models.autoregressive_lstm import LSTMTrajectoryGenerator
from autoreg.models.autoregressive_transformer import TransformerTrajectoryGenerator
from src.eval import (
    compute_ade, 
    compute_fde, 
    compute_collision_rate,
    compute_speed_distribution_distance,
    evaluate_trajectory_model,
    print_trajectory_report
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trajectory generation models')
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['all'],
        help='Models to evaluate: lstm, transformer, diffusion, or all'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='artifacts/autoregressive',
        help='Directory containing model checkpoints'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['val', 'test'],
        default='test',
        help='Which split to evaluate on'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu/mps)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (None = all)'
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


def load_model(
    model_type: str,
    checkpoint_path: Path,
    device: torch.device
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: 'lstm' or 'transformer'
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    hidden_dim = checkpoint['hidden_dim']
    config = checkpoint.get('config', {})
    ar_config = config.get('autoregressive', {})
    
    if model_type == 'lstm':
        lstm_config = ar_config.get('lstm', {})
        model = LSTMTrajectoryGenerator(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_config.get('num_layers', 2),
            dropout=lstm_config.get('dropout', 0.1)
        )
    elif model_type == 'transformer':
        tf_config = ar_config.get('transformer', {})
        model = TransformerTrajectoryGenerator(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=hidden_dim,
            nhead=tf_config.get('nhead', 8),
            num_layers=tf_config.get('num_layers', 4),
            dim_feedforward=tf_config.get('dim_feedforward', 512),
            dropout=tf_config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_model_on_split(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    config: Dict,
    num_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate a model on a data split.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation split
        device: Device to use
        config: Configuration dict
        num_samples: Optional limit on number of samples
        
    Returns:
        Dictionary of evaluation metrics
    """
    traj_config = config.get('trajectories', {})
    num_players = traj_config.get('num_players', 11)
    features_per_player = 4 if traj_config.get('include_velocity', True) else 2
    fps = traj_config.get('fps', 10.0)
    
    all_pred = []
    all_target = []
    all_mask = []
    
    samples_processed = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        X = batch['X'].to(device)
        Y = batch['Y'].to(device)
        mask = batch['mask'].to(device)
        
        batch_size = X.shape[0]
        
        # Extract context from first timestep
        context_dim = model.get_context_dim()
        init_context = X[:, 0, :context_dim]
        init_positions = Y[:, 0, :]
        
        # Determine horizon
        horizon = int(mask.sum(dim=1).max().item())
        
        # Generate trajectories via rollout
        pred = model.rollout(init_context, horizon, init_positions)
        
        # Store for batch evaluation
        all_pred.append(pred.cpu().numpy())
        all_target.append(Y[:, :horizon].cpu().numpy())
        all_mask.append(mask[:, :horizon].cpu().numpy())
        
        samples_processed += batch_size
        if num_samples is not None and samples_processed >= num_samples:
            break
    
    # Concatenate all batches
    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    
    # Compute metrics
    metrics = evaluate_trajectory_model(
        pred, target, mask,
        num_players=num_players,
        features_per_player=features_per_player,
        collision_threshold=1.0,
        fps=fps
    )
    
    return metrics


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Trajectory Model Evaluation")
    print("=" * 60)
    
    # Get device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")
    
    # Build datasets
    print(f"\nLoading {args.split} dataset...")
    datasets = make_trajectory_splits(config['data_dir'], config)
    
    eval_dataset = datasets[args.split]
    print(f"  {args.split.capitalize()} set: {len(eval_dataset)} sequences")
    
    # Create dataloader
    ar_config = config.get('autoregressive', {})
    batch_size = ar_config.get('batch_size', 32)
    dataloader = create_dataloaders(
        {args.split: eval_dataset}, 
        batch_size=batch_size
    )[args.split]
    
    # Determine which models to evaluate
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if 'all' in args.models:
        models_to_eval = ['lstm', 'transformer']
        # Check for diffusion model
        if (checkpoint_dir.parent / 'diffusion' / 'diffusion.pt').exists():
            models_to_eval.append('diffusion')
    else:
        models_to_eval = args.models
    
    # Evaluate each model
    results = {}
    
    for model_type in models_to_eval:
        checkpoint_path = checkpoint_dir / f'{model_type}.pt'
        
        if not checkpoint_path.exists():
            print(f"\nSkipping {model_type}: checkpoint not found at {checkpoint_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()}")
        print(f"{'='*60}")
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        model = load_model(model_type, checkpoint_path, device)
        
        # Evaluate
        metrics = evaluate_model_on_split(
            model, dataloader, device, config, args.num_samples
        )
        
        results[model_type] = metrics
        
        # Print comprehensive report
        print_trajectory_report(metrics, f"{model_type.upper()} Generator")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    
    if results:
        # Prepare comparison table with key metrics
        headers = ['Metric', *[m.upper() for m in results.keys()]]
        
        key_metrics = [
            ('ade', 'ADE (yards)'),
            ('fde', 'FDE (yards)'),
            ('within_1yd', 'Within 1yd (%)'),
            ('within_2yd', 'Within 2yd (%)'),
            ('within_5yd', 'Within 5yd (%)'),
            ('direction_accuracy', 'Direction Acc (%)'),
            ('phys_speed_violation_rate', 'Speed Violations (%)'),
            ('phys_accel_violation_rate', 'Accel Violations (%)'),
            ('collision_rate', 'Collision Rate (%)'),
        ]
        
        rows = []
        for metric_key, metric_name in key_metrics:
            row = [metric_name]
            for model_type in results.keys():
                val = results[model_type].get(metric_key, float('nan'))
                if 'rate' in metric_key or 'within' in metric_key or 'accuracy' in metric_key:
                    row.append(f"{val*100:.2f}")
                else:
                    row.append(f"{val:.3f}")
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    # Save results
    output_path = args.output or (checkpoint_dir / f'eval_results_{args.split}.json')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")
    
    # Print baseline comparison note
    print("\n" + "-" * 60)
    print("Note: These autoregressive models serve as the PRIMARY baselines")
    print("for comparing against the diffusion model in the paper.")
    print("-" * 60)


if __name__ == '__main__':
    main()

