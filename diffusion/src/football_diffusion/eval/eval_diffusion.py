"""
Evaluation script for diffusion model.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import yaml
import json
from tabulate import tabulate

from ..data.dataset import FootballPlayDataset, collate_fn
from ..models.diffusion_wrapper import FootballDiffusion
from ..training.train_diffusion import DiffusionLightningModule
from .metrics import compute_all_metrics, compute_diversity
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def evaluate_diffusion_model(
    model: FootballDiffusion,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 8,
    sample_steps: List[int] = [50],
    ddim: bool = False
) -> Dict[str, Dict]:
    """
    Evaluate diffusion model on dataset.
    
    Args:
        model: Trained diffusion model
        dataloader: DataLoader for evaluation
        device: Device to run on
        num_samples: Number of samples per context
        sample_steps: List of sampling steps to evaluate
        ddim: Use DDIM sampling
        
    Returns:
        Dict with metrics for each sampling step
    """
    model.eval()
    model = model.to(device)
    
    all_results = {steps: [] for steps in sample_steps}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            X = batch['X'].to(device)  # [B, T, P, F]
            context_cat = batch['context_categorical']
            context_cont = batch['context_continuous'].to(device)
            mask = batch['mask'].to(device)
            
            B, T, P, F = X.shape
            
            # Generate samples for each sampling step
            for steps in sample_steps:
                # Generate samples
                samples = []
                for _ in range(num_samples):
                    sample = model.sample(
                        shape=(T, P, F),
                        context_categorical=context_cat,
                        context_continuous=context_cont,
                        num_steps=steps,
                        ddim=ddim
                    )
                    samples.append(sample)
                
                samples = torch.stack(samples, dim=1)  # [B, num_samples, T, P, F]
                
                # Compute metrics for each sample
                batch_metrics = []
                for s_idx in range(num_samples):
                    sample = samples[:, s_idx]  # [B, T, P, F]
                    
                    metrics = compute_all_metrics(
                        sample, X, mask,
                        field_bounds=[0, 120, 0, 53.3],
                        speed_cap=12.0
                    )
                    batch_metrics.append(metrics)
                
                # Average metrics across samples
                avg_metrics = {
                    k: np.mean([m[k] for m in batch_metrics])
                    for k in batch_metrics[0].keys()
                }
                
                # Compute diversity
                # Flatten batch and samples: [B*num_samples, T, P, F]
                samples_flat = samples.reshape(-1, T, P, F)
                context_groups = [
                    i // num_samples for i in range(B * num_samples)
                ]
                diversity = compute_diversity(samples_flat, context_groups)
                avg_metrics['diversity'] = diversity
                
                all_results[steps].append(avg_metrics)
    
    # Aggregate results
    final_results = {}
    for steps, results in all_results.items():
        if len(results) == 0:
            continue
        
        aggregated = {
            k: np.mean([r[k] for r in results])
            for k in results[0].keys()
        }
        final_results[f'{steps}_steps'] = aggregated
    
    return final_results


def load_model(checkpoint_path: Path, config: Dict, device: torch.device) -> FootballDiffusion:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint
        config: Configuration dict
        device: Device to load on
        
    Returns:
        Loaded model
    """
    # Load Lightning module
    module = DiffusionLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        config=config
    )
    
    return module.model.to(device)


def print_results_table(results: Dict[str, Dict]):
    """Print evaluation results as a formatted table."""
    # Prepare table data
    headers = ['Sampling Steps', 'ADE', 'FDE', 'Validity', 'Diversity', 'Collision Rate']
    rows = []
    
    for steps_key, metrics in results.items():
        steps = steps_key.replace('_steps', '')
        row = [
            steps,
            f"{metrics['ade']:.4f}",
            f"{metrics['fde']:.4f}",
            f"{metrics['validity_rate']:.4f}",
            f"{metrics['diversity']:.4f}",
            f"{metrics['collision_rate']:.4f}"
        ]
        rows.append(row)
    
    print("\n" + "=" * 80)
    print("DIFFUSION MODEL EVALUATION RESULTS")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    print("=" * 80 + "\n")


def run_evaluation(
    checkpoint_path: str,
    cache_dir: str,
    config_path: str,
    split: str = 'test',
    batch_size: int = 8,
    num_samples: int = 8,
    sample_steps: List[int] = [20, 50, 100],
    ddim: bool = False,
    device: str = 'cuda'
):
    """
    Run full evaluation pipeline.
    
    Args:
        checkpoint_path: Path to model checkpoint
        cache_dir: Directory with cached data
        config_path: Path to config file
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for evaluation
        num_samples: Number of samples per context
        sample_steps: List of sampling steps to evaluate
        ddim: Use DDIM sampling
        device: Device to use
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup device with preference: mps -> cuda -> cpu
    if device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    # Check for pickle first (preferred), then parquet (backwards compat)
    cache_dir_path = Path(cache_dir)
    cache_file_pkl = cache_dir_path / 'processed_plays.pkl'
    cache_file_parquet = cache_dir_path / 'processed_plays.parquet'
    if cache_file_pkl.exists():
        cache_file = cache_file_pkl
    elif cache_file_parquet.exists():
        cache_file = cache_file_parquet
    else:
        raise FileNotFoundError(f"Cache file not found. Expected {cache_file_pkl} or {cache_file_parquet}")
    metadata_file = cache_dir_path / 'metadata.json'
    
    dataset = FootballPlayDataset(
        cache_file, metadata_file, split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Loaded {len(dataset)} plays from {split} set")
    
    # Load model
    checkpoint_path = Path(checkpoint_path)
    model = load_model(checkpoint_path, config, device)
    print(f"Loaded model from {checkpoint_path}")
    
    # Evaluate
    results = evaluate_diffusion_model(
        model, dataloader, device,
        num_samples=num_samples,
        sample_steps=sample_steps,
        ddim=ddim
    )
    
    # Print results
    print_results_table(results)
    
    # Save results
    output_file = Path(checkpoint_path).parent / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--sample_steps', type=int, nargs='+', default=[20, 50, 100])
    parser.add_argument('--ddim', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        config_path=args.config,
        split=args.split,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        sample_steps=args.sample_steps,
        ddim=args.ddim,
        device=args.device
    )
