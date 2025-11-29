"""
Main training script for diffusion model.
"""
import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.football_diffusion.data.dataset import FootballPlayDataset, collate_fn
from src.football_diffusion.training.train_diffusion import DiffusionLightningModule
from torch.utils.data import DataLoader
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/football_diffusion/config/train.yaml')
    parser.add_argument('--cache_dir', type=str, default='../data/cache')
    parser.add_argument('--output_dir', type=str, default='../artifacts/diffusion')
    parser.add_argument('--gpus', type=int, default=1, help='Deprecated: use --devices instead')
    parser.add_argument('--devices', type=int, default=None, help='Number of devices (0 for CPU, 1+ for GPU)')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Merge with default config
    default_config_path = Path(__file__).parent / 'src/football_diffusion/config/default.yaml'
    with open(default_config_path) as f:
        default_config = yaml.safe_load(f)
    
    # Merge configs (train.yaml overrides default.yaml)
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
    
    # Setup paths - resolve relative to script location
    script_dir = Path(__file__).parent  # diffusion/
    
    # Resolve paths (relative paths resolved from script_dir)
    if Path(args.cache_dir).is_absolute():
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = (script_dir / args.cache_dir).resolve()
    
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = (script_dir / args.output_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for cache in: {cache_dir}")
    print(f"Cache dir exists: {cache_dir.exists()}")
    
    # Load datasets
    # Check for pickle first (preferred), then parquet (backwards compat)
    cache_file_pkl = cache_dir / 'processed_plays.pkl'
    cache_file_parquet = cache_dir / 'processed_plays.parquet'
    
    if cache_file_pkl.exists():
        cache_file = cache_file_pkl
        print(f"Found cache file: {cache_file}")
    elif cache_file_parquet.exists():
        cache_file = cache_file_parquet
        print(f"Found cache file: {cache_file}")
    else:
        print(f"Cache file not found. Tried:")
        print(f"  - {cache_file_pkl} (exists: {cache_file_pkl.exists()})")
        print(f"  - {cache_file_parquet} (exists: {cache_file_parquet.exists()})")
        raise FileNotFoundError(f"Cache file not found. Expected {cache_file_pkl} or {cache_file_parquet}")
    
    metadata_file = cache_dir / 'metadata.json'
    
    # Determine accelerator early (needed for DataLoader pin_memory setting)
    num_devices = args.devices if args.devices is not None else args.gpus
    
    # Set accelerator and devices - check for MPS (Mac GPU) first, then CUDA, then CPU
    if num_devices is None or num_devices == 0:
        accelerator = 'cpu'
        devices = 'auto'
    else:
        if torch.backends.mps.is_available():
            accelerator = 'mps'
            devices = 1
            print("Using MPS (Metal Performance Shaders) accelerator")
        elif torch.cuda.is_available():
            accelerator = 'gpu'
            devices = num_devices
            print(f"Using CUDA with {devices} GPU(s)")
        else:
            print(f"Warning: Requested {num_devices} GPU(s) but no GPU available. Using CPU instead.")
            accelerator = 'cpu'
            devices = 'auto'
    
    train_dataset = FootballPlayDataset(cache_file, metadata_file, split='train')
    val_dataset = FootballPlayDataset(cache_file, metadata_file, split='val')
    
    # Determine if we should use pin_memory (not supported on MPS)
    use_pin_memory = accelerator != 'mps'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Reduced to 0 for MPS compatibility
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Reduced to 0 for MPS compatibility
        pin_memory=use_pin_memory
    )
    
    # Get dimensions from metadata
    metadata = train_dataset.metadata
    num_players = metadata.get('players', 22)
    num_features = len(metadata.get('features', ['x', 'y', 's']))
    
    # Create model
    model = DiffusionLightningModule(config, num_players, num_features)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='{epoch}-{val_loss:.6f}',  # More precision to see actual loss value
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop = EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min'
    )
    
    # Setup logger (optional - will use CSV logger if TensorBoard not available)
    try:
        logger = TensorBoardLogger(output_dir, name='logs')
    except (ModuleNotFoundError, ImportError):
        print("Warning: TensorBoard not available. Using CSV logger instead.")
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(output_dir, name='logs')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop],
        check_val_every_n_epoch=1,
        val_check_interval=config['train'].get('val_check_interval', 0.5),
        accumulate_grad_batches=config['train'].get('accumulate_grad_batches', 1),
        gradient_clip_val=config['train'].get('grad_clip', 1.0)
    )
    
    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume
    )
    
    print(f"\nTraining complete! Best model saved to {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()

