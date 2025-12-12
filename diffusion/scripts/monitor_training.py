#!/usr/bin/env python3
"""
Monitor training progress in real-time.
Usage: python scripts/monitor_training.py [--log-dir artifacts/diffusion/logs]
"""

import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

def find_latest_log_dir(log_base_dir: Path) -> Optional[Path]:
    """Find the most recent version directory."""
    if not log_base_dir.exists():
        return None
    
    versions = sorted([d for d in log_base_dir.iterdir() if d.is_dir() and d.name.startswith('version_')],
                     key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else -1,
                     reverse=True)
    return versions[0] if versions else None

def read_tensorboard_events(log_dir: Path) -> Optional[Dict]:
    """Read metrics from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Find event file
        event_files = list(log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            return None
        
        # Use the most recent event file
        event_file = sorted(event_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
        # Load events
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        # Extract scalar metrics
        metrics = {}
        scalar_tags = ea.Tags()['scalars']
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            if scalar_events:
                # Get the latest value
                latest = scalar_events[-1]
                metrics[tag] = {
                    'value': latest.value,
                    'step': latest.step,
                    'wall_time': latest.wall_time
                }
        
        return metrics if metrics else None
        
    except ImportError:
        # TensorBoard not installed - return None to trigger fallback
        return None
    except Exception as e:
        return None

def check_file_activity(log_dir: Path) -> Dict:
    """Check if TensorBoard event files are being updated (fallback when tensorboard not installed)."""
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}
    
    latest_file = sorted(event_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    stat = latest_file.stat()
    
    return {
        'file': latest_file.name,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S'),
        'age_seconds': time.time() - stat.st_mtime
    }

def read_training_metrics(log_dir: Path) -> Optional[pd.DataFrame]:
    """Read metrics from CSV logger."""
    csv_file = log_dir / "metrics.csv"
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            pass
    
    # Fallback to TensorBoard events
    metrics_dict = read_tensorboard_events(log_dir)
    if metrics_dict:
        # Convert to DataFrame-like structure
        data = {'step': [], 'epoch': []}
        for tag, values in metrics_dict.items():
            metric_name = tag.replace('/', '_')
            data[metric_name] = [values['value']]
            if 'step' not in data or len(data['step']) == 0:
                data['step'] = [values['step']]
        
        # Create a simple DataFrame-like object
        class MetricsDict:
            def __init__(self, data):
                self.data = data
                self.metrics = metrics_dict
            
            def __len__(self):
                return 1
            
            @property
            def columns(self):
                return list(self.data.keys())
        
        return MetricsDict(data)
    
    return None

def print_training_status(df, last_n: int = 10):
    """Print the latest training metrics."""
    if df is None:
        print("No metrics available yet.")
        return
    
    # Handle TensorBoard metrics dict
    if hasattr(df, 'metrics'):
        metrics = df.metrics
        print("\n📊 Training Metrics (TensorBoard):")
        print("-" * 70)
        
        # Map TensorBoard tags to friendly names
        tag_map = {
            'train/loss': '📈 Training Loss',
            'val/loss': '✅ Validation Loss',
            'train/noise_loss': '🔊 Noise Loss',
            'train/velocity_loss': '⚡ Velocity Loss',
            'train/boundary_loss': '🛡️  Boundary Loss',
            'epoch': '🔄 Epoch'
        }
        
        for tag, values in sorted(metrics.items()):
            friendly_name = tag_map.get(tag, tag.replace('/', ' ').title())
            print(f"{friendly_name:25} {values['value']:.6f} (step: {values['step']})")
        
        print("-" * 70)
        return
    
    # Handle CSV DataFrame
    if len(df) == 0:
        print("No metrics available yet.")
        return
    
    # Get latest metrics
    latest = df.tail(last_n)
    
    # Extract key metrics
    if 'train_loss_epoch' in latest.columns:
        train_loss = latest['train_loss_epoch'].dropna()
        if len(train_loss) > 0:
            print(f"\n📊 Training Loss (epoch): {train_loss.iloc[-1]:.6f} (avg last 5: {train_loss.tail(5).mean():.6f})")
    
    if 'train_loss_step' in latest.columns:
        train_loss_step = latest['train_loss_step'].dropna()
        if len(train_loss_step) > 0:
            print(f"📈 Training Loss (step): {train_loss_step.iloc[-1]:.6f}")
    
    if 'val_loss' in latest.columns:
        val_loss = latest['val_loss'].dropna()
        if len(val_loss) > 0:
            print(f"✅ Validation Loss: {val_loss.iloc[-1]:.6f} (best: {val_loss.min():.6f})")
    
    if 'epoch' in latest.columns:
        epoch = latest['epoch'].dropna()
        if len(epoch) > 0:
            print(f"🔄 Current Epoch: {int(epoch.iloc[-1])}")
    
    if 'train_noise_loss_epoch' in latest.columns:
        noise_loss = latest['train_noise_loss_epoch'].dropna()
        if len(noise_loss) > 0:
            print(f"🔊 Noise Loss: {noise_loss.iloc[-1]:.6f}")
    
    if 'train_velocity_loss_epoch' in latest.columns:
        vel_loss = latest['train_velocity_loss_epoch'].dropna()
        if len(vel_loss) > 0:
            print(f"⚡ Velocity Loss: {vel_loss.iloc[-1]:.6f}")
    
    print("-" * 70)

def monitor_loop(log_dir: Path, refresh_interval: int = 5):
    """Continuously monitor training metrics."""
    print("=" * 70)
    print("🔍 Training Monitor - Press Ctrl+C to exit")
    print("=" * 70)
    print(f"📁 Watching: {log_dir}")
    print(f"⏱️  Refresh interval: {refresh_interval} seconds")
    print("-" * 70)
    
    last_epoch = -1
    last_rows = 0
    
    try:
        while True:
            # Find latest log directory if it changed
            log_base = log_dir.parent if log_dir.name.startswith('version_') else log_dir
            latest_dir = find_latest_log_dir(log_base) if not log_dir.name.startswith('version_') else log_dir
            
            if latest_dir is None:
                print(f"⏳ Waiting for logs to appear... (checking {log_base})")
                time.sleep(refresh_interval)
                continue
            
            df = read_training_metrics(latest_dir)
            
            if df is not None:
                # Handle TensorBoard metrics
                if hasattr(df, 'metrics'):
                    # Check if metrics updated (compare by step)
                    current_step = 0
                    if 'train/loss' in df.metrics:
                        current_step = df.metrics['train/loss']['step']
                    
                    if current_step > last_rows:
                        print(f"\n{'='*70}")
                        print(f"🆕 New metrics detected! (step: {last_rows} → {current_step})")
                        print_training_status(df, last_n=5)
                        last_rows = current_step
                    else:
                        # Show current metrics
                        if last_rows == 0:
                            print_training_status(df, last_n=5)
                            last_rows = current_step
                        else:
                            print(f"⏳ Waiting for new metrics... (current step: {current_step})", end='\r')
                else:
                    # Handle CSV DataFrame
                    current_rows = len(df)
                    if current_rows > last_rows:
                        print(f"\n{'='*70}")
                        print(f"🆕 New metrics detected! ({last_rows} → {current_rows} rows)")
                        print_training_status(df, last_n=5)
                        last_rows = current_rows
                        
                        # Check for epoch completion
                        if 'epoch' in df.columns:
                            current_epoch = df['epoch'].dropna().iloc[-1] if len(df['epoch'].dropna()) > 0 else -1
                            if int(current_epoch) > last_epoch:
                                print(f"\n🎉 Epoch {int(current_epoch)} completed!")
                                last_epoch = int(current_epoch)
                    else:
                        # Still waiting for new data
                        print(f"⏳ Waiting for new metrics... (last seen: {current_rows} rows)", end='\r')
            else:
                # Check if TensorBoard events exist but couldn't be read
                event_files = list(latest_dir.glob("events.out.tfevents.*"))
                if event_files:
                    # Use file activity as fallback
                    activity = check_file_activity(latest_dir)
                    if activity:
                        age = activity['age_seconds']
                        if age < refresh_interval * 2:
                            # File was recently modified
                            print(f"✅ Training active! Log file: {activity['file']} ({activity['size_mb']:.1f} MB, updated {activity['modified']})", end='\r')
                        else:
                            print(f"⚠️  Log file exists but not updating ({age:.0f}s ago). Training may be paused. Install 'tensorboard' for detailed metrics.", end='\r')
                    else:
                        print("⏳ TensorBoard events found. Install 'pip install tensorboard' for detailed metrics.", end='\r')
                else:
                    print("⏳ Waiting for logs... (no CSV or TensorBoard events found)", end='\r')
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument(
        '--log-dir',
        type=str,
        default='artifacts/diffusion/logs',
        help='Path to logs directory (default: artifacts/diffusion/logs)'
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=5,
        help='Refresh interval in seconds (default: 5)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Print metrics once and exit (no continuous monitoring)'
    )
    
    args = parser.parse_args()
    
    # Resolve log directory
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / args.log_dir
    
    # Find latest version if not pointing to specific version
    if not log_dir.name.startswith('version_'):
        latest_dir = find_latest_log_dir(log_dir)
        if latest_dir:
            log_dir = latest_dir
            print(f"📁 Using latest log directory: {log_dir}")
    
    if args.once:
        # Print once and exit
        df = read_training_metrics(log_dir)
        if df is None:
            # Try fallback: check file activity
            activity = check_file_activity(log_dir)
            if activity:
                print("\n📊 Training Status (File Activity):")
                print("-" * 70)
                print(f"📁 Log file: {activity['file']}")
                print(f"📦 Size: {activity['size_mb']:.2f} MB")
                print(f"🕒 Last updated: {activity['modified']}")
                age_min = activity['age_seconds'] / 60
                if age_min < 1:
                    print(f"✅ Training appears active (updated {activity['age_seconds']:.0f}s ago)")
                else:
                    print(f"⚠️  File not updated recently ({age_min:.1f} min ago)")
                print("\n💡 Tip: Install 'tensorboard' for detailed metrics:")
                print("   pip install tensorboard")
                print("   tensorboard --logdir artifacts/diffusion/logs")
                print("-" * 70)
            else:
                print_training_status(df)
        else:
            print_training_status(df)
    else:
        # Continuous monitoring
        monitor_loop(log_dir, args.refresh)

if __name__ == '__main__':
    main()

