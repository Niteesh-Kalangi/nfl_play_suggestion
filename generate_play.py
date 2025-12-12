"""
Generate and visualize plays using trained autoregressive models.

Usage:
    python generate_play.py --model lstm --checkpoint artifacts/autoregressive/lstm.pt
    python generate_play.py --model transformer --checkpoint artifacts/autoregressive/transformer.pt
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from src.autoregressive import (
    LSTMTrajectoryGenerator,
    TransformerTrajectoryGenerator,
    generate_play_with_context
)
from src.autoregressive.viz import draw_field, plot_trajectory, animate_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description='Generate plays with autoregressive models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'transformer'],
        required=True,
        help='Model type'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
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
        help='Device to use (cuda/cpu/mps)'
    )
    parser.add_argument(
        '--down',
        type=int,
        default=1,
        help='Down (1-4)'
    )
    parser.add_argument(
        '--yards_to_go',
        type=float,
        default=10.0,
        help='Yards to go'
    )
    parser.add_argument(
        '--formation',
        type=str,
        default='SHOTGUN',
        help='Offensive formation'
    )
    parser.add_argument(
        '--personnel',
        type=str,
        default='1 RB, 1 TE, 3 WR',
        help='Personnel (e.g., "1 RB, 1 TE, 3 WR")'
    )
    parser.add_argument(
        '--def_team',
        type=str,
        default='BEN',
        help='Defensive team abbreviation'
    )
    parser.add_argument(
        '--yardline',
        type=int,
        default=50,
        help='Yardline (0-100)'
    )
    parser.add_argument(
        '--hash_mark',
        type=str,
        default='MIDDLE',
        choices=['LEFT', 'MIDDLE', 'RIGHT'],
        help='Hash mark position'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=60,
        help='Number of timesteps to generate'
    )
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Create animated visualization'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save animation to file (e.g., play.mp4)'
    )
    return parser.parse_args()


def get_device(device_str=None):
    """Get the best available device."""
    if device_str:
        return torch.device(device_str)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_model(model_type, checkpoint_path, config, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters
    num_players = checkpoint.get('num_players', 22)
    num_features = checkpoint.get('num_features', 3)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    
    # Get config
    ar_config = config.get('autoregressive', {})
    context_dim = 256
    
    # Build model
    if model_type == 'lstm':
        lstm_config = ar_config.get('lstm', {})
        model = LSTMTrajectoryGenerator(
            num_players=num_players,
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=lstm_config.get('num_layers', 2),
            dropout=lstm_config.get('dropout', 0.1),
            context_dim=context_dim
        )
    elif model_type == 'transformer':
        tf_config = ar_config.get('transformer', {})
        model = TransformerTrajectoryGenerator(
            num_players=num_players,
            num_features=num_features,
            d_model=hidden_dim,
            nhead=tf_config.get('nhead', 8),
            num_layers=tf_config.get('num_layers', 4),
            dim_feedforward=tf_config.get('dim_feedforward', 512),
            dropout=tf_config.get('dropout', 0.1),
            context_dim=context_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.model} model from {args.checkpoint}...")
    model = load_model(args.model, args.checkpoint, config, device)
    print("✓ Model loaded")
    
    # Generate play
    print(f"\n🎯 Generating play with context:")
    print(f"   Down: {args.down} & {args.yards_to_go}")
    print(f"   Formation: {args.formation}")
    print(f"   Personnel: {args.personnel}")
    print(f"   Defensive Team: {args.def_team}")
    print(f"   Yardline: {args.yardline}")
    print(f"   Hash Mark: {args.hash_mark}")
    
    trajectory = generate_play_with_context(
        model=model,
        down=args.down,
        yards_to_go=args.yards_to_go,
        offensive_formation=args.formation,
        personnel_o=args.personnel,
        def_team=args.def_team,
        yardline=args.yardline,
        hash_mark=args.hash_mark,
        horizon=args.horizon,
        device=device
    )
    
    print(f"✅ Play generated! Shape: {trajectory.shape}")
    
    # Visualize
    if args.animate:
        print("\n🎬 Creating animated visualization...")
        anim, fig = animate_trajectory(
            trajectory,
            animate_skill_only=True,
            show_trails=True
        )
        
        if args.save:
            print(f"💾 Saving animation to {args.save}...")
            anim.save(args.save, writer='ffmpeg', fps=10, bitrate=1800)
            print(f"✅ Animation saved!")
        else:
            plt.show()
    else:
        print("\n📊 Creating static visualization...")
        fig, ax = plt.subplots(figsize=(14, 7))
        draw_field(ax)
        plot_trajectory(trajectory, ax=ax, highlight_skill_only=True)
        ax.set_title(
            f'Generated Play - {args.formation} | {args.personnel} | Down {args.down} & {args.yards_to_go}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        
        if args.save:
            plt.savefig(args.save)
            print(f"✅ Visualization saved to {args.save}")
        else:
            plt.show()
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

