"""
Run trajectory visualization with custom inputs.

Usage:
    python run_visualization.py                    # Use defaults
    python run_visualization.py --interactive      # Interactive mode
    python run_visualization.py --from-model PATH  # Load trained model
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from src.visualize_trajectory import (
    animate_trajectory, 
    plot_trajectory_static,
    visualize_generated_play
)
from src.models.transformer_generator import (
    TransformerTrajectoryGenerator,
    TransformerTrajectoryGeneratorConfig,
    create_transformer_generator
)


# =============================================================================
# STEP 1: DEFINE YOUR INPUTS HERE
# =============================================================================

# Game context (6 features)
CONTEXT = {
    'down': 2,              # 1, 2, 3, or 4
    'yardsToGo': 7,         # Yards needed for first down
    'yardline_100': 35,     # Distance from opponent's end zone (0-100)
    'clock_seconds': 420,   # Seconds remaining in quarter (0-900)
    'score_diff': 3,        # Your score minus opponent's score
    'quarter': 2,           # 1, 2, 3, or 4
}

# Initial player positions (x, y) in yards
# x: 0-120 (0=your end zone, 60=midfield, 120=opponent end zone)
# y: 0-53.3 (sideline to sideline)
INITIAL_POSITIONS = {
    'QB':  (25, 26.65),     # Quarterback - behind center
    'RB':  (22, 26.65),     # Running back - behind QB
    'WR1': (25, 48),        # Wide receiver 1 - top of field
    'WR2': (25, 5),         # Wide receiver 2 - bottom of field  
    'WR3': (25, 38),        # Slot receiver - between WR1 and line
    'TE':  (23, 32),        # Tight end - on the line
    'C':   (20, 26.65),     # Center
    'LG':  (20, 29),        # Left guard
    'RG':  (20, 24),        # Right guard
    'LT':  (20, 32),        # Left tackle
    'RT':  (20, 21),        # Right tackle
}

# Generation settings
N_FRAMES = 30              # Number of frames to generate
OUTPUT_FILE = 'my_play.gif'  # Output filename (.gif or .mp4)


# =============================================================================
# PRESET FORMATIONS (optional - use these as starting points)
# =============================================================================

FORMATIONS = {
    'shotgun': {
        'QB':  (22, 26.65),
        'RB':  (22, 30),
        'WR1': (25, 50),
        'WR2': (25, 3),
        'WR3': (25, 40),
        'TE':  (23, 33),
        'C':   (25, 26.65),
        'LG':  (25, 29),
        'RG':  (25, 24),
        'LT':  (25, 32),
        'RT':  (25, 21),
    },
    'i_formation': {
        'QB':  (25, 26.65),
        'RB':  (20, 26.65),  # Fullback position
        'WR1': (25, 50),
        'WR2': (25, 3),
        'WR3': (17, 26.65),  # Tailback behind FB
        'TE':  (25, 33),
        'C':   (25, 26.65),
        'LG':  (25, 29),
        'RG':  (25, 24),
        'LT':  (25, 32),
        'RT':  (25, 21),
    },
    'spread': {
        'QB':  (20, 26.65),
        'RB':  (18, 26.65),
        'WR1': (25, 52),
        'WR2': (25, 1),
        'WR3': (25, 40),
        'TE':  (25, 13),     # TE as slot receiver
        'C':   (25, 26.65),
        'LG':  (25, 29),
        'RG':  (25, 24),
        'LT':  (25, 32),
        'RT':  (25, 21),
    },
}


def positions_to_tensor(positions: dict) -> torch.Tensor:
    """Convert position dict to flat tensor."""
    order = ['QB', 'RB', 'WR1', 'WR2', 'WR3', 'TE', 'C', 'LG', 'RG', 'LT', 'RT']
    flat = []
    for pos in order:
        x, y = positions[pos]
        flat.extend([x, y])
    return torch.tensor(flat, dtype=torch.float32)


def context_to_tensor(context: dict) -> torch.Tensor:
    """Convert context dict to tensor."""
    order = ['down', 'yardsToGo', 'yardline_100', 'clock_seconds', 'score_diff', 'quarter']
    return torch.tensor([context[k] for k in order], dtype=torch.float32)


def normalize_inputs(positions: torch.Tensor, context: torch.Tensor):
    """Normalize inputs (matching training normalization)."""
    # Position normalization (field center and scale)
    traj_mean = torch.tensor([60.0, 26.65] * 11)
    traj_std = torch.tensor([30.0, 15.0] * 11)
    positions_norm = (positions - traj_mean) / traj_std
    
    # Context normalization (approximate - ideally use training stats)
    ctx_mean = torch.tensor([2.5, 8.0, 50.0, 450.0, 0.0, 2.5])
    ctx_std = torch.tensor([1.0, 5.0, 25.0, 250.0, 14.0, 1.0])
    context_norm = (context - ctx_mean) / ctx_std
    
    return positions_norm, context_norm


def denormalize_trajectory(trajectory: np.ndarray, norm_stats=None) -> np.ndarray:
    """Denormalize generated trajectory back to field coordinates."""
    if norm_stats is not None:
        traj_mean = norm_stats['traj_mean']
        traj_std = norm_stats['traj_std']
    else:
        traj_mean = np.array([60.0, 26.65])
        traj_std = np.array([30.0, 15.0])
    return trajectory * traj_std + traj_mean


def run_with_model(model, positions, context, n_frames, output_file, norm_stats=None):
    """Generate trajectory using the model and visualize."""
    # Normalize inputs using saved stats if available
    if norm_stats is not None:
        traj_mean = torch.tensor(np.tile(norm_stats['traj_mean'], 11).flatten(), dtype=torch.float32)
        traj_std = torch.tensor(np.tile(norm_stats['traj_std'], 11).flatten(), dtype=torch.float32)
        ctx_mean = torch.tensor(norm_stats['ctx_mean'], dtype=torch.float32)
        ctx_std = torch.tensor(norm_stats['ctx_std'], dtype=torch.float32)
        
        pos_norm = (positions - traj_mean) / traj_std
        ctx_norm = (context - ctx_mean) / ctx_std
    else:
        pos_norm, ctx_norm = normalize_inputs(positions, context)
    
    # Add batch dimension
    pos_norm = pos_norm.unsqueeze(0)
    ctx_norm = ctx_norm.unsqueeze(0)
    
    # Generate
    model.eval()
    with torch.no_grad():
        generated = model.generate(pos_norm, ctx_norm, n_frames - 1)
    
    # Convert to numpy and reshape
    trajectory = generated.squeeze(0).numpy()  # (n_frames, 22)
    trajectory = trajectory.reshape(n_frames, 11, 2)
    
    # Denormalize
    trajectory = denormalize_trajectory(trajectory, norm_stats)
    
    # Animate
    print(f"\nGenerating animation with {n_frames} frames...")
    anim = animate_trajectory(
        trajectory,
        title="Model Generated Play",
        save_path=output_file,
        interval=100
    )
    print(f"✓ Animation saved to: {output_file}")
    
    # Also save static view
    static_file = output_file.replace('.gif', '_static.png').replace('.mp4', '_static.png')
    fig = plot_trajectory_static(trajectory, title="Model Generated Play - Static")
    fig.savefig(static_file, dpi=150, bbox_inches='tight')
    print(f"✓ Static view saved to: {static_file}")
    
    return trajectory


def run_demo(positions, n_frames, output_file):
    """Run demo with synthetic movement (no trained model needed)."""
    # Convert positions to array
    order = ['QB', 'RB', 'WR1', 'WR2', 'WR3', 'TE', 'C', 'LG', 'RG', 'LT', 'RT']
    start_pos = np.array([positions[p] for p in order])
    
    # Generate synthetic trajectory
    trajectory = np.zeros((n_frames, 11, 2))
    trajectory[0] = start_pos
    
    # Simple movement patterns
    for frame in range(1, n_frames):
        trajectory[frame] = trajectory[frame - 1].copy()
        
        # QB drops back slightly
        trajectory[frame, 0, 0] += np.random.normal(-0.2, 0.05)
        
        # RB moves forward
        trajectory[frame, 1, 0] += np.random.normal(0.4, 0.1)
        trajectory[frame, 1, 1] += np.random.normal(0.1, 0.1)
        
        # WRs run routes
        trajectory[frame, 2, 0] += np.random.normal(0.7, 0.1)  # WR1 go
        trajectory[frame, 3, 0] += np.random.normal(0.5, 0.1)  # WR2 slant
        trajectory[frame, 3, 1] += np.random.normal(0.3, 0.1)
        trajectory[frame, 4, 0] += np.random.normal(0.4, 0.1)  # WR3 out
        trajectory[frame, 4, 1] += np.random.normal(0.3, 0.1)
        
        # TE runs seam
        trajectory[frame, 5, 0] += np.random.normal(0.3, 0.1)
        
        # OL holds
        for i in range(6, 11):
            trajectory[frame, i, 0] += np.random.normal(0.02, 0.02)
    
    # Animate
    print(f"\nGenerating demo animation with {n_frames} frames...")
    anim = animate_trajectory(
        trajectory,
        title="Demo Play (Synthetic Movement)",
        save_path=output_file,
        interval=100
    )
    print(f"✓ Animation saved to: {output_file}")
    
    # Static view
    static_file = output_file.replace('.gif', '_static.png').replace('.mp4', '_static.png')
    fig = plot_trajectory_static(trajectory, title="Demo Play - Static View")
    fig.savefig(static_file, dpi=150, bbox_inches='tight')
    print(f"✓ Static view saved to: {static_file}")
    
    return trajectory


def interactive_mode():
    """Interactive CLI for setting inputs."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE - Set your play parameters")
    print("="*60)
    
    # Formation selection
    print("\nAvailable formations:")
    print("  1. Custom (use INITIAL_POSITIONS in script)")
    print("  2. Shotgun")
    print("  3. I-Formation")
    print("  4. Spread")
    
    choice = input("\nSelect formation [1-4, default=1]: ").strip() or "1"
    
    if choice == "2":
        positions = FORMATIONS['shotgun']
    elif choice == "3":
        positions = FORMATIONS['i_formation']
    elif choice == "4":
        positions = FORMATIONS['spread']
    else:
        positions = INITIAL_POSITIONS
    
    # Context
    print("\nGame Context (press Enter for defaults):")
    
    down = input(f"  Down [1-4, default={CONTEXT['down']}]: ").strip()
    down = int(down) if down else CONTEXT['down']
    
    yards = input(f"  Yards to go [default={CONTEXT['yardsToGo']}]: ").strip()
    yards = int(yards) if yards else CONTEXT['yardsToGo']
    
    yardline = input(f"  Yardline (0-100) [default={CONTEXT['yardline_100']}]: ").strip()
    yardline = int(yardline) if yardline else CONTEXT['yardline_100']
    
    context = {
        'down': down,
        'yardsToGo': yards,
        'yardline_100': yardline,
        'clock_seconds': CONTEXT['clock_seconds'],
        'score_diff': CONTEXT['score_diff'],
        'quarter': CONTEXT['quarter'],
    }
    
    # Frames
    frames = input(f"\nNumber of frames [default={N_FRAMES}]: ").strip()
    n_frames = int(frames) if frames else N_FRAMES
    
    # Output
    output = input(f"Output file [default={OUTPUT_FILE}]: ").strip() or OUTPUT_FILE
    
    return positions, context, n_frames, output


def main():
    parser = argparse.ArgumentParser(description='Run NFL play trajectory visualization')
    parser.add_argument('--interactive', '-i', action='store_true', 
                        help='Interactive mode for setting inputs')
    parser.add_argument('--from-model', type=str, 
                        default='artifacts/transformer_trajectory_generator.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--formation', type=str, choices=['shotgun', 'i_formation', 'spread'],
                        help='Use a preset formation')
    parser.add_argument('--frames', type=int, default=N_FRAMES,
                        help='Number of frames to generate')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_FILE,
                        help='Output file path')
    
    args = parser.parse_args()
    
    # Get inputs
    if args.interactive:
        positions, context, n_frames, output_file = interactive_mode()
    else:
        positions = FORMATIONS.get(args.formation, INITIAL_POSITIONS)
        context = CONTEXT
        n_frames = args.frames
        output_file = args.output
    
    print("\n" + "="*60)
    print("PLAY CONFIGURATION")
    print("="*60)
    print(f"\nContext: Down {context['down']} & {context['yardsToGo']} at the {context['yardline_100']}")
    print(f"Frames: {n_frames}")
    print(f"Output: {output_file}")
    
    # Run visualization
    if args.from_model and Path(args.from_model).exists():
        print(f"\nLoading model from: {args.from_model}")
        checkpoint = torch.load(args.from_model, map_location='cpu', weights_only=False)
        
        # Create model from saved config
        saved_config = checkpoint['config']
        config = TransformerTrajectoryGeneratorConfig(**saved_config)
        model = create_transformer_generator(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get normalization stats from checkpoint
        norm_stats = checkpoint.get('norm_stats', None)
        
        pos_tensor = positions_to_tensor(positions)
        ctx_tensor = context_to_tensor(context)
        
        trajectory = run_with_model(model, pos_tensor, ctx_tensor, n_frames, output_file, norm_stats)
    else:
        print("\nNo model found - running demo with synthetic movement")
        print(f"(Looked for: {args.from_model})")
        trajectory = run_demo(positions, n_frames, output_file)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
