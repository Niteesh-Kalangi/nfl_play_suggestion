"""
Visualization utilities for NFL play trajectory animations.
Animates player positions on a football field based on model-generated trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from typing import Optional, List, Dict, Tuple
import torch


# Player position colors and labels
POSITION_COLORS = {
    0: '#3498db',   # QB - Blue
    1: '#e74c3c',   # RB - Red  
    2: '#2ecc71',   # WR1 - Green
    3: '#2ecc71',   # WR2 - Green
    4: '#2ecc71',   # WR3 - Green
    5: '#f39c12',   # TE - Orange
    6: '#9b59b6',   # OL - Purple
    7: '#9b59b6',   # OL - Purple
    8: '#9b59b6',   # OL - Purple
    9: '#9b59b6',   # OL - Purple
    10: '#9b59b6',  # OL - Purple
}

POSITION_LABELS = {
    0: 'QB',
    1: 'RB',
    2: 'WR',
    3: 'WR',
    4: 'WR',
    5: 'TE',
    6: 'OL',
    7: 'OL',
    8: 'OL',
    9: 'OL',
    10: 'OL',
}


def draw_football_field(ax, field_color='#567d46', line_color='white', 
                         show_full_field=True, yard_numbers=True):
    """
    Draw a football field on the given axes.
    
    Args:
        ax: Matplotlib axes
        field_color: Background color of the field
        line_color: Color of yard lines
        show_full_field: If True, show full 120 yards (including end zones)
        yard_numbers: If True, show yard numbers
    """
    # Field dimensions (in yards)
    field_length = 120 if show_full_field else 100
    field_width = 53.3
    
    # Set background
    ax.set_facecolor(field_color)
    
    # Draw end zones
    if show_full_field:
        # Left end zone
        endzone_left = patches.Rectangle((0, 0), 10, field_width, 
                                          facecolor='#4a6b3a', edgecolor=line_color, linewidth=2)
        ax.add_patch(endzone_left)
        
        # Right end zone
        endzone_right = patches.Rectangle((110, 0), 10, field_width,
                                           facecolor='#4a6b3a', edgecolor=line_color, linewidth=2)
        ax.add_patch(endzone_right)
    
    # Draw yard lines every 5 yards
    start_yard = 10 if show_full_field else 0
    end_yard = 110 if show_full_field else 100
    
    for yard in range(start_yard, end_yard + 1, 5):
        linewidth = 2 if yard % 10 == 0 else 1
        ax.axvline(x=yard, color=line_color, linewidth=linewidth, alpha=0.8)
    
    # Draw hash marks
    hash_width = 0.5
    for yard in range(start_yard, end_yard + 1, 1):
        if yard % 5 != 0:
            # Top hash
            ax.plot([yard, yard], [field_width - 2, field_width - 2 + hash_width], 
                   color=line_color, linewidth=1, alpha=0.6)
            # Bottom hash
            ax.plot([yard, yard], [2, 2 - hash_width], 
                   color=line_color, linewidth=1, alpha=0.6)
    
    # Draw sidelines
    ax.axhline(y=0, color=line_color, linewidth=3)
    ax.axhline(y=field_width, color=line_color, linewidth=3)
    
    # Draw goal lines
    if show_full_field:
        ax.axvline(x=10, color=line_color, linewidth=3)
        ax.axvline(x=110, color=line_color, linewidth=3)
    
    # Add yard numbers
    if yard_numbers:
        for yard in range(10, 100, 10):
            display_yard = yard if yard <= 50 else 100 - yard
            x_pos = yard + 10 if show_full_field else yard
            
            # Top numbers
            ax.text(x_pos, field_width - 5, str(display_yard), 
                   fontsize=14, color=line_color, ha='center', va='center',
                   fontweight='bold', alpha=0.7)
            # Bottom numbers
            ax.text(x_pos, 5, str(display_yard), 
                   fontsize=14, color=line_color, ha='center', va='center',
                   fontweight='bold', alpha=0.7)
    
    # Set axis limits and aspect
    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_aspect('equal')
    ax.set_xlabel('Yards', fontsize=12)
    ax.set_ylabel('Field Width (yards)', fontsize=12)


def plot_trajectory_static(
    trajectory: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Play Trajectory",
    show_paths: bool = True,
    show_start: bool = True,
    show_end: bool = True,
    alpha_decay: bool = True
) -> plt.Figure:
    """
    Plot a static view of player trajectories.
    
    Args:
        trajectory: Shape (n_frames, n_players, 2) - x, y positions
        ax: Matplotlib axes (creates new figure if None)
        title: Plot title
        show_paths: Draw trajectory paths
        show_start: Mark starting positions
        show_end: Mark ending positions
        alpha_decay: Fade path from start to end
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.figure
    
    draw_football_field(ax)
    
    n_frames, n_players, _ = trajectory.shape
    
    for player_idx in range(n_players):
        color = POSITION_COLORS.get(player_idx, '#888888')
        label = POSITION_LABELS.get(player_idx, f'P{player_idx}')
        
        x = trajectory[:, player_idx, 0]
        y = trajectory[:, player_idx, 1]
        
        if show_paths:
            if alpha_decay:
                # Create line segments with varying alpha
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                alphas = np.linspace(0.3, 1.0, len(segments))
                
                for i, (seg, alpha) in enumerate(zip(segments, alphas)):
                    ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=2, alpha=alpha)
            else:
                ax.plot(x, y, color=color, linewidth=2, alpha=0.7)
        
        if show_start:
            ax.scatter(x[0], y[0], c=color, s=100, marker='o', 
                      edgecolors='white', linewidths=2, zorder=5)
            ax.annotate(label, (x[0], y[0]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, fontweight='bold')
        
        if show_end:
            ax.scatter(x[-1], y[-1], c=color, s=150, marker='o',
                      edgecolors='black', linewidths=2, zorder=6)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig


def animate_trajectory(
    trajectory: np.ndarray,
    interval: int = 100,
    title: str = "Play Animation",
    save_path: Optional[str] = None,
    show_trails: bool = True,
    trail_length: int = 10,
    figsize: Tuple[int, int] = (14, 6)
) -> animation.FuncAnimation:
    """
    Create an animated visualization of player trajectories.
    
    Args:
        trajectory: Shape (n_frames, n_players, 2) - x, y positions
        interval: Milliseconds between frames
        title: Animation title
        save_path: If provided, save animation to this path (.gif or .mp4)
        show_trails: Show trajectory trails behind players
        trail_length: Number of frames to show in trail
        figsize: Figure size
        
    Returns:
        Matplotlib animation object
    """
    n_frames, n_players, _ = trajectory.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    draw_football_field(ax)
    
    # Initialize player dots
    players = []
    labels = []
    trails = []
    
    for player_idx in range(n_players):
        color = POSITION_COLORS.get(player_idx, '#888888')
        label_text = POSITION_LABELS.get(player_idx, f'P{player_idx}')
        
        # Player marker
        player_dot, = ax.plot([], [], 'o', color=color, markersize=12,
                              markeredgecolor='white', markeredgewidth=2, zorder=10)
        players.append(player_dot)
        
        # Player label
        label = ax.text(0, 0, label_text, fontsize=8, fontweight='bold',
                       ha='center', va='bottom', zorder=11)
        labels.append(label)
        
        # Trail line
        if show_trails:
            trail, = ax.plot([], [], '-', color=color, linewidth=2, alpha=0.5, zorder=5)
            trails.append(trail)
    
    # Frame counter text
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    title_text = ax.set_title(title, fontsize=14, fontweight='bold')
    
    def init():
        for player_dot in players:
            player_dot.set_data([], [])
        for label in labels:
            label.set_position((0, 0))
        if show_trails:
            for trail in trails:
                trail.set_data([], [])
        frame_text.set_text('')
        return players + labels + trails + [frame_text]
    
    def animate(frame):
        for player_idx in range(n_players):
            x = trajectory[frame, player_idx, 0]
            y = trajectory[frame, player_idx, 1]
            
            # Update player position
            players[player_idx].set_data([x], [y])
            
            # Update label position
            labels[player_idx].set_position((x, y + 1.5))
            
            # Update trail
            if show_trails:
                start_frame = max(0, frame - trail_length)
                trail_x = trajectory[start_frame:frame+1, player_idx, 0]
                trail_y = trajectory[start_frame:frame+1, player_idx, 1]
                trails[player_idx].set_data(trail_x, trail_y)
        
        frame_text.set_text(f'Frame: {frame + 1}/{n_frames}')
        
        return players + labels + trails + [frame_text]
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=interval, blit=True
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        else:
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        print(f"Animation saved to {save_path}")
    
    return anim


def visualize_generated_play(
    model,
    initial_frame: torch.Tensor,
    context: torch.Tensor,
    n_steps: int = 30,
    stats: Optional[Dict] = None,
    title: str = "Generated Play",
    animate: bool = True,
    save_path: Optional[str] = None
):
    """
    Generate and visualize a play from the transformer model.
    
    Args:
        model: TransformerTrajectoryGenerator model
        initial_frame: (1, n_players * 2) or (n_players * 2,) - starting positions
        context: (1, context_dim) or (context_dim,) - game context
        n_steps: Number of frames to generate
        stats: Normalization stats for denormalization
        title: Visualization title
        animate: If True, create animation; else static plot
        save_path: Path to save visualization
        
    Returns:
        Animation or Figure object
    """
    model.eval()
    
    # Ensure correct shape
    if initial_frame.dim() == 1:
        initial_frame = initial_frame.unsqueeze(0)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    
    # Generate trajectory
    with torch.no_grad():
        generated = model.generate(initial_frame, context, n_steps)
    
    # Convert to numpy
    trajectory = generated.squeeze(0).cpu().numpy()  # (n_steps+1, n_players * 2)
    
    # Reshape to (n_frames, n_players, 2)
    n_frames = trajectory.shape[0]
    n_players = trajectory.shape[1] // 2
    trajectory = trajectory.reshape(n_frames, n_players, 2)
    
    # Denormalize if stats provided
    if stats is not None:
        trajectory = trajectory * stats['traj_std'] + stats['traj_mean']
    
    if animate:
        return animate_trajectory(trajectory, title=title, save_path=save_path)
    else:
        fig = plot_trajectory_static(trajectory, title=title)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        return fig


def compare_trajectories(
    real_trajectory: np.ndarray,
    generated_trajectory: np.ndarray,
    title: str = "Real vs Generated Trajectory",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare real and generated trajectories side by side.
    
    Args:
        real_trajectory: Shape (n_frames, n_players, 2)
        generated_trajectory: Shape (n_frames, n_players, 2)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    plot_trajectory_static(real_trajectory, ax=axes[0], title="Real Trajectory")
    plot_trajectory_static(generated_trajectory, ax=axes[1], title="Generated Trajectory")
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    
    return fig


# Demo function to test visualization with dummy data
def demo_visualization():
    """
    Demo visualization with synthetic trajectory data.
    """
    print("Creating demo visualization...")
    
    # Create synthetic trajectory (11 players, 30 frames)
    n_frames = 30
    n_players = 11
    
    # Starting positions (typical offensive formation)
    start_positions = np.array([
        [25, 26.65],   # QB
        [22, 26.65],   # RB
        [25, 45],      # WR1 (top)
        [25, 8],       # WR2 (bottom)
        [25, 35],      # WR3 (slot)
        [23, 30],      # TE
        [20, 24],      # C
        [20, 26],      # LG
        [20, 28],      # RG
        [20, 22],      # LT
        [20, 30],      # RT
    ])
    
    # Generate trajectories with some movement
    trajectory = np.zeros((n_frames, n_players, 2))
    trajectory[0] = start_positions
    
    # Movement patterns
    for frame in range(1, n_frames):
        trajectory[frame] = trajectory[frame - 1].copy()
        
        # QB drops back then moves
        trajectory[frame, 0, 0] += np.random.normal(-0.3, 0.1)
        trajectory[frame, 0, 1] += np.random.normal(0, 0.2)
        
        # RB moves forward
        trajectory[frame, 1, 0] += np.random.normal(0.5, 0.1)
        trajectory[frame, 1, 1] += np.random.normal(0.1, 0.1)
        
        # WRs run routes
        trajectory[frame, 2, 0] += np.random.normal(0.8, 0.1)  # WR1 go route
        trajectory[frame, 2, 1] += np.random.normal(-0.2, 0.1)
        
        trajectory[frame, 3, 0] += np.random.normal(0.6, 0.1)  # WR2 slant
        trajectory[frame, 3, 1] += np.random.normal(0.3, 0.1)
        
        trajectory[frame, 4, 0] += np.random.normal(0.5, 0.1)  # WR3 out route
        trajectory[frame, 4, 1] += np.random.normal(0.4, 0.1)
        
        # TE runs seam
        trajectory[frame, 5, 0] += np.random.normal(0.4, 0.1)
        trajectory[frame, 5, 1] += np.random.normal(0.1, 0.1)
        
        # OL holds position with slight movement
        for ol_idx in range(6, 11):
            trajectory[frame, ol_idx, 0] += np.random.normal(0.05, 0.05)
            trajectory[frame, ol_idx, 1] += np.random.normal(0, 0.05)
    
    # Static plot
    fig = plot_trajectory_static(trajectory, title="Demo Play - Static View")
    plt.savefig('/Users/namangoyal/Documents/GitHub/nfl_play_suggestion/demo_trajectory_static.png', 
                dpi=150, bbox_inches='tight')
    print("Static plot saved to demo_trajectory_static.png")
    
    # Animation
    anim = animate_trajectory(
        trajectory, 
        title="Demo Play - Animation",
        save_path='/Users/namangoyal/Documents/GitHub/nfl_play_suggestion/demo_trajectory.gif',
        interval=100
    )
    
    plt.show()
    
    return trajectory, anim


if __name__ == "__main__":
    demo_visualization()
