"""
Animation utilities for visualizing autoregressive play trajectories.

Matches diffusion model animation structure.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Optional, List
from .field import draw_field

SKILL_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'HB', 'FB']


def animate_trajectory(
    trajectory: np.ndarray,
    save_path: Optional[str] = None,
    interval: int = 100,
    show_field: bool = True,
    player_labels: Optional[List[str]] = None,
    animate_skill_only: bool = True,
    show_trails: bool = True
):
    """
    Animate player trajectory over time.
    
    Args:
        trajectory: [T, P, 2] or [T, P, F] - trajectory positions
        save_path: Optional path to save animation
        interval: Frame interval in milliseconds
        show_field: Whether to show field background
        player_labels: Optional list of position labels
        animate_skill_only: If True, only animate skill positions
        show_trails: Whether to show trajectory trails
        
    Returns:
        Animation object and figure
    """
    # Extract positions
    if trajectory.shape[-1] >= 2:
        positions = trajectory[:, :, :2]  # [T, P, 2]
    else:
        positions = trajectory
    
    T, P, _ = positions.shape
    
    # Default labels
    if player_labels is None:
        if P == 22:
            player_labels = ['QB', 'RB', 'WR', 'WR', 'WR', 'TE', 'OL', 'OL', 'OL', 'OL', 'OL'] + ['DB'] * 11
        else:
            player_labels = [f'P{i+1}' for i in range(P)]
    elif len(player_labels) < P:
        player_labels.extend([f'P{i+1}' for i in range(len(player_labels), P)])
    
    # Determine which players to animate
    if animate_skill_only:
        animate_mask = [label in SKILL_POSITIONS for label in player_labels]
    else:
        animate_mask = [True] * P
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_field:
        draw_field(ax)
    else:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 53.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Color scheme
    colors = []
    for i, label in enumerate(player_labels):
        if i < 11:  # Offense
            if label in SKILL_POSITIONS:
                colors.append('#00ff00' if label == 'QB' else 
                            '#ff6600' if label == 'RB' else 
                            '#0099ff' if 'WR' in label else '#ffff00')
            else:
                colors.append('#6666ff')
        else:  # Defense
            colors.append('#ff6666' if animate_mask[i] else '#999999')
    
    # Initialize scatter plots and labels
    scatters = []
    labels_text = []
    trails = []
    
    for p in range(P):
        if animate_mask[p]:
            # Animated player
            scatter = ax.scatter([], [], s=150, color=colors[p], 
                               alpha=0.9, zorder=10, edgecolors='white', linewidths=1.5)
            scatters.append(scatter)
            label_obj = ax.text(0, 0, player_labels[p], 
                              fontsize=8, fontweight='bold',
                              ha='center', va='center',
                              color='white', zorder=11,
                              bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=colors[p], 
                                       edgecolor='white', 
                                       linewidth=1, alpha=0.8))
            labels_text.append(label_obj)
            if show_trails:
                trail, = ax.plot([], [], color=colors[p], alpha=0.3, linewidth=2, zorder=5)
                trails.append(trail)
        else:
            # Static player
            initial_pos = positions[0, p, :]
            scatter = ax.scatter(initial_pos[0], initial_pos[1], 
                               s=50, color='#888888', alpha=0.5, 
                               zorder=3, marker='o')
            scatters.append(scatter)
            label_obj = ax.text(initial_pos[0], initial_pos[1] + 2, 
                              player_labels[p], fontsize=6, 
                              ha='center', va='bottom',
                              color='gray', alpha=0.6)
            labels_text.append(label_obj)
            trails.append(None)
    
    # Frame counter
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Track trail history
    trail_history = [[] for _ in range(P)]
    
    def animate(frame):
        """Update animation frame."""
        for p in range(P):
            if animate_mask[p]:
                # Update position
                pos = positions[frame, p, :]
                scatters[p].set_offsets([pos])
                labels_text[p].set_position((pos[0], pos[1] + 2))
                
                # Update trail
                if show_trails and trails[p] is not None:
                    trail_history[p].append(pos)
                    if len(trail_history[p]) > 20:  # Keep last 20 points
                        trail_history[p].pop(0)
                    if len(trail_history[p]) > 1:
                        trail_data = np.array(trail_history[p])
                        trails[p].set_data(trail_data[:, 0], trail_data[:, 1])
        
        # Update frame counter
        frame_text.set_text(f'Frame: {frame+1}/{T}')
        
        return scatters + labels_text + [frame_text] + [t for t in trails if t is not None]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=T, interval=interval, blit=True, repeat=True
    )
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=10, bitrate=1800)
    
    return anim, fig

