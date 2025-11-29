"""
Animation utilities for visualizing play trajectories.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
from .field import draw_field, DEFAULT_OFF_POSITIONS, DEFAULT_DEF_POSITIONS, SKILL_POSITIONS

# Note: Position constants are defined in field.py to avoid circular imports


def animate_trajectory(
    trajectory: np.ndarray,
    save_path: Optional[str] = None,
    interval: int = 100,
    show_field: bool = True,
    player_labels: Optional[List[str]] = None,
    animate_skill_only: bool = True,
    show_trails: bool = True,
    personnel_str: Optional[str] = None,
    raw_dir: Optional[Path] = None,
    game_id: Optional[int] = None,
    play_id: Optional[int] = None
):
    """
    Animate player trajectory over time with position labels.
    
    Args:
        trajectory: [T, P, 2] or [T, P, F] - trajectory positions
        save_path: Optional path to save animation (e.g., 'video.mp4')
        interval: Frame interval in milliseconds
        show_field: Whether to show field background
        player_labels: Optional list of position labels for each player (e.g., ['QB', 'RB', ...])
        animate_skill_only: If True, only animate skill positions (RB, WR, TE, QB)
        show_trails: Whether to show trajectory trails
    """
    # Extract positions
    if trajectory.shape[-1] >= 2:
        positions = trajectory[:, :, :2]  # [T, P, 2]
    else:
        positions = trajectory
    
    T, P, _ = positions.shape
    
    # Default labels - try to load actual positions from players.csv first
    # If not available, fall back to personnel-based inference (which is approximate)
    if player_labels is None:
        if P == 22:
            # Try to get real positions if we have game/play info
            if raw_dir and game_id and play_id:
                try:
                    from .load_real_positions import get_real_player_positions
                    real_positions = get_real_player_positions(raw_dir, game_id, play_id)
                    if real_positions:
                        player_labels = real_positions
                except Exception as e:
                    pass  # Silently fall through to fallback
            
            # Fallback to personnel-based (approximate - counts are right but player assignment may be wrong)
            if player_labels is None:
                if personnel_str:
                    from .position_utils import parse_personnel
                    counts = parse_personnel(personnel_str)
                    # Generate labels based on personnel counts only - may not match right players
                    labels_list = ['QB'] + ['WR']*counts['WR'] + ['RB']*counts['RB'] + ['TE']*counts['TE'] + ['OL']*counts['OL']
                    labels_list = (labels_list + ['OL']*11)[:11]
                    player_labels = labels_list + DEFAULT_DEF_POSITIONS
                else:
                    player_labels = DEFAULT_OFF_POSITIONS + DEFAULT_DEF_POSITIONS
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
    fig, ax = plt.subplots(figsize=(14, 7))
    if show_field:
        draw_field(ax)
    else:
        ax.set_xlim(0, 100)  # 100-yard field (middle at 50)
        ax.set_ylim(0, 53.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Color scheme: offense (blue shades), defense (red shades), skill positions (bright)
    colors = []
    for i, label in enumerate(player_labels):
        if i < 11:  # Offense
            if label in SKILL_POSITIONS:
                colors.append('#00ff00' if label == 'QB' else '#ff6600' if label == 'RB' else '#0099ff' if 'WR' in label else '#ffff00')
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
            # Animated player - larger, colored
            scatter = ax.scatter([], [], s=150, color=colors[p], 
                               alpha=0.9, zorder=10, edgecolors='white', linewidths=1.5)
            scatters.append(scatter)
            # Position label
            label_obj = ax.text(0, 0, player_labels[p], 
                              fontsize=8, fontweight='bold',
                              ha='center', va='center',
                              color='white', zorder=11,
                              bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=colors[p], 
                                       edgecolor='white', 
                                       linewidth=1, alpha=0.8))
            labels_text.append(label_obj)
            # Trail line
            if show_trails:
                trail, = ax.plot([], [], color=colors[p], alpha=0.3, linewidth=2, zorder=5)
                trails.append(trail)
        else:
            # Static player - smaller, gray
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
    
    # Add frame counter and info
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        """Update animation frame."""
        for p in range(P):
            if animate_mask[p]:
                # Update animated players
                x = positions[:frame+1, p, 0]
                y = positions[:frame+1, p, 1]
                
                # Current position
                if len(x) > 0:
                    scatters[p].set_offsets(np.c_[[x[-1]], [y[-1]]])
                    
                    # Update label position
                    labels_text[p].set_position((x[-1], y[-1] + 2.5))
                    
                    # Update trail
                    if show_trails and trails[p] is not None:
                        trails[p].set_data(x, y)
        
        frame_text.set_text(f'Frame: {frame+1}/{T} ({player_labels[:11].count("RB")+player_labels[:11].count("WR")+player_labels[:11].count("TE")} skill players animated)')
        
        return [s for s in scatters if s is not None] + [l for l in labels_text if l is not None] + [t for t in trails if t is not None] + [frame_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=T, interval=interval, blit=False, repeat=True
    )
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        try:
            anim.save(save_path, writer='ffmpeg', fps=10, bitrate=1800)
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")
            print("Install ffmpeg to save animations: conda install ffmpeg or brew install ffmpeg")
    
    plt.tight_layout()
    return anim, fig


def animate_comparison(
    trajectories: List[np.ndarray],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    interval: int = 100
):
    """
    Animate multiple trajectories side by side.
    
    Args:
        trajectories: List of [T, P, 2] trajectories
        labels: Optional list of labels
        save_path: Optional path to save
        interval: Frame interval in milliseconds
    """
    n_trajs = len(trajectories)
    fig, axes = plt.subplots(1, n_trajs, figsize=(6 * n_trajs, 6))
    
    if n_trajs == 1:
        axes = [axes]
    
    scatters_list = []
    colors_list = []
    
    for idx, traj in enumerate(trajectories):
        # Extract positions
        if traj.shape[-1] >= 2:
            positions = traj[:, :, :2]
        else:
            positions = traj
        
        T, P, _ = positions.shape
        
        # Setup field
        ax = axes[idx]
        draw_field(ax)
        if labels:
            ax.set_title(labels[idx], fontsize=14, fontweight='bold')
        
        # Colors
        colors = plt.cm.tab20(np.linspace(0, 1, P))
        colors_list.append(colors)
        
        # Initialize scatter
        scatters = []
        for p in range(P):
            scatter = ax.scatter([], [], s=100, color=colors[p], alpha=0.8, zorder=5)
            scatters.append(scatter)
        scatters_list.append(scatters)
    
    # Find max frames
    max_T = max([traj.shape[0] for traj in trajectories])
    
    def animate(frame):
        """Update animation frame."""
        for idx, (traj, scatters, colors) in enumerate(zip(trajectories, scatters_list, colors_list)):
            # Extract positions
            if traj.shape[-1] >= 2:
                positions = traj[:, :, :2]
            else:
                positions = traj
            
            T, P, _ = positions.shape
            frame_idx = min(frame, T - 1)
            
            for p in range(P):
                x = positions[frame_idx, p, 0]
                y = positions[frame_idx, p, 1]
                scatters[p].set_offsets(np.c_[x, y])
        
        return [s for scatters in scatters_list for s in scatters]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=max_T, interval=interval, blit=True, repeat=True
    )
    
    if save_path:
        print(f"Saving comparison animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=10)
    
    plt.tight_layout()
    return anim, fig

