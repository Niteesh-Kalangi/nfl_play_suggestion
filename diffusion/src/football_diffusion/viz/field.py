"""
Field visualization utilities.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, Tuple, List

# Position constants (defined here to avoid circular imports)
DEFAULT_OFF_POSITIONS = ['QB', 'RB', 'WR', 'WR', 'TE', 'OL', 'OL', 'OL', 'OL', 'OL', 'OL']
DEFAULT_DEF_POSITIONS = ['DB'] * 11
SKILL_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'HB', 'FB']


def draw_field(ax, field_length: float = 120, field_width: float = 53.3):
    """
    Draw football field outline with enhanced labels.
    
    Args:
        ax: Matplotlib axes
        field_length: Length of field in yards (including end zones)
        field_width: Width of field in yards
    """
    # Field outline - darker green background
    rect = patches.Rectangle(
        (0, 0), field_length, field_width,
        linewidth=2, edgecolor='white', facecolor='#2d5016', alpha=0.8
    )
    ax.add_patch(rect)
    
    # End zones
    endzone1 = patches.Rectangle(
        (0, 0), 10, field_width,
        linewidth=2, edgecolor='white', facecolor='#0066cc', alpha=0.3
    )
    ax.add_patch(endzone1)
    endzone2 = patches.Rectangle(
        (110, 0), 10, field_width,
        linewidth=2, edgecolor='white', facecolor='#0066cc', alpha=0.3
    )
    ax.add_patch(endzone2)
    
    # Yard lines - thicker for 10-yard markers
    for yard in range(0, field_length + 1, 5):
        if yard % 10 == 0:
            # Major yard lines (every 10 yards)
            ax.axvline(x=yard, color='white', linewidth=2, alpha=0.8)
            if 10 <= yard <= 110:
                # Label above and below
                ax.text(yard, field_width + 2, str(yard), 
                       ha='center', va='bottom', fontsize=10, 
                       fontweight='bold', color='black')
                ax.text(yard, -2, str(yard), 
                       ha='center', va='top', fontsize=10, 
                       fontweight='bold', color='black')
        else:
            # Minor yard lines (every 5 yards)
            ax.axvline(x=yard, color='white', linewidth=0.5, alpha=0.4)
    
    # Sidelines
    ax.axhline(y=0, color='white', linewidth=3, alpha=0.8)
    ax.axhline(y=field_width, color='white', linewidth=3, alpha=0.8)
    
    # Hash marks (simplified)
    for yard in range(10, 110, 5):
        ax.plot([yard, yard], [field_width * 0.43, field_width * 0.57],
               'w-', linewidth=0.3, alpha=0.5)
    
    # 50-yard line marker
    ax.axvline(x=60, color='white', linewidth=3, alpha=0.9)
    ax.text(60, field_width + 3, '50', 
           ha='center', va='bottom', fontsize=12, 
           fontweight='bold', color='black')
    
    ax.set_xlim(-5, field_length + 5)
    ax.set_ylim(-5, field_width + 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return ax


def plot_trajectory(
    trajectory: np.ndarray,
    ax=None,
    color: Optional[str] = None,
    alpha: float = 0.7,
    label: Optional[str] = None,
    show_path: bool = True,
    player_labels: Optional[List[str]] = None,
    highlight_skill_only: bool = False,
    personnel_str: Optional[str] = None
):
    """
    Plot player trajectory on field with optional position labels.
    
    Args:
        trajectory: [T, P, 2] or [T, P, F] - trajectory positions
        ax: Matplotlib axes (creates new if None)
        color: Color for trajectory
        alpha: Transparency
        label: Label for legend
        show_path: Whether to show path lines
        player_labels: Optional list of position labels
        highlight_skill_only: If True, only highlight skill positions
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
        draw_field(ax)
    
    # Extract x, y positions
    if trajectory.shape[-1] >= 2:
        positions = trajectory[:, :, :2]  # [T, P, 2]
    else:
        positions = trajectory
    
    T, P, _ = positions.shape
    
    # Default labels - try to derive from personnel if provided
    if player_labels is None:
        if P == 22:
            # Try to generate labels from personnel string
            if personnel_str:
                from .position_utils import assign_positions_by_formation
                # Use initial positions to infer formation
                initial_x = positions[0, :11, 0] if positions.shape[0] > 0 else None
                initial_y = positions[0, :11, 1] if positions.shape[0] > 0 else None
                if initial_x is not None and initial_y is not None:
                    off_labels = assign_positions_by_formation(
                        personnel_str, 
                        list(initial_x), 
                        list(initial_y)
                    )
                else:
                    from .position_utils import generate_position_labels
                    off_labels = generate_position_labels(personnel_str)
                player_labels = off_labels + DEFAULT_DEF_POSITIONS
            else:
                player_labels = DEFAULT_OFF_POSITIONS + DEFAULT_DEF_POSITIONS
        else:
            player_labels = [f'P{i+1}' for i in range(P)]
    
    # Determine which players to highlight
    if highlight_skill_only:
        highlight_mask = [label in SKILL_POSITIONS for label in player_labels]
    else:
        highlight_mask = [True] * P
    
    # Color scheme
    if color is None:
        colors = []
        for i, label in enumerate(player_labels):
            if i < 11:  # Offense
                if label in SKILL_POSITIONS:
                    colors.append('#00ff00' if label == 'QB' else '#ff6600' if label == 'RB' else '#0099ff' if 'WR' in label else '#ffff00')
                else:
                    colors.append('#6666ff')
            else:  # Defense
                colors.append('#ff6666' if highlight_mask[i] else '#999999')
    else:
        colors = [color] * P
    
    # Plot each player
    for p in range(P):
        player_traj = positions[:, p, :]  # [T, 2]
        x = player_traj[:, 0]
        y = player_traj[:, 1]
        
        # Plot path with appropriate style
        linewidth = 2.5 if highlight_mask[p] else 1.0
        path_alpha = alpha if highlight_mask[p] else alpha * 0.3
        
        if show_path:
            ax.plot(x, y, color=colors[p], alpha=path_alpha, linewidth=linewidth, 
                   label=label if p == 0 else None, zorder=5 if highlight_mask[p] else 3)
        
        # Plot start and end with labels
        if highlight_mask[p]:
            ax.scatter(x[0], y[0], color=colors[p], s=100, marker='o', 
                      zorder=7, edgecolors='white', linewidths=1.5)
            ax.scatter(x[-1], y[-1], color=colors[p], s=150, marker='*', 
                      zorder=7, edgecolors='white', linewidths=1.5)
            # Add position label at end point
            ax.text(x[-1], y[-1] + 2, player_labels[p], 
                   fontsize=8, fontweight='bold', ha='center', va='bottom',
                   color='white', zorder=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[p], 
                            edgecolor='white', linewidth=1, alpha=0.8))
        else:
            ax.scatter(x[0], y[0], color=colors[p], s=30, marker='o', zorder=4, alpha=0.5)
            ax.scatter(x[-1], y[-1], color=colors[p], s=50, marker='s', zorder=4, alpha=0.5)
    
    return ax


def plot_multiple_trajectories(
    trajectories: List[np.ndarray],
    labels: Optional[List[str]] = None,
    ax=None
):
    """
    Plot multiple trajectories on same field.
    
    Args:
        trajectories: List of [T, P, 2] trajectories
        labels: Optional list of labels
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        draw_field(ax)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        label = labels[i] if labels else f"Trajectory {i+1}"
        plot_trajectory(traj, ax=ax, color=colors[i], label=label, alpha=0.6)
    
    if labels:
        ax.legend()
    
    return ax

