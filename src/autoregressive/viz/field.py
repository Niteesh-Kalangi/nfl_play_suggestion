"""
Field visualization utilities for autoregressive models.

Matches diffusion model visualization structure.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, List


def draw_field(ax, field_length: float = 100, field_width: float = 53.3):
    """
    Draw football field outline with enhanced labels.
    
    Args:
        ax: Matplotlib axes
        field_length: Length of playable field in yards (0-100, middle at 50)
        field_width: Width of field in yards (53.3)
    """
    # Field outline - darker green background
    rect = patches.Rectangle(
        (0, 0), field_length, field_width,
        linewidth=2, edgecolor='white', facecolor='#2d5016', alpha=0.8
    )
    ax.add_patch(rect)
    
    # Yard lines
    for yard in range(0, field_length + 1, 5):
        if yard % 10 == 0:
            # Major yard lines (every 10 yards)
            ax.axvline(x=yard, color='white', linewidth=2, alpha=0.8)
            # Label above and below
            label = str(yard) if yard <= 50 else str(100 - yard)
            ax.text(yard, field_width + 2, label, 
                   ha='center', va='bottom', fontsize=10, 
                   fontweight='bold', color='black')
            ax.text(yard, -2, label, 
                   ha='center', va='top', fontsize=10, 
                   fontweight='bold', color='black')
        else:
            # Minor yard lines (every 5 yards)
            ax.axvline(x=yard, color='white', linewidth=0.5, alpha=0.4)
    
    # Sidelines
    ax.axhline(y=0, color='white', linewidth=3, alpha=0.8)
    ax.axhline(y=field_width, color='white', linewidth=3, alpha=0.8)
    
    # Hash marks
    for yard in range(5, field_length, 5):
        ax.plot([yard, yard], [field_width * 0.43, field_width * 0.57],
               'w-', linewidth=0.3, alpha=0.5)
    
    # 50-yard line marker
    ax.axvline(x=50, color='white', linewidth=3, alpha=0.9)
    ax.text(50, field_width + 3, '50', 
           ha='center', va='bottom', fontsize=12, 
           fontweight='bold', color='black')
    
    # Goal lines
    ax.axvline(x=0, color='yellow', linewidth=2.5, alpha=0.9, linestyle='-')
    ax.axvline(x=field_length, color='yellow', linewidth=2.5, alpha=0.9, linestyle='-')
    
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
    highlight_skill_only: bool = False
):
    """
    Plot player trajectory on field.
    
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
    
    # Default labels
    if player_labels is None:
        if P == 22:
            # Default: 11 offense + 11 defense
            player_labels = ['QB', 'RB', 'WR', 'WR', 'WR', 'TE', 'OL', 'OL', 'OL', 'OL', 'OL'] + ['DB'] * 11
        else:
            player_labels = [f'P{i+1}' for i in range(P)]
    
    # Skill positions
    SKILL_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'HB', 'FB']
    
    # Determine which players to highlight
    if highlight_skill_only:
        highlight_mask = [label in SKILL_POSITIONS for label in player_labels]
    else:
        highlight_mask = [True] * P
    
    # Plot trajectories
    for p in range(P):
        player_pos = positions[:, p, :]  # [T, 2]
        
        if highlight_mask[p]:
            # Skill position - colored
            if p < 11:  # Offense
                colors = ['#00ff00', '#ff6600', '#0099ff', '#0099ff', '#0099ff', '#ffff00']
                player_color = colors[min(p, len(colors)-1)]
            else:  # Defense
                player_color = '#ff6666'
        else:
            # Non-skill position - gray
            player_color = '#888888'
        
        # Plot path
        if show_path:
            ax.plot(player_pos[:, 0], player_pos[:, 1], 
                   color=player_color, alpha=alpha * 0.5, linewidth=1, zorder=1)
        
        # Plot start and end points
        ax.scatter(player_pos[0, 0], player_pos[0, 1], 
                  s=100, color=player_color, alpha=alpha, zorder=5, marker='o')
        ax.scatter(player_pos[-1, 0], player_pos[-1, 1], 
                  s=150, color=player_color, alpha=alpha, zorder=6, marker='s', edgecolors='white')
        
        # Add label
        if player_labels and p < len(player_labels):
            ax.text(player_pos[0, 0], player_pos[0, 1] + 2, 
                   player_labels[p], fontsize=8, fontweight='bold',
                   ha='center', va='bottom', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=player_color, 
                            edgecolor='white', linewidth=1, alpha=0.8),
                   zorder=7)
    
    return ax

