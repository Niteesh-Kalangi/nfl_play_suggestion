"""
Custom play generation API for autoregressive models.

Matches diffusion model generation structure for direct comparison.
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from .dataset import derive_situation


def generate_play_with_context(
    model: torch.nn.Module,
    down: int,
    yards_to_go: float,
    offensive_formation: str = "SHOTGUN",
    personnel_o: str = "1 RB, 1 TE, 3 WR",
    def_team: str = "BEN",
    yardline: int = 50,
    hash_mark: str = "MIDDLE",
    horizon: int = 60,
    init_positions: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Generate a play with custom conditioning parameters.
    
    Matches diffusion model's custom generation API for direct comparison.
    
    Args:
        model: Trained autoregressive model (LSTM or Transformer)
        down: Down (1-4)
        yards_to_go: Yards needed for first down
        offensive_formation: Formation (e.g., "SHOTGUN", "UNDER CENTER", "EMPTY")
        personnel_o: Personnel (e.g., "1 RB, 1 TE, 3 WR")
        def_team: Defensive team abbreviation
        yardline: Yardline (0-100, distance from own goal line)
        hash_mark: Hash mark position ("LEFT", "MIDDLE", "RIGHT")
        horizon: Number of timesteps to generate (default: 60)
        init_positions: Optional initial positions [P*F] (flattened)
        device: Device to run on (auto-detects if None)
        
    Returns:
        Generated trajectory [T, P, F] in numpy format
        - T: number of timesteps (horizon)
        - P: number of players (22)
        - F: number of features (3: x, y, s)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                            else 'cpu')
    
    model.eval()
    
    # Derive situation
    situation = derive_situation(down, yards_to_go)
    
    # Normalize yardline to [0, 1]
    yardline_norm = yardline / 100.0
    
    # Encode hash mark
    hash_map = {'LEFT': 0.0, 'MIDDLE': 0.5, 'RIGHT': 1.0, 
                'left': 0.0, 'middle': 0.5, 'right': 1.0}
    hash_encoded = hash_map.get(hash_mark.upper(), 0.5)
    
    # Build categorical context
    context_categorical = [{
        'down': down,
        'offensiveFormation': offensive_formation,
        'personnelO': personnel_o,
        'defTeam': def_team,
        'situation': situation
    }]
    
    # Build continuous context
    context_continuous = torch.tensor(
        [[yards_to_go, yardline_norm, hash_encoded]],
        dtype=torch.float32
    ).to(device)
    
    # Prepare initial positions
    if init_positions is None:
        # Default: zeros (model will generate from scratch)
        num_players = model.num_players
        num_features = model.num_features
        init_positions = torch.zeros(1, num_players * num_features, device=device)
    else:
        if isinstance(init_positions, np.ndarray):
            init_positions = torch.FloatTensor(init_positions).to(device)
        if init_positions.dim() == 1:
            init_positions = init_positions.unsqueeze(0)  # Add batch dimension
        init_positions = init_positions.to(device)
    
    # Generate trajectory
    with torch.no_grad():
        trajectory = model.rollout(
            context_categorical=context_categorical,
            context_continuous=context_continuous,
            horizon=horizon,
            init_positions=init_positions
        )  # [batch, horizon, P*F]
    
    # Reshape to [T, P, F] format
    batch_size, T, PF = trajectory.shape
    num_players = model.num_players
    num_features = model.num_features
    
    trajectory = trajectory[0].cpu().numpy()  # [T, P*F]
    trajectory = trajectory.reshape(T, num_players, num_features)  # [T, P, F]
    
    # Clip coordinates to valid field bounds
    trajectory[:, :, 0] = np.clip(trajectory[:, :, 0], 0, 100)  # x: 0-100
    trajectory[:, :, 1] = np.clip(trajectory[:, :, 1], 0, 53.3)  # y: 0-53.3
    
    return trajectory


def generate_play_from_formation_anchors(
    model: torch.nn.Module,
    down: int,
    yards_to_go: float,
    offensive_formation: str = "SHOTGUN",
    personnel_o: str = "1 RB, 1 TE, 3 WR",
    def_team: str = "BEN",
    yardline: int = 50,
    hash_mark: str = "MIDDLE",
    horizon: int = 60,
    anchors_t0: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Generate a play starting from formation anchors at t=0.
    
    This matches the diffusion model's anchor-based generation.
    
    Args:
        model: Trained autoregressive model
        down: Down (1-4)
        yards_to_go: Yards needed for first down
        offensive_formation: Formation
        personnel_o: Personnel
        def_team: Defensive team abbreviation
        yardline: Yardline (0-100)
        hash_mark: Hash mark position
        horizon: Number of timesteps to generate
        anchors_t0: Optional formation anchors [P, 2] (x, y positions for t=0)
        device: Device to run on
        
    Returns:
        Generated trajectory [T, P, F]
    """
    # If anchors provided, use them as initial positions
    if anchors_t0 is not None:
        # Convert anchors to flattened format [P*F]
        # For now, just use x, y (speed s will be 0)
        num_players = anchors_t0.shape[0]
        num_features = model.num_features
        
        init_positions = np.zeros(num_players * num_features, dtype=np.float32)
        for p in range(num_players):
            base_idx = p * num_features
            init_positions[base_idx] = anchors_t0[p, 0]  # x
            init_positions[base_idx + 1] = anchors_t0[p, 1]  # y
            # s (speed) remains 0
        
        init_positions = torch.FloatTensor(init_positions).to(device if device else torch.device('cpu'))
    else:
        init_positions = None
    
    return generate_play_with_context(
        model=model,
        down=down,
        yards_to_go=yards_to_go,
        offensive_formation=offensive_formation,
        personnel_o=personnel_o,
        def_team=def_team,
        yardline=yardline,
        hash_mark=hash_mark,
        horizon=horizon,
        init_positions=init_positions,
        device=device
    )

