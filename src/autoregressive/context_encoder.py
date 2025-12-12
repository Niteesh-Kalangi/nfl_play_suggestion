"""
Context encoder for conditioning autoregressive models on game situation.

Matches the diffusion model's context encoder structure for direct comparison.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ContextEncoder(nn.Module):
    """
    Encodes categorical and continuous context features for conditioning.
    
    Categorical features: down, offensiveFormation, personnelO, defTeam, situation
    Continuous features: yardsToGo, yardlineNorm, hash_mark (encoded as 0.0=LEFT, 0.5=MIDDLE, 1.0=RIGHT)
    
    This matches the diffusion model's ContextEncoder for direct comparison.
    """
    
    def __init__(
        self,
        embedding_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 256,
        output_dim: int = 256
    ):
        """
        Args:
            embedding_dims: Dict mapping categorical feature names to embedding dims
            hidden_dim: Hidden dimension for continuous features MLP
            output_dim: Output dimension of combined context embedding
        """
        super().__init__()
        
        # Default embedding dimensions (matching diffusion model)
        if embedding_dims is None:
            embedding_dims = {
                'down': 8,  # 4 downs
                'offensiveFormation': 32,  # Various formations
                'personnelO': 32,  # Various personnel packages
                'defTeam': 64,  # 32 teams
                'situation': 16  # short, medium, long
            }
        
        self.embedding_dims = embedding_dims
        
        # Categorical embeddings
        self.embeddings = nn.ModuleDict({
            'down': nn.Embedding(5, embedding_dims['down']),  # 1-4 + padding
            'offensiveFormation': nn.Embedding(50, embedding_dims['offensiveFormation']),
            'personnelO': nn.Embedding(50, embedding_dims['personnelO']),
            'defTeam': nn.Embedding(40, embedding_dims['defTeam']),  # 32 teams + padding
            'situation': nn.Embedding(4, embedding_dims['situation'])  # short, medium, long + padding
        })
        
        # Continuous features MLP (yardsToGo, yardlineNorm, hash_mark)
        # hash_mark: 0.0=LEFT, 0.5=MIDDLE, 1.0=RIGHT
        self.continuous_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 3 features: yardsToGo, yardlineNorm, hash_mark
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Combine all features
        total_cat_dim = sum(embedding_dims.values())
        total_dim = total_cat_dim + hidden_dim // 2
        
        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def encode_categorical(
        self,
        categorical: List[Dict[str, int]]
    ) -> torch.Tensor:
        """
        Encode categorical features.
        
        Args:
            categorical: List of dicts with categorical feature values
            
        Returns:
            Tensor [B, sum(embedding_dims)]
        """
        batch_size = len(categorical)
        embeddings_list = []
        
        # Get device from model parameters (handles MPS correctly)
        try:
            device = next(self.parameters()).device
        except:
            # Fallback: get from first embedding layer
            device = next(iter(self.embeddings.values())).weight.device
        
        # Get vocabularies (simplified - would need proper vocab mapping)
        for feat_name, embed_layer in self.embeddings.items():
            # Extract feature values and convert to indices
            feat_values = [cat.get(feat_name, 0) for cat in categorical]
            
            # Map to indices (simplified - should use proper vocab)
            # Create on CPU first, then move to device (MPS compatibility)
            if feat_name == 'down':
                indices = torch.tensor([min(v, 4) for v in feat_values], dtype=torch.long)
            elif feat_name == 'situation':
                # Map situation strings to indices
                situation_map = {'short': 0, 'medium': 1, 'long': 2}
                indices = torch.tensor([
                    situation_map.get(v, 3) if isinstance(v, str) else 3
                    for v in feat_values
                ], dtype=torch.long)
            else:
                # Hash-based encoding for other features (simplified)
                indices = torch.tensor([
                    hash(str(v)) % embed_layer.num_embeddings
                    for v in feat_values
                ], dtype=torch.long)
            
            # Move to device after creation (avoids MPS allocation issues)
            indices = indices.to(device)
            embeddings_list.append(embed_layer(indices))
        
        return torch.cat(embeddings_list, dim=1)  # [B, total_cat_dim]
    
    def forward(
        self,
        categorical: List[Dict],
        continuous: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode context features.
        
        Args:
            categorical: List of dicts with categorical features
            continuous: Tensor [B, 3] with (yardsToGo, yardlineNorm, hash_mark) - already on correct device
            
        Returns:
            Context embedding [B, output_dim]
        """
        # Encode categorical (will use device from embedding layers)
        cat_emb = self.encode_categorical(categorical)  # [B, total_cat_dim]
        
        # Ensure continuous is on same device as model
        device = next(self.parameters()).device
        continuous = continuous.to(device)
        
        # Encode continuous
        cont_emb = self.continuous_mlp(continuous)  # [B, hidden_dim // 2]
        
        # Combine and project
        combined = torch.cat([cat_emb, cont_emb], dim=1)  # [B, total_dim]
        output = self.projection(combined)  # [B, output_dim]
        
        return output

