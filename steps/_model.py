import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """A standard Residual Block for building deeply robust MLPs."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class GatedMultimodalMLP(nn.Module):
    def __init__(self, text_dim: int, img_dim: int, meta_dim: int):
        super().__init__()
        
        # 1. Compress Embeddings (using GELU for better deep gradients)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, 256), nn.GELU())
        self.img_proj  = nn.Sequential(nn.Linear(img_dim, 256), nn.GELU())
        
        # 2. Expand Meta Features (Tabular data)
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        # 256 (text) + 256 (image) + 64 (meta) = 576 dimensions natively concatenated
        fused_dim = 256 + 256 + 64
        
        # 3. Deep Residual Regressor
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # Deep Skip Connections to learn incredibly complex pricing combinations
            ResBlock(512, dropout=0.2),
            ResBlock(512, dropout=0.2),
            ResBlock(512, dropout=0.2),
            
            # Regression Head
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, text_emb, img_emb, meta):
        t = self.text_proj(text_emb)
        i = self.img_proj(img_emb)
        m = self.meta_proj(meta)
        
        fused = torch.cat([t, i, m], dim=1)
        
        return self.regressor(fused).squeeze()
