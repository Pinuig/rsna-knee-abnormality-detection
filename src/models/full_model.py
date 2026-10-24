import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GatedAttentionMIL(nn.Module):
    """Calculates adaptive attention weights across slices of an MRI sequence."""
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Slices, Features)
        scores = self.w(self.v(x) * self.u(x))  # (B, S, 1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # (B, Features)
        return pooled


class RSNAKneeModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 12,
        embedding_dim: int = 384,
        dropout: float = 0.3
    ):
        super().__init__()
        # 1. 2D Slice Feature Extractor (shared across all planes)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        in_feats = self.backbone.num_features

        self.proj = nn.Sequential(
            nn.Linear(in_feats, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )

        # 2. Slice Aggregators for each plane
        self.mil_sag = GatedAttentionMIL(embedding_dim)
        self.mil_cor = GatedAttentionMIL(embedding_dim)
        self.mil_axi = GatedAttentionMIL(embedding_dim)

        # 3. Cross-Plane Attention Layer (Coronal <-> Sagittal <-> Axial)
        self.plane_transformer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        # 4. Final Multi-Label Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _forward_plane(self, x: torch.Tensor, mil_head: nn.Module) -> torch.Tensor:
        # x: (Batch, Slices, 3, H, W)
        b, s, c, h, w = x.shape
        flat_x = x.view(b * s, c, h, w)
        feats = self.proj(self.backbone(flat_x))
        feats = feats.view(b, s, -1)
        return mil_head(feats)  # (Batch, embedding_dim)

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, axi: torch.Tensor) -> torch.Tensor:
        emb_sag = self._forward_plane(sag, self.mil_sag)
        emb_cor = self._forward_plane(cor, self.mil_cor)
        emb_axi = self._forward_plane(axi, self.mil_axi)

        # Stack planes: (Batch, 3, embedding_dim)
        planes = torch.stack([emb_sag, emb_cor, emb_axi], dim=1)
        contextualized = self.plane_transformer(planes)
        
        # Flatten representation across 3 planes
        fused = contextualized.reshape(planes.size(0), -1)
        logits = self.classifier(fused)
        return logits