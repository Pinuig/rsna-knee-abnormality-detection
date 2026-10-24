import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = 1.0 - p

        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)

        # Basic Cross Entropy with Focal weights
        los_pos = targets * torch.log(p_pos.clamp(min=self.eps)) * ((1.0 - p_pos) ** self.gamma_pos)
        los_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps)) * ((1.0 - p_neg) ** self.gamma_neg)
        loss = -(los_pos + los_neg)
        return loss.sum(dim=-1).mean()