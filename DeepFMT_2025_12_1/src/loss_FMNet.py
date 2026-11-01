# src/loss_mt_only.py

import torch
import torch.nn as nn

class GeodesicMSELossMTOnly(nn.Module):
    """
    Moment-tensor ONLY loss.
    - Optimizes direction on unit hypersphere
    - Ignores scalar magnitude completely
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, mt_pred, mt_true, mw_pred=None, mw_true=None):
        """
        mt_pred: (B, 6), normalized
        mt_true: (B, 6), normalized
        """

        # Cosine similarity per sample
        cos_sim = torch.sum(mt_pred * mt_true, dim=1)  # (B,)
        loss_geodesic = torch.mean(1.0 - cos_sim)

        # Optional small L2 regularizer
        loss_l2 = self.mse(mt_pred, mt_true)

        loss = self.alpha * loss_geodesic + (1.0 - self.alpha) * loss_l2

        return (
            loss,
            loss_geodesic.detach(),
            torch.tensor(0.0, device=mt_pred.device)
        )