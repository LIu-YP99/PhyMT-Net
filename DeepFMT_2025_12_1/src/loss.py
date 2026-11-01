import torch
import torch.nn as nn

class GeodesicMSELoss(nn.Module):
    def __init__(self, alpha=10.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, mt_pred, mt_true, mw_pred, mw_true):
        # Shape Loss: 1 - Cosine Similarity
        cosine_sim = torch.sum(mt_pred * mt_true, dim=1)
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        loss_shape = 1.0 - torch.mean(cosine_sim)
        
        # Mag Loss: MSE
        loss_mag = self.mse(mw_pred, mw_true)
        
        total = self.alpha * loss_shape + self.beta * loss_mag
        return total, loss_shape, loss_mag