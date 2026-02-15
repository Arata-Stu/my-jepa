import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeStdLoss(torch.nn.Module):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation.
        Features with std below the margin incur a penalty of (std_margin - std).
        Args:
            std_margin (float, default=1.0):
                Minimum desired standard deviation per feature.
        """
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss
    

class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calculate off-diagonal loss
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss

class VICRegLoss(nn.Module):
    def __init__(self, std_coeff=10.0, cov_coeff=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2):
        # z1, z2: [B, C, H, W]
        sim_loss = F.mse_loss(z1, z2)

        # Variance & Covariance Loss (空間次元 H, W をサンプルとして扱う)
        # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        z1_flat = z1.permute(0, 2, 3, 1).reshape(-1, z1.shape[1])
        z2_flat = z2.permute(0, 2, 3, 1).reshape(-1, z2.shape[1])
        
        z_all = torch.cat([z1_flat, z2_flat], dim=0)

        std_loss = self.std_loss_fn(z_all)
        cov_loss = self.cov_loss_fn(z_all)

        total_loss = sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "std_loss": std_loss,
            "cov_loss": cov_loss
        }
    
class MAELoss(nn.Module):
    def __init__(self, norm_pix_loss: bool = False):
        
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def forward(self, prediction, target, mask):
        
        # 1. 全ピクセルの二乗誤差を計算
        loss_all = F.mse_loss(prediction, target, reduction='none') # [B, 3, H, W]
        
        # 2. マスクされたピクセルのみを抽出
        # maskは [B, 1, H, W] なので、broadcastingにより [B, 3, H, W] に適用される
        mask_loss = (loss_all * mask).sum() / (mask.sum() + 1e-6)

        return {
            "loss": mask_loss,
            "reconstruction_loss": mask_loss
        }