import torch
import torch.nn as nn


class ImageSSL(nn.Module):
    """Image Self-Supervised Learning model implementation for 2D features."""

    def __init__(
        self, backbone, proj_hidden_dim=2048, proj_output_dim=2048, use_projector=True
    ):
        super().__init__()
        self.backbone = backbone
        self.use_projector = use_projector

        self.features_dim = self._get_features_dim()

        # Projector
        if use_projector:
            self.projector = ConvProjector(
                in_dim=self.features_dim,
                hidden_dim=proj_hidden_dim,
                out_dim=proj_output_dim
            )
        else:
            self.projector = nn.Identity()
    
    def _get_features_dim(self):

        device = next(self.backbone.parameters()).device
        dummy_input = torch.zeros(1, 3, 224, 224, device=device)
        
        with torch.no_grad():
            outputs = self.backbone(dummy_input)
            
        return outputs['c5'].shape[1]

    def forward(self, x):
        # use 'c5' for the main loss
        outputs = self.backbone(x)
        features = outputs['c5']  # [B, C, H, W]
        
        if self.use_projector:
            projections = self.projector(features)  # [B, 2048, H, W]
        else:
            projections = features
            
        return features, projections
    
class ConvProjector(nn.Module):
    """Convolutional Projector for 2D feature maps."""
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)
