import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import MIMDecoder

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
    
class ImageMIM(nn.Module):
    def __init__(self, backbone, decoder_hidden_dim=256, mask_ratio=0.6, patch_size=32):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # エンコーダーのチャンネル数を動的に取得
        encoder_channels = self._get_encoder_channels()
        
        # デコーダーの階層別チャンネル設計 (例: 256 -> 128 -> 64 -> 32)
        decoder_channels = [max(32, decoder_hidden_dim // (2**i)) for i in range(4)]
        
        self.decoder = MIMDecoder(
            encoder_channels=encoder_channels, 
            decoder_channels=decoder_channels
        )

    def _get_encoder_channels(self):
        # モデルのデバイスを確認
        device = next(self.backbone.parameters()).device
        dummy_input = torch.zeros(1, 3, 224, 224, device=device)
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(dummy_input)
        return {k: v.shape[1] for k, v in outputs.items()}

    def apply_mask(self, x):
        B, C, H, W = x.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size
        num_patches = grid_h * grid_w
        num_mask = int(self.mask_ratio * num_patches)

        # deviceを指定してマスクを作成
        mask = torch.zeros((B, num_patches), device=x.device)
        for i in range(B):
            perm = torch.randperm(num_patches, device=x.device)
            mask[i, perm[:num_mask]] = 1
        
        mask = mask.reshape(B, 1, grid_h, grid_w)
        # 画像サイズまで拡大
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        
        masked_x = x * (1 - mask)
        return masked_x, mask

    def forward(self, x, skip_mask=False):
        # 訓練中かつスキップフラグがない場合のみマスク適用
        if self.training and not skip_mask:
            masked_x, mask = self.apply_mask(x)
            features = self.backbone(masked_x)
            prediction = self.decoder(features)
            return prediction, mask
        else:
            features = self.backbone(x)
            prediction = self.decoder(features)
            return prediction