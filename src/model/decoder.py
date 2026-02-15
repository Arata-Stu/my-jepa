import torch
import torch.nn as nn
import torch.nn.functional as F

class MIMDecoder(nn.Module):
    def __init__(self, encoder_channels: dict, decoder_channels=[256, 128, 64, 32]):
        super().__init__()

        self.conv5 = nn.Conv2d(encoder_channels['c5'], decoder_channels[0], kernel_size=3, padding=1)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(decoder_channels[0] + encoder_channels['c4'], decoder_channels[1], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(decoder_channels[1] + encoder_channels['c3'], decoder_channels[2], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(decoder_channels[2] + encoder_channels['c2'], decoder_channels[3], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.recon_head = nn.Conv2d(decoder_channels[3], 3, kernel_size=3, padding=1)

    def forward(self, feats: dict):
        # 1. 最深部 (1/32)
        d5 = self.conv5(feats['c5'])
        
        # 2. 段階的なConcatとアップサンプリング
        # d5 (1/32) -> (1/16) に拡大して c4 と結合
        d4 = self.conv4(torch.cat([F.interpolate(d5, scale_factor=2), feats['c4']], dim=1))
        
        # d4 (1/16) -> (1/8) に拡大して c3 と結合
        d3 = self.conv3(torch.cat([F.interpolate(d4, scale_factor=2), feats['c3']], dim=1))
        
        # d3 (1/8) -> (1/4) に拡大して c2 と結合
        d2 = self.conv2(torch.cat([F.interpolate(d3, scale_factor=2), feats['c2']], dim=1))

        # 3. 最後に画像サイズに戻す (1/4 -> 1/1)
        x = F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=False)
        return self.recon_head(x)