import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet50,
    swin_t,
    swin_s,
    swin_b
)

class CustomResNet(nn.Module):
    def __init__(self,
                 model_name: str='resnet18',
                 out_features: list=['c2', 'c3', 'c4', 'c5']):
        
        super().__init__()
        self.out_features = out_features
        
        if model_name == 'resnet18':
            resnet = resnet18()
        elif model_name == 'resnet50':
            resnet = resnet50()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem(x)
        
        c2 = self.layer1(x)  # 1/4スケール
        c3 = self.layer2(c2) # 1/8スケール
        c4 = self.layer3(c3) # 1/16スケール
        c5 = self.layer4(c4) # 1/32スケール

        outputs = {}
        outputs['c2'] = c2
        outputs['c3'] = c3
        outputs['c4'] = c4
        outputs['c5'] = c5
        
        return {k: outputs[k] for k in self.out_features}
    

class CustomSwinTransformer(nn.Module):
    def __init__(self,
                 model_name: str='swin_t',
                 out_features: list=['c2', 'c3', 'c4', 'c5']):
        
        super().__init__()
        self.out_features = out_features
        
        # モデルの選択
        if model_name == 'swin_t':
            swin = swin_t()
        elif model_name == 'swin_s':
            swin = swin_s()
        elif model_name == 'swin_b':
            swin = swin_b()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        self.stage1 = nn.Sequential(swin.features[0], swin.features[1])
        self.stage2 = nn.Sequential(swin.features[2], swin.features[3])
        self.stage3 = nn.Sequential(swin.features[4], swin.features[5])
        self.stage4 = nn.Sequential(swin.features[6], swin.features[7])

    def forward(self, x):

        # torchvision model use not NCHW, but NHWC, so we need to permute the input and output
        x = self.stage1(x)
        c2 = x.permute(0, 3, 1, 2)  # 1/4スケール
        
        x = self.stage2(x)
        c3 = x.permute(0, 3, 1, 2)  # 1/8スケール
        
        x = self.stage3(x)
        c4 = x.permute(0, 3, 1, 2)  # 1/16スケール
        
        x = self.stage4(x)
        c5 = x.permute(0, 3, 1, 2)  # 1/32スケール

        outputs = {}
        outputs['c2'] = c2
        outputs['c3'] = c3
        outputs['c4'] = c4
        outputs['c5'] = c5
        
        return {k: outputs[k] for k in self.out_features}