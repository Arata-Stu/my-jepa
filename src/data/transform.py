import torch
from torchvision.transforms import v2

def get_train_transforms_coco(size=(640, 640)):
    return v2.Compose([
        v2.RandomResizedCrop(size=size, scale=(0.5, 1.2), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        
        v2.ToImage(), 
        v2.SanitizeBoundingBoxes(), 
        
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms_coco(size=(640, 640)):
    return v2.Compose([
        v2.Resize(size=size, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])