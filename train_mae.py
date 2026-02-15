import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm  

from src.data.dataset import MultiViewCocoDataset, collate_fn
from src.data.transform import get_train_transforms_coco
from src.model.build import get_backbone
from src.model.architecture import ImageMIM 
from src.utils.optimizer import LARS
from src.utils.scheduler import WarmupCosineScheduler
from src.utils.logging import log_model_info, log_data_info

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}]", leave=True)
    
    for images, _ in pbar:
        img = images[0].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            prediction, mask = model(img)
            
            loss = F.mse_loss(prediction * mask, img * mask) / (mask.mean() + 1e-6)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix({"mse_loss": f"{current_loss:.4f}"})

    return total_loss / len(dataloader)

@hydra.main(config_path="config", config_name="train_mae", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset = MultiViewCocoDataset(
        cfg.data.dataset_path, 
        split="train", 
        transforms=get_train_transforms_coco(cfg.data.size),
        num_crops=1, # 1つで十分
        use_labels=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn 
    )

    log_data_info(
        "COCO",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_dataset),
    )

    # --- Model ---
    backbone = get_backbone(model_cfg=cfg.model)
    model = ImageMIM(
        backbone=backbone,
        decoder_hidden_dim=cfg.model.decoder_dim,
        mask_ratio=cfg.training.mask_ratio,
        patch_size=cfg.training.patch_size
    ).to(device)

    log_model_info(model, {"total_params": sum(p.numel() for p in model.parameters())})

    # --- Optimizer & Scheduler ---
    optimizer = LARS(
        model.parameters(),
        lr=cfg.training.optim.lr,
        weight_decay=cfg.training.optim.weight_decay,
        momentum=0.9,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.training.optim.warmup_epochs,
        max_epochs=cfg.training.optim.epoch,
        base_lr=cfg.training.optim.lr,
    )

    # --- Training Loop ---
    for epoch in range(cfg.training.optim.epoch):
        avg_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step(epoch=epoch)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Summary -> Epoch {epoch+1}: Avg Loss={avg_loss:.4f}, LR={lr:.6f}")

if __name__ == "__main__":
    scaler = GradScaler()
    main()