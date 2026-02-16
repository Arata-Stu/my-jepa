import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm  

from src.data.dataset import MultiViewCocoDataset, collate_fn
from src.data.transform import get_train_transforms_coco, get_val_transforms_coco
from src.model.build import get_backbone
from src.model.architecture import ImageMIM 
from src.utils.optimizer import LARS
from src.utils.scheduler import WarmupCosineScheduler
from src.utils.logging import log_model_info, log_data_info

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, writer):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}]", leave=True)
    
    for i, (images, _) in enumerate(pbar):
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
        
        # ステップ単位のログは log_dir へ
        step = epoch * len(dataloader) + i
        writer.add_scalar("Loss/train_step", current_loss, step)
        
        pbar.set_postfix({"mse_loss": f"{current_loss:.4f}"})

    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Val   [{epoch+1}]", leave=True)
    
    last_img, last_pred, last_mask = None, None, None

    with torch.no_grad():
        for images, _ in pbar:
            img = images[0].to(device, non_blocking=True)
            with autocast(device_type=device.type):
                prediction, mask = model(img)
                loss = F.mse_loss(prediction * mask, img * mask) / (mask.mean() + 1e-6)
            
            total_loss += loss.item()
            last_img, last_pred, last_mask = img, prediction, mask

    # TensorBoardへの画像ログ
    if last_img is not None:
        n = min(last_img.shape[0], 4)
        combined = torch.cat([
            last_img[:n], 
            last_img[:n] * (1 - last_mask[:n]), 
            last_pred[:n]
        ], dim=3)
        writer.add_images("Reconstruction_MAE", combined.clamp(0, 1), epoch)

    return total_loss / len(dataloader)

@hydra.main(config_path="config", config_name="train_mae", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # --- ディレクトリの準備 ---
    os.makedirs(cfg.log_dir, exist_ok=True)   # TensorBoard用
    os.makedirs(cfg.ckpt_dir, exist_ok=True)  # モデル保存用
    
    writer = SummaryWriter(log_dir=cfg.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    train_dataset = MultiViewCocoDataset(
        cfg.data.dataset_path, split="train", 
        transforms=get_train_transforms_coco(cfg.data.size),
        num_crops=1, use_labels=False
    )
    val_dataset = MultiViewCocoDataset(
        cfg.data.dataset_path, split="val", 
        transforms=get_val_transforms_coco(cfg.data.size),
        num_crops=1, use_labels=False
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True, collate_fn=collate_fn)

    # --- Model ---
    backbone = get_backbone(model_cfg=cfg.model)
    model = ImageMIM(
        backbone=backbone,
        decoder_hidden_dim=cfg.model.decoder_dim,
        mask_ratio=cfg.training.mask_ratio,
        patch_size=cfg.training.patch_size
    ).to(device)

    log_model_info(model, {"total_params": sum(p.numel() for p in model.parameters())})

    # --- Optimizer ---
    optimizer = LARS(model.parameters(), lr=cfg.training.optim.lr, 
                     weight_decay=cfg.training.optim.weight_decay, momentum=0.9)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=cfg.training.optim.warmup_epochs,
                                     max_epochs=cfg.training.optim.epoch, base_lr=cfg.training.optim.lr)
    scaler = GradScaler()
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(cfg.training.optim.epoch):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, writer)
        val_loss = validate_one_epoch(model, val_loader, device, epoch, writer)
        
        scheduler.step(epoch=epoch)
        lr = optimizer.param_groups[0]['lr']
        
        # TensorBoard ログ
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("LearningRate", lr, epoch)

        print(f"Summary -> Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={lr:.6f}")

        # --- チェックポイント保存 (cfg.ckpt_dir を使用) ---
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        torch.save(checkpoint, os.path.join(cfg.ckpt_dir, "last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(cfg.ckpt_dir, "best.pt"))
            print(f"✨ Best model saved to: {cfg.ckpt_dir}")

    writer.close()

if __name__ == "__main__":
    main()