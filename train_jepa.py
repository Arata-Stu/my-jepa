import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm  

from src.data.dataset import MultiViewCocoDataset, collate_fn
from src.data.transform import get_train_transforms_coco, get_val_transforms_coco
from src.model.build import get_backbone
from src.model.architecture import ImageSSL
from src.utils.loss import VICRegLoss
from src.utils.optimizer import LARS
from src.utils.scheduler import WarmupCosineScheduler
from src.utils.logging import log_model_info, log_data_info

def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch, writer):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}]", leave=True)
    
    for i, (images, _) in enumerate(pbar):
        images = [img.to(device, non_blocking=True) for img in images]
        image1, image2 = images[0], images[1]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            _, z1 = model(image1)
            _, z2 = model(image2)
            loss_dict = loss_fn(z1, z2)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.item()
        total_loss += current_loss
        
        # ステップ単位のログ記録
        step = epoch * len(dataloader) + i
        writer.add_scalar("Loss/train_step", current_loss, step)
        # VICRegの詳細なLoss内訳を記録
        if "sim_loss" in loss_dict:
            writer.add_scalar("Loss_detail/sim", loss_dict["sim_loss"].item(), step)
            writer.add_scalar("Loss_detail/std", loss_dict["std_loss"].item(), step)
            writer.add_scalar("Loss_detail/cov", loss_dict["cov_loss"].item(), step)
        
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Val   [{epoch+1}]", leave=True)
    
    with torch.no_grad():
        for images, _ in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            image1, image2 = images[0], images[1]

            _, z1 = model(image1)
            _, z2 = model(image2)
            loss_dict = loss_fn(z1, z2)
            
            current_loss = loss_dict["loss"].item()
            total_loss += current_loss
            pbar.set_postfix({"val_loss": f"{current_loss:.4f}"})

    return total_loss / len(dataloader)

@hydra.main(config_path="config", config_name="train_jepa", version_base="1.2")
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
        num_crops=cfg.data.num_crops, use_labels=cfg.data.use_labels
    )
    val_dataset = MultiViewCocoDataset(
        cfg.data.dataset_path, split="val", 
        transforms=get_val_transforms_coco(cfg.data.size),
        num_crops=cfg.data.num_crops, use_labels=cfg.data.use_labels
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False, 
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn 
    )

    log_data_info("COCO", len(train_loader), cfg.data.batch_size, train_samples=len(train_dataset))

    # --- Model ---
    backbone = get_backbone(model_cfg=cfg.model)
    model = ImageSSL(
        backbone=backbone,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        proj_output_dim=cfg.model.proj_output_dim,
        use_projector=cfg.model.use_projector
    ).to(device)

    # パラメータ数ログ
    encoder_params = sum(p.numel() for p in backbone.parameters())
    projector_params = sum(p.numel() for p in model.projector.parameters()) if cfg.model.use_projector else 0
    log_model_info(model, {"encoder": encoder_params, "projector": projector_params})

    # --- Loss & Optimizer ---
    loss_fn = VICRegLoss(std_coeff=cfg.training.loss.std_coeff, cov_coeff=cfg.training.loss.cov_coeff)
    scaler = GradScaler()
    optimizer = LARS(
        model.parameters(), lr=cfg.training.optim.lr, weight_decay=cfg.training.optim.weight_decay,
        eta=0.02, clip_lr=True, exclude_bias_n_norm=True, momentum=0.9,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=cfg.training.optim.warmup_epochs,
        max_epochs=cfg.training.optim.epoch, base_lr=cfg.training.optim.lr,
    )

    # --- Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(cfg.training.optim.epoch):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch, writer)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device, epoch)
        
        scheduler.step(epoch=epoch)
        lr = optimizer.param_groups[0]['lr']
        
        # --- TensorBoard エポックログ ---
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("LearningRate", lr, epoch)

        print(f"Summary -> Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={lr:.6f}")

        # --- チェックポイント保存 (cfg.ckpt_dir を使用) ---
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'cfg': cfg,
        }
        
        torch.save(checkpoint, os.path.join(cfg.ckpt_dir, "last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(cfg.ckpt_dir, "best.pt"))
            print(f"🌟 Best model saved to {cfg.ckpt_dir}")

    writer.close()

if __name__ == "__main__":
    main()