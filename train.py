import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm  

from src.data.dataset import MultiViewCocoDataset, collate_fn
from src.data.transform import get_train_transforms_coco, get_val_transforms_coco
from src.model.build import get_backbone
from src.model.architecture import ImageSSL
from src.utils.loss import VICRegLoss
from src.utils.optimizer import LARS
from src.utils.scheduler import WarmupCosineScheduler
from src.utils.logging import log_model_info, log_data_info

def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    
    # tqdmによる進捗バーの設定
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}]", leave=True)
    
    for images, _ in pbar:
        # GPU転送の最適化 (pin_memory=Trueとセットで高速化)
        images = [img.to(device, non_blocking=True) for img in images]
        image1, image2 = images[0], images[1]

        # 勾配の初期化 (set_to_none=Trueで効率化)
        optimizer.zero_grad(set_to_none=True)

        # 順伝播 (Automatic Mixed Precision)
        with autocast(device_type=device.type):
            _, z1 = model(image1)
            _, z2 = model(image2)
            loss_dict = loss_fn(z1, z2)
            loss = loss_dict["loss"]

        # 逆伝播と最適化 (GradScalerを使用)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.item()
        total_loss += current_loss
        
        # tqdmの右側に現在のLossを表示
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss

@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    train_dataset = MultiViewCocoDataset(
        cfg.data.dataset_path, 
        split="train", 
        transforms=get_train_transforms_coco(cfg.data.size),
        num_crops=cfg.data.num_crops,
        use_labels=cfg.data.use_labels
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
    model = ImageSSL(
        backbone=backbone,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        proj_output_dim=cfg.model.proj_output_dim,
        use_projector=cfg.model.use_projector
    ).to(device)

    encoder_params = sum(p.numel() for p in backbone.parameters())
    
    projector_params = (
        sum(p.numel() for p in model.projector.parameters())
        if cfg.model.use_projector
        else 0
    )
    log_model_info(model, {"encoder": encoder_params, "projector": projector_params})


    # --- Loss & Optimizer ---
    loss_fn = VICRegLoss(std_coeff=cfg.training.loss.std_coeff, cov_coeff=cfg.training.loss.cov_coeff)

    scaler = GradScaler()
    
    optimizer = LARS(
        model.parameters(),
        lr=cfg.training.optim.lr,
        weight_decay=cfg.training.optim.weight_decay,
        eta=0.02,
        clip_lr=True,
        exclude_bias_n_norm=True,
        momentum=0.9,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.training.optim.warmup_epochs,
        max_epochs=cfg.training.optim.epoch,
        base_lr=cfg.training.optim.lr,
        min_lr=cfg.training.optim.min_lr,
        warmup_start_lr=cfg.training.optim.warmup_start_lr,
    )

    # --- Training Loop ---
    for epoch in range(cfg.training.optim.epoch):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch
        )
        
        # 学習率の更新
        scheduler.step(epoch=epoch)
        
        # エポック終了後のサマリー表示
        lr = optimizer.param_groups[0]['lr']
        print(f"Summary -> Epoch {epoch+1}: Avg Loss={train_loss:.4f}, LR={lr:.6f}")

if __name__ == "__main__":
    main()