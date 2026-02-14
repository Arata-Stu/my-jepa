from src.model.backbone import CustomResNet, CustomSwinTransformer

def get_backbone(model_cfg):
    if model_cfg.backbone.type == "cnn":
        return CustomResNet(model_cfg.backbone.name, out_features=model_cfg.out_features)
    elif model_cfg.backbone.type == "vit":
        return CustomSwinTransformer(model_cfg.backbone.name, out_features=model_cfg.out_features)
    else:
        raise ValueError(f"Unknown model type: {model_cfg.backbone.type}")