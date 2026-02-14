import logging
import sys
from typing import Dict, Optional
import torch.nn as nn

LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(name=None, force=False):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=force,
    )
    return logging.getLogger(name=name)

logger = get_logger(__name__)


def log_data_info(
    dataset_name: str,
    num_batches: int,
    batch_size: int,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
) -> None:
    """Log dataset information."""
    if train_samples is not None and val_samples is not None:
        logger.info(
            f"📦 Data: {dataset_name} | {num_batches} batches x {batch_size} samples | "
            f"train={train_samples:,} | val={val_samples:,}"
        )
    else:
        logger.info(
            f"📦 Data: {dataset_name} | {num_batches} batches x {batch_size} samples"
        )

def log_model_info(model: nn.Module, param_counts: Dict[str, int]) -> None:
    """Log model structure and parameter counts."""
    logger.info(f"🧠 Model:\n{model}")
    param_str = " | ".join(f"{k}={v:,}" for k, v in param_counts.items())
    logger.info(f"🔢 Parameters: {param_str}")
