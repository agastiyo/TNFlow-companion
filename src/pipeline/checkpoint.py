"""Checkpoint load / save for TNFlow."""

from pathlib import Path

import torch

from src.config import CHECKPOINT, get_device
from src.model.tnflow import TNFlow


def load_model(path=CHECKPOINT, device=None, verbose=True):
    """Load a checkpoint and rebuild the model in eval mode.

    Args:    path: str|Path checkpoint file; device: torch.device|None (default get_device());
             verbose: bool print a "Loaded ..." line.
    Returns: (model: TNFlow on device (eval), ckpt: dict with config/epoch/metrics).
    """
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    model = TNFlow(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if verbose:
        sel = ckpt.get("selected_by", "?")
        print(f"Loaded {Path(path).name} (epoch {ckpt['epoch'] + 1}, selected by {sel} "
              f"[{ckpt.get('splits', '?')}], "
              f"val loss {ckpt.get('val_loss', float('nan')):.3f}, "
              f"val top-N {ckpt.get('val_topN', float('nan')):.3f})")
    return model, ckpt


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics):
    """Atomically write full training state to ``path`` (temp file + replace).

    Args:    path: str|Path; model: TNFlow; optimizer, scheduler: torch objects;
             epoch: int; metrics: dict flattened into the checkpoint (including ``splits``).
    Returns: None.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": model.config,
        "epoch": epoch,
        **metrics,
    }, tmp)
    tmp.replace(path)
