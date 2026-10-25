"""Train TNFlow. Edit the settings below, then run from the project root:

    python -m src.train

Every epoch trains on the train split (shuffled, freshly noised), scores the val split (noised,
unshuffled), and steps the learning rate on the val loss. The loss is the composition NLL plus the
grain NLL. The checkpoint, checkpoints/<size>_<dataset>_<timestamp>.pt, is rewritten every epoch
with the weights from the best val epoch so far and the loss curves.
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT, SEED
from src.dataset import SpectraDataset
from src.model.tnflow import TNFlow

# --- settings ------------------------------------------------------------------------------
SIZE = "medium"                     # "small", "medium" or "large"
START_CHECKPOINT = None             # a checkpoint to start from, or None for a fresh model
DATASET = "data/spectra.parquet"
NOISE = "mult_add"
EPOCHS = 1000
LR = 1e-3
LR_FACTOR = 0.5                     # ReduceLROnPlateau on the val loss
LR_PATIENCE = 10
MIN_LR = 1e-6
GRAD_CLIP = 5.0
# -------------------------------------------------------------------------------------------

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    # model
    model = TNFlow(SIZE).to(device)
    if START_CHECKPOINT:
        state = torch.load(PROJECT_ROOT / START_CHECKPOINT, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    # dataset
    data = SpectraDataset(PROJECT_ROOT / DATASET)
    if not START_CHECKPOINT:
        # the grain flow works on standardized log10 grain: mean and std over the train split's
        # present components with a known grain
        rows = data.splits["train"]
        target, grains = data.target[rows], data.grains[rows]
        log_grain = np.log10(grains[(target > 0) & np.isfinite(grains)])
        model.grain.log_mean.fill_(float(log_grain.mean()))
        model.grain.log_std.fill_(float(log_grain.std()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{SIZE}_{Path(DATASET).stem}_{datetime.now():%Y%m%d-%H%M%S}.pt"

    n_train = len(data.splits["train"])
    train_losses, val_losses = [], []
    best_loss, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        total, n = 0.0, 0
        for batch in data.get("train", noise=NOISE, shuffle=True):
            lams, values, target, grains = (x.to(device) for x in batch)
            comp_nll, grain_nll = model.nll(model.encode(lams, values), target, grains)
            loss = comp_nll + grain_nll
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item() * len(values)
            n += len(values)
            print(f"Epoch {epoch}/{EPOCHS} | train {n:,}/{n_train:,} | loss {total / n:8.3f}", end="\r")
        train_loss = total / n

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in data.get("val", noise=NOISE, shuffle=False):
                lams, values, target, grains = (x.to(device) for x in batch)
                comp_nll, grain_nll = model.nll(model.encode(lams, values), target, grains)
                total += (comp_nll + grain_nll).item() * len(values)
                n += len(values)
        val_loss = total / n

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{EPOCHS} | train loss {train_loss:8.3f} | val loss {val_loss:8.3f} | "
              f"lr {optimizer.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s" + " " * 10)

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  new best val loss {best_loss:.3f}")
        torch.save({"model_state_dict": best_state, "size": SIZE, "dataset": DATASET,
                    "epoch": best_epoch, "train_loss": train_losses, "val_loss": val_losses}, path)

    print(f"Done. Best val loss {best_loss:.3f} at epoch {best_epoch} -> {path}")


if __name__ == "__main__":
    main()
