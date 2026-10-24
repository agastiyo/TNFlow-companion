"""Train TNFlow: combined composition + grain NLL, checkpoint selected on the validation split.

Data: ``data/splits/<splits>.parquet`` (default v2: train / val / test / ood). Training batches are
reshuffled and re-noised every epoch; val and OOD carry one frozen noise draw, so their curves are
comparable across epochs. Selection and the LR schedule use **val only**; OOD is logged, never
used, and test is never touched here (score it afterwards with ``src.pipeline.test``).

Run from the project root:
    python3.10 -m src.pipeline.train --run-name v2                          # full run
    python3.10 -m src.pipeline.train --run-name smoke --epochs 1 --limit 20000

Writes runs/<run-name>/: checkpoint.pt (best val loss), history.json (per-epoch curves), config.json.
"""

import argparse
import dataclasses
import json
import time

import numpy as np
import torch

from src import config
from src.data_utils import splits as splits_mod
from src.data_utils.dataset import SpectrumDataset, make_loader
from src.data_utils.noise import build_noise
from src.data_utils.store import mixture_id
from src.evaluation.metrics import grain_log_error, ranks_from_scores, recall_at_k
from src.evaluation.posterior import detection_score
from src.model.cnf import free_to_full_simplex
from src.model.tnflow import TNFlow
from src.pipeline.checkpoint import save_checkpoint

METRICS = ("loss", "topN", "top5", "grain_dex", "grain_within2")


# --------------------------------------------------------------------------- setup

def _subsample(ids, n, seed):
    """At most n of ids (seeded, original order kept); all of them when n is 0."""
    if not n or n >= len(ids):
        return ids
    keep = np.sort(np.random.default_rng(seed).choice(len(ids), n, replace=False))
    return [ids[i] for i in keep]


def build_datasets(splits, cfg: config.TrainConfig) -> dict:
    """{train, val, ood[, train_eval]} SpectrumDatasets.

    train is re-noised on every read; val / ood are noised once (frozen). The optional
    ``train_eval`` is a fixed CLEAN train subsample scored each epoch, as an invertibility-ceiling
    diagnostic.
    """
    noise = build_noise(cfg.noise)
    ids = {s: _subsample(splits[s], cfg.limit, cfg.seed) for s in ("train", "val", "ood")}
    datasets = {
        "train": SpectrumDataset(ids["train"], noise, mode="fresh", seed=cfg.seed),
        "val": SpectrumDataset(ids["val"], noise, mode="frozen", seed=cfg.seed),
        "ood": SpectrumDataset(ids["ood"], noise, mode="frozen", seed=cfg.seed + 1),
    }
    if cfg.train_eval_n:
        datasets["train_eval"] = SpectrumDataset(_subsample(ids["train"], cfg.train_eval_n, cfg.seed))
    return datasets


def build_model(cfg: config.TrainConfig, train_ds, device) -> TNFlow:
    """TNFlow from the size presets, with grain standardization fit on the training set:
    mean and std of log10(grain / µm) over its present components with a known grain."""
    target, grains = train_ds.targets()
    lg = np.log10(grains[(target > 0) & np.isfinite(grains)])
    return TNFlow(**dataclasses.asdict(cfg.model),
                  log_grain_mean=float(lg.mean()), log_grain_std=float(lg.std())).to(device)


def build_optimizer(model, cfg: config.TrainConfig):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    return optimizer, scheduler


# --------------------------------------------------------------------------- one epoch

def _to(batch, device):
    return [t.to(device) for t in batch]


def train_one_epoch(model, loader, optimizer, cfg: config.TrainConfig, device, epoch) -> float:
    """One pass over the (freshly shuffled, freshly noised) training set -> mean objective."""
    model.train()
    running, n = 0.0, 0
    for b, batch in enumerate(loader, start=1):
        lam, specs, target, grains = _to(batch, device)
        comp_nll, grain_nll = model.nll(model.encode(lam, specs), target, grains)
        loss = comp_nll + cfg.grain_weight * grain_nll
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        running += loss.item() * len(specs); n += len(specs)
        print(f"Epoch {epoch}/{cfg.epochs} | train batch {b}/{len(loader)} | avg loss {running/n:7.3f}",
              end='\r')
    return running / n


def evaluate(model, loader, cfg: config.TrainConfig, device, tag) -> dict:
    """Per-epoch diagnostics on one set: composition NLL, top-N / top-k recall, grain error.

    Top-N: the true components are the N highest by ``detection_score`` (N = how many are present).
    ``grain_dex`` is teacher-forced (conditioned on the true composition) -- a training
    diagnostic, not an achievable inference-time number.

    Returns: dict with keys METRICS.
    """
    model.eval()
    vl, vn, correct_N, correct_k, total = 0.0, 0, 0, 0, 0
    dlogs = []
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            lam, specs, target, grains = _to(batch, device)
            ctx = model.encode(lam, specs)
            comp_nll, _ = model.nll(ctx, target, grains)
            vl += comp_nll.item() * len(specs); vn += len(specs)

            draws = free_to_full_simplex(model.cnf.dist(ctx).sample((cfg.acc_samples,)))  # [S, B, K]
            present = target > 0
            ranks = ranks_from_scores(detection_score(draws))
            correct_N += recall_at_k(ranks, present, present.sum(dim=1, keepdim=True))
            correct_k += recall_at_k(ranks, present, cfg.acc_topk)
            total += len(specs)

            gmask = present & torch.isfinite(grains)
            if gmask.any():
                gpred = model.grain.predict_median(ctx, target, n=cfg.acc_samples)
                dlogs.append(grain_log_error(gpred, grains)[gmask].cpu())
            print(f"  {tag} batch {i}/{len(loader)} | loss {vl/vn:7.3f} | "
                  f"top-N {correct_N/total:5.3f} | top-{cfg.acc_topk} {correct_k/total:5.3f}", end='\r')
    nan = float("nan")
    if dlogs:
        d = torch.cat(dlogs)
        g_med, g_w2 = d.median().item(), (d < np.log10(2.0)).float().mean().item()
    else:
        g_med = g_w2 = nan
    return {
        "loss": vl / vn if vn else nan,
        "topN": correct_N / total if total else nan,
        "top5": correct_k / total if total else nan,
        "grain_dex": g_med,
        "grain_within2": g_w2,
    }


def select_checkpoint(scores, best_loss) -> bool:
    """Keep this epoch iff its VALIDATION composition NLL beats the best so far (never OOD/test)."""
    return scores["val"]["loss"] < best_loss


# --------------------------------------------------------------------------- bookkeeping

def _finite(x):
    """NaN/inf -> None so the history file stays strict-JSON readable."""
    return None if x is None or not np.isfinite(x) else float(x)


def save_history(path, history, meta):
    """Write the per-epoch curves (temp file + replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, **history}, indent=1))
    tmp.replace(path)


def _epoch_line(epoch, cfg, objective, scored, seconds):
    v, o = scored["val"], scored["ood"]
    k = cfg.acc_topk
    return (f"Epoch {epoch}/{cfg.epochs} | obj {objective:7.3f} | "
            f"val loss {v['loss']:7.3f} topN {v['topN']:5.3f} top{k} {v['top5']:5.3f} "
            f"gΔ {v['grain_dex']:4.2f} | ood loss {o['loss']:7.3f} topN {o['topN']:5.3f} "
            f"top{k} {o['top5']:5.3f} gΔ {o['grain_dex']:4.2f} | {seconds:.0f}s" + " " * 4)


# --------------------------------------------------------------------------- main

def main(cfg: config.TrainConfig = config.TrainConfig(), run_name="v2"):
    run_dir = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    torch.manual_seed(cfg.seed)
    device = config.get_device()

    splits = splits_mod.load_splits(cfg.splits)
    if "val" not in splits:
        raise ValueError(f"split file {cfg.splits!r} has no val split")
    datasets = build_datasets(splits, cfg)
    train_loader = make_loader(datasets["train"], cfg.batch_size, shuffle=True, seed=cfg.seed)
    eval_loaders = {s: make_loader(ds, cfg.batch_size) for s, ds in datasets.items() if s != "train"}
    held_out = sorted({mixture_id(t) for t in datasets["ood"].targets()[0]})
    print("  splits: " + " | ".join(f"{s} {len(ds):,}" for s, ds in datasets.items())
          + f"  ({cfg.splits}; {len(held_out)} held-out mixtures)")

    model = build_model(cfg, datasets["train"], device)
    optimizer, scheduler = build_optimizer(model, cfg)

    best_loss, best_epoch = float("inf"), None
    history = {"epochs": [], "train_objective": [],
               **{s: {m: [] for m in METRICS} for s in eval_loaders}}
    meta = {"model_config": model.config, "train_config": dataclasses.asdict(cfg),
            "split_sizes": {s: len(ds) for s, ds in datasets.items()},
            "held_out_mixtures": held_out,
            "notes": ("train_objective is the optimized comp+grain NLL on noised batches; val/ood "
                      "carry one frozen noise draw. Selection and LR schedule use val only.")}

    print("Training Start!")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        objective = train_one_epoch(model, train_loader, optimizer, cfg, device, epoch)
        scored = {s: evaluate(model, loader, cfg, device, s) for s, loader in eval_loaders.items()}
        scheduler.step(scored["val"]["loss"])
        print(_epoch_line(epoch, cfg, objective, scored, time.perf_counter() - t0))

        history["epochs"].append(epoch)
        history["train_objective"].append(_finite(objective))
        for s in eval_loaders:
            for m in METRICS:
                history[s][m].append(_finite(scored[s][m]))

        if select_checkpoint(scored, best_loss):
            best_loss, best_epoch = scored["val"]["loss"], epoch
            save_checkpoint(run_dir / "checkpoint.pt", model, optimizer, scheduler, epoch - 1, {
                "selected_by": "val_loss", "splits": cfg.splits, "train_objective": objective,
                **{f"{s}_{m}": scored[s][m] for s in eval_loaders for m in METRICS},
                "train_config": dataclasses.asdict(cfg)})
            print(f"  new best val loss ({best_loss:.3f}) -> {run_dir / 'checkpoint.pt'}")

        meta["best"] = {"val_loss": {"epoch": best_epoch, "value": _finite(best_loss)}}
        save_history(run_dir / "history.json", history, meta)

    print(f"Training complete!  best val loss {best_loss:.3f} (epoch {best_epoch})")
    print(f"  run -> {run_dir}")


def parse_args():
    d = config.TrainConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-name", default="v2")
    p.add_argument("--splits", default=d.splits)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--limit", type=int, default=d.limit,
                   help="cap every split at this many spectra (smoke runs; 0 = off)")
    p.add_argument("--train-eval-n", type=int, default=d.train_eval_n)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    main(config.TrainConfig(splits=a.splits, epochs=a.epochs, batch_size=a.batch_size,
                            limit=a.limit, train_eval_n=a.train_eval_n), a.run_name)
