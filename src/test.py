"""Score a checkpoint on the test and ood splits. Edit the settings below, then run from the
project root:

    OMP_NUM_THREADS=1 python -m src.test

Every spectrum (clean, unshuffled) gets N_SAMPLES posterior draws and a joint comp+grain BIC-GMM
mode fit, so each mode is a complete solution. Per split it reports (Table 1):

    top / best-mode comp TV     TV distance to the truth of the highest-weight mode / of the closest mode
    top / best-mode grain dex   mean |Δlog10 grain| over the true components, for the same two modes
    top = best (%)              how often the highest-weight mode is also the closest one
    mean mode count, multimodal (%)
    comp / grain PIT W²         calibration: Cramér–von Mises W² of the PIT values against uniform
    sharpness                   mean pairwise distance between composition draws

plus per-stage timing. Writes results/<size>_<dataset>_<timestamp>/: metrics.json,
queries.parquet (one row per spectrum), pits.parquet (one row per scored component), and the
PIT figures pit_histograms.png and pit_trace_histograms.png.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT, SEED
from src.dataset import SpectraDataset, mixture_id
from src.evaluation.posterior import grain_draws, modes
from src.model.grain import GRAIN_EPS
from src.model.tnflow import load

# --- settings ------------------------------------------------------------------------------
SIZE = "medium"                     # must match the checkpoint
CHECKPOINT = "tnflow_checkpoint.pt"
DATASET = "data/spectra.parquet"
N_SAMPLES = 500                     # posterior draws per spectrum
TRACE = 0.05                        # present components at or below this abundance count as trace
PIT_BINS = 10
# -------------------------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS = {"test": "In-distribution", "ood": "OOD"}
STAGES = ("encode", "sample", "modes")


# --------------------------------------------------------------------------- metrics

def tv_distance(a, b):
    """a, b [K] -> total-variation distance 0.5·Σ|a−b|."""
    return float(0.5 * (a - b).abs().sum())


def grain_dex(pred_um, true_um):
    """|Δlog10| between grain sizes (same shape), clamped at GRAIN_EPS."""
    return (torch.log10(pred_um.clamp(min=GRAIN_EPS)) - torch.log10(true_um.clamp(min=GRAIN_EPS))).abs()


def mode_grain_dex(mode, true_grains):
    """Mean grain dex of one mode over the components where both it and the truth have a grain."""
    g = mode["grain_um"]
    ok = torch.isfinite(g) & torch.isfinite(true_grains) & (true_grains > 0) & (g > 0)
    return float(grain_dex(g[ok], true_grains[ok]).mean()) if ok.any() else float("nan")


def sharpness(draws):
    """draws [S, K] -> mean distance between two different draws."""
    s = draws.shape[0]
    return float(torch.cdist(draws, draws).sum() / (s * (s - 1)))


def pit(draws, truth):
    """Fraction of draws strictly below the truth; Uniform(0, 1) when calibrated."""
    return float((draws < truth).to(draws.dtype).mean())


def pit_cvm(u):
    """PIT values -> Cramér–von Mises W² against Uniform(0, 1); 0 = uniform, lower is better."""
    u, _ = torch.sort(torch.as_tensor(np.asarray(u), dtype=torch.float64))
    n = u.numel()
    if n == 0:
        return float("nan")
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return float(1.0 / (12 * n) + ((u - (2 * i - 1) / (2 * n)) ** 2).sum())


# --------------------------------------------------------------------------- one spectrum

def score(model, lams, values, target, true_grains):
    """One labelled spectrum -> (query record, list of PIT rows). Inputs are CPU tensors."""
    device = next(model.parameters()).device
    target, true_grains = target.to(device), true_grains.to(device)
    present = target > 0

    t0 = time.perf_counter()
    post = model.posterior(values.numpy(), lams.numpy(), seed=SEED)
    t1 = time.perf_counter()
    comp, grain = post.sample(N_SAMPLES)
    t2 = time.perf_counter()
    found = modes(comp, grain)
    t3 = time.perf_counter()

    tvs = [tv_distance(m["composition"], target) for m in found]
    best = int(np.argmin(tvs))
    record = dict(mixture=mixture_id(target.cpu().numpy()), n_modes=len(found),
                  top_tv=tvs[0], best_tv=tvs[best],
                  top_grain_dex=mode_grain_dex(found[0], true_grains),
                  best_grain_dex=mode_grain_dex(found[best], true_grains),
                  top_is_best=best == 0, sharpness=sharpness(comp),
                  encode_s=t1 - t0, sample_s=t2 - t1, modes_s=t3 - t2)

    # calibration: PIT of the truth under each present component's marginal, and under its
    # log-grain marginal where the truth is known and the draws contain the component
    pits = [dict(kind="comp", component=c, truth=float(target[c]), pit=pit(comp[:, c], target[c]))
            for c in present.nonzero().flatten().tolist()]
    known = present & torch.isfinite(true_grains) & (true_grains > 0)
    for c in known.nonzero().flatten().tolist():
        g = grain_draws(grain, c)
        if g.numel():
            pits.append(dict(kind="grain", component=c, truth=float(true_grains[c]),
                             pit=pit(torch.log10(g), torch.log10(true_grains[c]))))
    return record, pits


# --------------------------------------------------------------------------- Table 1

def _mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def _std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def summarize(q, p):
    """One split's queries and PIT rows -> Table 1 numbers (means ± std over spectra)."""
    m = {"n_spectra": len(q)}
    for name, col in (("top_tv", "top_tv"), ("best_tv", "best_tv"),
                      ("top_grain_dex", "top_grain_dex"), ("best_grain_dex", "best_grain_dex"),
                      ("sharpness", "sharpness")):
        m[name], m[name + "_std"] = _mean(q[col]), _std(q[col])
    m["top_is_best_pct"] = 100.0 * float(q["top_is_best"].mean())
    m["mean_mode_count"] = float(q["n_modes"].mean())
    m["multimodal_pct"] = 100.0 * float((q["n_modes"] > 1).mean())
    comp, grain = p[p["kind"] == "comp"], p[p["kind"] == "grain"]
    m["comp_pit_w2"], m["grain_pit_w2"] = pit_cvm(comp["pit"]), pit_cvm(grain["pit"])
    m["n_present"], m["n_grain"] = len(comp), len(grain)
    m["n_trace"] = int((comp["truth"] <= TRACE).sum())
    for s in STAGES:
        m[f"{s}_ms"] = 1e3 * float(q[f"{s}_s"].mean())
    return m


def text_hist(u, width=40):
    """PIT values -> lines of a text histogram."""
    if len(u) == 0:
        return ["    (no scored components)"]
    counts, _ = np.histogram(np.asarray(u), bins=PIT_BINS, range=(0.0, 1.0))
    peak = max(int(counts.max()), 1)
    lines = [f"    [{b / PIT_BINS:.1f},{(b + 1) / PIT_BINS:.1f}) {counts[b]:6d} "
             + "#" * int(round(width * counts[b] / peak)) for b in range(PIT_BINS)]
    return lines + [f"    (flat reference ≈ {len(u) / PIT_BINS:.0f} per bin; U-shape = overconfident)"]


def report(title, m, p):
    """Print one split's Table 1 numbers, PIT histograms and timings."""
    line = "=" * 66
    print(f"\n{line}\n{title} ({m['n_spectra']} spectra)\n{line}")
    print("  per-solution accuracy (joint comp+grain BIC-GMM modes)")
    for label, key in (("Top-mode comp TV", "top_tv"), ("Best-mode comp TV", "best_tv"),
                       ("Top-mode grain dex", "top_grain_dex"), ("Best-mode grain dex", "best_grain_dex")):
        print(f"    {label:>24s} : {m[key]:8.4f} ± {m[key + '_std']:.4f}")
    print(f"    {'Top = Best (%)':>24s} : {m['top_is_best_pct']:8.1f}")
    print(f"    {'Mean mode count':>24s} : {m['mean_mode_count']:8.2f}")
    print(f"    {'Multimodal (%)':>24s} : {m['multimodal_pct']:8.1f}")

    print(f"\n  calibration ({m['n_present']} present components, {m['n_grain']} grains)")
    print(f"    {'comp PIT W2':>24s} : {m['comp_pit_w2']:8.4f}")
    print(f"    {'grain PIT W2':>24s} : {m['grain_pit_w2']:8.4f}")
    print(f"    {'sharpness':>24s} : {m['sharpness']:8.4f} ± {m['sharpness_std']:.4f}")

    comp, grain = p[p["kind"] == "comp"], p[p["kind"] == "grain"]
    trace = comp["truth"] <= TRACE
    for name, u in (("composition, all present components", comp["pit"]),
                    (f"composition, trace only (≤{TRACE:.0%})", comp.loc[trace, "pit"]),
                    (f"composition, non-trace only (>{TRACE:.0%})", comp.loc[~trace, "pit"]),
                    ("grain (log10 µm)", grain["pit"])):
        print(f"\n  PIT histogram: {name}")
        print("\n".join(text_hist(u)))

    print("\n  inference cost per spectrum (ms)")
    for s in STAGES:
        print(f"    {s:>14s} : {m[f'{s}_ms']:8.1f}")
    print(f"    {'total':>14s} : {sum(m[f'{s}_ms'] for s in STAGES):8.1f}\n{line}")


def pit_figure(rows, path):
    """rows: list of (title, {split: PIT values}) -> one figure row each, one column per split."""
    fig, axes = plt.subplots(len(rows), len(SPLITS), figsize=(3.5 * len(SPLITS), 2.75 * len(rows)),
                             sharex=True, sharey="row", squeeze=False)
    for r, (title, by_split) in enumerate(rows):
        for c, (split, label) in enumerate(SPLITS.items()):
            u = np.asarray(by_split[split])
            counts, edges = np.histogram(u, bins=PIT_BINS, range=(0.0, 1.0))
            ax = axes[r, c]
            ax.stairs(counts / max(len(u), 1), edges, color="#4C72B0", linewidth=1.5)
            ax.axhline(1.0 / PIT_BINS, color="red", ls="--", lw=1.0)
            ax.set_title(f"{title} — {label}", fontsize=10)
        axes[r, 0].set_ylabel("Frequency")
    for ax in axes[-1]:
        ax.set_xlabel("PIT value")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = load(SIZE, PROJECT_ROOT / CHECKPOINT, device)

    # dataset
    data = SpectraDataset(PROJECT_ROOT / DATASET)
    out = RESULTS_DIR / f"{SIZE}_{Path(DATASET).stem}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)

    # one untimed pass, so the first timed spectrum doesn't pay one-off setup costs
    lams, values, _, _ = next(data.get("test"))
    modes(*model.posterior(values[0].numpy(), lams[0].numpy(), seed=SEED).sample(10))

    queries, pits, metrics = [], [], {}
    for split, label in SPLITS.items():
        records, pit_rows = [], []
        n = len(data.splits[split])
        for lams, values, target, grains in data.get(split, noise="none", shuffle=False):
            for i in range(len(values)):
                record, rows = score(model, lams[i], values[i], target[i], grains[i])
                rows = [dict(query=len(records), **r) for r in rows]
                records.append(dict(query=len(records), **record))
                pit_rows += rows
                print(f"  {split} {len(records)}/{n}", end="\r")
        q = pd.DataFrame(records).assign(split=split)
        p = pd.DataFrame(pit_rows, columns=["query", "kind", "component", "truth", "pit"]).assign(split=split)
        metrics[split] = summarize(q, p)
        report(f"{label} ({split})", metrics[split], p)
        queries.append(q)
        pits.append(p)

    queries, pits = pd.concat(queries, ignore_index=True), pd.concat(pits, ignore_index=True)
    queries.to_parquet(out / "queries.parquet")
    pits.to_parquet(out / "pits.parquet")
    (out / "metrics.json").write_text(json.dumps(
        {"size": SIZE, "checkpoint": CHECKPOINT, "dataset": DATASET, "n_samples": N_SAMPLES,
         "noise": "none", "metrics": metrics}, indent=1))

    comp, grain = pits[pits["kind"] == "comp"], pits[pits["kind"] == "grain"]
    trace = comp["truth"] <= TRACE
    by_split = lambda df: {s: df.loc[df["split"] == s, "pit"] for s in SPLITS}
    pit_figure([("Composition", by_split(comp)), ("Grain (log µm)", by_split(grain))],
               out / "pit_histograms.png")
    pit_figure([(f"Trace (≤{TRACE:.0%})", by_split(comp[trace])),
                (f"Non-trace (>{TRACE:.0%})", by_split(comp[~trace]))],
               out / "pit_trace_histograms.png")
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
