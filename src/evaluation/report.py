"""Text output: the Table 1 report, PIT histograms, and posterior summaries for one spectrum."""

import numpy as np
import torch

from src.config import SEED, EvalConfig
from src.data_utils.components import Component
from src.evaluation.evaluate import STAGES
from src.evaluation.posterior import (credible, grain_draws, grain_frequency, modes,
                                      order_by_detection)

ACCURACY_KEYS = ("Top-mode comp TV", "Best-mode comp TV",
                 "Top-mode grain dex", "Best-mode grain dex", "Top = Best (%)",
                 "Mean mode count", "Multimodal (%)")
CALIB_KEYS = ("comp PIT W2", "grain PIT W2", "sharpness")


# --------------------------------------------------------------------------- Table 1

def report(desc, m, timings, cfg: EvalConfig = EvalConfig()):
    """Print one split's ``summarize`` dict, PIT histograms and per-stage timings."""
    line = "=" * 66
    print("\n" + line)
    print(desc)
    print(line)

    print("  per-solution accuracy (joint comp+grain BIC-GMM modes)")
    for k in ACCURACY_KEYS:
        if "%" in k:
            print(f"    {k:>24s} : {m[k]:8.1f}")
        elif k + " std" in m:
            print(f"    {k:>24s} : {m[k]:8.4f} ± {m[k + ' std']:.4f}")
        else:
            print(f"    {k:>24s} : {m[k]:8.2f}")

    print(f"\n  calibration  ({m['n_present']} present components, {m['n_grain']} grains)")
    for k in CALIB_KEYS:
        if k + " std" in m:
            print(f"    {k:>24s} : {m[k]:8.4f} ± {m[k + ' std']:.4f}")
        else:
            print(f"    {k:>24s} : {m[k]:8.4f}")
    print("      (PIT W² 0 = uniform/calibrated; sharpness = mean pairwise draw dist)")

    t = cfg.trace_thresh
    print(f"\n  composition PIT: {m['n_trace']} trace (≤{t:.0%}), "
          f"{m['n_nontrace']} non-trace (>{t:.0%})")
    for title, key in (("all present components; want flat", "comp_pit"),
                       (f"trace only, ≤{t:.0%}", "comp_pit_trace"),
                       (f"non-trace only, >{t:.0%}", "comp_pit_nontrace")):
        print(f"\n  composition PIT histogram ({title})")
        for ln in pit_hist(m[key], cfg.pit_bins):
            print(ln)
    print("\n  grain PIT histogram (log10 µm, where present)")
    for ln in pit_hist(m["grain_pit"], cfg.pit_bins):
        print(ln)

    if timings and timings[STAGES[0]]:
        n = len(timings[STAGES[0]])
        per_query = np.zeros(n)
        print(f"\n  inference cost per query, mean ± std over {n} queries (ms)")
        for s in STAGES:
            a = np.asarray(timings[s]) * 1e3
            per_query += a
            print(f"    {s:>14s} : {a.mean():8.1f} ± {a.std():6.1f}")
        print(f"    {'TOTAL':>14s} : {per_query.mean():8.1f} ± {per_query.std():6.1f}"
              f"   ({per_query.sum() / 1e3:.1f}s total)")
    print(line)


def pit_hist(u, bins=10, width=40):
    """PIT values -> lines of a text histogram."""
    if not u:
        return ["    (no scored components)"]
    counts, _ = np.histogram(np.asarray(u), bins=bins, range=(0.0, 1.0))
    peak = max(int(counts.max()), 1)
    ideal = len(u) / bins
    lines = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bar = "#" * int(round(width * counts[b] / peak))
        lines.append(f"    [{lo:.1f},{hi:.1f}) {counts[b]:5d} {bar}")
    lines.append(f"    (flat reference ≈ {ideal:.0f} per bin; U-shape = overconfident)")
    return lines


# --------------------------------------------------------------------------- one spectrum

def print_marginals(comp, true=None, n=5):
    """comp [N, K] -> print median + 68%/90% CIs for the top n components. true [K] optional."""
    print(f"Composition posterior  (top {n} components by detection frequency)")
    print(f"  {'component':>14s}   {'median':>6s}   {'68% CI':^16s}   {'90% CI':^16s}")
    for c in order_by_detection(comp, n).tolist():
        med, (lo68, hi68), (lo90, hi90) = credible(comp[:, c])
        tag = f"   <- TRUE {true[c]:.3f}" if (true is not None and true[c] > 0) else ""
        print(f"  {Component.decode(c):>14s}   {med:6.3f}   "
              f"[{lo68:5.3f}, {hi68:5.3f}]   [{lo90:5.3f}, {hi90:5.3f}]{tag}")


def print_grains(comp, grain, true_grains=None, n=5):
    """comp/grain [N, K] -> print grain median + CIs for the top n components.

    ``freq`` = share of draws containing the component; a low freq means the grain is conditional
    on a rare branch of the composition posterior.
    """
    freq = grain_frequency(grain)
    print(f"Grain-size posterior (µm)  (top {n} components by detection frequency, where present)")
    print(f"  {'component':>14s}   {'freq':>6s}   {'median':>8s}   {'68% CI':^18s}   {'90% CI':^18s}")
    for c in order_by_detection(comp, n).tolist():
        med, (lo68, hi68), (lo90, hi90) = credible(grain_draws(grain, c))
        tag = (f"   <- TRUE {true_grains[c]:.1f}"
               if true_grains is not None and torch.isfinite(torch.as_tensor(true_grains[c])) else "")
        print(f"  {Component.decode(c):>14s}   {freq[c]:6.1%}   {med:8.1f}   "
              f"[{lo68:7.1f}, {hi68:7.1f}]   [{lo90:7.1f}, {hi90:7.1f}]{tag}")


def print_modes(comp, grain, true_comp=None, true_grains=None, max_modes=None, seed=SEED):
    """Print BIC-selected joint comp+grain modes, components above 1% each."""
    found = modes(comp, grain, max_modes=max_modes, seed=seed)
    print(f"Found {len(found)} joint mode(s) (BIC-selected, comp+grain) over {len(comp)} samples:")
    for r, m in enumerate(found, 1):
        w = m["composition"]
        g = m["grain_um"]
        parts = []
        for c in w.argsort(descending=True).tolist():
            if w[c] <= 0.01:
                continue
            s = f"{Component.decode(c)} {w[c]:.2f}"
            if torch.isfinite(g[c]):
                s += f" ({g[c]:.0f}µm)"
            parts.append(s)
        print(f"  {r}: wgt {m['weight']:.2f} | spr {m['spread']:.3f} | {', '.join(parts)}")
    if true_comp is not None:
        true_comp = torch.as_tensor(true_comp)
        parts = []
        for c in true_comp.argsort(descending=True).tolist():
            if true_comp[c] <= 0:
                continue
            s = f"{Component.decode(c)} {true_comp[c]:.2f}"
            if true_grains is not None:
                tg = torch.as_tensor(true_grains)
                if torch.isfinite(tg[c]) and tg[c] > 0:
                    s += f" ({tg[c]:.0f}µm)"
            parts.append(s)
        print(f"  (true: {', '.join(parts)})")


def describe_attention(lam, weights, n=10):
    """Human-readable listing of the top-n highest-attention wavelengths.

    Args:    lam, weights: ndarray [L] (from ``TNFlow.attention``); n: how many to list.
    Returns: str.
    """
    order = np.argsort(weights)[::-1][:n]
    lines = [f"Top {n} wavelengths by query-pooling attention weight (of {len(lam)} total):"]
    for i in order:
        lines.append(f"  {lam[i]:7.4f} µm   weight {weights[i]:.4f}")
    return "\n".join(lines)


def print_mode_report(r):
    """Print one ``apply_jwst.analyze`` result: every mode, then detection frequencies."""
    print(f"  {r['n_modes']} mode(s) found")
    for j, m in enumerate(r["modes"], 1):
        parts = ", ".join(f"{c['name']} {c['proportion']:.3f}" for c in m["components"])
        print(f"    mode {j} (w={m['weight']:.3f}): {parts}")
    top = sorted(r["detection_frequency"].items(), key=lambda kv: -kv[1])[:8]
    print("  detection frequency (top 8): " + ", ".join(f"{n} {f:.2f}" for n, f in top))
    never = [n for n, f in r["detection_frequency"].items() if f == 0.0]
    print(f"  never detected in any draw ({len(never)}): {', '.join(never)}")
