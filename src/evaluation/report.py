"""Text output: posterior summaries for one spectrum (used by inference.py and apply_jwst.py)."""

import numpy as np
import torch

from src.config import SEED
from src.data_utils.components import Component
from src.evaluation.posterior import (credible, grain_draws, grain_frequency, modes,
                                      order_by_detection)


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
