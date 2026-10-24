"""Summaries of posterior draws. Everything takes the two matrices from SimplexPosterior.sample().

comp  [N, K] simplex rows (each sums to 1)
grain [N, K] µm, NaN wherever that draw does not contain the component

All torch, so it stays on whatever device the model is on. Only the BIC-GMM mode fit (scikit-learn)
drops to CPU/numpy, and it does so internally. Printing lives in ``report.py``, figures in ``plots.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import SEED
from src.model.grain import GRAIN_EPS, PRESENT_THRESH

MIN_GRAIN_DRAWS = 1      # draws needed to report a grain; the frequency is reported alongside


def detection_score(comp_draws, dim=0):
    """comp_draws [S, ..., K] -> [..., K] score ranking components for Top-N / Top-5.

    Primary key is the **detection frequency**: the fraction of posterior draws in which the
    component clears ``PRESENT_THRESH`` -- i.e. is present enough that ``GrainFlow.sample_grid``
    gives it a grain draw instead of NaN. Top-N asks *which components are present*, so the
    marginal detection probability is the right quantity; posterior mean abundance conflates "how
    often" with "how much" and ranks a trace species detected in every draw below one proposed
    rarely but large.

    Frequencies are quantised to 1/S and every never-detected component sits at exactly 0, so ties
    are pervasive -- and ranking by argsort breaks ties by enum index, which would hand an
    undetected component a lucky rank. Mean abundance enters as a secondary key scaled to stay
    strictly below one frequency step (max 1/(S+1) < 1/S), so it can only order *within* a tie
    group, never override the frequency.
    """
    s = comp_draws.shape[dim]
    freq = (comp_draws > PRESENT_THRESH).to(comp_draws.dtype).mean(dim)
    return freq + comp_draws.mean(dim) / (s + 1)


def order_by_median(comp, n=None):
    """comp [N, K] -> component indices by descending median proportion (top n, or all)."""
    order = comp.median(dim=0).values.argsort(descending=True)
    return order if n is None else order[:n]


def order_by_detection(comp, n=None):
    """comp [N, K] -> component indices by descending ``detection_score`` (top n, or all).

    The same ranking key as the Top-N metric, so the "top" components a human reads off the
    inference display are the ones the reported metric scores.
    """
    order = detection_score(comp).argsort(descending=True)
    return order if n is None else order[:n]


def credible(x):
    """x [N] -> (median, (lo68, hi68), (lo90, hi90)) floats; NaNs if empty."""
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan"), (float("nan"),) * 2, (float("nan"),) * 2
    q = torch.quantile(x, torch.tensor([0.05, 0.16, 0.50, 0.84, 0.95], device=x.device))
    lo90, lo68, med, hi68, hi90 = q.tolist()
    return med, (lo68, hi68), (lo90, hi90)


def grain_draws(grain, c, min_draws=MIN_GRAIN_DRAWS):
    """grain [N, K], c: int -> that component's finite positive draws (empty if under min_draws)."""
    g = grain[:, c]
    g = g[torch.isfinite(g) & (g > 0)]
    return g if g.numel() >= min_draws else g[:0]


def grain_median(grain, min_draws=MIN_GRAIN_DRAWS):
    """grain [N, K] -> [K] median µm over the draws containing each component; NaN where too few."""
    med = torch.nanmedian(grain, dim=0).values
    enough = (~torch.isnan(grain)).sum(dim=0) >= min_draws
    return med.masked_fill(~enough, float("nan"))


def grain_frequency(grain):
    """grain [N, K] -> [K] fraction of draws containing each component."""
    return (~torch.isnan(grain)).to(grain.dtype).mean(dim=0)


def _bic_gmm(X, max_k, reg_covar=1e-4, seed=SEED, patience=2, min_scan=4):
    """Fit GMMs with 1, 2, ... components and keep the lowest-BIC one.

    Full covariances, one init each (``random_state=seed``). The scan stops early once BIC has
    failed to improve ``patience`` times in a row, but never before ``min_scan`` components.
    """
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    best, best_bic, rises = None, float("inf"), 0
    for k in range(1, max_k + 1):
        gm = GaussianMixture(k, covariance_type="full", reg_covar=reg_covar, random_state=seed)
        # One OpenMP thread for the fit: torch (libiomp5) and scikit-learn (libomp) load two
        # OpenMP runtimes, and scikit-learn's multi-threaded k-means init deadlocks next to torch.
        with threadpool_limits(limits=1, user_api="openmp"):
            gm.fit(X)
        if (bic := gm.bic(X)) < best_bic:
            best, best_bic, rises = gm, bic, 0
        else:
            rises += 1
            if patience is not None and k >= min_scan and rises >= patience:
                break
    return best


def modes(comp, grain, max_modes=None, reg_covar=1e-4, seed=SEED, patience=2, min_scan=4):
    """Joint comp+grain BIC-GMM: each mode is a complete solution (composition + grain sizes).

    Fits in [comp free coords | log10(grain)] space so modes capture the full joint posterior.
    NaN grains (absent components) are zero-filled before fitting.

    Returns list of dicts sorted by descending weight:
        {"weight", "composition" [K], "grain_um" [K] (NaN where absent), "spread"}
    """
    comp_free = comp[:, :-1].detach().cpu().numpy()
    log_g = torch.log10(grain.clamp(min=GRAIN_EPS)).detach().cpu()
    log_g = torch.where(torch.isfinite(log_g), log_g, torch.zeros_like(log_g)).numpy()
    X = np.concatenate([comp_free, log_g], axis=1)

    Kc = comp.shape[1]
    best = _bic_gmm(X, max_modes or Kc, reg_covar, seed, patience, min_scan)

    kf = comp_free.shape[1]
    out = []
    for j in best.weights_.argsort()[::-1]:
        mu_comp = torch.as_tensor(best.means_[j, :kf], dtype=comp.dtype, device=comp.device)
        full = torch.cat([mu_comp, (1.0 - mu_comp.sum()).reshape(1)]).clamp(min=0.0)
        full = full / full.sum()

        mu_log_g = torch.as_tensor(best.means_[j, kf:], dtype=comp.dtype, device=comp.device)
        grain_um = 10.0 ** mu_log_g
        present = full > PRESENT_THRESH
        grain_um = torch.where(present, grain_um,
                               torch.tensor(float("nan"), dtype=grain_um.dtype, device=grain_um.device))

        var = torch.as_tensor(best.covariances_[j].diagonal().copy(),
                              dtype=comp.dtype, device=comp.device)
        out.append({"weight": float(best.weights_[j]),
                    "composition": full,
                    "grain_um": grain_um,
                    "spread": float(var[:kf].clamp(min=0).mean().sqrt())})
    return out
