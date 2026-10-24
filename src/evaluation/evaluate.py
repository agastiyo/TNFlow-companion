"""Score a trained TNFlow on held-out spectra: per-solution accuracy, calibration, timing.

Every query gets ``n_samples`` posterior draws and a joint comp+grain BIC-GMM mode fit, so each
mode is a complete solution. ``evaluate_split`` returns two DataFrames:

    queries  one row per spectrum: mode count, top/best-mode TV and grain dex, sharpness, timings
    pits     one row per scored (spectrum, component): kind ("comp" or "grain"), truth, PIT value

New breakdowns (per material, duplicated vs unique spectra, by mixture size) are a groupby on
these, not new evaluation code. ``summarize`` turns them into the Table 1 numbers.

Reproducibility: query choice, noise and sampling are seeded. One ``noise_rng`` is shared, in
order, by the warmup spectrum, the test queries and the OOD queries, and ``model.posterior``
reseeds torch before every query.
"""

import time

import numpy as np
import pandas as pd
import torch

from src.config import EvalConfig
from src.data_utils.spectrum_sample import SpectrumSample
from src.data_utils import store
from src.evaluation.metrics import energy_score, grain_log_error, pit, pit_cvm, tv_distance
from src.evaluation.posterior import grain_draws, modes

STAGES = ("encode", "sample", "modes")


# --------------------------------------------------------------------------- queries

def choose_queries(ids, n, seed) -> list:
    """Up to n ids drawn without replacement (seeded), in draw order."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    return [ids[i] for i in idx]


def noised(sample, noise, rng) -> SpectrumSample:
    """A noised copy of a synthetic sample (real spectra pass through unchanged)."""
    if not sample.synthetic:
        return sample
    values = noise(sample.values[None, :], sample.lams[None, :], rng)[0]
    return SpectrumSample(sample.lams, values, sample.target, sample.synthetic, sample.grains,
                          sample_id=sample.sample_id)


def make_queries(ids, noise, noise_rng, payload=None) -> list:
    """ids -> noised SpectrumSamples, noised in order from ``noise_rng``."""
    return [noised(s, noise, noise_rng) for s in store.load_samples(ids, payload)]


def warmup(model, sample, cfg: EvalConfig):
    """One untimed pass so the first timed query doesn't pay one-off setup costs."""
    c, g = model.posterior(sample.values, sample.lams, seed=cfg.seed).sample(10)
    modes(c, g)


# --------------------------------------------------------------------------- one query

def _mode_grain_dex(mode, truth_g):
    g_mode = mode["grain_um"]
    ok = torch.isfinite(g_mode) & torch.isfinite(truth_g) & (truth_g > 0) & (g_mode > 0)
    if not ok.any():
        return float("nan")
    return float(grain_log_error(g_mode[ok], truth_g[ok]).mean())


def score_query(model, sample, cfg: EvalConfig):
    """One labelled spectrum -> (query record dict, list of PIT row dicts)."""
    device = next(model.parameters()).device
    target = torch.as_tensor(sample.target, dtype=torch.float32, device=device)
    truth_g = torch.as_tensor(sample.grains, dtype=torch.float32, device=device)
    present = target > 0

    t0 = time.perf_counter()
    post = model.posterior(sample.values, sample.lams, seed=cfg.seed)
    t1 = time.perf_counter()
    comp, grain = post.sample(cfg.n_samples)
    t2 = time.perf_counter()
    found = modes(comp, grain)
    t3 = time.perf_counter()

    # per-solution accuracy: the top (highest-weight) mode and the mode closest to the truth
    tvs = [tv_distance(m["composition"], target) for m in found]
    best = int(np.argmin(tvs))
    _, spread = energy_score(comp, target)
    record = dict(
        sample_id=sample.sample_id, n_modes=len(found),
        top_tv=tvs[0], best_tv=tvs[best],
        top_grain_dex=_mode_grain_dex(found[0], truth_g),
        best_grain_dex=_mode_grain_dex(found[best], truth_g),
        top_is_best=best == 0, sharpness=spread,
        encode_s=t1 - t0, sample_s=t2 - t1, modes_s=t3 - t2,
    )

    # calibration: PIT of the truth under each present component's marginal ...
    pits = [dict(sample_id=sample.sample_id, kind="comp", component=c,
                 truth=float(target[c]), pit=pit(comp[:, c], target[c]))
            for c in present.nonzero().flatten().tolist()]
    # ... and under its log-grain marginal, where the truth is known and the draws contain it
    scorable = present & torch.isfinite(truth_g) & (truth_g > 0)
    for c in scorable.nonzero().flatten().tolist():
        g = grain_draws(grain, c)
        if g.numel():
            pits.append(dict(sample_id=sample.sample_id, kind="grain", component=c,
                             truth=float(truth_g[c]),
                             pit=pit(torch.log10(g), torch.log10(truth_g[c]))))
    return record, pits


def evaluate_split(model, queries, cfg: EvalConfig, label=""):
    """list[SpectrumSample] -> (queries DataFrame, pits DataFrame), in query order."""
    records, pit_rows = [], []
    for i, s in enumerate(queries, 1):
        record, pits = score_query(model, s, cfg)
        records.append(record)
        pit_rows.extend(pits)
        print(f"    {label} {i}/{len(queries)}", end="\r")
    print()
    pit_cols = ["sample_id", "kind", "component", "truth", "pit"]
    return pd.DataFrame(records), pd.DataFrame(pit_rows, columns=pit_cols)


# --------------------------------------------------------------------------- Table 1

def _nanmean(xs):
    finite = [x for x in xs if np.isfinite(x)]
    return float(np.mean(finite)) if finite else float("nan")


def _nanstd(xs):
    finite = [x for x in xs if np.isfinite(x)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")


def summarize(queries: pd.DataFrame, pits: pd.DataFrame, cfg: EvalConfig) -> dict:
    """Per-query and PIT tables -> Table 1 metrics (means ± std over queries) + the PIT lists."""
    out = {}
    for name, col in (("Top-mode comp TV", "top_tv"), ("Best-mode comp TV", "best_tv"),
                      ("Top-mode grain dex", "top_grain_dex"), ("Best-mode grain dex", "best_grain_dex")):
        out[name] = _nanmean(queries[col].tolist())
        out[name + " std"] = _nanstd(queries[col].tolist())

    top_is_best = queries["top_is_best"].tolist()
    mc = queries["n_modes"].to_numpy()
    out["Top = Best (%)"] = 100.0 * sum(top_is_best) / len(top_is_best) if top_is_best else float("nan")
    out["Mean mode count"] = float(mc.mean())
    out["Multimodal (%)"] = 100.0 * float((mc > 1).sum()) / len(mc) if len(mc) else float("nan")

    comp = pits[pits["kind"] == "comp"]
    grain = pits[pits["kind"] == "grain"]
    trace = comp["truth"] <= cfg.trace_thresh
    out["comp PIT W2"] = pit_cvm(comp["pit"].tolist())
    out["grain PIT W2"] = pit_cvm(grain["pit"].tolist())
    out["sharpness"] = _nanmean(queries["sharpness"].tolist())
    out["sharpness std"] = _nanstd(queries["sharpness"].tolist())
    out.update({
        "comp_pit": comp["pit"].tolist(),
        "comp_pit_trace": comp.loc[trace, "pit"].tolist(),
        "comp_pit_nontrace": comp.loc[~trace, "pit"].tolist(),
        "grain_pit": grain["pit"].tolist(),
        "n_present": len(comp), "n_grain": len(grain),
        "n_trace": int(trace.sum()), "n_nontrace": int((~trace).sum()),
    })
    return out


def timings(queries: pd.DataFrame) -> dict:
    """Per-stage wall times (s) per query, keyed by STAGES."""
    return {s: queries[f"{s}_s"].tolist() for s in STAGES}
