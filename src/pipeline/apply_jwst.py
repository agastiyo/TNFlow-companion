"""Apply TNFlow to representative real JWST DiSCo-TNOs spectra (appendix figure + mode report).

Runs the trained model on one exemplar of each DiSCo spectral group (bowl / double-dip / cliff)
and reports **every** BIC-GMM mode, not just the top one, alongside each component's detection
frequency across the posterior draws. Real spectra are unlabelled, so nothing here is scored --
this is a qualitative illustration of the sim-to-real gap, not an evaluation.

Exemplars are the highest-band-area object of each group among those with a high PCA silhouette
coefficient in Pinilla-Alonso et al. (2025) Table 1, chosen so each group's defining species is
tested where its signal is strongest.

Run from the project root:  python3.10 -m src.pipeline.apply_jwst [--checkpoint PATH] [--out DIR]
"""

import argparse
import json
from pathlib import Path

import torch

from src import config
from src.data_utils.components import Component
from src.data_utils.processing import jwst
from src.evaluation import plots, report
from src.evaluation.posterior import modes
from src.model.grain import PRESENT_THRESH
from src.pipeline.checkpoint import load_model

N_SAMPLES = 500
NORM_LAM, NORM_WIN = 0.9, 0.05
REPORT_THRESH = 0.01          # components below this in a mode are omitted from the printed table

# designation -> (DiSCo group, band-area note from Pinilla-Alonso et al. 2025 Table 1)
TARGETS = {
    "1998SN165": ("Bowl",       "strongest H2O Fresnel band area of the bowl exemplars (0.0075)"),
    "2007UK126": ("Double-dip", "highest CO2 band area in the DiSCo sample (0.0050)"),
    "2004PF115": ("Cliff",      "highest CH3OH band area in the DiSCo sample (0.0074)"),
}

NAMES = {c.value: c.name for c in Component}


def analyze(model, sample):
    """One spectrum -> dict with every mode's composition + per-component detection frequency."""
    comp, grain = model.posterior(sample.values, sample.lams, seed=config.SEED).sample(N_SAMPLES)

    detect = (comp > PRESENT_THRESH).float().mean(dim=0)   # [K] fraction of draws containing each
    found = modes(comp, grain)

    out_modes = []
    for m in found:
        c = m["composition"].detach().cpu()
        g = m["grain_um"].detach().cpu()
        present = (c > REPORT_THRESH).nonzero().flatten().tolist()
        out_modes.append({
            "weight": float(m["weight"]),
            "spread": float(m["spread"]),
            "components": [
                {"code": int(i), "name": NAMES[int(i)], "proportion": float(c[i]),
                 "grain_um": (float(g[i]) if torch.isfinite(g[i]) else None)}
                for i in sorted(present, key=lambda j: -float(c[j]))
            ],
        })
    return {
        "n_modes": len(found),
        "modes": out_modes,
        "detection_frequency": {NAMES[i]: float(detect[i]) for i in range(len(detect))},
    }


def main(checkpoint=config.CHECKPOINT, out=None):
    model, _ = load_model(checkpoint, config.get_device())
    out = Path(out or config.RUNS_DIR / f"jwst_{Path(checkpoint).stem}")
    out.mkdir(parents=True, exist_ok=True)

    by_desig = dict(zip(jwst.designations(), jwst.process()))
    entries, results = [], {}
    for desig, (group, note) in TARGETS.items():
        s = by_desig[desig]
        entries.append((desig, group, s))
        print(f"\n{'=' * 72}\n{group} — {desig}\n  selected: {note}"
              f"\n  spectrum: {s.lams.min():.3f}–{s.lams.max():.3f} µm, {len(s.lams)} channels")
        r = analyze(model, s)
        r.update(group=group, selection_note=note,
                 lam_min=float(s.lams.min()), lam_max=float(s.lams.max()),
                 n_channels=int(len(s.lams)))
        results[desig] = r
        report.print_mode_report(r)

    (out / "jwst_apply.json").write_text(json.dumps(results, indent=1))
    print(f"\n  results -> {out / 'jwst_apply.json'}")
    plots.jwst_exemplars(entries, out / "jwst_exemplars.png", NORM_LAM, NORM_WIN)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=str(config.CHECKPOINT))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    main(args.checkpoint, args.out)
