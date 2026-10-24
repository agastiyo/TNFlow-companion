"""Evaluate a checkpoint on the held-out test and OOD splits (Table 1, PIT calibration, timing).

Scores up to ``--n-test`` test spectra and every OOD spectrum, with 500 posterior draws each
(see ``src/evaluation/evaluate.py``). The validation split is never scored here.

Run from the project root:
    python3.10 -m src.pipeline.test                                   # runs/v2/checkpoint.pt
    python3.10 -m src.pipeline.test --checkpoint runs/<run>/checkpoint.pt

Writes to ``--out`` (default runs/eval_<checkpoint>_<splits>/): queries_<split>.parquet (one row
per spectrum), pits_<split>.parquet (one row per scored component), metrics.json, pit_data.json,
and the PIT figures.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from src import config
from src.data_utils import splits as splits_mod
from src.data_utils import store
from src.data_utils.noise import build_noise
from src.evaluation import evaluate, plots, report
from src.pipeline.checkpoint import load_model

HELD_OUT = {"test": "In-distribution", "ood": "OOD"}


def describe(split, queries):
    if split == "ood":
        n_mix = len({store.mixture_id(s.target) for s in queries})
        return f"Group-holdout OOD ({len(queries)} spectra, {n_mix} held-out mixtures)"
    return f"In-distribution held-out ({len(queries)} spectra)"


def save(out, split, queries, pits, manifest):
    """Per-query and PIT tables, with each spectrum's manifest metadata joined on."""
    meta = manifest.set_index("id")[["source", "mixture", "proportion", "n_components", "n_channels"]]
    queries.join(meta, on="sample_id").assign(split=split).to_parquet(out / f"queries_{split}.parquet")
    pits.assign(split=split).to_parquet(out / f"pits_{split}.parquet")


def main(checkpoint=config.CHECKPOINT, split_name=None, cfg=config.EvalConfig(), out=None):
    device = config.get_device()
    model, ckpt = load_model(checkpoint, device)
    print(f"  config: {ckpt['config']}")
    split_name = split_name or ckpt["splits"]
    out = Path(out or config.RUNS_DIR / f"eval_{Path(checkpoint).stem}_{split_name}")
    out.mkdir(parents=True, exist_ok=True)

    manifest = store.load_manifest()
    splits = splits_mod.load_splits(split_name)
    chosen = {s: evaluate.choose_queries(splits[s], cfg.n_test if s == "test" else len(splits[s]),
                                         cfg.seed)
              for s in HELD_OUT}
    warm_id = splits["test"][0]
    payload = store.load_payload(ids=[warm_id] + [i for ids in chosen.values() for i in ids])

    noise, noise_rng = build_noise(cfg.noise), np.random.default_rng(cfg.seed)
    evaluate.warmup(model, evaluate.make_queries([warm_id], noise, noise_rng, payload)[0], cfg)

    metrics, columns = {}, []
    for split, label in HELD_OUT.items():
        queries = evaluate.make_queries(chosen[split], noise, noise_rng, payload)
        desc = describe(split, queries)
        print(f"\n{desc}")
        q, pits = evaluate.evaluate_split(model, queries, cfg, split)
        m = evaluate.summarize(q, pits, cfg)
        report.report(desc, m, evaluate.timings(q), cfg)
        save(out, split, q, pits, manifest)
        metrics[split] = {k: v for k, v in m.items() if not isinstance(v, list)}
        columns.append((label, m))

    (out / "metrics.json").write_text(json.dumps(
        {"checkpoint": str(checkpoint), "splits": split_name,
         "eval_config": dataclasses.asdict(cfg), "metrics": metrics}, indent=1))
    pit_data = {}
    for label, m in columns:
        key = label.lower().replace("-", "").replace(" ", "_")
        for name in ("comp_pit", "grain_pit", "comp_pit_trace", "comp_pit_nontrace"):
            pit_data[f"{key}_{name}"] = m[name]
    (out / "pit_data.json").write_text(json.dumps(pit_data))

    t = cfg.trace_thresh
    plots.pit_grid([(label, m["comp_pit"], m["grain_pit"]) for label, m in columns],
                   ("Composition — {label}", "Grain (log µm) — {label}"),
                   out / "pit_histograms.png", cfg.pit_bins)
    plots.pit_grid([(label, m["comp_pit_trace"], m["comp_pit_nontrace"]) for label, m in columns],
                   (f"Trace (≤{t:.0%}) — {{label}}", f"Non-trace (>{t:.0%}) — {{label}}"),
                   out / "pit_trace_histograms.png", cfg.pit_bins)
    print(f"  results -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=str(config.CHECKPOINT))
    p.add_argument("--splits", default=None, help="split file name (default: the checkpoint's)")
    p.add_argument("--n-test", type=int, default=config.EvalConfig.n_test)
    p.add_argument("--n-samples", type=int, default=config.EvalConfig.n_samples)
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.checkpoint, args.splits,
         config.EvalConfig(n_test=args.n_test, n_samples=args.n_samples), args.out)
