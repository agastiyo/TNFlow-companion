"""Grouped train / val / test / ood splits, saved once to ``data/splits/<name>.parquet``.

The split unit is a duplicate group: every spectrum sharing a composition (the same ``mixture`` and
``proportion`` ids, see ``store``) lands in the same split, so near-identical spectra can't leak
between splits. Whole component sets are held out first as the group-holdout OOD split; the rest
is split 80 / 10 / 10 into train / val / test (``make_splits``).

Build:  python3.10 -m src.data_utils.splits [--name v2] [--force]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.data_utils import store

SPLITS = ("train", "val", "test", "ood")


def group_ids(manifest: pd.DataFrame) -> pd.Series:
    """Duplicate-group key per row: ``<mixture>/<proportion>``."""
    return manifest["mixture"] + "/" + manifest["proportion"]


def _group_shuffle_split(groups: pd.Series, frac: float, seed: int):
    """Positions of (rest, held) rows, holding out ~``frac`` of the groups."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    rest, held = next(gss.split(np.zeros(len(groups)), groups=groups.to_numpy()))
    return rest, held


def make_splits(manifest: pd.DataFrame, cfg: config.SplitConfig = config.SplitConfig()) -> pd.DataFrame:
    """manifest -> DataFrame [id, split] with train / val / test / ood, deterministic from cfg.seed.

    1. hold out ``cfg.n_ood_mixtures`` component sets, drawn at random (seeded) -> ood
    2. take ``test_frac`` of the remaining groups -> test
    3. take ``val_frac`` (of the same pool) of the groups left after that -> val; the rest is train

    ``GroupShuffleSplit`` counts its fraction in groups, not rows; groups hold 1-875 rows (mean 1.5),
    so the row fractions come out close to, not exactly, the configured ones.
    """
    rows = manifest[manifest["source"].isin(cfg.sources)]
    rng = np.random.default_rng(cfg.seed)
    ood_mixtures = rng.choice(sorted(set(rows["mixture"])), cfg.n_ood_mixtures, replace=False)

    is_ood = rows["mixture"].isin(ood_mixtures).to_numpy()
    pool = rows[~is_ood]
    groups = group_ids(pool)

    rest, test = _group_shuffle_split(groups, cfg.test_frac, cfg.seed)
    train, val = _group_shuffle_split(groups.iloc[rest], cfg.val_frac / (1.0 - cfg.test_frac), cfg.seed)

    split = np.empty(len(pool), dtype=object)
    split[test] = "test"
    split[rest[val]] = "val"
    split[rest[train]] = "train"
    out = pd.concat([
        pd.DataFrame({"id": pool["id"].to_numpy(), "split": split}),
        pd.DataFrame({"id": rows.loc[is_ood, "id"].to_numpy(), "split": "ood"}),
    ], ignore_index=True)
    out["split"] = pd.Categorical(out["split"], categories=SPLITS)
    return out


def check_splits(splits: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Raise AssertionError unless every id, every group, and every OOD mixture sits in one split."""
    assert splits["id"].is_unique, "an id appears more than once"
    assert set(splits["split"].astype(str)) <= set(SPLITS), "unknown split name"
    m = manifest.set_index("id").loc[splits["id"], ["mixture", "proportion"]]
    m = m.assign(split=splits["split"].astype(str).to_numpy(), group=group_ids(m).to_numpy())
    leaked = m.groupby("group")["split"].nunique()
    assert (leaked == 1).all(), f"{int((leaked > 1).sum())} duplicate groups span two splits"
    ood_mix = set(m.loc[m["split"] == "ood", "mixture"])
    assert not (set(m.loc[m["split"] != "ood", "mixture"]) & ood_mix), "an OOD mixture leaks"


def save_splits(splits: pd.DataFrame, name: str, force: bool = False):
    path = config.SPLITS_DIR / f"{name}.parquet"
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass force=True to overwrite")
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(path, index=False)
    return path


def load_splits(name: str, check: bool = True) -> dict:
    """data/splits/<name>.parquet -> {split: [ids]} in file order (only the splits present)."""
    df = pd.read_parquet(config.SPLITS_DIR / f"{name}.parquet")
    if check:
        check_splits(df, store.load_manifest())
    split = df["split"].astype(str).to_numpy()
    ids = df["id"].to_numpy()
    return {s: ids[split == s].tolist() for s in SPLITS if (split == s).any()}


def summary(splits: dict) -> str:
    n = sum(len(v) for v in splits.values())
    return " | ".join(f"{s} {len(v):,} ({len(v) / n:.1%})" for s, v in splits.items())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", default="v2")
    p.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = p.parse_args()

    manifest = store.load_manifest()
    df = make_splits(manifest)
    check_splits(df, manifest)
    path = save_splits(df, args.name, force=args.force)
    print(f"{path.name}: {summary(load_splits(args.name, check=False))}")


if __name__ == "__main__":
    main()
