"""The spectra dataset: one parquet in, train / val / test / ood splits, batches out.

    data = SpectraDataset("data/spectra.parquet")
    for lams, values, target, grains in data.get("train", noise="mult_add", shuffle=True):
        ...   # tensors [B, L] µm, [B, L] reflectance, [B, K] proportions, [B, K] µm (NaN = unknown)

Splits. The split unit is a duplicate group: every spectrum with the same composition (the same
components in the same proportions) lands in the same split, so near-identical spectra can't leak
between splits. First N_OOD_MIXTURES whole component sets are held out as the ood split; the rest
of the groups are split 80 / 10 / 10 into train / val / test. The split is rebuilt from SEED every
time, so train and test always see the same one (for spectra.parquet it is the v2 split:
370,369 / 48,376 / 44,156 / 374).

Batches. Spectra come at several resolutions (466, 477, ... up to 9601 channels). Self-attention
needs equal-length rows, so every batch holds spectra of one length, and long spectra get small
batches (self-attention memory grows as B·L²).
"""

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.config import SEED
from src.data_utils.components import Component, K

BATCH_SIZE = 724
ATTN_BUDGET = 300_000_000     # rows per batch are capped at ATTN_BUDGET // L²

N_OOD_MIXTURES = 8            # whole component sets held out as the ood split
TEST_FRAC = 0.1               # fractions of the non-ood groups
VAL_FRAC = 0.1

# mult_add noise: R -> R·(1 + ε_mult) + ε_add. Per spectrum σ0 ~ N(0, SIGMA0_MAX) and
# a ~ N(0, ADD_MAX); per channel ε_mult ~ N(0, 1)·σ0·sqrt(λ / LAM_REF) (grows to the red) and
# ε_add ~ N(0, 1)·a·median(R).
SIGMA0_MAX = 0.04
ADD_MAX = 0.04
LAM_REF = 0.9                 # µm


def mixture_id(target) -> str:
    """Component-set id: sorted present-component names joined by '+'."""
    return "+".join(sorted(Component.decode(int(i)) for i in np.nonzero(target)[0]))


def proportion_id(target) -> str:
    """The present components' proportions, in component order, rounded to 6 decimals."""
    active = np.nonzero(target)[0]
    vals = np.round(np.asarray(target, dtype=np.float64)[active], 6)
    return "|".join(f"{v:.6f}" for v in vals)


def mult_add_noise(values, lams, rng):
    """values, lams [B, L] -> noised values [B, L] float32."""
    b = values.shape[0]
    sig0 = rng.normal(0.0, SIGMA0_MAX, (b, 1))
    eps_mult = rng.standard_normal(values.shape) * sig0 * np.sqrt(lams / LAM_REF)
    a = rng.normal(0.0, ADD_MAX, (b, 1))
    eps_add = rng.standard_normal(values.shape) * a * np.median(values, axis=1, keepdims=True)
    return (values * (1.0 + eps_mult) + eps_add).astype(np.float32)


def _group_shuffle_split(groups, frac):
    """Positions (rest, held) of groups, holding out ~frac of the distinct groups."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=SEED)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


class SpectraDataset:
    """Every spectrum of a parquet file (columns lams, values, target, grains), split four ways."""

    def __init__(self, path):
        table = pq.read_table(path, columns=["lams", "values", "target", "grains"])
        cols = {}
        for name in table.column_names:
            arr = table.column(name).combine_chunks()
            cols[name] = arr.values.to_numpy(zero_copy_only=False), arr.offsets.to_numpy()
        self.lams, offsets = cols["lams"]
        self.values, self.offsets = cols["values"]
        assert np.array_equal(offsets, self.offsets), "lams and values differ in length"
        self.lengths = np.diff(self.offsets)
        self.target = cols["target"][0].reshape(-1, K)
        self.grains = cols["grains"][0].reshape(-1, K)
        self.splits = self.make_splits()
        self.rng = np.random.default_rng(SEED)
        print(f"{path}: " + " | ".join(f"{s} {len(r):,}" for s, r in self.splits.items()))

    def make_splits(self):
        """{"train", "val", "test", "ood"} -> row indices, rebuilt from SEED."""
        mixture = np.array([mixture_id(t) for t in self.target], dtype=object)
        group = np.array([m + "/" + proportion_id(t) for m, t in zip(mixture, self.target)],
                         dtype=object)

        ood_mixtures = np.random.default_rng(SEED).choice(
            sorted(set(mixture)), N_OOD_MIXTURES, replace=False)
        is_ood = np.isin(mixture, ood_mixtures)
        pool = np.nonzero(~is_ood)[0]
        rest, test = _group_shuffle_split(group[pool], TEST_FRAC)
        train, val = _group_shuffle_split(group[pool[rest]], VAL_FRAC / (1.0 - TEST_FRAC))
        splits = {"train": pool[rest[train]], "val": pool[rest[val]], "test": pool[test],
                  "ood": np.nonzero(is_ood)[0]}

        # no duplicate group in two splits, and no ood component set outside ood
        names = list(splits)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not set(group[splits[a]]) & set(group[splits[b]]), f"{a}/{b} share a group"
        assert not set(mixture[splits["ood"]]) & set(mixture[pool]), "an ood mixture leaks"
        return splits

    def get(self, split, noise="none", shuffle=False):
        """Batches of one split, as (lams, values, target, grains) tensors.

        split: "train", "val", "test" or "ood". noise: "none" or "mult_add" (a fresh draw every
        call). shuffle: new spectrum and batch order every call; otherwise the same order every time.
        """
        if noise not in ("none", "mult_add"):
            raise ValueError(f"noise must be 'none' or 'mult_add', got {noise!r}")
        rows = self.splits[split]
        lengths = self.lengths[rows]
        batches = []
        for length in np.unique(lengths):
            bucket = rows[lengths == length]
            if shuffle:
                bucket = self.rng.permutation(bucket)
            size = max(1, min(BATCH_SIZE, ATTN_BUDGET // int(length) ** 2))
            batches += [bucket[i:i + size] for i in range(0, len(bucket), size)]
        if shuffle:
            batches = [batches[i] for i in self.rng.permutation(len(batches))]

        for batch in batches:
            idx = self.offsets[batch, None] + np.arange(self.lengths[batch[0]])   # [B, L]
            lams, values = self.lams[idx], self.values[idx]
            if noise == "mult_add":
                values = mult_add_noise(values, lams, self.rng)
            yield (torch.from_numpy(lams), torch.from_numpy(values),
                   torch.from_numpy(self.target[batch]), torch.from_numpy(self.grains[batch]))
