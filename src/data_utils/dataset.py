"""PyTorch data pipeline: Dataset of spectra, resolution-bucketed batches, DataLoader.

    ds = SpectrumDataset(ids, noise=build_noise(cfg.noise), mode="fresh")
    for batch in make_loader(ds, batch_size=724, shuffle=True, seed=SEED):   # one epoch
        batch.lams, batch.values, batch.target, batch.grains                 # [B, L], [B, L], [B, K], [B, K]

Spectra come at several resolutions (466, 477, ... up to 9601 channels). Self-attention needs
equal-length rows, and padding to the longest would make every batch 9601 wide, so each batch is
drawn from one resolution bucket (``BucketBatchSampler``) instead.
"""

from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src import config
from src.data_utils import store

ATTN_BL2_BUDGET = 300_000_000  # self-attention ~ O(B·L²); caps rows/batch on long buckets (L up to 9601)

NOISE_MODES = ("none", "fresh", "frozen")


class Batch(NamedTuple):
    lams: torch.Tensor      # [B, L] µm
    values: torch.Tensor    # [B, L] reflectance
    target: torch.Tensor    # [B, K] proportions
    grains: torch.Tensor    # [B, K] µm, NaN where unknown


class SpectrumDataset(Dataset):
    """Spectra by manifest id, with noise on synthetic rows only.

    mode="fresh"   new noise every time a spectrum is read, so every epoch differs (training)
    mode="frozen"  one noise draw per spectrum, made once from a per-index seed, so the set is the
                   same every epoch and in every run (val / OOD scored during training)
    mode="none"    clean spectra
    """

    def __init__(self, ids, noise=None, mode="none", seed=config.SEED, payload=None):
        if mode not in NOISE_MODES:
            raise ValueError(f"mode must be one of {NOISE_MODES}, got {mode!r}")
        if mode != "none" and noise is None:
            raise ValueError(f"mode={mode!r} needs a noise model")
        self.payload = store.load_payload() if payload is None else payload
        self.ids = list(ids)
        self.rows = self.payload.rows(self.ids)
        self.lengths = self.payload.lengths(self.rows)
        self.noise, self.mode = noise, mode
        self.rng = np.random.default_rng(seed)
        self._frozen = {}
        if mode == "frozen":
            for i in range(len(self)):
                r = self.payload.row(self.rows[i])
                if r["synthetic"]:
                    rng = np.random.default_rng([seed, i])
                    self._frozen[i] = self.noise(r["values"], r["lams"], rng).astype(np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.payload.row(self.rows[i])
        values = r["values"]
        if self.mode == "frozen":
            values = self._frozen.get(i, values)
        elif self.mode == "fresh" and r["synthetic"]:
            values = self.noise(values, r["lams"], self.rng).astype(np.float32)
        return r["lams"], values, r["target"], r["grains"]

    def __getitems__(self, indices):
        """One batch of equal-length spectra, stacked, with "fresh" noise drawn for the whole batch
        in one vectorized call (same noise model as ``__getitem__``, far less per-row overhead).
        ``DataLoader`` calls this instead of ``__getitem__`` when it exists.
        Returns: list [lams, values, target, grains] of stacked ndarrays.
        """
        rows = [self.payload.row(self.rows[i]) for i in indices]
        lams, values, target, grains = (np.stack([r[n] for r in rows]) for n in store.ARRAY_COLUMNS)
        if self.mode == "frozen":
            values = np.stack([self._frozen.get(i, r["values"]) for i, r in zip(indices, rows)])
        elif self.mode == "fresh":
            synthetic = np.array([r["synthetic"] for r in rows])
            if synthetic.any():
                noised = self.noise(values, lams, self.rng).astype(np.float32)
                values = np.where(synthetic[:, None], noised, values)
        return [lams, values, target, grains]

    def targets(self):
        """(targets [N, K], grains [N, K]) for the whole set, without reading the spectra."""
        return self.payload.targets(self.rows)


class BucketBatchSampler(Sampler):
    """Batches of dataset indices, each from one resolution bucket.

    Rows per batch: ``min(batch_size, ATTN_BL2_BUDGET // L²)``, so long buckets get small batches.
    With ``shuffle``, every pass shuffles inside each bucket and then shuffles the batch order, so
    an epoch doesn't run one bucket at a time; the seeded ``generator`` makes it reproducible.
    """

    def __init__(self, lengths, batch_size, shuffle=False, generator=None,
                 attn_budget=ATTN_BL2_BUDGET):
        self.lengths = np.asarray(lengths)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.generator, self.attn_budget = generator, attn_budget
        self.buckets = {int(L): np.nonzero(self.lengths == L)[0] for L in np.unique(self.lengths)}

    def _cap(self, length):
        return max(1, min(self.batch_size, self.attn_budget // length ** 2))

    def _perm(self, n):
        return torch.randperm(n, generator=self.generator).numpy()

    def __iter__(self):
        batches = []
        for length, idx in self.buckets.items():
            if self.shuffle:
                idx = idx[self._perm(len(idx))]
            cap = self._cap(length)
            batches.extend(idx[s:s + cap].tolist() for s in range(0, len(idx), cap))
        if self.shuffle:
            batches = [batches[j] for j in self._perm(len(batches))]
        return iter(batches)

    def __len__(self):
        return sum(-(-len(idx) // self._cap(length)) for length, idx in self.buckets.items())


def collate(items) -> Batch:
    """Stacked arrays from ``__getitems__`` (or a list of per-row tuples with equal L) -> Batch."""
    if isinstance(items[0], np.ndarray) and items[0].ndim == 2:     # already stacked
        return Batch(*(torch.from_numpy(a) for a in items))
    return Batch(*(torch.from_numpy(np.stack(col)) for col in zip(*items)))


def make_loader(dataset, batch_size, shuffle=False, seed=config.SEED) -> DataLoader:
    """Standard ``torch.utils.data.DataLoader`` over ``BucketBatchSampler`` batches.

    ``num_workers=0``: the corpus is already in memory, and worker processes would each need
    their own noise stream.
    """
    generator = torch.Generator().manual_seed(seed)
    sampler = BucketBatchSampler(dataset.lengths, batch_size, shuffle=shuffle, generator=generator)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0)
