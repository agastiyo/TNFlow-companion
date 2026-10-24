"""R-2: per-epoch reshuffling, one resolution per batch, noise on synthetic rows only."""

import numpy as np
import pytest

from src.config import NoiseConfig
from src.data_utils.dataset import BucketBatchSampler, SpectrumDataset, make_loader
from src.data_utils.noise import MultAddNoise, build_noise
from tests.conftest import make_payload


def test_batches_have_one_length(payload):
    ds = SpectrumDataset(payload.ids, payload=payload)
    for batch in make_loader(ds, batch_size=2, shuffle=True):
        assert batch.values.shape == batch.lams.shape
        assert batch.target.shape[0] == batch.values.shape[0]


def test_every_row_once_per_epoch(payload):
    sampler = BucketBatchSampler(SpectrumDataset(payload.ids, payload=payload).lengths,
                                 batch_size=2, shuffle=True)
    seen = sorted(i for b in sampler for i in b)
    assert seen == list(range(len(payload)))
    assert len(sampler) == len(list(iter(sampler)))


def test_long_bucket_is_capped():
    sampler = BucketBatchSampler(np.full(10, 9601), batch_size=724)
    assert all(len(b) <= 3 for b in sampler)   # 300M // 9601² = 3


def test_order_changes_between_epochs_and_is_seeded():
    payload = make_payload(lengths=(466,) * 64)
    ds = SpectrumDataset(payload.ids, payload=payload)
    order = lambda loader: [b.values[:, 0].tolist() for b in loader]
    a, b = make_loader(ds, 8, shuffle=True, seed=1), make_loader(ds, 8, shuffle=True, seed=1)
    epoch1, epoch2 = order(a), order(a)
    assert epoch1 != epoch2                      # reshuffled each epoch
    assert epoch1 == order(b)                    # same seed, same first epoch


def test_noise_only_on_synthetic_rows():
    payload = make_payload(lengths=(466, 466), synthetic=(True, False))
    ds = SpectrumDataset(payload.ids, noise=build_noise(NoiseConfig()), mode="fresh", payload=payload)
    clean = SpectrumDataset(payload.ids, payload=payload)
    assert not np.array_equal(ds[0][1], clean[0][1])
    assert np.array_equal(ds[1][1], clean[1][1])


def test_fresh_noise_differs_frozen_noise_does_not(payload):
    noise = build_noise(NoiseConfig())
    fresh = SpectrumDataset(payload.ids, noise=noise, mode="fresh", payload=payload)
    frozen = SpectrumDataset(payload.ids, noise=noise, mode="frozen", payload=payload)
    again = SpectrumDataset(payload.ids, noise=noise, mode="frozen", payload=payload)
    assert not np.array_equal(fresh[0][1], fresh[0][1])
    assert np.array_equal(frozen[0][1], frozen[0][1])
    assert np.array_equal(frozen[0][1], again[0][1])


def test_batched_read_matches_row_reads():
    payload = make_payload(lengths=(466,) * 6, synthetic=(True, False) * 3)
    noise = build_noise(NoiseConfig())
    for mode in ("none", "frozen"):
        ds = SpectrumDataset(payload.ids, noise=noise, mode=mode, payload=payload)
        batch = ds.__getitems__([4, 1, 2])
        for col, stacked in enumerate(batch):
            assert np.array_equal(stacked, np.stack([ds[i][col] for i in (4, 1, 2)]), equal_nan=True)
    fresh = SpectrumDataset(payload.ids, noise=noise, mode="fresh", payload=payload)
    clean = SpectrumDataset(payload.ids, payload=payload)
    values = fresh.__getitems__([0, 1])[1]
    assert not np.array_equal(values[0], clean[0][1])      # synthetic row: noised
    assert np.array_equal(values[1], clean[1][1])          # real row: untouched


def test_noise_matches_formula():
    # Same draws, in the same order, as the documented formula.
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    values = np.random.default_rng(0).uniform(0.1, 0.5, (3, 50))
    lams = np.tile(np.linspace(0.4, 5.0, 50), (3, 1))
    sig0 = rng_b.normal(0.0, 0.04, (3, 1))
    eps_mult = rng_b.standard_normal(values.shape) * sig0 * np.sqrt(lams / 0.9)
    a = rng_b.normal(0.0, 0.04, (3, 1))
    eps_add = rng_b.standard_normal(values.shape) * a * np.median(values, axis=1, keepdims=True)
    expected = values * (1.0 + eps_mult) + eps_add
    assert np.array_equal(MultAddNoise()(values, lams, rng_a), expected)


def test_unknown_noise_kind():
    with pytest.raises(ValueError):
        build_noise(NoiseConfig(kind="nope"))
