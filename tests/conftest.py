"""Shared fixtures: a tiny in-memory corpus, so most tests don't need data/spectra.parquet."""

import numpy as np
import pyarrow as pa
import pytest

from src.data_utils.components import K
from src.data_utils.store import Payload


def make_payload(lengths=(466, 466, 466, 477, 477, 9601), synthetic=None, seed=0) -> Payload:
    """Payload with one spectrum per entry of ``lengths`` (ids s0, s1, ...)."""
    rng = np.random.default_rng(seed)
    n = len(lengths)
    synthetic = [True] * n if synthetic is None else list(synthetic)
    lams = [np.linspace(0.35, 5.0, L, dtype=np.float32) for L in lengths]
    values = [rng.uniform(0.05, 0.5, L).astype(np.float32) for L in lengths]
    targets = []
    for _ in range(n):
        t = np.zeros(K, np.float32)
        t[rng.choice(K, 2, replace=False)] = (0.7, 0.3)
        targets.append(t)
    grains = [np.where(t > 0, rng.uniform(5.0, 100.0, K), np.nan).astype(np.float32) for t in targets]
    f32 = pa.list_(pa.float32())
    return Payload(pa.table({
        "id": [f"s{i}" for i in range(n)], "synthetic": synthetic,
        "lams": pa.array(lams, f32), "values": pa.array(values, f32),
        "target": pa.array(targets, f32), "grains": pa.array(grains, f32),
    }))


@pytest.fixture
def payload():
    return make_payload()
