"""Parquet-backed sample store: build the corpus once, then read rows by id.

Build:  python3.10 -m src.data_utils.store   (writes manifest.parquet + spectra.parquet)

``manifest.parquet`` holds one metadata row per spectrum; ``spectra.parquet`` holds the ragged
arrays. ``mixture`` + ``proportion`` ids together identify a composition, and every spectrum
sharing one is a duplicate group (the split unit, see ``splits.py``).
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config
from src.data_utils.components import Component, K
from src.data_utils.processing import more_shkuratov, shkuratov
from src.data_utils.spectrum_sample import SpectrumSample

SOURCES = {"shkuratov": shkuratov, "more_shkuratov": more_shkuratov}

PROPORTION_DECIMALS = 6   # round before hashing so float noise can't split a true duplicate group

ARRAY_COLUMNS = ("lams", "values", "target", "grains")


def mixture_id(target) -> str:
    """Component-set id: sorted present-component names joined by '+' (proportions/grain ignored)."""
    return "+".join(sorted(Component.decode(int(i)) for i in np.nonzero(target)[0]))


def proportion_id(target) -> str:
    """Proportion id: the present components' proportions, in component-index order, rounded.

    Orthogonal to ``mixture_id`` (which components vs. how much of each)."""
    active = np.nonzero(target)[0]
    vals = np.round(np.asarray(target, dtype=np.float64)[active], PROPORTION_DECIMALS)
    return "|".join(f"{v:.{PROPORTION_DECIMALS}f}" for v in vals)


def build(rebuild=False):
    if config.MANIFEST_PATH.exists() and not rebuild:
        return
    rows, lams, values, targets, grains = [], [], [], [], []
    for source, module in SOURCES.items():
        for i, s in enumerate(module.process()):
            sid = f"{source}:{i:06d}"
            rows.append(dict(id=sid, source=source, synthetic=s.synthetic,
                             mixture=mixture_id(s.target), proportion=proportion_id(s.target),
                             n_components=int(np.count_nonzero(s.target)),
                             n_channels=len(s.values), lam_min=float(s.lams.min()), lam_max=float(s.lams.max())))
            lams.append(s.lams); values.append(s.values); targets.append(s.target); grains.append(s.grains)

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(config.MANIFEST_PATH, index=False)
    f32 = pa.list_(pa.float32())
    pq.write_table(pa.table({
        "id": manifest["id"].tolist(),
        "synthetic": manifest["synthetic"].tolist(),
        "lams": pa.array(lams, f32), "values": pa.array(values, f32),
        "target": pa.array(targets, f32), "grains": pa.array(grains, f32),
    }), config.SPECTRA_PATH)
    print(f"Built {len(manifest)} samples -> {config.MANIFEST_PATH.name} + {config.SPECTRA_PATH.name}")


def load_manifest() -> pd.DataFrame:
    return pd.read_parquet(config.MANIFEST_PATH)


class Payload:
    """Columnar view of ``spectra.parquet``: O(1) row slicing without one object per spectrum.

    Each list<float32> column is kept as (flat values, offsets [N+1]); row i is
    ``values[offsets[i]:offsets[i+1]]``. ``target`` and ``grains`` always have K entries per row.
    """

    def __init__(self, table: pa.Table):
        self.ids = table.column("id").to_pylist()
        self.synthetic = np.asarray(table.column("synthetic").to_pylist(), dtype=bool)
        self.cols = {n: self._unpack(table.column(n)) for n in ARRAY_COLUMNS}
        self._index = {sid: i for i, sid in enumerate(self.ids)}

    @staticmethod
    def _unpack(col):
        arr = col.combine_chunks()
        return arr.values.to_numpy(zero_copy_only=False), arr.offsets.to_numpy()

    def __len__(self):
        return len(self.ids)

    def rows(self, ids) -> np.ndarray:
        """ids -> int row positions."""
        return np.fromiter((self._index[sid] for sid in ids), dtype=np.int64, count=len(ids))

    def row(self, i: int) -> dict:
        """Row position -> dict of lams, values, target, grains (float32 views) and synthetic."""
        out = {n: v[o[i]:o[i + 1]] for n, (v, o) in self.cols.items()}
        out["synthetic"] = bool(self.synthetic[i])
        return out

    def lengths(self, rows) -> np.ndarray:
        """Row positions -> channel count per row."""
        o = self.cols["values"][1]
        return o[np.asarray(rows) + 1] - o[np.asarray(rows)]

    def targets(self, rows) -> tuple:
        """Row positions -> (targets [n, K], grains [n, K]) float32."""
        rows = np.asarray(rows)
        return (self.cols["target"][0].reshape(-1, K)[rows],
                self.cols["grains"][0].reshape(-1, K)[rows])


_payload = None


def load_payload(ids=None) -> Payload:
    """The full corpus (read once, cached), or just ``ids`` (read fresh, not cached)."""
    global _payload
    if ids is not None:
        return Payload(pq.read_table(config.SPECTRA_PATH, filters=[("id", "in", list(ids))]))
    if _payload is None:
        _payload = Payload(pq.read_table(config.SPECTRA_PATH))
    return _payload


def load_samples(ids, payload=None) -> list:
    """ids -> list[SpectrumSample] in the same order (each carries its ``sample_id``)."""
    payload = load_payload() if payload is None else payload
    out = []
    for sid, i in zip(ids, payload.rows(ids)):
        r = payload.row(i)
        out.append(SpectrumSample(r["lams"], r["values"], r["target"], synthetic=r["synthetic"],
                                  grains=r["grains"], sample_id=sid))
    return out


if __name__ == "__main__":
    build(rebuild=True)
