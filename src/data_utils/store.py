"""Build the training corpus once: every bestfit / generated Shkuratov spectrum -> spectra.parquet.

Build:  python -m src.data_utils.store

``spectra.parquet`` holds one row per spectrum: id (``<source>:<index>``), synthetic, and the
ragged arrays lams, values, target, grains. ``src/dataset.py`` reads it.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import DATA_DIR
from src.data_utils.processing import more_shkuratov, shkuratov

SOURCES = {"shkuratov": shkuratov, "more_shkuratov": more_shkuratov}
SPECTRA_PATH = DATA_DIR / "spectra.parquet"


def build():
    ids, synthetic, lams, values, targets, grains = [], [], [], [], [], []
    for source, module in SOURCES.items():
        for i, s in enumerate(module.process()):
            ids.append(f"{source}:{i:06d}")
            synthetic.append(s.synthetic)
            lams.append(s.lams); values.append(s.values); targets.append(s.target); grains.append(s.grains)

    f32 = pa.list_(pa.float32())
    pq.write_table(pa.table({
        "id": ids, "synthetic": synthetic,
        "lams": pa.array(lams, f32), "values": pa.array(values, f32),
        "target": pa.array(targets, f32), "grains": pa.array(grains, f32),
    }), SPECTRA_PATH)
    print(f"Built {len(ids)} samples -> {SPECTRA_PATH.name}")


if __name__ == "__main__":
    build()
