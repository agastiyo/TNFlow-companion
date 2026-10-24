"""Generate synthetic reflectance spectra with the Shkuratov forward model (``forward.py``).

For every spectrum the components, their proportions, per-component grain sizes and the (shared)
porosity are drawn at random; each spectrum is written as its own text file under ``OUT_DIR``,
in the format read by ``src/data_utils/processing/more_shkuratov.py``::

    POROSITY
    <value>
    COMPONENTS
    <name> <proportion> <grain_size_um>
    ...
    SPECTRUM
    <wavelength_um> <reflectance>
    ...

Batches: ``N_PER_BATCH`` spectra for each mixture size in ``N_COMPONENTS_BATCHES``. Note that
these committed settings do not reproduce the published corpus, whose ``more_shkuratov`` part has
299,990 spectra with 2-7 components.

Run with::

    python3.10 -m src.simulator.generate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import DATA_DIR, SEED
from src.simulator.forward import COMPONENT_LIBRARY, WAVELENGTHS, load_optical_constants, simulate

# --- configuration -------------------------------------------------------------------------

OUT_DIR = DATA_DIR / "train" / "more_shkuratov_080326"

N_PER_BATCH = 50_000
N_COMPONENTS_BATCHES = (2, 3, 4)

# Grain size (um): log-uniform in [5, 100], drawn independently per component.
GRAIN_MIN_UM, GRAIN_MAX_UM = 5.0, 100.0
# Porosity: uniform in [0.0, 0.7], one value per spectrum.
POROSITY_MIN, POROSITY_MAX = 0.0, 0.7
# Floor on the smallest mixture proportion so no component is effectively absent.
MIN_PROPORTION = 0.05
# Max redraws when a parameter combo yields a non-finite spectrum (Shkuratov albedo formula
# has no real solution -- aux < 1 -- for some highly absorbing / porous mixtures).
MAX_REDRAWS = 50

# -------------------------------------------------------------------------------------------


def draw_proportions(rng: np.random.Generator, k: int) -> np.ndarray:
    """Draw ``k`` proportions summing to 1, each at least ``MIN_PROPORTION`` (Dirichlet)."""
    while True:
        p = rng.dirichlet(np.ones(k))
        if p.min() >= MIN_PROPORTION:
            return p


def draw_grainsizes(rng: np.random.Generator, k: int) -> np.ndarray:
    """Draw ``k`` grain sizes (um), log-uniform in [GRAIN_MIN_UM, GRAIN_MAX_UM]."""
    log_lo, log_hi = np.log10(GRAIN_MIN_UM), np.log10(GRAIN_MAX_UM)
    return 10.0 ** rng.uniform(log_lo, log_hi, size=k)


def write_spectrum(path: Path, names, proportions, grains, porosity, reflectance) -> None:
    """Write one spectrum file in the COMPONENTS / SPECTRUM text format."""
    lines = [
        "POROSITY",
        f"{porosity:.6f}",
        "COMPONENTS",
    ]
    for name, prop, grain in zip(names, proportions, grains):
        lines.append(f"{name} {prop:.6f} {grain:.4f}")
    lines.append("SPECTRUM")
    for wl, refl in zip(WAVELENGTHS, reflectance):
        lines.append(f"{wl:.2f} {refl:.6f}")
    path.write_text("\n".join(lines) + "\n")


def generate() -> None:
    """Generate all batches and write files to ``OUT_DIR``."""
    rng = np.random.default_rng(SEED)
    ocs = load_optical_constants()
    names_all = list(COMPONENT_LIBRARY.keys())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total, redraws = 0, 0
    for k in N_COMPONENTS_BATCHES:
        print(f"Generating {N_PER_BATCH} spectra with {k} components ...", flush=True)
        for i in range(N_PER_BATCH):
            # Redraw the parameters (and, as a last resort, the component set) until the model
            # returns a fully finite spectrum. NaNs arise where the Shkuratov albedo has no
            # real solution; such combos are physically degenerate and are simply resampled.
            for attempt in range(MAX_REDRAWS):
                names = list(rng.choice(names_all, size=k, replace=False))
                proportions = draw_proportions(rng, k)
                grains = draw_grainsizes(rng, k)
                porosity = float(rng.uniform(POROSITY_MIN, POROSITY_MAX))
                reflectance = simulate(names, proportions, grains, porosity, ocs=ocs)
                if np.isfinite(reflectance).all():
                    break
                redraws += 1
            else:
                raise RuntimeError(
                    f"Could not draw a finite {k}-component spectrum in {MAX_REDRAWS} attempts"
                )

            total += 1
            path = OUT_DIR / f"shk_{k}c_{i:05d}.txt"
            write_spectrum(path, names, proportions, grains, porosity, reflectance)

            if (i + 1) % 1000 == 0:
                print(f"  {k}c: {i + 1}/{N_PER_BATCH}", flush=True)

    print(f"Done: wrote {total} spectra to {OUT_DIR} ({redraws} redraws)", flush=True)


if __name__ == "__main__":
    generate()
