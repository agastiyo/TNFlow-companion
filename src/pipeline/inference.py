"""Apply a checkpoint to a JWST / other-object / generated-Shkuratov spectrum: plot it, print
posterior + modes + grains.

Sources: JWST, other objects, and "Shkuratov (generated)" -- a fresh intimate-mixture spectrum built
from the MIXTURE / POROSITY block below (edit it here for a manual input).

Run from the project root:  python3.10 -m src.pipeline.inference
"""
#%%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.data_utils.components import Component, K
from src.data_utils.processing import jwst, other_objects
from src.data_utils.spectrum_sample import SpectrumSample
from src.evaluation import plots, report
from src.model.tnflow import load

warnings.filterwarnings('ignore')

SIZE = "medium"                     # must match the checkpoint
CHECKPOINT = "tnflow_checkpoint.pt"
N_SAMPLES = 5000

# --- Shkuratov (generated) source input: edit for a manual mixture ------------------------
# name -> (proportion, grain_size_um); names must be Component enum names, proportions ~sum to 1.
MIXTURE = {
    "H2O" : (0.35, 10.0),
    "AC" : (0.28, 45.0),
    "TTH" : (0.17, 23.0),
    "CO" : (0.2, 6.0)
}
POROSITY = 0.1
# ------------------------------------------------------------------------------------------

#%%
def choose(options, prompt="Select", n_cols=1):
    """Print options in a grid and prompt for an index.

    Args:    options: list[str]; prompt: input prompt text; n_cols: names per line.
    Returns: int index.
    """
    n_cols = max(1, min(n_cols, len(options)))
    width = max(len(o) for o in options) + 7
    n_rows = -(-len(options) // n_cols)  # ceil
    for r in range(n_rows):
        row = "".join(
            f"[{r * n_cols + c:2d}] {options[r * n_cols + c]}".ljust(width)
            for c in range(n_cols) if r * n_cols + c < len(options)
        )
        print(row.rstrip())
    raw = input(f"{prompt} [0-{len(options) - 1}] (default 0): ").strip()
    idx = int(raw) if raw else 0
    if not 0 <= idx < len(options):
        raise SystemExit(f"index {idx} out of range")
    return idx


#%%
class _ShkuratovSource:
    """Generate one Shkuratov intimate-mixture spectrum from the module-level MIXTURE / POROSITY.

    The forward model is imported lazily, so cana/sdoc load only when this source is selected and
    the JWST / other-object paths pay nothing. The generated sample carries the true composition +
    grains as its target.
    """
    _cache = None

    def _gen(self):
        if _ShkuratovSource._cache is None:
            from src.simulator.forward import WAVELENGTHS, simulate
            names = list(MIXTURE.keys())
            props = [MIXTURE[n][0] for n in names]
            grains = [MIXTURE[n][1] for n in names]
            refl = simulate(names, props, grains, POROSITY)
            label = ", ".join(f"{n} {p:.2f}" for n, p in zip(names, props))
            _ShkuratovSource._cache = (np.asarray(WAVELENGTHS, float), refl, label)
        return _ShkuratovSource._cache

    def designations(self):
        return [self._gen()[2]]

    def process(self):
        lams, refl, _ = self._gen()
        target = np.zeros(K, dtype=np.float32)
        grains = np.full(K, np.nan, dtype=np.float32)
        for n, (p, g) in MIXTURE.items():
            k = Component[n].value
            target[k], grains[k] = p, g
        return [SpectrumSample(lams, refl, target, synthetic=True, grains=grains)]


def main():
    model = load(SIZE, config.PROJECT_ROOT / CHECKPOINT)

    sources = [("JWST", jwst), ("Other objects", other_objects),
               ("Shkuratov (generated)", _ShkuratovSource())]
    print("Choose a spectrum source:")
    src_idx = choose([label for label, _ in sources], prompt="Select a source")
    label, module = sources[src_idx]

    names = module.designations()
    samples = module.process()

    print(f"\nChoose a {label} spectrum:")
    idx = choose(names, prompt="Select a spectrum", n_cols=4)
    sample, name = samples[idx], names[idx]
    print(f"\nSelected {name} (D={len(sample.values)})")

    plots.spectrum(sample, name)
    plt.show()

    attn_lam, attn_weights = model.attention(spec=sample.values, lams=sample.lams)
    print(f"\n===== {name}: query-pooling attention =====")
    print(report.describe_attention(attn_lam, attn_weights))
    plots.attention(sample, attn_lam, attn_weights, name)
    plt.show()

    comp, grain = model.posterior(sample.values, sample.lams, seed=config.SEED).sample(N_SAMPLES)
    labelled = bool(sample.target.any())
    true_comp = sample.target if labelled else None
    true_grains = sample.grains if labelled else None
    print(f"\n===== {name}: posterior (median, 68% & 90% credible intervals) =====")
    print()
    report.print_modes(comp, grain, true_comp=true_comp, true_grains=true_grains)
    print()
    report.print_marginals(comp, true=true_comp, n=10)
    print()
    report.print_grains(comp, grain, true_grains=true_grains, n=10)

    plots.hist_marginals(comp, n=3)
    plots.hist_grains(comp, grain, n=3)
    plt.show()


if __name__ == "__main__":
    main()

# %%
