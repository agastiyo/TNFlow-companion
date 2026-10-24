# TNFlow

Companion repository for the paper: [TNFlow: Amortized Posterior Inference for Trans-Neptunian Object Surface Composition](https://arxiv.org/abs/2609.04305)

This repository contains the code and pretrained model needed to reproduce the results reported in the paper. It is not an official release of TNFlow.

TNFlow is a transformer and conditional neural spline flow that learns to invert the Shkuratov radiative transfer model for TNO surface composition. Given a reflectance spectrum, it returns a full multimodal posterior over simplex-valid compositions and grain sizes in under a second on CPU. The model is trained entirely on synthetic spectra (463,275 Shkuratov mixtures over 20 surface components) and generalizes to unseen combinations of known components. Qualitative tests on real JWST spectra reveal a sim-to-real gap that future work will address.

## Requirements

Python 3.10+ is required. CUDA is optional (CPU inference takes ~0.7 s per object).

```
pip install $(grep -v -e '^#' -e 'cana' requirements.txt)
pip install --no-deps -r requirements.txt
```

The first line installs everything except the simulator packages. The second adds [CANA](https://github.com/cana-asteroids/cana) and [SDOC](https://github.com/cana-asteroids/sdoc), pinned to commits, without their declared dependencies (numba, dask and the deprecated `sklearn` package, none of which this code needs; `src/simulator/forward.py` stubs out numba). They are needed only to generate new spectra (`src/simulator/`); evaluation and training use the pre-built corpus.

On macOS, if evaluation hangs during mode fitting, run it with `OMP_NUM_THREADS=1`. PyTorch and scikit-learn can load two different OpenMP runtimes, which can deadlock scikit-learn's k-means.

## Data

Small data files are included in the repository:

| File | Description | Size |
|------|-------------|------|
| `data/manifest.parquet` | Metadata: spectrum IDs, source, mixture and proportion codes, channel counts | 11 MB |
| `data/splits/v2.parquet` | Train / val / test / OOD split | 3 MB |
| `data/JWST_DiSCO-TNOs/` | Real JWST spectra for the transfer experiment (Appendix C) | 1.6 MB |
| `data/other_objects/` | Pholus, Quaoar, Iapetus spectra for inference demo | 180 KB |

The training corpus is hosted on Zenodo:

| File | Description | Size | Download |
|------|-------------|------|----------|
| `spectra.parquet` | Pre-built synthetic training corpus (463,275 spectra) | 827 MB | [Zenodo](https://doi.org/10.5281/zenodo.22240289) |

Download `spectra.parquet` and place it in `data/`:

```
data/
├── spectra.parquet        ← download from Zenodo
├── manifest.parquet
├── splits/
├── JWST_DiSCO-TNOs/
└── other_objects/
```

## Pre-trained Model

The retrained checkpoint (trained on the `v2` split, selected on val) will be released here once training finishes. Training writes it to `runs/v2/checkpoint.pt`.

## Reproduction

To reproduce the results from a trained checkpoint:

```
bash reproduce.sh [path/to/checkpoint.pt]   # default runs/v2/checkpoint.pt
```

This runs two scripts in sequence:

1. **`python3.10 -m src.pipeline.test`** evaluates the model on the in-distribution test split (5,000 spectra) and the group-holdout OOD split (374 spectra), with 500 posterior draws per query. It produces the Table 1 metrics and the PIT calibration figures, on the split the checkpoint was trained with.

2. **`python3.10 -m src.pipeline.apply_jwst`** applies the model to real JWST DiSCo-TNOs spectra (bowl / double-dip / cliff exemplars). It produces the appendix figure and the mode report.

Both scripts use the checkpoint at `runs/v2/checkpoint.pt` by default (`--checkpoint` to change it) and write to `runs/`. The noise added to the synthetic query spectra is seeded, so repeated runs on the same machine give the same numbers.

## Splits

Splits are saved files, `data/splits/<name>.parquet` (columns `id`, `split`), checked on every load. Every spectrum sharing a composition (same mixture and proportions) is in the same split. The OOD split holds out 8 whole component sets, drawn at random from the seed; the rest is split 80 / 10 / 10 into train / val / test.

| Split file | train | val | test | ood |
|------------|-------|-----|------|-----|
| `v2` | 370,369 | 48,376 | 44,156 | 374 |

Checkpoints are selected on val only.

To rebuild a split file: `python3.10 -m src.data_utils.splits --name v2 --force`.

## Training

To train the model from scratch on the synthetic corpus:

```
python3.10 -m src.pipeline.train --run-name v2
```

Training requires `data/spectra.parquet`, `data/manifest.parquet` and a split file (default `v2`). Batches are reshuffled and re-noised every epoch. The checkpoint with the lowest validation loss is saved to `runs/<run-name>/checkpoint.pt`, alongside `history.json` (per-epoch curves) and `config.json`. To evaluate it: `python3.10 -m src.pipeline.test --checkpoint runs/<run-name>/checkpoint.pt`. The paper's model was trained on a single NVIDIA RTX 6000 Ada GPU (48 GB).

For a quick check that training runs: `python3.10 -m src.pipeline.train --run-name smoke --epochs 1 --limit 3000`.

To rebuild the parquet corpus from raw Shkuratov model output (not required for training from the provided corpus):

```
python3.10 -m src.data_utils.store
```

## Tests

```
pip install pytest
python3.10 -m pytest
```

The tests check the split file, the data pipeline, the model, and the training loop. They need `data/manifest.parquet` and the split files, but not `spectra.parquet`.

## Results

Paper (V1) checkpoint. Per-solution error on the test and group-holdout OOD splits (500 posterior draws per query, joint composition + grain BIC-GMM mode search). Top mode is the highest-weight mode; best mode is the mode closest to the target composition in total-variation distance.

| Metric | Test (*n* = 5,000) | OOD (*n* = 374) |
|--------|-------------------|-----------------|
| Top-mode comp. TV | 0.149 ± 0.125 | 0.156 ± 0.091 |
| Best-mode comp. TV | 0.121 ± 0.098 | 0.132 ± 0.080 |
| Top-mode grain (dex) | 0.242 ± 0.212 | 0.279 ± 0.162 |
| Best-mode grain (dex) | 0.253 ± 0.201 | 0.283 ± 0.161 |
| Top mode = best (%) | 67.7 | 60.4 |
| Mean mode count | 2.50 | 2.64 |
| Multimodal (%) | 88.9 | 91.7 |
| Posterior sharpness | 0.203 ± 0.106 | 0.228 ± 0.066 |

## Source File Index

### `src/model/`

| File | Description |
|------|-------------|
| `tnflow.py` | Top-level model: tokenizer, transformer, composition CNF, and grain flow; training objective (`nll`) and posterior sampling |
| `transformer.py` | Pre-norm transformer encoder with learned query pooling |
| `tokenizers.py` | Raw (λ, R) spectrum to per-channel tokens with positional encoding |
| `cnf.py` | Conditional normalizing flow over the K-component composition simplex |
| `grain.py` | Per-component grain-size posterior (shared 1-D NSF) |

### `src/evaluation/`

| File | Description |
|------|-------------|
| `evaluate.py` | Score held-out spectra: one record per query and per scored component, summarized into Table 1 |
| `posterior.py` | Summaries of posterior draws: credible intervals, detection ranking, BIC-GMM mode search |
| `metrics.py` | Scores against the truth (top-N recall, TV distance, grain dex, PIT, energy score) |
| `report.py` | Text output: Table 1 report, PIT histograms, per-spectrum posterior summaries |
| `plots.py` | All figures: PIT histograms, posterior marginals, JWST exemplars, attention |

### `src/pipeline/` (entry points)

| File | Description |
|------|-------------|
| `train.py` | Training loop: combined composition + grain NLL, checkpoint selected on val |
| `test.py` | Evaluation on test and OOD: per-solution accuracy, calibration (PIT), and timing |
| `apply_jwst.py` | Apply TNFlow to real JWST DiSCo-TNOs spectra (appendix figure + mode report) |
| `inference.py` | Interactive single-spectrum inference with plots and printed summaries |
| `checkpoint.py` | Checkpoint load / save |

### `src/data_utils/`

| File | Description |
|------|-------------|
| `splits.py` | Grouped train / val / test / OOD splits, saved to and checked from `data/splits/` |
| `dataset.py` | PyTorch `Dataset`, resolution-bucketed batch sampler, and `DataLoader` |
| `noise.py` | Noise augmentation models for synthetic spectra |
| `store.py` | Parquet-backed corpus: build, and read rows by id |
| `components.py` | Component material to integer code mapping (K=20 enum) |
| `spectrum_sample.py` | SpectrumSample: wavelength, reflectance, composition target, grains |

### `src/data_utils/processing/`

| File | Description |
|------|-------------|
| `shkuratov.py` | Shkuratov RT model grid (bestfit files) to synthetic SpectrumSamples |
| `more_shkuratov.py` | CANA/SDOC synthetic spectra to SpectrumSamples |
| `jwst.py` | JWST DiSCo-TNOs PRISM spectra to unlabelled SpectrumSamples |
| `other_objects.py` | Miscellaneous published spectra (Pholus, Quaoar, Iapetus) to SpectrumSamples |

### `src/simulator/`

| File | Description |
|------|-------------|
| `forward.py` | The Shkuratov forward model (CANA + SDOC optical constants): composition to spectrum |
| `generate.py` | Generate a synthetic corpus with random mixtures |
| `plot_one.py` | Generate and plot a single Shkuratov spectrum |

### `src/`

| File | Description |
|------|-------------|
| `config.py` | Project paths, device selection, and every shared default (split, noise, model, training, evaluation) |

## License

Code: MIT

Data (`spectra.parquet`, `manifest.parquet`): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
