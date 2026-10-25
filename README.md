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

The first line installs everything except the simulator packages. The second adds [CANA](https://github.com/cana-asteroids/cana) and [SDOC](https://github.com/cana-asteroids/sdoc), pinned to commits, without their declared dependencies (numba, dask and the deprecated `sklearn` package, none of which this code needs; `src/simulator/forward.py` stubs out numba). They are needed only to generate new spectra (`src/simulator/`); training and testing use the pre-built corpus.

## Data

Small data files are included in the repository:

| File | Description | Size |
|------|-------------|------|
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
├── JWST_DiSCO-TNOs/
└── other_objects/
```

To rebuild `spectra.parquet` from raw Shkuratov model output (not required): `python -m src.data_utils.store`.

## How the code works

Every script is configured by a short block of constants at its top. Edit the block, then run the script from the project root.

**Dataset** (`src/dataset.py`). `SpectraDataset("data/spectra.parquet")` loads the corpus and splits it into train / val / test / ood. `get(split, noise, shuffle)` returns that split's batches, with `noise` set to `"none"` or `"mult_add"`.

Every spectrum with the same composition (same components, same proportions) lands in the same split. The ood split holds out 8 whole component sets, drawn from the seed. The rest is split 80 / 10 / 10 into train / val / test. The split is rebuilt from the seed every time, so train and test always agree.

| Corpus | train | val | test | ood |
|--------|-------|-----|------|-----|
| `spectra.parquet` | 370,369 | 48,376 | 44,156 | 374 |

**Model** (`src/model/`). `TNFlow(size)` with `size` set to `"small"`, `"medium"` or `"large"`. The paper model is `"medium"`. `load(size, path)` builds a model and loads a checkpoint into it.

## Training

```
python -m src.train
```

Set the model size, an optional checkpoint to start from, the dataset path and the training settings at the top of `src/train.py`. Each epoch trains on the train split (shuffled, freshly noised), computes the loss on the val split (noised, not shuffled), and steps the learning rate on the val loss. The checkpoint is written to `checkpoints/<size>_<dataset>_<timestamp>.pt`. Every epoch it is rewritten with the weights from the best val epoch so far, plus the train and val loss curves. The paper's model was trained on a single NVIDIA RTX 6000 Ada GPU (48 GB).

## Testing

```
OMP_NUM_THREADS=1 python -m src.test
```

Set the model size, the checkpoint and the dataset path at the top of `src/test.py`. It scores every spectrum of the test and ood splits, clean (no noise) and in file order, with 500 posterior draws each. It prints the Table 1 metrics, the PIT calibration histograms and the per-spectrum timing.

It writes `results/<size>_<dataset>_<timestamp>/`:
- `metrics.json`
- `queries.parquet`: one row per spectrum
- `pits.parquet`: one row per scored component
- `pit_histograms.png` and `pit_trace_histograms.png`

The full test split takes several hours on CPU.

On macOS, if testing hangs during mode fitting, keep `OMP_NUM_THREADS=1`. PyTorch and scikit-learn can load two different OpenMP runtimes, which can deadlock scikit-learn's k-means.

## Reproduction

```
bash reproduce.sh
```

This runs `src.test` (Table 1 and the PIT figures), then `src.pipeline.apply_jwst`. `apply_jwst` applies the model to real JWST DiSCo-TNOs spectra, one bowl, one double-dip and one cliff exemplar, and writes the appendix figure and the mode report to `results/`. Both scripts use the checkpoint named at their top. The default is the paper checkpoint, `tnflow_checkpoint.pt`.

For interactive single-spectrum inference with plots: `python -m src.pipeline.inference`.

## Results

These are the paper (V1) numbers for the paper checkpoint, on noised queries and 5,000 test spectra. `src/test.py` now scores clean spectra over the full test split, so its numbers will differ.

Each metric is a per-solution error on the test and group-holdout OOD splits (500 posterior draws per query, joint composition + grain BIC-GMM mode search). The top mode is the highest-weight mode; the best mode is the mode closest to the target composition in total-variation distance.

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

| Path | Description |
|------|-------------|
| `src/model/` | The full model. `tnflow.py`: TNFlow, its sizes, the training objective (`nll`), posterior sampling, `load`. `transformer.py`: transformer encoder with query pooling. `tokenizer.py`: (λ, R) to per-channel tokens. `cnf.py`: composition flow on the simplex. `grain.py`: per-component grain-size flow |
| `src/dataset.py` | `SpectraDataset`: parquet in, train / val / test / ood splits, batches out (with optional noise) |
| `src/train.py` | Training |
| `src/test.py` | Test + OOD scoring: Table 1 metrics, PIT calibration, timing |
| `src/pipeline/inference.py` | Interactive single-spectrum inference with plots and printed summaries |
| `src/pipeline/apply_jwst.py` | Apply TNFlow to real JWST DiSCo-TNOs spectra (appendix figure + mode report) |
| `src/evaluation/` | Helpers for test and inference: posterior summaries and mode search (`posterior.py`), printed reports (`report.py`), figures (`plots.py`) |
| `src/simulator/` | The Shkuratov forward model (`forward.py`), corpus generation (`generate.py`), plot one spectrum (`plot_one.py`) |
| `src/data_utils/` | Readers for the bestfit / generated / JWST / other-object spectra (`processing/`), the component enum (`components.py`), `SpectrumSample`, and the corpus builder (`store.py`) |
| `src/config.py` | Project paths and the seed |

## License

Code: MIT

Data (`spectra.parquet`): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
