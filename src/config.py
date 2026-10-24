"""Project paths, device, and every shared default.

Defaults are frozen dataclasses. Entry points override fields from
the command line, and a training run writes the config it used to ``runs/<run>/config.json``.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.parquet"
SPECTRA_PATH = DATA_DIR / "spectra.parquet"
SPLITS_DIR = DATA_DIR / "splits"            # <name>.parquet: id -> split
RUNS_DIR = PROJECT_ROOT / "runs"            # one folder per training / evaluation run
#CHECKPOINT = RUNS_DIR / "v2" / "checkpoint.pt"          # default checkpoint for evaluation / inference
CHECKPOINT = "tnflow_checkpoint.pt" # old checkpoint

SEED = 10025734   # every seeded stream: splits, weight init, noise, posterior sampling, GMM fits


@dataclass(frozen=True)
class SplitConfig:
    """Grouped train / val / test / ood split (see ``src/data_utils/splits.py``).

    ``test_frac`` and ``val_frac`` are fractions of the non-OOD pool, counted in duplicate groups.
    """
    sources: tuple = ("shkuratov", "more_shkuratov")
    n_ood_mixtures: int = 8         # whole component sets held out as the OOD split
    test_frac: float = 0.1
    val_frac: float = 0.1
    seed: int = SEED


@dataclass(frozen=True)
class NoiseConfig:
    """Noise augmentation on synthetic spectra (see ``src/data_utils/noise.py``)."""
    kind: str = "mult_add"
    sigma0_max: float = 0.04    # scale of the per-spectrum multiplicative noise level
    add_max: float = 0.04       # scale of the per-spectrum additive noise level (× median R)
    lam_ref: float = 0.9        # µm; multiplicative noise grows as sqrt(lam / lam_ref)


@dataclass(frozen=True)
class ModelConfig:
    """TNFlow size presets (see ``src/model/tnflow.py``)."""
    transformer_size: str = "medium"
    comp_flow_size: str = "medium"
    grain_flow_size: str = "small"
    tokenizer: str = "norm"


@dataclass(frozen=True)
class TrainConfig:
    splits: str = "v2"                  # data/splits/<splits>.parquet
    epochs: int = 1000
    batch_size: int = 724
    lr: float = 1e-3
    lr_factor: float = 0.5              # ReduceLROnPlateau on val loss
    lr_patience: int = 10
    min_lr: float = 1e-6
    grad_clip: float = 5.0
    grain_weight: float = 1.0           # weight on the (masked) grain NLL
    seed: int = SEED
    acc_topk: int = 5                   # per-epoch top-k recall
    acc_samples: int = 64               # posterior draws per spectrum for the per-epoch metrics
    train_eval_n: int = 0               # fixed clean train subsample scored each epoch (0 = off)
    limit: int = 0                      # cap every split at this many spectra (smoke runs; 0 = off)
    model: ModelConfig = field(default_factory=ModelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass(frozen=True)
class EvalConfig:
    n_test: int = 5000          # test queries (OOD is always scored in full)
    n_samples: int = 500        # posterior draws per query
    seed: int = SEED
    trace_thresh: float = 0.05  # present components at or below this abundance count as trace
    pit_bins: int = 10
    noise: NoiseConfig = field(default_factory=NoiseConfig)


def get_device() -> torch.device:
    """CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
