"""Project paths and the one seed used everywhere."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SEED = 10025734   # every seeded stream: splits, weight init, noise, posterior sampling, GMM fits
