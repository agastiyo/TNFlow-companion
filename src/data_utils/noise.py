"""Noise models: augmentation applied to synthetic spectra (training, and the held-out queries).

Each model is a callable ``(values, lams, rng) -> noised values`` on ndarrays [L] or [B, L] with an
``np.random.Generator``. ``build_noise`` picks one by ``NoiseConfig.kind``; to try another (e.g. a
per-channel σ calibrated to NIRSpec PRISM, roadmap item 13), add a class and register it below.
"""

import numpy as np

from src.config import NoiseConfig


class MultAddNoise:
    """R -> R·(1 + ε_mult) + ε_add.

    Per spectrum: σ0 ~ N(0, sigma0_max) and a ~ N(0, add_max). Per channel:
    ε_mult ~ N(0, 1)·σ0·sqrt(λ / lam_ref) (grows to the red) and ε_add ~ N(0, 1)·a·median(R).
    """

    def __init__(self, cfg: NoiseConfig = NoiseConfig()):
        self.cfg = cfg

    def __call__(self, values, lams, rng):
        squeeze = np.ndim(values) == 1
        values, lams = np.atleast_2d(values), np.atleast_2d(lams)
        b = values.shape[0]
        sig0 = rng.normal(0.0, self.cfg.sigma0_max, (b, 1))
        eps_mult = rng.standard_normal(values.shape) * sig0 * np.sqrt(lams / self.cfg.lam_ref)
        a = rng.normal(0.0, self.cfg.add_max, (b, 1))
        eps_add = rng.standard_normal(values.shape) * a * np.median(values, axis=1, keepdims=True)
        out = values * (1.0 + eps_mult) + eps_add
        return out[0] if squeeze else out


NOISE_MODELS = {"mult_add": MultAddNoise}


def build_noise(cfg: NoiseConfig = NoiseConfig()):
    try:
        return NOISE_MODELS[cfg.kind](cfg)
    except KeyError:
        raise ValueError(f"noise kind must be one of {list(NOISE_MODELS)}, got {cfg.kind!r}")
