"""Tokenizer: raw (λ, R) spectrum -> per-channel tokens [R / R(0.9 µm), sin(ωλ), cos(ωλ)...].

Reflectance is normalized at the NORM_LAM µm continuum (NORM_WINDOW µm window), then each channel
gets Fourier features of its wavelength.
"""

import torch
import torch.nn as nn

N_FREQS = 6
NORM_LAM = 0.9        # µm reference wavelength to normalize at (per Cristina; in-range for all datasets)
NORM_WINDOW = 0.1     # µm width of the window averaged to get the reference level
NORM_EPS = 1e-6       # numerical floor on the divisor (avoid div-by-zero); not a design knob


class NormTokenizer(nn.Module):
    """Raw (λ, R) -> [R / R(NORM_LAM µm window), sin(ωλ), cos(ωλ)...]; width 1 + 2·N_FREQS."""

    def __init__(self):
        super().__init__()
        self.d_token = 1 + 2 * N_FREQS
        self.register_buffer("omegas", 2.0 * torch.pi * 2.0 ** torch.arange(N_FREQS))

    def forward(self, lam: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Args: lam, R: Tensor [B, L]. Returns: tokens Tensor [B, L, d_token]."""
        win = ((lam - NORM_LAM).abs() <= NORM_WINDOW / 2) & torch.isfinite(R)
        wsum = win.sum(1, keepdim=True)
        window_mean = (R * win).sum(1, keepdim=True) / wsum.clamp(min=1)
        nearest = (lam - NORM_LAM).abs().argmin(dim=1, keepdim=True)
        ref = torch.where(wsum > 0, window_mean, R.gather(1, nearest)).clamp(min=NORM_EPS)
        phases = lam.unsqueeze(-1) * self.omegas
        return torch.cat([(R / ref).unsqueeze(-1), phases.sin(), phases.cos()], dim=-1)
