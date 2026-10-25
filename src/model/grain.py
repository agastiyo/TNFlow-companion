"""GrainFlow: per-component grain-size posterior q(g_k | h, c, k).

One 1-D NSF over standardized log10 grain, z = (log10(g / µm) − mean) / std, shared by all K
components; the component enters through a learned embedding. Conditioning vector per
(spectrum, component) pair: [context h, composition c, embed(k)].

mean and std are buffers (saved in the checkpoint), 0 and 1 until train.py sets them from the
training split.
"""

import torch
import torch.nn as nn
import zuko

from src.data_utils.components import K

GRAIN_EPS = 1e-6  # floor for log10 on the grain target

# A composition draw "contains" component k when c_k exceeds this threshold.
PRESENT_THRESH = 0.01


class GrainFlow(nn.Module):
    """1-D NSF over standardized log10(grain/µm), conditioned on [context, composition, component embed]."""

    def __init__(self, context: int, embed_dim: int = 16, **flow_kwargs):
        super().__init__()
        self.embed = nn.Embedding(K, embed_dim)
        self.flow = zuko.flows.NSF(features=1, context=context + K + embed_dim, **flow_kwargs)
        self.register_buffer("log_mean", torch.tensor(0.0))
        self.register_buffer("log_std", torch.tensor(1.0))

    def _cond(self, c, comp, idx):
        return torch.cat([c, comp, self.embed(idx)], dim=-1)

    def _cond_all(self, c, comp):
        """Conditioning for every (row, component) pair: c [B, d], comp [B, K] -> [B·K, d + K + embed]."""
        b = comp.shape[0]
        idx = torch.arange(K, device=comp.device).repeat(b)
        return self._cond(c.repeat_interleave(K, dim=0), comp.repeat_interleave(K, dim=0), idx)

    def _standardize(self, grain_um):
        return (torch.log10(grain_um.clamp(min=GRAIN_EPS)) - self.log_mean) / self.log_std

    def _destandardize(self, z):
        return 10.0 ** (z * self.log_std + self.log_mean)

    def log_prob(self, context, comp, grain_um, mask):
        """Summed log-density over the masked (spectrum, component) pairs, and their count.

        Args:    context: Tensor [B, d_model]; comp: Tensor [B, K] composition to condition on;
                 grain_um: Tensor [B, K] µm; mask: BoolTensor [B, K] pairs to score.
        Returns: (Tensor scalar log-density sum, int count).
        """
        b, k = mask.nonzero(as_tuple=True)
        if b.numel() == 0:
            return context.new_zeros(()), 0
        z = self._standardize(grain_um[b, k]).unsqueeze(-1)
        lp = self.flow(self._cond(context[b], comp[b], k)).log_prob(z)
        return lp.sum(), int(b.numel())

    def sample_grid(self, c, comp, present_thresh: float = PRESENT_THRESH):
        """One grain sample (µm) per component per composition sample; NaN where absent.

        Args:    c: Tensor [1, d_model] or [d_model]; comp: Tensor [N, K]; present_thresh: float.
        Returns: Tensor [N, K] µm, NaN where comp <= present_thresh.
        """
        n = comp.shape[0]
        cond = self._cond_all(c.reshape(1, -1).expand(n, -1), comp)
        z = self.flow(cond).sample().squeeze(-1)
        grain = self._destandardize(z).reshape(n, K)
        return grain.masked_fill(comp <= present_thresh, float("nan"))
