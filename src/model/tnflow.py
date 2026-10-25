"""TNFlow: spectrum -> context vector -> joint posterior over composition and grain size.

Model      spectrum x = (λ, R) [L] -> tokens [L, d_token] -> context h(x) [d_model]
           q(c, g | x) = q(c | h) · ∏_{k : c_k > 0} q(g_k | h, c, k)
           c on the simplex Δ^{K-1} (composition CNF), g_k in µm (one shared grain flow).
Training   minimize E_{p(c, g, x)}[−log q(c, g | x)] over the synthetic corpus; the grain factor is
           teacher-forced (conditioned on the true c), see ``nll``.
Sampling   one draw = composition [K] + grain [K] µm (NaN where that draw lacks the component).

    model = load("medium", "tnflow_checkpoint.pt")
    post = model.posterior(spec, lams)   # encode once
    comp, grain = post.sample(2000)      # (2000, K), (2000, K) tensors; resample freely

Everything stays on the model's device. Summaries of the draws live in src/evaluation/posterior.py.
"""

import numpy as np
import torch
import torch.nn as nn

from src.config import SEED
from src.data_utils.components import K
from src.model.cnf import CNF, free_to_full_simplex
from src.model.grain import GrainFlow
from src.model.tokenizer import NormTokenizer
from src.model.transformer import Transformer

# The transformer and the composition flow grow together; the grain flow is the same at every size.
# "medium" is the paper model.
SIZES = {
    "small":  dict(d_model=64,  n_head=4,  n_layers=2, transforms=2, bins=8,  hidden=(64, 64)),
    "medium": dict(d_model=128, n_head=8,  n_layers=4, transforms=4, bins=16, hidden=(128, 128)),
    "large":  dict(d_model=256, n_head=16, n_layers=8, transforms=8, bins=32, hidden=(256, 256)),
}
GRAIN_FLOW = dict(transforms=2, bins=8, hidden=(64, 64))

DROPOUT = 0.1
COMP_SCALE = 2.5        # pre-shrink keeping stick-breaking preimages inside the NSF spline bound
GRAIN_EMBED_DIM = 16


class TNFlow(nn.Module):
    """Tokenizer -> transformer -> {composition CNF, grain flow}.

    Submodule names (tokenizer, transformer, cnf, grain) are the checkpoint's state_dict keys.
    """

    def __init__(self, size: str = "medium"):
        super().__init__()
        s = SIZES[size]
        self.tokenizer = NormTokenizer()
        self.transformer = Transformer(self.tokenizer.d_token, s["d_model"], s["n_head"],
                                       s["n_layers"], DROPOUT)
        self.cnf = CNF(features=K - 1, context=s["d_model"], scale=COMP_SCALE,
                       transforms=s["transforms"], bins=s["bins"], hidden_features=s["hidden"])
        self.grain = GrainFlow(context=s["d_model"], embed_dim=GRAIN_EMBED_DIM,
                               transforms=GRAIN_FLOW["transforms"], bins=GRAIN_FLOW["bins"],
                               hidden_features=GRAIN_FLOW["hidden"])

    def encode(self, lam, R):
        """lam, R: Tensor [B, L] -> context h(x) Tensor [B, d_model]."""
        return self.transformer(self.tokenizer(lam, R))

    def nll(self, context, target, grains):
        """Negative log-likelihood terms for one batch (the training objective).

        Args:    context: Tensor [B, d_model] from ``encode``; target: Tensor [B, K] true
                 composition; grains: Tensor [B, K] true grain sizes µm (NaN where unknown).
        Returns: (comp_nll, grain_nll) scalar Tensors. comp_nll = −log q(c | h) averaged over the
                 B spectra. grain_nll = −log q(g_k | h, c, k) averaged over the present
                 (spectrum, component) pairs with a known grain, conditioned on the TRUE c
                 (teacher forcing); 0 when there are none.
        """
        comp_nll = -self.cnf.log_prob(context, target[:, :-1]).mean()
        mask = (target > 0) & torch.isfinite(grains)
        g_sum, g_n = self.grain.log_prob(context, target, grains, mask)
        return comp_nll, -g_sum / max(g_n, 1)

    def _as_batch(self, spec, lams):
        """spec, lams: ndarray [L] or [B, L] -> (R, lam) Tensor [B, L] on the model device."""
        device = next(self.parameters()).device
        R = torch.as_tensor(np.asarray(spec), dtype=torch.float32)
        lam = torch.as_tensor(np.asarray(lams), dtype=torch.float32)
        if R.ndim == 1:
            R, lam = R.unsqueeze(0), lam.unsqueeze(0)
        return R.to(device), lam.to(device)

    def posterior(self, spec, lams, seed=SEED):
        """spec, lams: ndarray|Tensor [L] -> SimplexPosterior (runs the encoder once).

        ``seed`` reseeds torch's GLOBAL generator (``torch.manual_seed``) before encoding, so the
        draws that follow are reproducible; pass ``seed=None`` to leave the generator alone.
        """
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        R, lam = self._as_batch(spec, lams)
        with torch.no_grad():
            context = self.encode(lam, R)
        return SimplexPosterior(self, context)

    def attention(self, spec, lams):
        """spec, lams: ndarray [L] -> (lam [L], query-pooling weights [L] summing to 1)."""
        self.eval()
        R, lam = self._as_batch(spec, lams)
        with torch.no_grad():
            _, weights = self.transformer(self.tokenizer(lam, R), return_weights=True)
        return lam.squeeze(0).cpu().numpy(), weights.squeeze(0).cpu().numpy()


class SimplexPosterior:
    """The posterior q(c, g | x) for one spectrum. Sample it as often as you like."""

    def __init__(self, model, context):
        self.model = model
        self.context = context      # [1, d_model]

    def __repr__(self):
        return f"SimplexPosterior(d_model={self.context.shape[-1]}, device={self.context.device})"

    def sample(self, n_samples: int = 2000):
        """n_samples -> (composition [n, K] summing to 1, grain [n, K] µm, NaN where absent).

        Ancestral: c ~ q(c | h), then g_k ~ q(g_k | h, c, k) for every component the draw contains.
        """
        with torch.no_grad():
            free = self.model.cnf.dist(self.context).sample((n_samples,)).squeeze(1)
            comp = free_to_full_simplex(free)
            return comp, self.model.grain.sample_grid(self.context, comp)


def load(size, path, device="cpu"):
    """TNFlow of the given size with the weights from a checkpoint file, in eval mode."""
    model = TNFlow(size).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
    return model.eval()
