"""R-5: checkpoints round-trip, the objective is finite, and draws are valid mixtures."""

import torch

from src.data_utils.components import K
from src.model.grain import PRESENT_THRESH
from src.model.tnflow import TNFlow
from src.pipeline.checkpoint import load_model, save_checkpoint


def _tiny_model():
    torch.manual_seed(0)
    return TNFlow("small", "small", "small", log_grain_mean=1.3, log_grain_std=0.4)


def _batch(b=4, length=60):
    lam = torch.linspace(0.35, 5.0, length).repeat(b, 1)
    R = torch.rand(b, length) * 0.4 + 0.05
    target = torch.zeros(b, K)
    target[:, 0], target[:, 3] = 0.6, 0.4
    grains = torch.full((b, K), float("nan"))
    grains[:, 0], grains[:, 3] = 20.0, 50.0
    return lam, R, target, grains


def test_checkpoint_round_trip(tmp_path):
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    save_checkpoint(tmp_path / "c.pt", model, opt, sched, 4, {"splits": "v2"})
    loaded, ckpt = load_model(tmp_path / "c.pt", torch.device("cpu"), verbose=False)
    assert ckpt["epoch"] == 4 and ckpt["splits"] == "v2"
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_nll_is_finite_and_trains():
    model = _tiny_model()
    lam, R, target, grains = _batch()
    comp_nll, grain_nll = model.nll(model.encode(lam, R), target, grains)
    assert torch.isfinite(comp_nll) and torch.isfinite(grain_nll)
    (comp_nll + grain_nll).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_draws_are_valid_mixtures():
    model = _tiny_model()
    lam, R, _, _ = _batch(b=1)
    comp, grain = model.posterior(R[0], lam[0]).sample(200)
    assert comp.shape == grain.shape == (200, K)
    assert torch.allclose(comp.sum(1), torch.ones(200), atol=1e-4)
    assert (comp >= -1e-6).all()
    assert torch.isnan(grain[comp <= PRESENT_THRESH]).all()
    assert (grain[comp > PRESENT_THRESH] > 0).all()


def test_posterior_is_reproducible():
    model = _tiny_model()
    lam, R, _, _ = _batch(b=1)
    a = model.posterior(R[0], lam[0], seed=3).sample(50)
    b = model.posterior(R[0], lam[0], seed=3).sample(50)
    assert torch.equal(a[0], b[0])
    assert torch.equal(torch.nan_to_num(a[1]), torch.nan_to_num(b[1]))
