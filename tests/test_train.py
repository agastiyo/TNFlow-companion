"""R-3: the training loop's pieces run end to end on a tiny in-memory corpus."""

import math

import numpy as np
import torch

from src.config import ModelConfig, NoiseConfig, TrainConfig
from src.data_utils.dataset import SpectrumDataset, make_loader
from src.data_utils.noise import build_noise
from src.pipeline import train
from tests.conftest import make_payload

CFG = TrainConfig(epochs=1, batch_size=4, acc_samples=8,
                  model=ModelConfig("small", "small", "small", "norm"))


def test_one_epoch_and_evaluate():
    payload = make_payload(lengths=(120,) * 10 + (90,) * 6)
    noise = build_noise(NoiseConfig())
    train_ds = SpectrumDataset(payload.ids[:12], noise, mode="fresh", payload=payload)
    val_ds = SpectrumDataset(payload.ids[12:], noise, mode="frozen", payload=payload)
    device = torch.device("cpu")

    torch.manual_seed(0)
    model = train.build_model(CFG, train_ds, device)
    target, grains = train_ds.targets()
    present = grains[target > 0]
    assert math.isclose(model.config["log_grain_mean"], float(np.log10(present).mean()), rel_tol=1e-5)
    assert model.config["log_grain_std"] > 0
    optimizer, scheduler = train.build_optimizer(model, CFG)

    before = [p.detach().clone() for p in model.parameters()]
    objective = train.train_one_epoch(model, make_loader(train_ds, CFG.batch_size, shuffle=True),
                                      optimizer, CFG, device, epoch=1)
    assert math.isfinite(objective)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    scores = train.evaluate(model, make_loader(val_ds, CFG.batch_size), CFG, device, "val")
    assert set(scores) == set(train.METRICS)
    assert math.isfinite(scores["loss"]) and 0.0 <= scores["topN"] <= 1.0
    assert train.select_checkpoint({"val": scores}, float("inf"))
    assert not train.select_checkpoint({"val": scores}, scores["loss"])
