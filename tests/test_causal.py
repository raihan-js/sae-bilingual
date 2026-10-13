"""Causal harness tests on CPU with a tiny fake transformer (no GPU)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch import nn

from sae.causal import ablate_hook, delta_logprob, steer_hook
from sae.model import TopKSAE


class FakeLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin(h)


class FakeModel(nn.Module):
    def __init__(self, d: int = 16, v: int = 32):
        super().__init__()
        self.emb = nn.Embedding(v, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer(d)])
        self.head = nn.Linear(d, v)

    def forward(self, ids: torch.Tensor) -> object:
        from types import SimpleNamespace

        h = self.emb(ids)
        for layer in self.model.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.head(h))


class FakeTok:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def test_ablate_changes_output():
    torch.manual_seed(0)
    m, sae = FakeModel(), TopKSAE(16, 32, 4)
    base = delta_logprob(m, FakeTok(), ["hi"], None)
    hooked = delta_logprob(m, FakeTok(), ["hi"], ablate_hook(sae, 0))
    assert hooked["delta"] != 0.0
    assert base["base_mean"] == hooked["base_mean"]


def test_steer_changes_output():
    torch.manual_seed(0)
    m, sae = FakeModel(), TopKSAE(16, 32, 4)
    hooked = delta_logprob(m, FakeTok(), ["hi"], steer_hook(sae, 0, 5.0))
    assert hooked["delta"] != 0.0


def test_no_hook_returns_base_only():
    m = FakeModel()
    d = delta_logprob(m, FakeTok(), ["a", "b"], None)
    assert "base_mean" in d and "delta" not in d
