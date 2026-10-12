"""TopK sparse autoencoder (hand-rolled; SAELens needs TransformerLens models,
unsupported for llm-jp — spec fallback). Standard recipe: pre-encoder bias,
TopK activation, unit-norm decoder columns, MSE loss, Adam."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        with torch.no_grad():
            self.decoder.weight.data = self.decoder.weight.data / \
                self.decoder.weight.data.norm(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        h = self.encoder(x - self.pre_bias)
        topk = torch.topk(h, self.k, dim=-1)
        z = torch.zeros_like(h).scatter_(-1, topk.indices, topk.values)
        recon = self.decoder(z) + self.pre_bias
        l0 = (z != 0).float().sum(-1).mean()
        mse = ((recon - x) ** 2).mean()
        return {"recon": recon, "acts": z, "mse": mse, "l0": l0}

    @torch.no_grad()
    def resample_dead(self, dead_idx: torch.Tensor, data_mean: torch.Tensor) -> int:
        """Reinit dead features (standard resampling): random unit decoder
        directions, encoder pointing at data mean. Returns count reset."""
        n = int(dead_idx.sum())
        if n == 0:
            return 0
        d = self.decoder.weight.data.shape[0]
        r = torch.randn(n, d, device=self.decoder.weight.device)
        unit = (r / r.norm(dim=1, keepdim=True))
        self.decoder.weight.data[:, dead_idx] = unit.T
        # Tie encoder to the new direction so it can actually fire.
        self.encoder.weight.data[dead_idx] = unit * 0.2
        self.encoder.bias.data[dead_idx] = 0.0
        self.pre_bias.data.copy_(data_mean.to(self.pre_bias.device))
        return n

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.data = self.decoder.weight.data / \
            self.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def state(self) -> dict[str, Any]:
        return {"d_model": self.encoder.in_features,
                "d_sae": self.encoder.out_features, "k": self.k,
                "encoder_w": self.encoder.weight.data.cpu(),
                "encoder_b": self.encoder.bias.data.cpu(),
                "decoder_w": self.decoder.weight.data.cpu(),
                "pre_bias": self.pre_bias.data.cpu()}

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> TopKSAE:
        m = cls(s["d_model"], s["d_sae"], s["k"])
        with torch.no_grad():
            m.encoder.weight.data.copy_(s["encoder_w"])
            m.encoder.bias.data.copy_(s["encoder_b"])
            m.decoder.weight.data.copy_(s["decoder_w"])
            m.pre_bias.data.copy_(s["pre_bias"])
        return m
