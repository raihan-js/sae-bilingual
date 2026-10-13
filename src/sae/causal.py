"""Causal harness: feature ablation + steering via hooks.

operate on (model, sae, tokenizer) handles; unit-tested on CPU with a tiny
fake transformer. GPU runs use the real llm-jp + trained SAEs.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def ablate_hook(sae: Any, feat: int) -> Callable[..., object]:
    def fn(_mod: object, _inp: object, o: Any) -> Any:
        t = o[0] if isinstance(o, tuple) else o
        with torch.no_grad():
            z = sae(t.float())["acts"]
            z[..., feat] = 0.0
            recon = sae.decoder(z.to(sae.decoder.weight.dtype)).to(t.dtype) + \
                sae.pre_bias.to(t.dtype)
        return (recon,) if isinstance(o, tuple) else recon
    return fn


def steer_hook(sae: Any, feat: int, alpha: float) -> Callable[..., object]:
    def fn(_mod: object, _inp: object, o: Any) -> Any:
        t = o[0] if isinstance(o, tuple) else o
        with torch.no_grad():
            direction = sae.decoder.weight.data[:, feat].to(t.dtype)
            t = t + alpha * direction.unsqueeze(0).unsqueeze(0)
        return (t,) if isinstance(o, tuple) else t
    return fn


def delta_logprob(model: Any, tok: Any, prompts: list[str],
                  hook_fn: Callable[..., object] | None, layer: int = 0,
                  device: str = "cpu") -> dict[str, Any]:
    """Mean Δ top-5-token logprob with vs without hook. Base probs fixed from
    the unhooked pass (same prompts); hook pass reuses them."""
    import torch.nn.functional as F

    base_lps, hook_lps = [], []
    for text in prompts:
        ids = tok(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids)
            lp = F.log_softmax(out.logits[0, -1], dim=-1)
            top5 = lp.topk(5)
            base_lps.append(top5.values)
            if hook_fn is not None:
                h = model.model.layers[layer].register_forward_hook(hook_fn)
                try:
                    out2 = model(ids)
                finally:
                    h.remove()
                lp2 = F.log_softmax(out2.logits[0, -1], dim=-1)
                hook_lps.append(lp2[top5.indices])
    base = torch.stack(base_lps)
    out_d: dict[str, Any] = {"base_mean": round(float(base.mean()), 4)}
    if hook_lps:
        hook_t = torch.stack(hook_lps)
        out_d["hook_mean"] = round(float(hook_t.mean()), 4)
        out_d["delta"] = round(float((hook_t - base).mean()), 4)
    return out_d
