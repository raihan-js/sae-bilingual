"""Dump SAE activations + token ids for the 20k Tatoeba pairs (all layers,
one model pass, 3 hooks). Writes data/analysis/l{L}.pt {acts f16, ids i32,
lang u8 (0=ja,1=en), pair i32}. ~2.5GB/layer, deleted after analysis.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = [4, 12, 20]


def main() -> None:
    tok = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "llm-jp/llm-jp-3-1.8b", dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager").eval()
    pairs = [json.loads(line) for line in open("data/pairs.jsonl", encoding="utf-8")]
    out = Path("data/analysis")
    out.mkdir(parents=True, exist_ok=True)
    bufs: dict[int, list] = {L: [] for L in LAYERS}

    def hook(L: int) -> object:
        def fn(_m: object, _i: object, o: object) -> None:
            t = o[0] if isinstance(o, tuple) else o
            bufs[L].append(t.detach().float().cpu())
        return fn

    hs = [model.model.layers[L].register_forward_hook(hook(L)) for L in LAYERS]
    acc: dict[int, list] = {L: [] for L in LAYERS}
    ids_acc: list[torch.Tensor] = []
    lang_acc: list[torch.Tensor] = []
    pair_acc: list[torch.Tensor] = []
    t0 = time.time()
    B = 32
    try:
        for b in range(0, len(pairs), B):
            chunk = pairs[b:b + B]
            for li, key in [(0, "ja"), (1, "en")]:
                enc = tok([p[key] for p in chunk], max_length=64, truncation=True,
                          padding=True, return_tensors="pt")
                ids = enc["input_ids"].to("cuda:0")
                mask = enc["attention_mask"].bool()
                with torch.no_grad():
                    model(ids, attention_mask=mask.to("cuda:0"))
                for L in LAYERS:
                    a = bufs[L].pop()[mask].clone()
                    acc[L].append(a)
                ids_acc.append(enc["input_ids"][mask].clone())
                lang_acc.append(torch.full((int(mask.sum()),), li, dtype=torch.uint8))
                pair_acc.append(torch.tensor(
                    [i for i in range(b, b + len(chunk)) for _ in range(int(mask[i - b].sum()))],
                    dtype=torch.int32))
            if (b // B + 1) % 20 == 0:
                print(f"  {b + B}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for h in hs:
            h.remove()
    ids_all = torch.cat(ids_acc)
    lang_all = torch.cat(lang_acc)
    pair_all = torch.cat(pair_acc)
    for L in LAYERS:
        A = torch.cat(acc[L])
        torch.save({"acts": A.half(), "ids": ids_all, "lang": lang_all,
                    "pair": pair_all}, out / f"l{L}.pt")
        print(f"l{L}: {tuple(A.shape)}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
