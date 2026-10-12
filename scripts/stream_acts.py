"""Two-phase activation dump (datasets streaming threads + CUDA in one
process fatally crashes: PyGILState_Release).
Phase 1: stream Wikipedia JA+EN to data/corpus.txt (no torch).
Phase 2: batched hook dump to fp16 shards (no streaming).
Usage: phase1 [N_DOCS] / phase2 LAYER N_TOKENS
"""
import sys
import time
from pathlib import Path

N_DOCS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "phase1" else 0


def phase1(n_docs: int = 60000) -> None:
    from datasets import load_dataset

    out = Path("data/corpus.txt")
    if out.exists():
        print("corpus exists, skipping")
        return
    streams = [load_dataset("wikimedia/wikipedia", "20231101.ja", split="train",
                            streaming=True, trust_remote_code=True),
               load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                            streaming=True, trust_remote_code=True)]
    iters = [iter(s) for s in streams]
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        i = 0
        while n < n_docs:
            try:
                row = next(iters[i % 2])
            except StopIteration:
                break
            i += 1
            t = row["text"].replace("\n", " ")[:1500]
            if len(t) > 200:
                f.write(t + "\n")
                n += 1
    print("docs:", n)


def phase2(layer: int, n_tok: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(f"data/acts/l{layer}")
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("shard_*.pt")):
        print("shards exist, skipping")
        return
    tok = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "llm-jp/llm-jp-3-1.8b", dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager").eval()
    buf: list[torch.Tensor] = []

    def hook(_mod: object, _inp: object, o: object) -> None:
        t = o[0] if isinstance(o, tuple) else o  # v5 returns plain tensor
        buf.append(t.detach().float().cpu())

    h = model.model.layers[layer].register_forward_hook(hook)
    lines = Path("data/corpus.txt").read_text(encoding="utf-8").splitlines()
    t0 = time.time()
    total = shard = 0
    acc: list[torch.Tensor] = []
    B = 16
    try:
        for b in range(0, len(lines), B):
            enc = tok(lines[b:b + B], max_length=128, truncation=True,
                      padding=True, return_tensors="pt")
            ids = enc["input_ids"].to("cuda:0")
            mask = enc["attention_mask"].to("cuda:0")
            with torch.no_grad():
                model(ids, attention_mask=mask)
            a = buf.pop()[mask.cpu().bool()].clone()
            acc.append(a)
            total += a.shape[0]
            if total - shard * 100_000 >= 100_000:
                torch.save({"acts": torch.cat(acc).half()},
                           out / f"shard_{shard:03d}.pt")
                shard += 1
                acc = []
                print(f"  shard {shard} total={total} ({time.time() - t0:.0f}s)",
                      flush=True)
            if total >= n_tok:
                break
    finally:
        h.remove()
    if acc:
        torch.save({"acts": torch.cat(acc).half()}, out / f"shard_{shard:03d}.pt")
    print(f"done: {total} tokens in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] == "phase1":
        phase1(int(sys.argv[2]) if len(sys.argv) > 2 else 60000)
    else:
        phase2(int(sys.argv[1]), int(sys.argv[2]))
