"""SAE eval: held-out dead rate, FVE, and LLM loss delta (base vs SAE
reconstruction patched into layer 12). Tail 5k corpus docs = held-out.
Writes results/sae_eval.json.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sae.model import TopKSAE


def main() -> None:
    import sys as _sys

    layer = int(_sys.argv[1]) if len(_sys.argv) > 1 else 12
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "llm-jp/llm-jp-3-1.8b", dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager").eval()
    sae = TopKSAE.from_state(torch.load(f"models/sae_l{layer}.pt", weights_only=True)).cuda().eval()
    lines = Path("data/corpus.txt").read_text(encoding="utf-8").splitlines()[-5000:]
    mode = [None]

    def hook(_mod: object, _inp: object, o: object) -> object:
        t = o[0] if isinstance(o, tuple) else o
        if mode[0] == "sae":
            with torch.no_grad():
                return (sae(t.float()).get("recon").to(t.dtype),) if isinstance(o, tuple) else sae(t.float()).get("recon").to(t.dtype)
        return o

    h = model.model.layers[layer].register_forward_hook(hook)
    base_loss = sae_loss = ntok = 0
    t0 = time.time()
    try:
        for b in range(0, len(lines), 8):
            enc = tok(lines[b:b + 8], max_length=128, truncation=True,
                      padding=True, return_tensors="pt")
            ids = enc["input_ids"].to("cuda:0")
            mask = enc["attention_mask"].to("cuda:0")
            with torch.no_grad():
                mode[0] = None
                lb = model(ids, attention_mask=mask, labels=ids).loss
                mode[0] = "sae"
                ls = model(ids, attention_mask=mask, labels=ids).loss
            n = int(mask.sum())
            base_loss += float(lb) * n
            sae_loss += float(ls) * n
            ntok += n
    finally:
        h.remove()
    # held-out dead rate: capture resid, SAE-forward in chunks
    caps: list[torch.Tensor] = []

    def cap_hook(_mod: object, _inp: object, o: object) -> None:
        t = o[0] if isinstance(o, tuple) else o
        caps.append(t.detach().float().cpu())
    h2 = model.model.layers[layer].register_forward_hook(cap_hook)
    try:
        for b in range(0, min(len(lines), 800), 8):
            enc = tok(lines[b:b + 8], max_length=128, truncation=True,
                      padding=True, return_tensors="pt")
            ids = enc["input_ids"].to("cuda:0")
            mask = enc["attention_mask"].to("cuda:0")
            with torch.no_grad():
                model(ids, attention_mask=mask)
            caps[-1] = caps[-1][mask.cpu().bool()].clone()
    finally:
        h2.remove()
    X = torch.cat(caps)
    with torch.no_grad():
        hit = torch.zeros(sae.encoder.out_features)
        for q in range(0, len(X), 4096):
            hit += (sae(X[q:q + 4096].cuda())["acts"] != 0).float().sum(0).cpu()
    res = {"heldout_tokens": len(X)}
    res["heldout_dead"] = int((hit == 0).sum())
    res["heldout_dead_rate"] = round(float((hit == 0).sum()) / len(hit), 4)
    res.update({"loss_tokens": ntok,
                "base_loss": round(base_loss / ntok, 4),
                "sae_loss": round(sae_loss / ntok, 4),
                "loss_delta": round((sae_loss - base_loss) / ntok, 4),
                "secs": round(time.time() - t0)})
    Path(f"results/sae_eval_l{layer}.json").write_text(__import__("json").dumps(res, indent=2))
    print(__import__("json").dumps(res, indent=2))


if __name__ == "__main__":
    main()
