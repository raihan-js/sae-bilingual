"""M3 causal run (NEEDS GPU): paired design — for each of 30 shared L12
features, ablate shared AND one random control on the SAME concept prompts;
steer both on the same neutrals. Per results/prereg_m3.md. ~40 min GPU.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sae.causal import ablate_hook, delta_logprob, steer_hook
from sae.model import TopKSAE

ALPHA = 8.0  # steer strength, fixed before the run (see prereg amendment)


def ci(vals: list[float]) -> tuple[float, float]:
    import math

    n = len(vals)
    m = sum(vals) / n
    v = sum((x - m) ** 2 for x in vals) / max(n - 1, 1)
    h = 1.96 * math.sqrt(v / n)
    return (round(m - h, 4), round(m + h, 4))


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "llm-jp/llm-jp-3-1.8b", dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager").eval()
    sae = TopKSAE.from_state(
        torch.load("models/sae_l12.pt", weights_only=True)).cuda().eval()
    feats = json.load(open("results/m3_features.json"))
    prompts = json.load(open("results/m3_prompts.json"))
    randoms = json.load(open("results/m3_random.json"))
    assert len(feats) == len(randoms) == 30
    recs = []
    t0 = time.time()
    for f, rf in zip(feats, randoms):
        fi, key = f["feature"], str(f["feature"])
        cp_ja = prompts[key]["concept_ja"][:20]
        cp_en = prompts[key]["concept_en"][:20]
        nt_ja = prompts[key]["neutral_ja"]
        nt_en = prompts[key]["neutral_en"]
        rec = {"feature": fi, "random": rf}
        for lang, cp, nt in [("ja", cp_ja, nt_ja), ("en", cp_en, nt_en)]:
            a_s = delta_logprob(model, tok, cp, ablate_hook(sae, fi),
                                layer=12, device="cuda:0")
            a_r = delta_logprob(model, tok, cp, ablate_hook(sae, rf),
                                layer=12, device="cuda:0")
            s_s = delta_logprob(model, tok, nt, steer_hook(sae, fi, ALPHA),
                                layer=12, device="cuda:0")
            s_r = delta_logprob(model, tok, nt, steer_hook(sae, rf, ALPHA),
                                layer=12, device="cuda:0")
            rec[lang] = {"ablate_shared": a_s.get("delta"),
                         "ablate_random": a_r.get("delta"),
                         "steer_shared": s_s.get("delta"),
                         "steer_random": s_r.get("delta")}
        recs.append(rec)
        print(f"  {len(recs)}/30 ({time.time() - t0:.0f}s)", flush=True)
    Path("results/m3_recs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")

    def paired(key: str, lang: str) -> dict:
        d = [(r[lang][f"{key}_shared"] or 0.0) - (r[lang][f"{key}_random"] or 0.0)
             for r in recs]
        m = sum(d) / len(d)
        import math
        v = sum((x - m) ** 2 for x in d) / (len(d) - 1)
        h = 1.96 * math.sqrt(v / len(d))
        return {"mean_shared_minus_random": round(m, 4), "n": len(d),
                "ci95": [round(m - h, 4), round(m + h, 4)]}
    res = {"alpha": ALPHA, "n_features": 30,
           "ablate_ja": paired("ablate", "ja"),
           "ablate_en": paired("ablate", "en"),
           "steer_ja": paired("steer", "ja"),
           "steer_en": paired("steer", "en"),
           "secs": round(time.time() - t0)}
    Path("results/m3.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
