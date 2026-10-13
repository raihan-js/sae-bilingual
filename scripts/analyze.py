"""M2 sharing analysis: per-feature JA/EN means (all 32k), paired Pearson r
on top-2k both-firing candidates, share with r>thr (sensitivity 0.3/0.5/0.7),
top-100 shared with bilingual examples. Writes results/sharing_l{L}.json.
Sentence value = max over tokens (spec: sentence-level, disclosed).
"""
import heapq
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sae.model import TopKSAE  # noqa: E402


def main() -> None:
    layer = int(sys.argv[1])
    d = torch.load(f"data/analysis/l{layer}.pt", weights_only=True)
    acts, lang, pair = d["acts"].float(), d["lang"], d["pair"]
    del d
    pairs = [json.loads(line) for line in open("data/pairs.jsonl", encoding="utf-8")]
    sae = TopKSAE.from_state(
        torch.load(f"models/sae_l{layer}.pt", weights_only=True)).cuda().eval()
    n_feat = sae.encoder.out_features
    ja_sum = torch.zeros(n_feat, dtype=torch.float64)
    en_sum = torch.zeros(n_feat, dtype=torch.float64)
    ja_n = en_n = 0
    t0 = time.time()
    # pass 1: means (SAE forward in chunks; acts already resid)
    with torch.no_grad():
        for q in range(0, len(acts), 2048):
            z = sae(acts[q:q + 2048].cuda())["acts"].float().cpu()
            m = lang[q:q + 2048] == 0
            ja_sum += z[m].sum(0)
            en_sum += z[~m].sum(0)
            ja_n += int(m.sum())
            en_n += int((~m).sum())
            del z
            torch.cuda.empty_cache()
    ja_mean = ja_sum / max(ja_n, 1)
    en_mean = en_sum / max(en_n, 1)
    both = torch.minimum(ja_mean, en_mean)
    cand = torch.topk(both, 2000).indices.tolist()
    print(f"pass1 done ({time.time() - t0:.0f}s) ja_n={ja_n} en_n={en_n}", flush=True)
    # pass 2: paired Pearson on candidates (sentence max), streaming accumulators
    sx = torch.zeros(2000, dtype=torch.float64)
    sy = torch.zeros(2000, dtype=torch.float64)
    sxx = torch.zeros(2000, dtype=torch.float64)
    syy = torch.zeros(2000, dtype=torch.float64)
    sxy = torch.zeros(2000, dtype=torch.float64)
    n_pairs = 0
    tops: list[list] = [[] for _ in range(2000)]  # heaps of (val, pair, lang)
    # re-iterate per pair from raw acts to bound memory
    with torch.no_grad():
        for p in range(0, 20000, 500):
            sel = (pair >= p) & (pair < p + 500)
            z = sae(acts[sel].cuda())["acts"][:, cand].float().cpu()
            lp, pp = lang[sel], pair[sel]
            for j in range(p, min(p + 500, 20000)):
                m = (pp == j)
                if m.sum() == 0:
                    continue
                zj, ze = z[m & (lp == 0)], z[m & (lp == 1)]
                if len(zj) == 0 or len(ze) == 0:
                    continue
                vj, ve = zj.double().max(0).values, ze.double().max(0).values
                sx += vj
                sy += ve
                sxx += vj * vj
                syy += ve * ve
                sxy += vj * ve
                n_pairs += 1
                for k in range(2000):
                    heapq.heappush(tops[k], (float(vj[k]), j, 0))
                    heapq.heappush(tops[k], (float(ve[k]), j, 1))
                    if len(tops[k]) > 10:
                        heapq.heappop(tops[k])
            del z
            if (p // 500 + 1) % 8 == 0:
                print(f"  pairs {p} ({time.time() - t0:.0f}s)", flush=True)
    cov = sxy / n_pairs - (sx / n_pairs) * (sy / n_pairs)
    vx = (sxx / n_pairs - (sx / n_pairs) ** 2).clamp_min(1e-12)
    vy = (syy / n_pairs - (sy / n_pairs) ** 2).clamp_min(1e-12)
    r = (cov / torch.sqrt(vx * vy)).clamp(-1.0, 1.0)
    res = {"layer": layer, "n_pairs": n_pairs, "n_tokens": len(acts),
           "share_r03": round(float((r > 0.3).sum()) / 2000, 3),
           "share_r05": round(float((r > 0.5).sum()) / 2000, 3),
           "share_r07": round(float((r > 0.7).sum()) / 2000, 3),
           "mean_r": round(float(r.mean()), 3)}
    top100 = torch.topk(r, 100)
    feats = []
    for rank, (rv, k) in enumerate(zip(top100.values.tolist(), top100.indices.tolist())):
        f = cand[k]
        ex = sorted(tops[k], reverse=True)[:6]
        feats.append({"feature": f, "r": round(rv, 3),
                      "ja_mean": round(float(ja_mean[f]), 4),
                      "en_mean": round(float(en_mean[f]), 4),
                      "examples": [{"text": pairs[j][["ja", "en"][lg]],
                                    "lang": ["ja", "en"][lg],
                                    "act": round(v, 3)} for v, j, lg in ex]})
    res["top100"] = feats
    Path(f"results/sharing_l{layer}.json").write_text(
        json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "top100"}, indent=2))


if __name__ == "__main__":
    main()
