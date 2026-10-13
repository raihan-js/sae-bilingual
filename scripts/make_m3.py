"""Freeze M3 feature list + prompts (CPU; no GPU needed).

Features: L12 top-100 by paired-r, keep those with >=3 JA and >=3 EN
examples in top-6, drop proper-name features (>=4/6 examples sharing one
capitalized word). Prompts per feature per lang: top-6 examples + 14 fixed-seed
random Tatoeba pairs (neutral, same for all features). Writes
results/m3_features.json + results/m3_prompts.json (FROZEN before the run).
"""
import json
import random
from collections import Counter

TOP_N = 30


def is_name(feat: dict) -> bool:
    words = Counter()
    for e in feat["examples"]:
        for w in e["text"].replace(".", " ").replace(",", " ").split():
            if len(w) > 2 and w[0].isupper() and w[1:].islower():
                words[w] += 1
    return any(c >= 4 for c in words.values())


def main() -> None:
    d = json.load(open("results/sharing_l12.json"))
    pairs = json.load(open("data/pairs.jsonl", encoding="utf-8"))
    feats = []
    for f in d["top100"]:
        langs = Counter(e["lang"] for e in f["examples"])
        if langs["ja"] >= 2 and langs["en"] >= 2 and not is_name(f):
            feats.append(f)
        if len(feats) >= TOP_N:
            break
    print("selected:", len(feats), "(skipped names/non-bilingual)")
    rng = random.Random(0)
    neutrals = rng.sample(pairs, 14)
    prompts = {}
    for f in feats:
        ex = {"ja": [e["text"] for e in f["examples"] if e["lang"] == "ja"][:6],
              "en": [e["text"] for e in f["examples"] if e["lang"] == "en"][:6]}
        prompts[str(f["feature"])] = {
            "concept_ja": ex["ja"], "concept_en": ex["en"],
            "neutral_ja": [p["ja"] for p in neutrals],
            "neutral_en": [p["en"] for p in neutrals]}
    json.dump([{"feature": f["feature"], "r": f["r"],
                "ja_mean": f["ja_mean"], "en_mean": f["en_mean"]}
               for f in feats],
              open("results/m3_features.json", "w"), indent=2)
    json.dump(prompts, open("results/m3_prompts.json", "w"),
              ensure_ascii=False, indent=2)
    # random-feature controls (seed 0), disjoint from selected
    sel = {f["feature"] for f in feats}
    pool = [i for i in range(32768) if i not in sel]
    json.dump(rng.sample(pool, TOP_N),
              open("results/m3_random.json", "w"))
    print("frozen: m3_features/m3_prompts/m3_random")


if __name__ == "__main__":
    main()
