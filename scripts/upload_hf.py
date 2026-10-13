"""Upload release artifacts to HF dataset raihan-js/sae-bilingual.

Contents: 3 SAE weights (.pt: encoder/decoder/bias dicts, see src/sae/model.py),
per-layer train/eval/sharing JSONs, M3 records + prereg, top100 browser data.
Model weights are OURS (trained here) — no licence encumbrance. Eval text:
Tatoeba (CC-BY) + Wikipedia (CC-BY-SA) excerpts in examples (short quotes).
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO = "sae-bilingual"
FILES = [
    "models/sae_l4.pt",
    "models/sae_l12.pt",
    "models/sae_l20.pt",
    "results/sae_train_l4.json",
    "results/sae_train_l12.json",
    "results/sae_train_l20.json",
    "results/sae_eval_l4.json",
    "results/sae_eval_l12.json",
    "results/sae_eval_l20.json",
    "results/sharing_l4.json",
    "results/sharing_l12.json",
    "results/sharing_l20.json",
    "results/m3.json",
    "results/m3_recs.jsonl",
    "results/m3_features.json",
    "results/prereg_m3.md",
    "browser/index.html",
]

CARD = """---
license: apache-2.0
---
# sae-bilingual

TopK sparse autoencoders (2048→32768, k=64) on llm-jp-3-1.8B residual stream
(layers 4/12/20), trained streaming on Wikipedia JA+EN. For
[sae-bilingual](https://github.com/raihan-js/sae-bilingual).

- `models/sae_l*.pt`: SAE weights (ours, Apache-2.0): encoder_w/bias,
  decoder_w, pre_bias, k. Load with the repo's `src/sae/model.py`.
- `sharing_l*.json`: paired-sentence Pearson sharing on 20k Tatoeba pairs +
  top-100 shared features with bilingual examples.
- `m3*.json(l)`: pre-registered causal tests (ablate/steer, paired random
  controls) + frozen feature list. Verdict: weak JA, null EN.
- `browser/index.html`: static feature browser (open in any browser).

Headline: sharing rises then falls with depth (r>.5 share 0.25/0.29/0.13);
L4 = names, L12 = concepts, L20 = mixed incl. script features.
"""


def main() -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    for f in FILES:
        api.upload_file(path_or_fileobj=str(root / f), path_in_repo=f,
                        repo_id=f"raihan-js/{REPO}", repo_type="dataset")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=f"raihan-js/{REPO}", repo_type="dataset")
    print(f"uploaded {len(FILES)} files + card to raihan-js/{REPO}")


if __name__ == "__main__":
    main()
