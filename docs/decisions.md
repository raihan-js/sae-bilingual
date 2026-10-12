# sae-bilingual — decisions (append-only, newest first)

## 2026-10-12 — readiness: model loads, hooks work, TL named-support missing
- Base: llm-jp/llm-jp-3-1.8b (24 layers, d_model 2048, vocab 99584).
  transformers 5.19 sdpa default is BROKEN for it (RoPE dim mismatch);
  attn_implementation="eager" works. Documented, must be pinned everywhere.
- TransformerLens has no registered llm-jp config -> plain PyTorch forward
  hooks (spec fallback). Hook captures (seq, 2048) resid_post cleanly.
- Tokenizer splits Japanese at word level (契約/とは/何か) — good for M3.
- Throughput ~1,100 tok/s (1.8B bf16, batch 100) -> 5M-token layer pass ~75 min.
- sae-lens 6.54.0 installed. Disk 84G free: no 100GB cache; plan is
  streaming/online activation generation, 1 layer at a time.

## 2026-10-13 — M0 first SAE trained; held-out numbers rule (results/sae_*.json)
- Layer-12 TopK SAE (2048 -> 32768, k=64), 2M wiki tokens (50/50 ja/en):
  train MSE 0.10, FVE 0.977, dead 0/32768. HELD-OUT: dead 19,336/32,768
  (0.59!), LLM loss 2.626 -> 2.936 (+0.31). In-train dead 0 is meaningless;
  quote held-out only. M1 needs dead-feature handling (resampling/aux loss).
- Fights won: datasets-streaming threads + CUDA = fatal GIL crash ->
  two-phase (corpus.txt then batched dump); v5 layer hook returns plain
  tensor; val forward on 20k tokens OOMs (2.44GB zeros_like) -> batched;
  Adam foreach buffers hog 8GB -> foreach=False + micro-batch 256.
- Shards deleted post-training except 1 test shard (disk discipline).

## 2026-10-13 — three-layer table (held-out): resampling halves death
- L4: dead 0.27, loss +0.28 | L12 (no resample): dead 0.59, loss +0.31 |
  L20: dead 0.58, loss +0.74. Late layer hardest (MSE 0.97, churn 115k).
- Retraining L12 WITH resampling for a fair comparison.
