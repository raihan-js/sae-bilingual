# sae-bilingual — M0 spec (approved 2026-10-12)

Goal: prove the activation pipeline + train the FIRST SAE (layer 12) with
reported quality. Approach A (owner-approved): streaming activations, one
layer at a time, no giant cache.

## Streaming (scripts/stream_acts.py)
- Source: wikimedia/wikipedia streaming (20231101.ja + 20231101.en, 50/50
  mix), tokenized to 128-token blocks, BOS prepended.
- Hook: layer-12 resid_post via plain PyTorch forward hook (no TL support),
  bf16 model, fp16 shards of 100k tokens (~410MB) to data/acts/l12/.
- Target: 2M tokens (~8GB, deleted after SAE training except 1 test shard).
- Throughput + disk recorded to results/m0.json.

## SAE training
- First try SAELens cache-train path (timebox 1h); fallback: hand-rolled
  TopK SAE (~80 lines, fully owned). Config: d_model 2048, 16x (32,768
  features), k=64 active, lr 1e-3, 2M tokens.
- Metrics: fraction of variance explained, L0, dead-feature rate, LLM loss
  delta ( Pile sample, base vs reconstructed).

## Non-goals (M0)
Early/late layers, feature analysis, steering (M1-M3). No activation text
stored (int token IDs + fp16 acts only; Wikipedia is CC-BY-SA, fine).
