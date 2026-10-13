# SAE-Bilingual — shared or separate? Sparse autoencoders on llm-jp-3-1.8B

Do Japanese and English share features inside one bilingual model? One TopK
SAE per layer (4/12/20, 2048→32768, k=64), trained streaming on Wikipedia
JA+EN. Sharing measured as paired-sentence Pearson r on 20k Tatoeba pairs;
causality tested by ablation/steering with random controls.

Plan: `docs/PLAN.md`. Decisions: `docs/decisions.md` (append-only).
Pre-registration: `results/prereg_m3.md` (+2 amendments, all pre-run).

## Results

### M0–M1 — three SAEs (held-out numbers only)

| layer | dead | LLM loss Δ | FVE | notes |
|-------|------|-----------|-----|-------|
| 4 | 0.27 | +0.28 | 0.91 | resampling helped |
| 12 | 0.59→0.72 | +0.30 | 0.98 | per-shard resampling churned (236k resets); cadence lesson |
| 20 | 0.58 | +0.74 | 0.84 | hardest layer |

In-train dead 0.0 is meaningless; quote held-out. Loads only with
`attn_implementation="eager"` (transformers-5 sdpa is broken for this model).

### M2 — sharing rises then falls

Share (r>0.5) / mean r on top-2k both-firing candidates, 20k pairs:

| layer | share r>.5 | mean r |
|-------|-----------|--------|
| 4 | 0.25 | 0.33 |
| 12 | 0.29 | 0.37 |
| 20 | 0.13 | 0.22 |

Qualitative ladder: L4 = NAME detectors (Mary/Maria, r=.99); L12 = CONCEPTS
(burning: *house burned* + 家は燃え…); L20 = mixed incl. script features
(英 kanji-prefix, r=.90). L12 also has 55/100 top features with both-language
examples (vs 25/22). Caveat: L20's SAE is worst-trained, so its drop may be
SAE quality, not model property. Browser: `browser/index.html`.

### M3 — causal: weak JA, null EN (pre-registered, paired, n=30)

| test | shared−random | 95% CI | verdict |
|------|--------------|--------|---------|
| ablate JA | −0.046 | [−0.068, −0.023] | weak (10% atop −0.48 generic damage) |
| ablate EN | −0.010 | [−0.020, +0.0001] | null |
| steer JA/EN | −0.084/−0.030 | excl. zero, wrong sign | uninformative (alpha overshoot) |

Zeroing SAE features off-manifold hurts everything; the design can't cleanly
separate feature importance from SAE fragility. Correlational ladder stands;
causal story doesn't close. Stated, not spun.

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q   # CPU only, no weights needed
```

## Limits
- Hand-rolled TopK SAE (SAELens needs TransformerLens models; llm-jp
  unsupported). No ghost grads; resampling cadence crude.
- Sentence value = token-max (token alignment imperfect, disclosed).
- Tatoeba pairs are short/simple; Wikipedia-only training.
- L20 SAE undertrained; late-layer claims are conditional on better SAEs.
