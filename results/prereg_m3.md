# M3 pre-registration (written BEFORE running; GPU busy at write time)

Question: do shared features DRIVE behavior in both languages, or only
correlate? Layer: 12 (strongest sharing, best mid-layer SAE).

## Features
30 concept features from L12 top-100 by paired-r, filtered: must have ≥3 JA
AND ≥3 EN examples among top-6 (excludes script/prefix artifacts like 英),
no proper-name features (Mary/Robert — entities, not concepts). List frozen
in results/m3_features.json before running (feature ids + witness pairs).

## Interventions (per feature, per language JA/EN)
- **Ablate**: zero the feature's activation during a forward pass on 20
  concept prompts per language (short sentences containing the concept,
  from Tatoeba pairs not used in M2); metric = mean Δ logprob of the
  concept's characteristic next tokens (top-5 tokens by base prob, fixed
  before). Control: same with 30 RANDOM features (seed 0).
- **Steer**: add +α·decoder_direction (α = 2× the feature's p95 activation
  on M2 pairs, fixed before) on 20 neutral prompts per language; metric =
  mean Δ logprob of the same concept tokens. Same random controls.
- Stats: mean Δ with 95% CI (Student-t, n=20 prompts); shared-vs-random
  compared by paired bootstrap CI excluding zero.

## Decision rules
- Causal iff shared features move probits beyond random controls (CI
  excludes zero) in BOTH languages for ablation; steering is supportive
  (may fail while ablation holds — stated, not spun).
- "Correlational only" is a valid outcome. Feature labels from examples are
  subjective (spec risk #1) — only the Δ numbers carry weight.
- One run; prompts frozen in results/m3_prompts.json before running.

## Amendment (before run): bilingual-example threshold 3+3 -> 2+2
Top-6 example lists skew EN (L12: 358 en vs 242 ja examples), so >=3 JA
admits only 9 features; >=2 JA + >=2 EN admits 32 (take top 30 by r).
Bilingual-evidence requirement kept, threshold fit to the skew. Name filter
unchanged.

## Amendment 2 (before run): paired design + fixed alpha
Random controls run on the SAME concept prompts as their paired shared
feature (paired shared-minus-random per feature; cleaner than group means).
Steer alpha fixed at 8.0 for all features/languages (constant chosen before
the run; p95-adaptive dropped as unmeasurable without GPU at design time).
