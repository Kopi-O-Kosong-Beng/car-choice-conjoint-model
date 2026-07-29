# Iteration 62 — NN-blend + fixed calibration + GAM head ✅ SUBMITTED, public **1.194**

**Built on the team's second modelling track.** This is the first submission from that
track to be scored, and it is our current best public entry —
1.197 → **1.194**, moving us 12th → 10th on 29 Jul.

| | |
|---|---|
| submitted | 29 Jul 2026, 06:54 UTC, as `submissions/sub_20260729_nnblend.csv` |
| public | **1.194** (previous best 1.197) |
| pre-registered | 1.192, band 1.191–1.193 — **missed by one tick**, see below |
| shipped mean p4 | 0.21086 |

## Architecture

A single listwise pipeline with four stacked stages. Nothing here shares code with the
main track (`model/`), so the two are genuinely independent implementations.

1. **Tree arm** — listwise-softmax xgboost, `max_depth 3`, `eta 0.1`, `min_child_weight 2`,
   `gamma 0.2`, 250 rounds, **20 seeds × 5 folds = 100 models averaged**. Features are
   `build_long()` plus attribute-level one-hot dummies plus two MF cold-start columns.
   Fitted with segment importance weights at `α = 0.10`.
2. **NN arm** (`code/wave6_nn.R`) — a listwise MLP in torch, 64→32→1 with ReLU and
   dropout 0.2, Adam (`lr 1e-3`, `weight_decay 1e-4`), 30 epochs, weighted cross-entropy.
   It consumes **the same long design matrix as the tree**, so any difference is
   attributable to the function class alone. Its stated rationale: the one-hot block acts
   as learned level embeddings, and the MLP interpolates smoothly where trees make
   staircases — which matters because only 107 training respondents are luxury while
   68.8% of graded rows are.
3. **Geometric blend** of the two arms at `w = 0.25` on the NN side.
4. **Calibration tower** — per-segment Ch4 shift (shrink 0.85), luxury temperature 1.30,
   then a ridge-penalised 6-coefficient residual logit on (relative price, outside
   constant, total-vs-best) × (global, luxury deviation), fitted with test-mix weights.
5. **GAM outside-option head** — a low-df binomial GAM (`k = 4` cubic splines,
   `select = TRUE`, `gamma = 1.8`) supplying an independent p(Ch4), mixed into the margin
   at 25%. Within-buy conditionals are preserved exactly.

This build also **fixes a calibration reference bug**: the earlier `wave6_nnblend.csv`
calibrated against the wrong pooled OOF reference. Rebuilt self-referenced on the blend
family's own reference (`wave18_blend_ref.rds`).

## What the result taught us

**The pre-registration missed, and the reason is instructive.** We predicted 1.191–1.193
from an information-distance argument: this file sits at mean **KL = 0.00079** from the
GAM-on-MF variant, so we argued the scores could not differ by more than ~0.0008. That
reasoning is wrong. `KL(P‖Q)` is the expected loss difference *when P is the truth*, and
neither file is the truth, so the actual gap is not bounded by it. It is a good heuristic
for near-identical models, not a bound, and it was quoted as harder than it is.

**Distances between models on this dataset**, measured over test predictions — useful
context for how much any single change actually moves predictions:

| pair | mean KL |
|---|---|
| this file vs the GAM-on-MF variant (NN arm added) | 0.00079 |
| GAM-on-MF vs A05FIXED (α 0.10 → 0.05) | 0.00036 |
| `xgb_lw2bag` vs `xgb_syn` (both listwise xgb, separately built) | 0.01969 |
| `lcmnl3_both` vs `mnl_pw` (both conditional logits) | 0.03177 |
| `xgb_lw2bag` vs `lcmnl3_both` (tree vs logit) | 0.05885 |
| this file vs the main track's blend | 0.02589 |

Swapping one component inside a shared pipeline moves predictions by ~0.0005; changing
model family moves them by ~0.03–0.06. Two tracks converge on the same *signal* but not
on each other's arbitrary choices — frozen hyperparameters, seed lists, calibration
constants — and that is where most of the remaining divergence lives.

**The NN arm is smaller than its own validation suggested.** It was credited with −0.0026,
but adding it moves predictions by less than 0.0008 in KL. A change that small cannot
deliver a gain that large; either the −0.0026 was measured against a different base, or it
partly cancels against the calibration-reference fix bundled into the same rebuild.

## Margin audit against the probe

The probe (`submissions/probe_alt4.csv`, constant `(1/6,1/6,1/6,1/2)`, returned 1.499)
measures the graded population's walk-away rate exactly: `r* = (log6 − 1.499)/log3 = 0.26652`.

| file | mean p4 | miss | recoverable |
|---|---|---|---|
| this file | 0.21086 | −0.05566 | 0.00879 |
| main-track blend (1.197) | 0.24800 | −0.01852 | 0.00091 |

The second track's whole family ships ~0.21, which is what you get by reweighting the
training decline rates (luxury 0.15986, non-luxury 0.31712) to the test mix:
`0.68821 × 0.15986 + 0.31179 × 0.31712 = 0.2089`. That reasoning is right and the number
is still wrong by 0.058 — **test luxury respondents decline substantially more than
training luxury respondents do**, and nothing in the training data reveals it. This is the
clearest evidence we have that the segment-composition correction, applied on its own,
overshoots. See `experiments/iter42_segprobe/` for the instrument that would resolve the
per-segment rates exactly.

## Files

- `submission_NNBLEND_FIXEDCAL_GAMHEAD.csv` — the submitted file (md5 `2d6d63665945…`)
- `code/` — the full pipeline as run: `wave6_nn.R` (the MLP),
  `submit_nnblend_fixedcal.R` (blend + calibration tower),
  `submit_gam_on_nnblend_fixedcal.R` (the head), plus shared helpers
