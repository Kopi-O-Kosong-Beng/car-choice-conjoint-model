# Iteration 08 — parallel track: what the second workstream found

**Status: reference material, NOT a competing pipeline.** A second member of the team
built an independent codebase (`../../../Comp/`, not under version control) before the two
tracks were compared. It scored **worse overall** and is being abandoned as a pipeline.
This note records the parts worth keeping, and — more usefully — the ~25 ideas it
tested and rejected, so nobody spends a day rediscovering them.

**Read the caveat in "Why the scores are not comparable" before quoting any number here.**

---

## Why the scores are not comparable

The parallel track used a **different CV protocol**, so its numbers cannot be placed
alongside this repo's 1.13883:

| | this repo | parallel track |
|---|---|---|
| fold seed | 42 | 2026 |
| grouping | by `Case` | by `Case` (same principle) |
| decision number | **nested** blend OOF | **plain** OOF |
| best score | **1.13883** nested | 1.1514 plain |

A plain OOF is optimistic relative to a nested one, so the repo's true lead is *larger*
than the 0.013 headline difference. Any claim below is reported as a **paired delta
within the parallel track**, never as an absolute comparison against this repo.

---

## Finding 1 — a tuned listwise configuration (PORTABLE, highest value)

A 40-config random search over the listwise/shared-utility xgboost, on enriched features.
The search consistently preferred **much heavier regularisation** than hand-set values:

```
max_depth 5, min_child_weight 80, eta 0.04, subsample 0.85,
colsample_bytree 0.30, alpha 1, lambda 10, nrounds 800
```

The top four configs all chose `colsample_bytree` in 0.3–0.9 with the winner at **0.30**,
and the top two used **800 rounds**. Interpretation: once the long-format feature matrix
is enriched, the columns are highly redundant, and the optimum moves toward aggressive
column subsampling and longer, slower training.

**Evidence.** Adding this config plus an exact-softmax twin of it to the mate side was
confirmed on **5 fresh, independently seeded fold partitions** — better on 5/5, mean
−0.00214, 95% CI [−0.00246, −0.00182], **p = 0.0001**. It also survived a Hansen SPA test
(p = 0.006) that prices in the ~30 comparisons made that day, and PBO via CSCV was 0.067.

**Do not assume it transfers.** It was tuned on the parallel track's feature set, which
differs from `model/00_load.R`'s. Re-run the search — or at minimum re-validate this exact
config — against `long.rds` before adopting. Acceptance test: `model/compare.R` against
the incumbent `xgb_lw`, on the repo's fixed folds.

---

## Finding 2 — SEGMENT shift is more extreme than INCOME shift (PORTABLE)

`model/shift_audit.R` reweights on **income**. The parallel track reweighted on
**`segment`**, and the shift there is far more violent:

| segment | train | test | weight (test/train) |
|---|---|---|---|
| Midsize Luxury Utility | 4.3% | 31.9% | **7.40** |
| Prestige Luxury Sedan | 5.1% | 36.9% | **7.22** |
| Full-size Pickup | 11.3% | 21.7% | 1.92 |
| Midsize Utility | 18.6% | 6.1% | 0.33 |
| Midsize Car | 33.7% | 3.4% | 0.10 |
| **Small Car** | **27.0%** | **0%** | **0.00** |

60.7% of training respondents belong to segments that are 3.4% of test. Reweighting leaves
**~208 effective respondents of 1,135** — an 82% loss of statistical power, and the reason
the reweighted metric can only resolve differences of roughly 0.012.

**Why it matters more than a diagnostic.** Per-segment performance shows the model is
weakest exactly where the grade is decided:

| segment | n resp | logloss | alt-4 observed | alt-4 predicted |
|---|---|---|---|---|
| Midsize Luxury Utility | 49 | **1.2483** | 0.161 | **0.225** |
| Prestige Luxury Sedan | 58 | **1.2689** | 0.159 | **0.216** |
| Full-size Pickup | 128 | 1.1069 | 0.305 | 0.312 |
| Midsize Car | 383 | 1.1424 | 0.320 | 0.311 |
| Small Car | 306 | 1.1473 | 0.321 | 0.308 |

The two segments that make up **69% of test** have the worst logloss *and* a systematic
**+0.06 over-prediction of the outside option** — luxury buyers opt out ~16% of the time
and the model says ~22%. That is a specific, addressable bias, not generic difficulty.

**Concrete consequence in the parallel track.** A wide 4-class xgboost scored 1.1659 on
plain OOF but ~1.2477 segment-reweighted — the worst component on the graded population
while looking fine locally. Dropping it was worth −0.0079 reweighted
(95% CI [−0.0160, −0.0003]) and cost nothing plain.

**Open question for this repo:** are income shift and segment shift the same phenomenon?
Luxury segments and high income are obviously correlated. Cheapest test: add segment
reweighting alongside income in `shift_audit.R` and check whether a change's retention
percentage differs materially under the two. If they agree, keep income and note the
robustness. If they disagree, segment is the sharper instrument.

---

## Finding 3 — multiple-testing machinery (PORTABLE, decision-making not score)

After ~30 comparisons on one fold partition, the expected maximum t-statistic under the
pure null is ~2.07 — so a "winner" at t = 2.2 is indistinguishable from noise. Three tools
were implemented on a respondent-clustered bootstrap and are worth reusing before any
end-of-competition selection:

- **Hansen's Model Confidence Set** — reduces N models to the set that cannot be
  distinguished from the best. Take the max *within* each bootstrap resample, so
  correlation between near-identical candidates is handled exactly.
- **White's Reality Check / Hansen's SPA** — "does the best of N genuinely beat the
  incumbent after pricing in the search?"
- **PBO via CSCV** — probability that the in-sample winner lands below median
  out-of-sample. Needs no retraining; runs on an existing per-respondent loss matrix.

Script: `scripts/quant_selection.R`. It expects a per-respondent mean-logloss matrix
(respondents × candidate models), which this repo can build from its `oof_*.rds` artifacts.

---

## What was tested and REJECTED (saves ~2 days)

All rejected on a respondent-level paired bootstrap, 95% CI excluding zero in the wrong
direction or spanning it. Roughly in order of how promising they looked first:

**Blend / calibration layer — exhausted, all five failed**
- Cross-fitted re-tuning of blend weights and temperature (the hand-set values were
  already optimal; per-fold picks jittered, i.e. the grid was fitting noise)
- Per-alternative (vector-scaling) recalibration — the fitted map came out as the
  **identity** (a ≈ 0, b = 0.99), so there was no miscalibration to exploit
- Arithmetic instead of geometric blending
- A free 4-way simplex over all base models, cross-fitted — *worse* than a fixed nested
  structure; more freedom, worse result
- Task-conditioned alt-4 calibration. Note the fatigue trend is **real** (outside-option
  rate rises 24% → 34% across the 19 tasks, correlation 0.86) and the residual still
  correlates 0.55 with task number — but correcting it *hurts*, because shifting the mean
  level of alt 4 does not improve discrimination *between* respondents within a task.

**Structural**
- Monotone price constraints on the trees. Diagnostic: mean |probability change| on
  top-decile-price rows was 0.0098 vs 0.0091 on all rows — the constraint **never binds**,
  because unconstrained trees are already effectively monotone in own price.
- Slot-permutation augmentation (permute alternatives 1–3, relabel, 3× data) — the
  relative/rank features are already slot-symmetric, so it adds redundancy not signal
- `rank:pairwise` objective with a cross-fitted temperature — ordering alone is not enough
  when the metric scores calibrated probabilities
- Within-respondent context features (each attribute z-scored against everything that
  respondent was shown). **Worst result of the whole track** (+0.0066). The designs are
  randomised per respondent, so "the range this person saw" is design noise, not an
  anchoring reference point.

**Ensembling**
- Bagged greedy ensemble selection (Caruana) over a ~50-matrix library — reached 1.1515,
  i.e. it did **not** beat a simple hand-built 4-member blend at 1.1513
- Learned simplex weights by EM, bagged over respondent resamples — no concentration
  problem appeared and no gain over equal weights

**Already known to this repo, confirmed independently**
- The exact softmax/listwise objective (gradient unit-tested against numerical
  differentiation, max error 5e-10) — same idea as `03_xgb_listwise.R`, arrived at
  independently. Two of us finding it separately is decent evidence it is right.

---

## Files copied here

- `scripts/quant_selection.R` — MCS / SPA / PBO / bagged EM weights
- `scripts/segment_shift.R` — segment reweighting, per-segment breakdown, reweighted
  paired bootstrap
- `scripts/softmax_core.R` — exact softmax objective + its gradient self-test
- `PROJECT_LOG.md` — the parallel track's full handover (its own version history)

All of them assume the parallel track's harness and **will not run as-is here.** They are
reference implementations; port the logic, not the files.
