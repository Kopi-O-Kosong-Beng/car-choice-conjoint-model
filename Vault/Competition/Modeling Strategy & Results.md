---
title: Modeling Strategy & Results
type: strategy
updated: 2026-07-26
tags: [competition, models, results]
---

# Modeling Strategy & Results

Code: `model/` (entry point `model/run_all.R`) · Research log: `EXPERIMENTS.md` ·
Scores: `submissions/log.md` · How to run: `README.md`

## Current standing

| | logloss |
|---|---|
| Benchmark (25% each) | 1.38629 |
| Team's first submission | 1.2230 public |
| Rival team's best known | 1.210 public |
| **Ours** | **1.201 public** / 1.13044 local nested CV |

## The submitted model

A blend of exactly **two** models, combined in log-space with a fitted temperature (0.939)
and a small uniform mixture (2.7%):

| model | file | OOF | weight |
|---|---|---|---|
| xgboost, **listwise softmax objective** + design-encoding features | `model/03_xgb_listwise.R` | **1.14477** | 0.62 |
| conditional logit, **part-worth coded** levels + price×demographic interactions | `model/02_mnl_partworth.R` | 1.15686 | 0.38 |

Six other models were fitted and given **zero weight** (linear-coded logit, elastic net,
wide 4-class xgboost, pointwise long xgboost, mixed logit v1 and v2). They live in
`model/legacy/`. The optimizer keeps exactly one member from each family — trees and
linear utility — and discards the rest as redundant, which says the two families make
genuinely different errors while models within a family do not.

## Principles we hold to

1. **Honest validation or nothing.** Five folds grouped by **respondent** (`Case`), fixed
   in `model/artifacts/folds.rds`, shared by every model. The test set is 263 people who
   appear nowhere in training, so held-out folds must be new people too. Splitting by row
   would let a model learn a person from 15 of their tasks and be graded on the other 4.
2. **Nested blending.** Blend weights are refit five times, each excluding the fold it is
   scored on. No number we act on has seen its own tuning data.
3. **Paired tests, not headline numbers.** Fold-to-fold SD is ±0.013, so a genuine +0.005
   improvement is invisible in raw comparisons but clear when paired on identical rows
   with respondent-clustered SEs (`model/compare.R`).
4. **One change per experiment.** We broke this once (iteration 05) and lost the ability
   to attribute a result.
5. **Spend submissions on information.** Two per day. Record every public score.

## Score history

| stage | nested OOF |
|---|---|
| first pipeline (mnl + mixl + xgb_long + xgb_wide) | 1.15294 |
| + part-worth logit, + design encoding | 1.14211 |
| + listwise objective | **1.13044** |

## What's queued

1. Residual-based design encoding — encode the model's *error* per design, not the raw share
2. Bundle-level rather than choice-set-level encoding (better support per unit)
3. Retune xgboost for the listwise objective — hyperparameters were inherited from a
   different objective and round counts jumped 250 → 600, so the optimum moved
4. Latent-class MNL — discrete taste segments, strong report material
5. Hierarchical Bayes (`bayesm`) — highest ceiling, slowest

**Diminishing returns:** the private leaderboard is ~1,500 rows with SE ≈ ±0.02, larger
than everything gained so far. The report is worth 15 of 30 marks. Past a point, writing
beats tuning.

Findings for the report: [[Key Findings]] ·
Theory: [[Topic 3 - Discrete Choice]] ·
Data: [[Data Dictionary]]
