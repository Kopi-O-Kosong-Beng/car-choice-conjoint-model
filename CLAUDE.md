# Guide for AI assistants working in this repository

*(Claude Code reads `CLAUDE.md` automatically; `AGENTS.md` is a symlink-equivalent copy
for other tools. If you are a human, read `README.md` instead — it covers the same ground
in a friendlier order.)*

## What this project is

Kaggle competition for SUTD's *The Analytics Edge* (2026), graded coursework. Predict
which of 4 car safety-feature bundles a respondent picks. Metric: mean multiclass logloss.
**R only** — this is a hard competition rule, never propose Python.

Current state: local nested CV **1.13044**, public leaderboard **1.199**
(benchmark 1.38629, rival team 1.210).

## Environment

- R 4.6.0. On the original machine it is **not on PATH**: call
  `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe"`. On other machines `Rscript` may work.
- Always run from the repository root — all paths in the scripts are relative to it.
- Packages: `data.table`, `xgboost` (≥3.0), `mlogit`, `dfidx`, `glmnet`, `Matrix`.
- Long runs (10–60 min) are normal. Run them in the background rather than blocking.

## Rules that must not be broken

1. **Folds are grouped by respondent (`Case`)** and fixed in
   `model/artifacts/folds.rds` (seed 42). Never regenerate them, never split by row.
   Test respondents are entirely new people; row-wise splitting silently inflates every
   score by letting a model learn individuals it will then be graded on.
2. **The decision number is the nested blend OOF** from `model/06_blend.R`. Not a plain
   OOF, not a training score, not a single fold.
3. **Judge changes with `model/compare.R`**, which does a paired test with
   respondent-clustered SEs. Fold-to-fold SD is ±0.013, so a real +0.005 improvement is
   invisible in headline numbers but clear when paired. Example:
   `Rscript model/compare.R xgb_de xgb_lw`
4. **One change per experiment.** Violating this already cost us an unattributable result
   (see Iteration 05 in `EXPERIMENTS.md`).
5. **Blend membership is explicit** in `model/members.txt`. Auto-discovery once pulled an
   unvetted artifact into a real submission.
6. **Two Kaggle submissions per day, one team account.** Never suggest a second account.

## How to add a model

1. Write it in `experiments/iterNN_<name>/run.R`, stating the hypothesis in the header
   *before* running.
2. Consume `model/artifacts/long.rds` (or `wide.rds`) and `folds.rds`.
3. Emit exactly two artifacts, both `data.table`s ordered by `No` with columns
   `No, p1, p2, p3, p4`:
   - `model/artifacts/oof_<name>.rds` — 21,565 rows, out-of-fold predictions
   - `model/artifacts/test_<name>.rds` — 4,997 rows, from a refit on all training data
4. Verify with `model/compare.R` against the incumbent.
5. If it wins, add the name to `model/members.txt` and rerun `06_blend.R`.
6. Record hypothesis, result, and an honest reflection in `EXPERIMENTS.md` — **including
   for failures**, which are the most useful entries.

## Data shape

- `long.rds`: one row per (choice task × alternative), 4 rows per task, sorted by
  `(No, alt)`. Several models rely on that contiguity — do not reorder without care.
- **Alternative 4 is the all-zero "none of these" option** and is chosen 30.2% of the
  time, more than any single bundle.
- Train: 1,135 respondents × 19 tasks. Test: 263 *different* respondents × 19 tasks.
- Attributes are ordinal tiers with 3–7 levels (Price has 12). **Code them as part-worths,
  not as numbers** — that was worth 0.020 logloss.

## Known traps

- **xgboost ≥3.0** returns a *matrix* from `predict` for `multi:softprob`. The old
  flat-vector reshape scrambles predictions silently and produced a 1.54 score with no
  error raised.
- **Conditional logit identification:** anything constant within a choice set is
  unidentified. `Price` is 0 only on the none-option, so using 0 as the part-worth
  reference makes price dummies collinear with the none-constant (singular Hessian).
  Reference must be the lowest level occurring on a *real* bundle. Use a rank-revealing
  QR on the **task-demeaned** design matrix to catch the rest — it found `HU_L2`, which
  hand-reasoning missed.
- **Artifact collisions:** two different scripts once wrote to the same `oof_mixl2.rds`,
  making provenance unrecoverable. One artifact name, one producing script.
- **Local gains transfer at roughly 58%** to the leaderboard. Discount accordingly, and
  prefer structurally justified changes. `model/shift_audit.R` tells you whether a gain
  survives reweighting toward the test population.

## Where things are

| question | file |
|---|---|
| How do I run it? | `README.md`, `model/run_all.R` |
| What's been tried? | `EXPERIMENTS.md` (hypothesis → result → reflection) |
| What are the findings? | `report_notes.md` |
| What scored what on Kaggle? | `submissions/log.md` |
| Course theory | `Vault/` (Obsidian; `Vault/Topics/Topic 3 - Discrete Choice.md` is the relevant one) |

## Open ideas

**Read the "👉 PICK UP HERE" section at the top of `EXPERIMENTS.md`.** It is the single
source of truth: current state, five ranked ideas with expected payoffs and failure modes,
and a ⛔ table of things already tested so you do not repeat them.

The top-ranked idea already has a runnable script at
`experiments/iter08_mono_tuned/run.R` — combine the monotone price constraint with the
tuned hyperparameters, which beat the old settings separately (1.14298 and 1.14152 versus
1.14477) but have never been run together.

Do not re-run: listwise hyperparameter tuning (iteration 06, done), blend softening or
income-reweighted blend tuning (iteration 07, both refuted).

Diminishing returns warning: the private leaderboard is ~1,500 rows with SE ≈ ±0.02,
which is larger than everything gained so far. The report carries 15 of 30 marks; past a
point, effort there beats another 0.002 of logloss.
