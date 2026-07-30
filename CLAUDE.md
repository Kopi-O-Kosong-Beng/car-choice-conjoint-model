# Guide for AI assistants working in this repository

*(Claude Code reads `CLAUDE.md` automatically; `AGENTS.md` is a copy for other tools.
Humans want `README.md`. The full strategic reasoning behind everything below is in
[`STRATEGY_REVIEW.md`](STRATEGY_REVIEW.md) — read it before proposing any plan.)*

## What this project is

Kaggle competition for SUTD's *The Analytics Edge* (2026), graded coursework. Predict which
of 4 car safety-feature bundles a respondent picks. Metric: mean multiclass logloss.
**R only** — hard competition rule, never propose Python.

**State (30 Jul 2026, late):** best public **1.193** (`sub_20260730_final00.csv`), auto-selected
for private scoring; its forecast was 1.1930, so the segment-reweighted anchor landed exactly.
Previous best was 1.194 (`sub_20260729_nnblend.csv`, second modelling track —
`experiments/iter62_nnblend/`). **Iteration 80 measured the luxury none-rate directly:
r_lux = 0.2236, against final00's implied 0.2314 — inside one sampling sd, so the
segment-margin channel is CLOSED (exact correction worth +0.00037).** The `model/` blend reads nested
1.12819 / public 1.197. Current candidate: `submissions/sub_20260730_final00.csv` — the 2-member
blend + a nested 6-coefficient residual-logit correction + the probe anchor; forecast public
**1.1930** (iter67/68). Kaggle closes **1 Aug 12:00 SGT**, report due **10 Aug** (15 of 30 marks).

> ⛔ **ITERATION 48 CHANGES HOW YOU READ EVERY NUMBER ABOVE THIS LINE.** The design-share
> encoding leaks: `apply_design_encoding()` is called ONCE, before the CV loop, so training
> rows are encoded from a set containing the scored fold. Its honest value is **−0.00596**,
> not the +0.0218 it appears to be worth. **Nested OOF 1.12819 is inflated by ~0.0077**
> (honest ≈ 1.1359), and iteration 47's hyperparameter sweep is void — the honest depth
> optimum is **4–5**, we ship 8. (The corollary "so w_tree should be ~0.2" was measured on
> 30 Jul and does NOT hold on the shipped artifacts — see Corrections below.)
> Read `EXPERIMENTS.md` iteration 48 before quoting any plain-OOF figure.

---


**Allowed without asking:** report writing, documentation, analysis of existing artifacts,
diagnostics that emit no artifacts, reproducibility fixes.

**Requires the user to explicitly re-open the freeze:** any new model, feature, member, or
retune. Say plainly that the project is frozen and why, then ask. *(The user re-opened the
freeze on 30 Jul for the endgame — iterations 67–71. It closes for good when Kaggle does,
1 Aug 12:00 SGT; after that, report work only.)*

**Also true, and it cuts the other way:** the private board is decided by *ranking*, and the
`±0.02` figure quoted in older notes is the **absolute**-score wobble, not the ranking noise.
All teams are scored on the same ~1,500 rows, so differences are *paired* and the relevant SE
is ~0.006–0.012. Our lead over the known rival is ~1–2 SE — probable, not safe. So genuine
gains do matter; it is *search* that is exhausted, not the value of accuracy.
See `STRATEGY_REVIEW.md` Part II.1.

---

## Environment

- R 4.6.0, **not on PATH** on the original machine: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe"`.
  In Bash tool: `"/c/Program Files/R/R-4.6.0/bin/Rscript.exe"`.
- Always run from the repository root; all script paths are relative to it.
- Packages: `data.table`, `xgboost` (≥3.0), `mlogit`, `dfidx`, `glmnet`, `Matrix`, `bayesm`.
- Long runs (10–60 min) are normal — background them, do not block.
- R quirk that has bitten twice: a top-level `else` on its own line is a parse error.
  Wrap in braces: `x <- if (c) { a } else { b }`.

## Rules that must not be broken

1. **Folds are grouped by respondent (`Case`)**, fixed in `model/artifacts/folds.rds`
   (seed 42). **Never regenerate, never split by row.** Test respondents are entirely new
   people; row-wise splitting silently inflates every score. Independent splits for
   *validation* exist as `folds_b.rds` / `folds_c.rds` — new fold structures go to new
   filenames, never over `folds.rds`.
2. **The decision number is the nested blend OOF** from `model/06_blend.R`. Never a plain
   OOF, a training score, or a single fold.
3. **Judge with `model/compare.R`** — paired, respondent-clustered SEs.
   `Rscript model/compare.R <baseline> <challenger>`.
4. **One change per experiment.** Iteration 05 lost an unattributable result to this.
5. **Blend membership is explicit** in `model/members.txt`. Auto-discovery once pulled an
   unvetted artifact into a real submission; `blend.rds` has since disagreed with
   `members.txt` once more. One artifact name, one producing script.
6. **Two Kaggle submissions per day, one team account.** Never suggest a second account.
7. **Nest everything that is fitted** — weights, temperatures, encodings, *and the baselines
   those encodings are built on*. Both leakage incidents were un-nested quantities riding on
   honest-looking reference sets.

## How to judge a number (this is where results die)

**There is no single noise floor. Pick the one matching your decision.**

| comparing | use | measured |
|---|---|---|
| two single models | **model-level seed sd** | **0.00283** |
| two blends | **blend-level seed sd** | **0.00048** |
| fold-to-fold spread within one model | fold SD | 0.013 |

Conflating the first two is exactly how iteration 08's monotone price constraint passed and
survived eighteen iterations before being retracted. **Anything quoted below 0.003 on a
single model, from any iteration before 26, is unresolved.**

**AND THE FLOOR IS NOT THE HARD PART — THE METRIC IS.** Plain nested OOF is a *poor* predictor
of the board and since 1.197 has been an *inverted* one: every local improvement produced a
public regression (1.12341 → 1.209, 1.12341 → 1.211, 1.11210 → 1.205). Use the
**segment-reweighted nested OOF** — reweight training respondents by
`p_test(segmentind) / p_train(segmentind)`, unclipped. It reads **1.19610** for the production
blend against an actual public of **1.197**, closing 98.7% of the +0.069 offset.

Three cautions on it, all measured:
- It is **noisy, and its ESS must be counted in respondents, not tasks**: ESS is **208 of
  1,135 respondents** (the previously quoted "≈3,900 of 21,565" was task-level — exactly 19×
  inflated because weights are constant within a respondent; `model/predict_lb.R` documents
  this). There is **no single floor**: the paired respondent-clustered SE runs from ~0.0004
  (near-twin blends) to ~0.006 (dissimilar candidates) — compute it (`paired_segrw` in
  `experiments/iter67_caltower/harness.R`) instead of eyeballing a threshold. As a *level*
  forecaster it resolves ~0.012 at best.
- It is **blind to the encoding leak**, which is population-independent and inflates plain and
  reweighted alike. It predicted the rotation blend at 1.187; the board said 1.205.
- **Judge with it, never fit on it** (iterations 07 and 46 both refuted fitting on a reweighted
  objective — reweighting cuts effective sample, and variance beats bias; iteration 68
  corroborated it a third time: uniform observation weights beat test-mix weights in the
  residual logit even when judged on the reweighted metric).

Then apply, in order:

- **× 0.8** — measured replication on an independent fold structure (iteration 21: member
  79%, blend 81%). Any accepted change should be re-run under `folds_b` before production.
- **× ~⅓** — measured transfer to the public leaderboard **for leak-free gains in the
  pre-1.197 era**; only ≈0.001 is even *visible* at three decimals. Since 1.197 this
  multiplier is unusable on plain-OOF gains (the correlation went negative — that was the
  leak); predict public from the segment-reweighted anchor instead.
- **Shift audit** (`model/shift_audit.R`): does the gain survive reweighting toward the
  wealthier test population? ~100%+ is structural; 77% was the design encoding; 64% was a
  fatigue term we rejected for exactly this reason. **Predict public from the
  income-reweighted OOF, not the plain one** — but never *optimise* on it (iteration 07).
- **Artifact vs procedure.** A lucky single seed is not a good procedure. Ask which question
  your test answers before running it — `compare.R` compares two artifacts, and the decision
  usually needs two procedures. See iteration 26 and `experiments/iter26_seedbag/expected_blend.R`.

## Data shape

- `long.rds`: one row per (task × alternative), 4 rows per task, sorted by `(No, alt)`.
  Several models rely on that contiguity — do not reorder without care.
- **Alternative 4 is the all-zero "none of these" option**, chosen 30.2% of the time. Its
  `Price` is 0, so it is always the cheapest alternative present — this is *why* rising price
  sensitivity produces a rising decline rate (iteration 25).
- Train: 1,135 respondents × 19 tasks. Test: 263 *different* respondents × 19 tasks.
- Attributes are ordinal tiers, 3–7 levels (Price has 12). **Code as part-worths, not
  numbers** — worth 0.020.

## Known traps

- **xgboost ≥3.0** returns a *matrix* from `predict` for `multi:softprob`. The old flat-vector
  reshape scrambles predictions silently and scored 1.54 with no error raised.
- **Conditional-logit identification:** anything constant within a choice set is unidentified.
  `Price` is 0 only on the none-option, so 0 as reference makes price dummies collinear with
  the none-constant (singular Hessian). Reference must be the lowest level on a *real* bundle.
  Use a rank-revealing QR on the **task-demeaned** design — it caught `HU_L2`, which
  hand-reasoning missed.
- **Long jobs must write their own assembled artifacts as their last act.** Two experiments
  looked like failures for hours when they had in fact finished, because the caller died
  before assembling them.
- ⛔ **THE OLD LEAK HEURISTIC IS WRONG AND COST US THREE SUBMISSIONS.** This file used to say
  "a leak concentrates in one fold." That is true only of *accidental* leaks. A **structural**
  leak — one baked into how a feature is constructed — appears **uniformly in every fold**.
  Iteration 48's encoding leak wins 5/5 folds at every depth and every seed, which is exactly
  why per-fold checking never flagged it. **Per-fold consistency is not evidence of honesty.**
  The tests that do work: (a) build an *isomorphic honest arm* with matched support and compare;
  (b) check whether the gain **grows monotonically with model capacity** — real effects
  saturate, leaks do not (this one went −0.0004 at depth 4 to +0.0777 at depth 10/1400r);
  (c) ablate the suspect feature entirely and see if the gain survives.
- **A contrast is only valid if leak exposure is MATCHED on both sides.** Changing *capacity*
  changes leak exposure, so a capacity comparison on leaked features is contaminated even when
  both arms use identical features (iteration 54–59, which passed five gates and scored 1.205).
  Changing *data* does not — iteration 39's carve removal is a valid contrast for that reason.
- **Old wording, retained so the failure is legible:** a real gain appears in *every* fold; a
  leak concentrates in one. Check
  per-fold before believing anything.
- **`model/artifacts/` is a graveyard, not a menu.** It holds oof_/test_ pairs for ~40 dead
  members. Three are actively dangerous if ever blended: `xgb_pt` (iter30's run was killed at
  fold 3 — the artifact is misnamed, its own header says never cite it), `xgb_resenc*`
  (proven 100% leakage, iterations 12/15), and `blend_freepool5` (a *blend* stored in member
  format — it is the pool that scored 1.209). `members.txt` is the only source of truth.
- **The production tree's own scripts still carry the voided settings.**
  `model/03_xgb_listwise.R` and `experiments/iter26_seedbag/run.R` both call
  `apply_design_encoding()` before the CV loop at depth 8 — the exact iteration-48 defect.
  Kept because removing it is board-neutral (measured, iter48) and a rebuild would spend the
  replication budget; but any *new* work must not source or copy these scripts' feature step.

## If the user re-opens the freeze

1. Write it in `experiments/iterNN_<name>/run.R`, hypothesis **and decision rule** in the
   header *before* running.
2. Consume `model/artifacts/long.rds` (or `wide.rds`) and `folds.rds`.
3. Emit exactly two `data.table`s ordered by `No`, columns `No, p1, p2, p3, p4`:
   `oof_<name>.rds` (21,565 rows) and `test_<name>.rds` (4,997 rows).
4. Verify with `compare.R`, then `shift_audit.R`, then **replicate under `folds_b`**.
5. Only then add to `members.txt` and rerun `06_blend.R`.
6. Record hypothesis, result, and an honest reflection in `EXPERIMENTS.md` — **including for
   failures**, which are the most useful entries.

## Where things are

| question | file |
|---|---|
| **What should I be doing right now?** | **`STRATEGY_REVIEW.md`** — plan, freeze rule, report skeleton |
| How do I run it? | `README.md`, `model/run_all.R` |
| What's been tried, and what is already dead? | `EXPERIMENTS.md` — the ⛔ table especially |
| What are the findings? | `report_notes.md` (the 15-mark deliverable) |
| What scored what on Kaggle? | `submissions/log.md` |
| Course theory | `Vault/` (Obsidian; `Vault/Topics/Topic 3 - Discrete Choice.md`) |

## Corrections to older notes still quoted in this repo

- ~~"the monotone price constraint is worth +0.00172"~~ — **retracted**, iteration 26.
  Paired across 10 seeds: −0.00034, 95% CI [−0.00159, +0.00092], wins 5 of 10.
- ~~"local gains transfer at ~58%"~~ — early rate; decayed to ~⅓; then, from 1.197 to the
  iteration-48 discovery, the plain-OOF→public correlation was **negative** (1.12341 → 1.209,
  1.11210 → 1.205). There is no usable transfer rate for plain-OOF gains any more.
- ~~"private SE ±0.02 swamps everything"~~ — that is *absolute* noise; ranking noise is
  paired and ~0.006–0.012.
- ~~"bundle-level encoding is the next idea"~~ — structurally impossible (bijection),
  iteration 16.
- ~~"the design-share encoding is worth +0.0218"~~ — **retracted, iteration 48.** Honest value
  **−0.00596** (z −3.53), confirmed by a second independent construction at −0.00440 (z −3.45).
  All of the apparent gain is the leave-own-fold-out complement identity.
- ~~"depth 8 / the iteration-47 hyperparameter ranking"~~ — **retracted, iteration 48.** That
  sweep ranked 27 configs on contaminated OOF; the honest ranking is the reverse and the honest
  depth optimum is **4–5**.
- ~~"nested blend OOF 1.12819 is the decision number"~~ — it is inflated by ~0.0077 (honest
  ≈ 1.1359) and, worse, it is the wrong *metric*. Use the segment-reweighted OOF to judge.
- ~~"w_tree = 0.528"~~ — fitted on leaked OOF; an *honest* tree would deserve **0.194–0.278**
  (iterations 48, 61). **But the follow-up claim — "over-weighting the shipped tree is an
  uncorrected defect" — was measured on 30 Jul and refuted** (`iter67` `agent_synth`): forcing
  w_tree to 0.194–0.278 on the *shipped* (leaky) artifacts is flat-to-negative on the
  segment-reweighted metric (w 0.194: −0.00188, z −0.37; w 0.278: +0.00004, z +0.01), and the
  seg-rw optimum ≈0.45 is indistinguishable from 0.528 (+0.00117, z +0.88). Down-weighting a
  leaky member is not the same intervention as replacing it with an honest one. There is no
  free correction here.
- ~~"the segment metric's ESS is ≈3,900 of 21,565"~~ — task-level counting; the honest figure
  is **208 of 1,135 respondents** (`model/predict_lb.R`).
- ~~"a leak concentrates in one fold"~~ — **false for structural leaks**, see Known traps.

## Open — endgame state (30 Jul, ~18h to close)

The former standing item here — "refit the blend weight honestly" — was **measured on 30 Jul
and closed**: flat-to-negative on the segment metric at every honest weight (see Corrections).

What actually remains, in order of evidence:

1. **The one survivor of ~160 arms tested on 30 Jul is the nested residual-logit correction**
   (6 coefficients, uniform weights, penalty 0.03): **+0.00300** seg-rw on production folds
   (z +0.77) and **+0.00628** on `folds_b` (z +1.60) — same sign on two independent fold
   structures, combined ≈1.7σ. It never clears the project's own z ≥ 2 bar; it is the
   best-supported remaining gain, not a proven one. It ships in `sub_20260730_final00.csv`
   together with the probe anchor (a *measured* constant, worth +0.00104 on our margin).
2. **The probe anchor is the only intervention that ever beat its own forecast on the board**
   (predicted 0.00879 recoverable for a file at p4 ≈ 0.211; the board returned ~0.010). On our
   file, which is already near the right margin, it is worth only ~0.001.
3. Pending at time of writing: iter70 (six-family model search), iter71 (capacity ×
   invariance × population), and the symmetry forest. Judge any survivor by the iter67
   harness gates before it goes near a submission.

After Kaggle closes the freeze is permanent and the only deliverable is the report.
