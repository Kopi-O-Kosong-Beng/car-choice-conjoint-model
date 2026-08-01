# Guide for AI assistants working in this repository

*(Claude Code reads `CLAUDE.md` automatically; `AGENTS.md` is a copy for other tools.
Humans want `README.md`. The full strategic reasoning behind everything below is in
[`STRATEGY_REVIEW.md`](STRATEGY_REVIEW.md) — read it before proposing any plan.)*

## What this project is

Kaggle competition for SUTD's *The Analytics Edge* (2026), graded coursework. Predict which
of 4 car safety-feature bundles a respondent picks. Metric: mean multiclass logloss.
**R only** — hard competition rule, never propose Python.

## ✅ FINAL STATE — the competition is over, only the report remains

**Kaggle closed 1 Aug 2026, 12:00 SGT. Selected submission:
`submissions/cand_pool5050_final00.csv` — public 1.185 (3rd), private 1.185 (4th).**

The graded file is a log-opinion pool at **fixed `w = 0.5`** of our two modelling tracks, both
anchored to the probe-measured `r* = 0.266481153`:

- **A, main track** — `sub_20260730_final00.csv` (public 1.193): the 2-member nested blend
  (`xgb_lw2bag` 0.528 + `lcmnl3_both` 0.472) → a nested 6-coefficient residual logit → the anchor.
- **B, second track** — `sub_20260729_nnblend.csv` (public 1.194) anchored: the team's second
  modelling track, `experiments/iter62_nnblend/`, built by a teammate and sharing no code with
  `model/`. Iteration 82 found it had never been anchored (mean p4 0.21086 vs `r*` 0.26648);
  correcting that is where most of the 1.193 → 1.185 gain came from.

`w` was never tuned and could not honestly have been: the second track has no OOF on
`folds.rds` and can never have one (adjusted Rand ≈ 0.002 against it — independent partitions).
0.5 spends no selection budget, and Hölder gives `loss(pool) ≤ max(L_A, L_B)` on every row set.
`experiments/iter82_provenance/track_distances.R` confirms the pool is buying real diversity:
the tracks sit at conditional distance **0.01291**, 2.3× above the same-model-reseeded floor
of 0.00552.

**The result that matters most for the report: public 1.185, private 1.185 — drift of zero to
three decimals, against a ~1,499-row private draw whose own sd is ~0.011.** Nothing overfitted
the leaderboard, and every constant calibrated on the public 70% transferred intact. The rank
still slipped 3rd → 4th on an unchanged score, which is precisely the paired ranking noise
(SE 0.006–0.012) this project spent two weeks warning about. **The rank moved; the model did not.**

Three pre-registered forecasts, all committed to git before upload, all correct: `r*` by
algebra; 1.1930 for `final00` (returned 1.193); 1.186 band 1.184–1.189 for the shipped pool
(returned **1.185**).

**Report due 10 Aug** (15 of 30 marks). `report_notes.md` now carries a drafted "The graded
model, described" section for rubric item (i), and the post-mortem for item (ii) is at the end
of `submissions/log.md`.

> ⛔ **ITERATION 48 CHANGES HOW YOU READ EVERY NUMBER ABOVE THIS LINE.** The design-share
> encoding leaks: `apply_design_encoding()` is called ONCE, before the CV loop, so training
> rows are encoded from a set containing the scored fold. Its honest value is **−0.00596**,
> not the +0.0218 it appears to be worth. **Nested OOF 1.12819 is inflated by ~0.0077**
> (honest ≈ 1.1359), and iteration 47's hyperparameter sweep is void — the honest depth
> optimum is **4–5**, we ship 8. (The corollary "so w_tree should be ~0.2" was measured on
> 30 Jul and does NOT hold on the shipped artifacts — see Corrections below.)
> Read `EXPERIMENTS.md` iteration 48 before quoting any plain-OOF figure.

**⚠️ Corroborating this from the other track: our OOF is an anti-signal below ~1.128.** Six
measured (local, public) pairs exist. The four inside the established model family transfer at
~37%. Everything that departed from it failed: free-sign weights (1.12341 → **1.209/1.211**),
rotation blend (1.11210 → **1.205**), and demographic feature ablation (1.09690 → **1.265**).
That last one passed 5/5 folds, three out-of-population holdouts and a breadth test — and is
now explained by iteration 48: dropping demographics made the model lean harder on the leaky
encoding. **A better local score is not evidence.**

**The one durable gain that is not fitted at all.** The alt-4 probe — a constant
`(1/6,1/6,1/6,1/2)` submission — scored 1.499, and `r = (log6 − score)/log3` gives the test
none-rate **0.26648** exactly. Correcting the shipped marginal to it is worth **+0.00104** and
transfers at ~100%, because it is *measured on the test set* rather than fitted on our folds.
See `EXPERIMENTS.md` "Round 4" for three mechanisms that were confidently argued and refuted
in the same round — including an encoding-leak test whose instrument (an AUC contrast) could
not detect a *structural* leak, which iteration 48 then found by the capacity test.

---

## ⛔ READ THIS FIRST: the freeze is now PERMANENT — the only deliverable is the report

**Kaggle closed on 1 Aug 2026. There is nothing left to submit and no score left to move.**
The only remaining deliverable is the 8-page report, due **10 Aug**, worth 15 of 30 marks.

**The correct action is report writing.** If you arrive wanting to propose a new model,
feature, encoding, or blend architecture — there is no longer any mechanism by which it could
matter, and `EXPERIMENTS.md`'s ⛔ table probably already has your idea with the number that
killed it anyway. The search space was measured-exhausted before the close: the 30 Jul endgame
ran ~160 arms and produced exactly one survivor at ≈1.7σ.

**Allowed without asking:** report writing, documentation, analysis of existing artifacts,
diagnostics that emit no artifacts, reproducibility fixes.

**Do not do without the user explicitly asking:** any new model, feature, member, or retune.
Say plainly that the competition is closed and the model is final, then ask what they actually
want. The one legitimate reason to run a model now is to *produce a figure or number the
report needs* — and even then, prefer reading an existing artifact.

**Why the freeze is not laziness:** every additional selection event spends part of a measured
replication budget. Wins here replicate on an independent fold structure at only **~80%**
(measured, iteration 21), and the plain-OOF→public transfer rate is no longer usable at all —
it went *negative* during the leak era (see Corrections). Searching more does not just fail to
help; it degrades what already works.

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
- Train: 1,135 respondents × 19 tasks. Test: 263 *different* respondents × 19 tasks (4,997 rows).
  **The public leaderboard scores ~70% of those rows (~3,498); the private/graded score is the
  other ~30% (~1,499).** Every score we have ever read — including the probe measurements — is
  a statement about the public 70% only.
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
- ⛔ **THE 30 JUL ENDGAME IS DOCUMENTED BUT ONLY PARTLY COMMITTED.** Iterations 63–79 exist in
  prose only; `experiments/` jumps from `iter62_nnblend` to `iter80_mprobe`. Still **not in the
  repo**, though cited as evidence by `CLAUDE.md`, `EXPERIMENTS.md` and `report_notes.md`:
  `model/predict_lb.R`, `experiments/iter67_caltower/harness.R` (and its `agent_*.log`),
  `experiments/iter68_final/replicate_foldsb.R`, `sweep.rds`, `oof_hurdle_rank.rds`,
  `oof_xgb_tau00/10.rds`. So the +0.00300/+0.00628 residual-logit numbers and the ESS-208
  figure **cannot be verified from this checkout** — cite them in the report only as recorded
  measurements, and say so if precision matters.
  **Partly resolved:** `sub_20260730_final00.csv` was likewise missing, and
  `experiments/iter82_provenance/build_candidates.R` now reconstructs it as
  `sub_20260730_final00_reconstructed.csv` by inverting `mprobe285`. Every within-buy
  conditional is recovered exactly; the segment split is pinned only to ~1e−4 by the 4-d.p.
  `r_lux`. It unblocks `iter80_mprobe/run.R` and `invert.R`, and it is what the shipped pool
  was built from — but it is a **faithful reconstruction, not** the byte-identical file that
  scored 1.193, and must never be described as such.

## If the user asks for a new run anyway (report figures, or curiosity)

*The competition is closed, so nothing here can change a score. Follow it regardless — the
report's credibility rests on every number in it having been produced this way.*

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
| **What should I be doing right now?** | **Writing the report.** `report_notes.md` for material, `STRATEGY_REVIEW.md` for the skeleton and the noise argument |
| How did it end, and what did that settle? | `submissions/log.md`, final section — the post-mortem |
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

## Closed — how the competition actually ended, and what it settled

Kaggle closed 1 Aug 2026. **`cand_pool5050_final00.csv`: public 1.185 (3rd), private 1.185
(4th).** Nothing about the model is open any more. What the close *settled*, in order of how
much the report should lean on it:

1. **Public 1.185, private 1.185 — zero drift.** The private set is ~1,499 rows with a
   sampling sd of ~0.011 on the none-rate alone, and the score did not move a tick. This is
   the direct answer to rubric item (ii) and the strongest fact the project owns. It says the
   pipeline did not overfit the public board, and that `r*` — calibrated purely against public
   rows — transferred intact to rows it had never touched.
2. **Rank moved 3rd → 4th on an unchanged score.** Exactly the paired ranking noise
   (SE 0.006–0.012) this repo spent two weeks distinguishing from the ±0.02 absolute wobble.
   The prediction was that public rank is nearly uninformative about private rank; the outcome
   demonstrates it on our own file.
3. **The gain came from measurement, not search.** 1.193 → 1.185 came from noticing the second
   track had never been anchored to the already-measured `r*`, plus a fixed-weight pool with a
   Hölder bound. Neither spent selection budget. ~160 searched arms on 30 Jul produced one
   ≈1.7σ survivor; a constant recovered by algebra from one returned score was worth more.
4. **Three pre-registered forecasts, three hits**, all committed before upload: `r*`; 1.1930
   for `final00` (returned 1.193); 1.186 band 1.184–1.189 for the pool (returned 1.185).

Still-unverifiable-from-this-checkout items are listed under Known traps — do not quote the
+0.00300/+0.00628 residual-logit numbers as reproducible.

**The only deliverable now is the report, due 10 Aug.** Start from `report_notes.md`.
