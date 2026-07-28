# Full-Test Robustness Design

**Date:** 29 July 2026  
**Status:** Approved  
**Decision owner:** Codex, acting on the user's instruction to choose the statistically
strongest course that protects both public and private performance

## Objective

Minimise expected logloss on the complete unseen test population. Public and private scores
are treated as two samples from that population, not as separate objectives. The work will
not select a model because it happens to score well on the public subset.

The current production system is the reference:

- `xgb_lw2bag` with blend weight approximately 0.528;
- `lcmnl3_both` with blend weight approximately 0.472;
- nested respondent-grouped OOF logloss 1.12819;
- public logloss 1.197.

A public score below 1.18 is an aspiration, not an adoption criterion. The measured evidence
does not support promising a 0.017 improvement without taking unacceptable private-score
risk.

## Non-goals

This work will not:

- introduce a new model family, feature, encoding, or blend architecture;
- choose among several submissions using public leaderboard performance;
- use row-wise validation, regenerate `model/artifacts/folds.rds`, or modify existing fold
  assignments;
- use free-sign blend weights, demographic ablation, calibration-curve probing, or a model
  slate;
- optimise against income-reweighted OOF;
- use Python.

## Chosen strategy

The final prediction will use two changes at most:

1. a measured alternative-4 marginal correction, already justified by the leaderboard probe;
2. bagging over latent-class EM starts, but only if a pre-registered procedure-level audit
   shows that the bagged procedure is more robust than a random single-start procedure.

The marginal correction is accepted now. EM-start bagging is conditional on the gates below.
If any EM gate fails or the computation cannot finish safely before the competition deadline,
the final candidate is the corrected existing production blend.

## Fixed inputs

The implementation must consume the existing artifacts without regenerating them:

- `model/artifacts/long.rds`;
- `model/artifacts/wide.rds`;
- `model/artifacts/folds.rds`;
- `model/artifacts/folds_b.rds`;
- `model/artifacts/oof_xgb_lw2bag.rds`;
- `model/artifacts/test_xgb_lw2bag.rds`;
- `model/artifacts/oof_xgb_lw2_b.rds`;
- `model/artifacts/oof_lcmnl3_both.rds`;
- `model/artifacts/oof_lcmnl3_both_b.rds`;
- `model/artifacts/test_lcmnl3_both.rds`.

All prediction tables must be ordered by `No`, contain exactly `No, p1, p2, p3, p4`, have
strictly positive finite probabilities, and have row sums equal to one within `1e-9`.

## EM-start experiment

### One change under test

The only varying quantity is the random initialisation used alongside the deterministic EM
start. Features, penalties, convergence tolerance, iteration budgets, task-position terms,
folds, and prediction code remain identical to `lcmnl3_both`.

Five procedures are fixed before fitting. Their random-start seeds are:

`4242, 1103, 2207, 3301, 4409`.

Seed 4242 is represented by the existing production artifact where its producing settings
match byte-for-byte. The other four are new fits. Every procedure screens the same
deterministic start and one random start using training likelihood only. Held-out choices are
never used to select a start.

Each procedure produces:

- a five-fold OOF prediction matrix under `folds.rds`;
- a five-fold OOF prediction matrix under `folds_b.rds` if the primary variance gate passes;
- a full-training test prediction matrix.

Per-seed outputs live under `experiments/iter36_emstart/`. No per-seed run may overwrite a
production artifact.

### Bagging operator

Bagged probabilities are the arithmetic mean of the five probability matrices, followed by
row normalisation. Arithmetic averaging is fixed in advance because the five fits represent
uncertainty over latent-class solutions; it integrates predictive distributions. Geometric
versus arithmetic averaging will not be selected using OOF performance.

The candidate member names are:

- `lcmnl3_bothbag` for the primary fold structure;
- `lcmnl3_bothbag_b` for the independent fold structure.

## Statistical estimand

The relevant comparison is not the stored production artifact against the bag. A stored
artifact may be a lucky EM draw, while its full-data test refit is a separate draw.

For each of the five start procedures:

1. pair that procedure's latent-class OOF predictions with the fixed `xgb_lw2bag` member;
2. fit the existing simplex log-opinion combiner inside each outer fold;
3. score the held-out fold;
4. average the five fold losses to obtain one nested blend score.

The single-start estimand is the mean of those five nested blend scores. The bagged estimand
is the nested blend score obtained when the latent-class member is the arithmetic average of
the same five procedures. The primary gain is:

`mean(single-start nested scores) - bagged-member nested score`.

The same calculation is repeated using `folds_b.rds`, with `xgb_lw2_b` as the fixed tree
member for both the single-start and bagged arms. Using the fold-B tree artifact is necessary:
reusing a member OOF artifact built under the primary grouping would make the fold-B combiner
evaluation non-nested. Artifact-versus-artifact results from `model/compare.R` are reported as
diagnostics but do not decide adoption.

## Pre-registered gates

The bagged latent-class member is adopted only if every applicable gate passes in order.

### Gate 1: material start variance

Compute the sample standard deviation of the five single-start model OOF logloss values under
`folds.rds`.

- If SD is below 0.00100, stop. EM-start variance is immaterial and production remains
  unchanged.
- If SD is at least 0.00100, create the primary bag and continue.

### Gate 2: primary procedure gain

The primary procedure-level gain must be strictly positive. The income-reweighted
procedure-level gain must also be non-negative.

If either value is negative, reject bagging.

### Gate 3: independent-fold replication

Repeat the five procedures and the procedure-level comparison using `folds_b.rds`.

The independent-fold procedure gain must be strictly positive. The ratio

`independent-fold gain / primary gain`

must be at least 0.50. This is stricter than merely observing the expected 0.80 replication
factor and protects against a gain tied to one respondent grouping.

### Gate 4: concentration audit

Across the ten held-out folds from the two fold structures:

- at least seven must improve;
- no fold may worsen by more than 0.00100 logloss;
- the respondent-clustered mean improvement must be positive.

If the gain is concentrated in one fold or a small respondent group, reject bagging.

### Gate 5: artifact integrity

Before promotion:

- prediction identifiers must match the reference identifiers exactly;
- row counts must be 21,565 for OOF and 4,997 for test;
- all probabilities must be finite and strictly positive;
- row sums must equal one within `1e-9`;
- recomputing each reported score directly from the saved artifact must reproduce the logged
  score within `1e-10`.

## Blend construction

If all gates pass, replace only `lcmnl3_both` with `lcmnl3_bothbag` in an experimental
two-member simplex blend. The other member remains `xgb_lw2bag`. Blend weights, temperature,
and epsilon are fitted using the existing nested procedure in `model/06_blend.R`.

The experiment must use `BLEND_MEMBERS` and `BLEND_OUT` so it cannot overwrite
`model/artifacts/blend.rds` or change `model/members.txt` before promotion is approved by the
gates.

If any gate fails, use the existing `model/artifacts/test_blend.rds` unchanged.

## Measured marginal correction

The alternative-4 target is fixed at:

`r4 = (log(6) - 1.499) / log(3) = 0.26648...`

After choosing either the accepted bagged blend or the unchanged production blend, solve one
constant log-odds shift for alternative 4 so that mean test `p4` equals `r4`.

For every row:

- preserve the rank order of `p4`;
- preserve the relative odds among alternatives 1, 2, and 3;
- renormalise to a valid probability vector.

The optional heterogeneity-mixing correction from iteration 31 is excluded. Its estimated
gain of approximately 0.0003 is too small relative to its modelling assumptions for a
public-and-private robustness objective.

## Final submission rule

Exactly one new competitive submission is produced:

- accepted EM bag plus measured marginal correction, if all gates pass in time; or
- existing production blend plus measured marginal correction otherwise.

No alternative from the six-file slate is uploaded as part of this design. The public score
is recorded after upload, but it does not trigger another model or submission search.

## Failure handling and resumability

Each long-running seed writes its own complete artifact as its final action. Re-running skips
an artifact only after validating its seed, fold structure, settings fingerprint, dimensions,
identifiers, and probabilities. Partial or invalid files are rejected rather than silently
reused.

The combine and evaluation stages refuse to run unless all five required seed procedures are
present and valid. A failure on `folds_b` rejects promotion; it does not fall back to the
primary-only result.

## Verification and research record

The implementation must:

- run `model/compare.R` for the member-level artifact contrast;
- report the procedure-level nested estimand on both fold structures;
- report income-reweighted gains without optimising against them;
- report per-fold and respondent-clustered effects;
- append the hypothesis, decision rules, results, verdict, and reflection to
  `EXPERIMENTS.md`, including a failed result;
- leave `model/members.txt` unchanged unless every adoption gate passes.

## Expected outcome

The measured marginal correction has an expected gain of approximately 0.00104 and is the
main improvement. EM-start bagging is expected to contribute between zero and roughly 0.001;
the audit may correctly decide to contribute nothing.

This design maximises the probability that any accepted gain transfers to both public and
private scoring. It deliberately gives up low-probability leaderboard upside in exchange for
protection against the empirically observed failure of locally selected structural changes.
