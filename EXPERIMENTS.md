# Experiment Log — Analytics Edge Competition

**Purpose:** a running research record. Every iteration keeps its own program under
`experiments/iterNN_<name>/`, and every iteration ends with a written reflection:
*why did it score what it scored, and what would I do differently?*
This file is the index. Anyone (including a future session with no memory of today)
should be able to read it top to bottom and continue the work.

---

## Protocol (the rules we hold ourselves to)

1. **One decision number.** Nested, respondent-grouped OOF from `model/06_blend.R`.
   Never a plain in-sample or self-tuned number.
2. **Fixed folds.** `model/artifacts/folds.rds`, seed 42, grouped by `Case`. Never regenerate.
   Every model, every iteration, same split — otherwise scores aren't comparable.
3. **The 1-SD rule.** Fold-to-fold SD is ~0.013. A change smaller than that is noise,
   not progress. Report deltas with that yardstick attached.
4. **Explicit membership.** `model/members.txt` lists which models enter the blend.
   No auto-discovery of stray artifacts into a graded submission.
5. **Hypothesis before result.** Each iteration states what it expects and why *before*
   running. Being wrong on the record is the point — that's where the learning is.
6. **Submissions are scarce.** 2/day. Upload only when nested OOF beats the incumbent
   by more than 1 SD, or when the public score would resolve a genuine question.

---

## Adversarial review of the plan

Five perspectives, argued honestly, then a ruling. Written after Iteration 0's
diagnostics, so the arguments have facts to stand on.

### 🎓 The choice econometrician
> "You're treating a structural problem as a prediction problem. This is a random-utility
> system with panel structure: 19 observations per person, a genuine outside option, and
> demonstrably heterogeneous tastes — sd of the none-constant came out at 2.1, which is
> enormous. Hierarchical Bayes is the standard tool for exactly this data-generating
> process and it is not close. Your mixed logit scored 1.187 only because you put a
> *normal* prior on the price coefficient: with mean −0.19 and sd 0.46, a third of your
> simulated population *likes* paying more. That's not evidence against heterogeneity,
> it's evidence you mis-specified the sign constraint. Fix it with a log-normal and
> refit before you conclude anything."

### ⚙️ The competition ML engineer
> "Respectfully: the leaderboard doesn't award marks for a correct generative story.
> xgboost beat all three structural models on the first try and it isn't close either.
> And your xgboost is *itself* mis-specified — you're training it with a pointwise binary
> objective and normalizing afterwards. The metric is a softmax over four alternatives
> within a choice set. Write a custom listwise objective so the gradient actually matches
> the loss you're scored on. That's a two-hour job with a bigger expected payoff than a
> week of Bayesian machinery."

### 📐 The statistician
> "Both of you are ignoring the sample size. The private leaderboard is 30% of 4,997
> rows — about 1,500 observations from perhaps 80 respondents. The standard error on a
> logloss difference that small is roughly ±0.02. You are proposing to spend days chasing
> improvements of 0.005 that the private leaderboard *cannot reliably measure*.
> Worse, every iteration you run against the same fixed folds is another draw in a
> multiple-comparison problem: run thirty experiments and the best one is optimistic by
> construction. My position: pursue changes with a *structural* reason to generalize —
> correct functional form, correct objective, robustness to the income shift — and treat
> anything under 1 SD as unproven. And keep a hard cap on how much the blend weights are
> allowed to chase the OOF."

### 📊 The conjoint / marketing scientist
> "All three of you skipped the obvious. This is a *designed* experiment. I'd want to know
> the design before I model anything — and the diagnostics just told us: 299 designs per
> task, replicated across ~4.7 respondents, and **98.5% of test rows reuse a design that
> appears in training**. That is a nonparametric estimate of the population choice
> probability sitting there for free, for almost every test row. Also: you coded 3-to-7
> level attributes as linear numerics. In conjoint you estimate *part-worths* — a separate
> utility per level — because feature levels are qualitative tiers, not quantities. Both of
> these are more fundamental than anyone's favourite estimator."

### ⚖️ Referee's ruling

The marketing scientist wins the opening. Design-level information is (a) cheap to test,
(b) applies to 98.5% of the test set, and (c) is orthogonal to everything we've built —
it can only be captured by a model that *knows* the design repeats.

The statistician's constraint is binding and I accept it: prefer changes justified by
structure over changes justified by a decimal, log every experiment (including the
failures) so the multiple-comparison risk stays visible, and require >1 SD before
changing what we submit.

The ML engineer's listwise-objective point is correct and cheap — our current objective
genuinely doesn't match the metric.

The econometrician is right about *why* mixed logit failed, and that diagnosis is worth
the refit — but as a blend member and report material, not as the centrepiece. Note that
report marks (15) exceed leaderboard marks (15 combined but split), so interpretable
models earn their keep regardless of blend weight.

**Priority queue:** design encoding → part-worths → listwise objective → fixed mixed
logit → latent class/HB if time allows.

---

## Iteration index

| # | Hypothesis | Result (single-model OOF) | Verdict |
|---|---|---|---|
| 00 | Diagnostics: where is the headroom? | — | ✅ found design reuse (98.5%) |
| 01 | Design-level empirical shares add signal no attribute model captures | 1.15516 → **1.15055** | ✅ confirmed, z = 2.94 |
| 02 | Part-worth level coding beats linear coding for utility models | 1.17731 → **1.15686** | ✅ confirmed, z = 11.84 |
| 03 | Listwise softmax objective beats pointwise binary + renormalize | 1.15055 → **1.14477** | ✅ confirmed, z = 4.30 |
| 04 | Part-worths + demographic interactions inside glmnet | running | |
| 05 | Mixed logit respecified (log-normal price, log-price, part-worths) | running | |

### Blend progression

| members | nested OOF | submission |
|---|---|---|
| mnl + mixl + xgb_long + xgb_wide | 1.15294 | `sub_20260725_2204.csv` |
| + mnl_pw + xgb_de + glmnet | 1.14211 | `sub_20260725_2326.csv` |
| + xgb_lw | **1.13888** | **`sub_20260725_2334.csv`** ← current best |

Weights at 1.13888: xgb_lw 0.616, mnl_pw 0.384, everything else 0. Temperature 0.938,
uniform mix 2.7%. Note the blend keeps exactly one member from each family — the tree
and the linear-utility model — and discards the rest as redundant.

**Incumbent blend to beat: 1.15294** (nested).

---

## Iteration 00 — Diagnostics

**Program:** `experiments/iter00_diagnostics/diag.R`

**Findings:**
1. **Design reuse — the big one.** 5,624 unique designs in train, 3,325 in test;
   **4,921 of 4,997 test rows (98.5%) reuse a design seen in training.** Support is
   thin per design (~3.8 rows on average, 1,881 designs with ≥5) so raw empirical
   shares are noisy — shrinkage is mandatory, not optional.
2. **Attributes are low-cardinality tiers** (3–7 levels; Price 1–12), currently coded
   as linear numerics. Part-worth coding is the textbook treatment and is untested here.
3. **Income shift is real and heavy-tailed.** Test is richer at every quantile
   (median $60k → $80k, p75 $85k → $125k) with extreme outliers ($3.9M / $5.0M).
   Raw `incomea` is a poor feature; log-income deserves a try.
4. **Calibration is good**, mildly underconfident above 0.4 — consistent with the
   blend independently choosing a sharpening temperature of 0.89.
5. **Irreducible noise is substantial.** Per-respondent mean loss spans 0.59 (p0) to
   2.71 (p100), median 1.09. Some respondents are simply not predictable from
   attributes and demographics; no model will fix them.

**Reflection:** the most valuable hour of the night was spent *not* fitting a model.
Every model so far (ours and the team's) implicitly assumed each choice set is unique
and must be understood through its attributes. The data says otherwise. The lesson to
carry forward: **interrogate the experimental design before choosing an estimator** —
had we run this first, the mixed logit would have waited its turn.

**Self-criticism:** the loss-decomposition section printed `<multi-column>` instead of
numbers (a `tapply`-returns-array slip). It was never repaired because the calibration
table answered the same question. Left as a known gap — if a future iteration needs to
know *which class* we lose on, fix that block first.

---

## Iteration 01 — Design-level empirical choice shares

**Program:** `experiments/iter01_design_encoding/run.R` · encoder promoted to
`model/encode_design.R` for reuse.

**Hypothesis:** with 98.5% of test rows reusing a training design, the empirical choice
share within a design estimates the population choice probability directly — capturing
whatever the attribute encoding fails to express.

**Method:** shrunk shares `(n_chosen + α·prior) / (n + α)` at three shrinkage strengths
(α = 1, 5, 20) plus the support count, handed to xgboost as features so the model learns
how far to trust them. Leakage control: a training row in fold k is encoded only from
respondents in other folds; respondents never span folds, so nobody informs their own row.

**Result:** 1.15516 → **1.15055**. Paired, respondent-clustered: **+0.00461, SE 0.00157,
z = 2.94**, 95% CI [+0.0015, +0.0077]; better on 54.5% of respondents. The encoding
features rank 4th, 7th and 10th by gain.

**Why this score and not more.** The shares alone score only 1.307 (α = 5) — barely
better than the 1.386 benchmark. The reason is support: ~3.8 rows per design, dropping
to ~3 once a fold is held out. A share estimated from 3 observations has enormous
variance, so most of the signal is shrunk away, and what survives is a small consistent
nudge. Note α = 1 (1.433) is *worse than the benchmark* while α = 5 is much better —
under-shrinking thin counts is actively harmful, which is exactly why handing xgboost
several strengths beat picking one.

**What I'd do differently.** Three untried variants, in order of promise:
1. **Residual encoding** — encode the average *residual* (observed − model-predicted)
   per design rather than the raw share. This isolates precisely what the attribute
   model misses and should shrink better, since residuals have smaller variance than
   raw choice indicators.
2. **Partial-credit matching** — exact design matching wastes the near-misses. Designs
   differing in one attribute could contribute at a discount.
3. **Bundle-level rather than set-level** encoding — 17,043 unique bundles appear across
   79,686 slots, so a given bundle recurs in *different* choice sets. Its across-context
   win rate is a different (and better-supported) signal than the set-level share.

**Process note:** the honest verdict came from the paired clustered test, not the
headline number. A raw −0.0046 sits well inside the ±0.013 fold-to-fold SD and would
have looked like noise; the paired test correctly identifies it as real because both
models were scored on identical rows. Any future iteration must be judged this way.

---

## Iteration 02 — Part-worth coding of attribute levels

**Program:** `experiments/iter02_partworth/run.R` · coefficients saved to `coefs.rds`

**Hypothesis:** attributes are 3–7 level qualitative tiers (Price 1–12), but we coded
them as linear numerics, forcing utility to move by a constant per level step.
Estimating a separate part-worth per level should help the linear-utility models most,
since they are the ones the constraint actually binds on.

**Result:** MNL 1.17731 → **1.15686**. Paired clustered: **+0.02044, SE 0.00173,
z = 11.84**, better on 64.8% of respondents. Four times the gain of iteration 01, from
a change requiring no new information — only a correct functional form. The MNL now
essentially matches the untuned xgboost (1.15516) despite being a transparent linear model.

**First attempt failed, and the failure was informative.** The initial run died with a
singular Hessian. Cause: `Price` is 1–12 on real bundles and 0 *only* on the all-zero
"none" option, so taking 0 as the reference level made the price dummies sum to exactly
(1 − none) — collinear with the none-constant. In a conditional logit anything constant
within a choice set is unidentified. Fix: reference = lowest level occurring on a *real*
bundle, plus a rank-revealing QR on the task-demeaned design matrix to catch anything
else. That QR then dropped `HU_L2` — a second dependency I had not reasoned my way to.
**Lesson: verify identification numerically; don't trust the by-hand argument.**

**The interesting finding (report material).** Price part-worths, relative to level 1:

| level | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| utility | −0.52 | −0.76 | −0.90 | −1.08 | −1.11 | −1.22 | −1.38 | −1.47 | −1.71 | −1.93 | −1.92 |

Strikingly **concave**: each additional price step hurts less than the last, and levels
11 and 12 are statistically indistinguishable — sensitivity saturates. Consumers respond
to price roughly logarithmically (Weber–Fechner), not linearly. Our linear coding was
fitting a straight line through a clearly curved response, which is exactly why freeing
the shape paid so much. Top-valued features: NV, NS, BU, BZ at their higher levels.

**What I'd do differently.** (a) Test a monotonicity-constrained or smoothed version —
11 free price parameters is more flexibility than the curve needs, and a 2–3 parameter
concave form (log-price, or a spline) might generalize better to the richer test
population. (b) The same coding error is still present in the mixed logit and was
present in glmnet — iteration 04 propagates the fix. (c) Some rare levels have thin
support and their part-worths are noisy; partial pooling toward a linear trend would be
the principled middle ground.

---

## Iteration 03 — Listwise softmax objective

**Program:** `experiments/iter03_listwise/run.R`

**Hypothesis:** the xgboost was optimizing the wrong loss. It scored each alternative as
an independent binary "was this chosen?" and we renormalized afterwards — but the metric
is a softmax over the four alternatives in a set, where only *relative* utility matters.
Under a binary objective a model is penalized for shifting all four scores together, even
though that changes nothing about the prediction. Supplying a custom gradient makes the
training loss identical to the competition metric.

**Method:** custom objective with `p = softmax(scores within task)`, `grad = p − y`,
`hess = 2p(1−p)`; custom eval metric = per-task logloss for early stopping;
`base_score = 0` so scores are raw utilities. Rows sorted by `(No, alt)` so each task is
four consecutive rows. Features, folds and hyperparameters held identical to iteration 01,
so the objective is the only thing under test.

**Result:** 1.15055 → **1.14477**. Paired clustered: **+0.00578, SE 0.00134, z = 4.30**,
better on 55.2% of respondents. Now the best single model — it alone beats the entire
four-member blend we submitted at 1.15294.

**Why this size and not larger.** Renormalizing after a pointwise fit already recovers
much of what the listwise objective provides; the gain is in the *training signal*, not
the prediction transform. The model now spends capacity on within-set contrasts rather
than on the absolute level of each alternative's score, which shows up as deeper trees
being useful (best iterations rose from ~250 to ~400–600).

**What I'd do differently.** (a) Re-tune hyperparameters — they were inherited from the
pointwise setup and the optimal depth/eta almost certainly shifted, as the higher round
counts hint. (b) Try a monotone constraint forcing utility to decrease in price: a free
lunch under the income shift, since it enforces sensible extrapolation into the richer
test population. (c) Subsample by *task* rather than by row, so partial choice sets never
appear during tree construction.

**Compounding note.** Iterations 01 and 03 stack: 1.15516 → 1.15055 → 1.14477, a total of
+0.0104 on the same model. Blended with the part-worth MNL, the nested decision number
went 1.15294 → 1.13888.

---

## Iteration 05 — Mixed logit, respecified (partial success, flawed experiment)

**Program:** `experiments/iter05_mixl_v2/run.R`

**Hypothesis:** v1 (1.18743) underperformed a plain MNL because a *normal* prior on the
price coefficient put ~⅓ of the simulated population at a positive price coefficient —
people who prefer paying more. Fixing the sign constraint should unlock the heterogeneity
the data clearly contains (v1 estimated sd of the none-constant at 2.1).

**Result:** **1.17281**, an improvement of +0.0146 over v1 — but still **worse than the
part-worth MNL at 1.15686**, and it earned zero blend weight.

**The experiment was badly designed, and that is the main lesson.** I changed three things
at once: (1) log-normal instead of normal on price, (2) log-price instead of linear price,
(3) part-worth coding for the other attributes. The net is +0.0146, but the attribution is
unrecoverable. Almost certainly (1) and (3) helped while (2) *hurt* — iteration 02 showed
the price response has a specific concave shape worth 11 free parameters, and I replaced
it with a single log term. **I violated my own protocol of one change per iteration**,
having just written that protocol. Worth remembering: the discipline is easiest to abandon
exactly when several fixes look obviously right at once.

**Why mixed logit may genuinely lose here.** Every test respondent is new, so we can only
ever predict the *population-averaged* probability. Integrating over a taste distribution
buys calibration but costs sharpness, and with 19 observations per training respondent the
fixed-coefficient part-worth model already captures the average response well. The
heterogeneity is real — it just isn't *conditionable* at prediction time. This is a genuine
finding for the report, not a failure: it explains why a simpler model wins on a metric
that only ever sees strangers.

**Next version (untested):** keep part-worth price (11 parameters, the shape that works),
put the log-normal random coefficient on a *linear* price term added alongside, or
randomize only the none-constant. One change at a time.

---

## Iteration 07 — Shift-aware blend calibration ❌ HYPOTHESIS REFUTED

**Program:** `experiments/iter07_shift_blend/run.R`

**Hypothesis:** the test population is measurably harder than our CV (local 1.13878 →
public 1.201). A model facing a harder population than it was calibrated on is
systematically *overconfident*, so the blend — which chose T = 0.968 and eps = 0, i.e.
slight sharpening and no insurance — should do better if softened. Tuning the blend
against an income-reweighted objective should find that softening automatically.

**Result: wrong on both counts.**

| blend tuned on | scored plain | scored reweighted |
|---|---|---|
| plain OOF | **1.13893** | **1.14323** |
| reweighted OOF | 1.13955 | 1.14524 |

Tuning on the reweighted objective is worse on the plain metric *and worse on its own
target metric*. Extra softening degrades monotonically:

| temperature multiplier | plain | reweighted |
|---|---|---|
| ×1.00 | **1.13794** | **1.14269** |
| ×1.05 | 1.13836 | 1.14309 |
| ×1.10 | 1.13948 | 1.14419 |
| ×1.20 | 1.14327 | 1.14791 |
| ×1.30 | 1.14839 | 1.15295 |

**Why the hypothesis failed.** Two distinct reasons, and separating them matters:

1. **The blend is not overconfident.** The calibration table from iteration 00 already
   showed predicted probabilities tracking observed frequencies closely, with mild
   *under*-confidence above 0.4. Softening a well-calibrated model can only hurt. The
   local↔public gap is therefore not a calibration failure — it is that the test
   respondents are harder to predict *at all*, which no monotone transform of our
   probabilities can fix. Sharpening and softening both lose.
2. **Importance weighting is self-defeating here.** Reweighting cuts the effective
   sample from 21,565 to 13,689 rows, so the reweighted objective is noisier than the
   plain one. Fitting 10 parameters against a noisier target produced a worse fit even
   when judged on that same noisy target — the variance cost exceeded the bias benefit.

**Decision: no change.** The blend stays tuned on plain OOF, and **we do not spend a
submission on a softened variant** — the idea was proposed before this evidence existed
and is now withdrawn.

**What this teaches for the remaining days.** "The test set is harder" and "our
probabilities are miscalibrated for the test set" are different claims, and only the
first is supported. Effort should go to models that are genuinely more accurate on
unseen respondents, not to post-hoc recalibration of the ones we have.

**Cost of being wrong: about 25 minutes of compute and no submission slots.** Cheap,
because it was tested locally before being tested on Kaggle — which is the entire
argument for maintaining an honest local metric.
