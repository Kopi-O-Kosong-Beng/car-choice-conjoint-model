# Experiment Log — Analytics Edge Competition

**Purpose:** a running research record. Every iteration keeps its own program under
`experiments/iterNN_<name>/`, and every iteration ends with a written reflection:
*why did it score what it scored, and what would I do differently?*
This file is the index. Anyone (including a future session with no memory of today)
should be able to read it top to bottom and continue the work.

---

# 👉 PICK UP HERE — next ideas, ranked

**State as of 26 Jul 2026:** nested blend **1.13044**, public leaderboard **1.199**
(rival reference 1.210, benchmark 1.38629). `sub_20260726_0651.csv` is uploaded and is
currently our best public score, so it is what Kaggle will score privately unless beaten.

**Transfer is decaying — read `submissions/log.md` before planning more modelling.** Early
gains reached the leaderboard at ~58%; the most recent step transferred at ~24–39%, and the
local→public offset has grown monotonically (+0.046 → +0.062 → +0.065 → +0.069) across
eighteen experiments against one fixed fold structure. Expect roughly a third of any further
local gain to show. Moving 1.199 → 1.196 needs ~0.009 more local, more than round 2 produced
in total.

Production blend is now **xgb_lw2 + xgb_mono + lcmnl3** (`mnl_pw` still listed but earns
weight 0.000 — the latent-class model strictly generalises it).

### ⚠️ Read this before optimising anything else

Two submissions with local 1.13878 and 1.13556 **both scored 1.201**. At a ~58% transfer
rate and a three-decimal display, gains under ~0.005 local are *invisible* on the public
board. Incremental tuning can no longer be confirmed or refuted publicly. The private
leaderboard still scores at full precision, so genuine gains remain worth having — but
prefer structurally justified changes, and remember the report is 15 of 30 marks.

### ❌ SETTLED: the residual design encoding is dead. Do not revive it.

Iteration 15 built the properly nested double-OOF baseline (20 fits: for every ordered pair
(k, j), fit on folds ∉ {k, j} and predict fold j) and rebuilt the encoding on top of it,
changing nothing else. Result:

| model | OOF |
|---|---|
| `xgb_lw2` — no residual encoding at all | 1.14152 |
| `xgb_resenc3` — nested (honest) baseline | **1.14151** |
| `xgb_resenc2` — single-OOF `mnl_pw` baseline | 1.13721 |

**The entire +0.0043 was leakage.** `resenc2` vs `resenc3` share every feature,
hyperparameter and seed and differ only in baseline honesty: −0.00430, SE 0.00095,
**z = −4.54**. That difference *is* the leak, measured directly. And the null is not a
degraded-baseline artefact — sd(residual) is 0.39723 nested vs 0.39698 single-OOF, and the
nested encoding correlates *more* strongly with each row's own held-out residual.

Even ~150 global coefficients with no design-level features absorb enough fold-k choice
information to be worth 0.0043 when subtracted. "Structurally unable to memorise a choice
set" was too strong a claim.

**The methodological lesson, which is sharper than the original one and belongs in the
report.** Both leak detectors *passed* on the nested encoding — and passed **harder** than
on the leaky one:

| | cor with own held-out residual (α=50) | best direct correction |
|---|---|---|
| iter12, tree baseline (leaky) | **−0.071** | 1.14515 — *worse* than doing nothing |
| `resenc2`, single-OOF baseline | +0.0248 | 1.15608 (+0.00078) |
| `resenc3`, nested baseline | **+0.0282** | 1.15587 (+0.00099) |

Yet `resenc3`'s tree gain is exactly zero. **The detectors were right about the sign and
useless about the size.** There is genuine design signal here — worth about +0.001, and
redundant with the share encoding the model already carries. Everything above that was
fold-k label information.

*Correction to the record:* on 26 Jul I ran the direct-signal test, saw it pass, and wrote
that "the +0.0043 tree result is credible." That was wrong. Passing a signal test
establishes that a feature contains real information; it says nothing about whether a
flexible learner's gain comes from that information or from a leak riding alongside it.
Only the nested baseline could separate them. Holding the model back from the submission
was right, but the reasoning I gave for keeping it as a candidate was not.

### ❌ SETTLED: bundle-level encoding is structurally impossible

Iteration 16. The briefed idea assumed a bundle recurs in *different* choice sets. It does
not: there are 5,681 designs and exactly 5,681 × 3 = 17,043 real bundles, and
bundle → (design, position) is a **bijection**. A bundle-keyed win rate is the same
partition as the design-keyed one — differencing them gives max absolute difference
**0.000e+00**. The presence-pattern fallback is constant within a task in
**26,562 of 26,562 tasks**, so its within-task contrast is identically zero and it cannot
move a softmax. There is no bundle-level statistic between the attribute level and the
choice-set level. Do not revisit.

### ❌ SETTLED: latent class is already at its best configuration

Iteration 14 tested every obvious extension. All measured on the fixed folds:

| variant | OOF |
|---|---|
| `lcmnl3p` class-specific price part-worths | 1.14379 (z = 1.57 vs lcmnl3 — noise; blend 1.13037 vs 1.13044) |
| **`lcmnl3` — production, 3 classes** | **1.14396** |
| `lcmnl3b` richer membership (demographic interactions, log-income) | 1.14506 |
| `lcmnl5` 5 classes | 1.14612 |
| `lcmnl4` 4 classes | 1.14617 |

**Three classes is the optimum**; more classes and a richer membership model both overfit.
Do not spend time here again.

One systematic observation worth carrying: *every* latent-class variant predicts a test
"none" rate near 0.223–0.228 against an OOF rate of ~0.304, while the tree models predict
~0.275. That ~0.08 gap is a property of the demographic membership channel, not a quirk of
one fit — the wealthier test respondents are consistently routed toward buying classes.
Either richer people genuinely buy more, or the membership model over-extrapolates. The
26 Jul 06:51 submission is partly a test of which.

### ❌ SETTLED: hierarchical Bayes is the worst model in the repo

Iteration 17, ~15 hours of compute. `hbmnl` OOF **1.23703**, blend weight **0.000**.
Ablating the demographic channel *improves* it to 1.16405 — the opposite sign to
`lcmnl3`, where demographics carry 93% of the gain. The difference is dimensionality:
`lcmnl3` routes demographics through a 3-way membership softmax (58 parameters), HB
shifts all 73 part-worths per demographic (438). **Heterogeneity helps when it is
discrete and low-dimensional and hurts when it is continuous and high-dimensional.**

The cold-start gap is now quantified, and it is the best single number in the report:
predicting with each respondent's own posterior-mean β gives logloss **0.35246**;
population-averaging the same rows — which is all a new respondent can get — gives
**1.22914**. That 0.877 is structurally unavailable at test time. It explains the mixed
logit's failure too.

### ⛔ Do not repeat — already tested and settled

| idea | outcome |
|---|---|
| Two-stage none-vs-buy decomposition | **decisively rejected**, iteration 09 — 1.17169, z = −11.04 |
| Nested logit, {1,2,3} vs {4} | iteration 10 — 1.15681, statistically identical to `mnl_pw`, zero blend weight |
| Residual encoding off a tree baseline | **leakage**, iteration 12 — 1.09962 is fake; use the `mnl_pw` baseline |
| Retuning listwise hyperparameters | iteration 06 — `slow_deep` won, in production |
| Softening the blend for the harder test set | **refuted**, iteration 07 — degrades monotonically |
| Tuning the blend on an income-reweighted objective | **refuted**, iteration 07 — worse on its own metric |
| Mixed logit, continuous heterogeneity | iteration 05 — loses to the part-worth MNL, zero weight |
| Part-worth glmnet with demographic interactions | iteration 04 — weight 0.001, kept for its coefficients |
| Wide 4-class xgboost, elastic net, linear-coded MNL | all zero weight, in `model/legacy/` |
| Latent class with 4/5 classes or richer membership | iteration 14 — all worse than 3 classes |
| Bundle-level encoding | iteration 16 — **structurally impossible**, bundle→(design,pos) is a bijection |
| Hierarchical Bayes | iteration 17 — 1.23703, blend weight 0.000, dead |

### Method note earned the hard way

`model/compare.R` compares single models. To compare two *blends* you must reproduce
`06_blend.R`'s nested loop, save the held-out blend predictions per fold, and apply the
same respondent-clustered paired statistic to those. Also: evaluating many member sets
against the nested number introduces search bias — the cleanest statistic is a contrast
where the challenger was pre-selected by its own single-model result.

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
| 04 | Part-worths + demographic interactions inside glmnet | 1.18118 → 1.16718 | ⚠️ no blend value, real insight |
| 05 | Mixed logit respecified (log-normal price, log-price, part-worths) | 1.18743 → 1.17281 | ⚠️ partial, zero blend weight |
| 06 | Hyperparameters retuned for the listwise objective | 1.14477 → **1.14152** | ✅ confirmed, z = 2.48 |
| 07 | Soften the blend for the harder test population | no improvement | ❌ refuted |
| 08 | Monotone price constraint + tuned hyperparameters | 1.14152 → 1.13980 | ⚠️ z = 1.35 alone, but earns 0.34 blend weight |
| 09 | Two-stage none-vs-buy decomposition | 1.17169 | ❌ decisively worse, z = −11.04 |
| 10 | Nested logit, real bundles vs outside option | 1.15681 | ❌ indistinguishable from MNL, zero weight |
| 11 | **Latent-class MNL (3 classes)** | mnl_pw 1.15686 → **1.14396** | ✅ **confirmed, z = 3.77** |
| 12 | Residual-based design encoding | 1.09962 | 🚨 **leakage** |
| 14 | Latent class: 4 and 5 classes, richer membership, class-specific price | all ≥ 1.14379 | ❌ 3 classes is optimal |
| 15 | Residual encoding with a **nested double-OOF** baseline | 1.14151 | 🚨 **the +0.0043 was 100% leak** |
| 16 | Bundle-level encoding has better support than choice-set-level | — | ❌ **structurally impossible**, bundle→(design,pos) is a bijection |
| 17 | Hierarchical Bayes: mixture-of-normals population + demographic channel | 1.23703 | ❌ **worst model in the repo**; ablating demographics gives 1.16405 |

### Blend progression, continued

| members | nested OOF | submission |
|---|---|---|
| mnl_pw + xgb_lw2 | 1.13556 | `sub_20260726_0116.csv` → public **1.201** |
| + xgb_mono + lcmnl3 (`mnl_pw` falls to weight 0) | **1.13044** | `sub_20260726_0651.csv` → public **1.199** ← current best |
| + `hbmnl` (hierarchical Bayes) | 1.13060 | rejected — weight 0.000 |
| + `hbmnl_nod` (HB, demographic channel ablated) | 1.13055 | rejected — weight 0.017, nested *worse* |

### Blend progression

| members | nested OOF | submission |
|---|---|---|
| mnl + mixl + xgb_long + xgb_wide | 1.15294 | `sub_20260725_2204.csv` |
| + mnl_pw + xgb_de + glmnet | 1.14211 | `sub_20260725_2326.csv` |
| + xgb_lw | 1.13888 | `sub_20260725_2349.csv` → **public 1.201** |
| xgb_lw replaced by tuned xgb_lw2 | **1.13556** | **`sub_20260726_0116.csv`** ← current best |

Weights at 1.13556: xgb_lw2 0.636, mnl_pw 0.364. Temperature 0.941, uniform mix 3.3%.
The blend keeps exactly one member from each family — the tree and the linear-utility
model — and discards everything else as redundant. Nine members were tried; adding the
seven zero-weight ones changes the nested score by 0.00005, so production runs two.

---

## Iteration 06 — Hyperparameters retuned for the listwise objective

**Program:** `experiments/iter06_lw_tuning/run.R` · full results in `all_results.rds`

**Hypothesis:** iteration 03 changed the objective but kept hyperparameters chosen for the
old pointwise one. Best-iteration counts jumped from ~250 to 400–600, which suggests the
optimum moved. Separately, economics says utility must decrease in price, so a monotone
constraint is a free prior — most valuable where training support is thinnest, which is
the richer end of the test distribution.

**Method:** five configurations, each given a full honest 5-fold OOF on the fixed folds.
All reported, not only the winner.

| config | OOF | rounds used |
|---|---|---|
| **`slow_deep`** eta 0.03, depth 8, mcw 20 | **1.14152** | 375–875 |
| `mono` monotone price constraint, base settings | 1.14298 | 424–921 |
| `base` (inherited settings) | 1.14462 | 207–894 |
| `slow_shal` eta 0.03, depth 4 | 1.14814 | 774–1798 |
| `reg` heavy L1/L2, mcw 30 | 1.14989 | 108–737 |

**Result:** `slow_deep` confirmed by paired clustered test: **+0.00325, SE 0.00131,
z = 2.48**, better on 56.3% of respondents. Promoted to production; blend went
1.13883 → **1.13556**.

**Two things worth noting.**

*The monotone constraint also worked* (+0.0018 over base), which is a genuinely principled
result — it encodes "utility decreases in price" as a hard structural prior rather than
something the trees must learn. It lost to `slow_deep` only because that config was tuned
harder. **The two have never been run together**, and there is no reason they should
conflict. That is the most promising untested idea remaining.

*Depth was the active ingredient, not the learning rate.* `slow_shal` (same eta 0.03,
depth 4) was the second-worst config and needed ~1,700 rounds to get there, while
`slow_deep` (depth 8) won with ~600. The listwise objective evidently rewards deeper
interactions — consistent with it optimising *contrasts within a choice set*, which are
inherently interaction-shaped, rather than each alternative in isolation.

**Selection-bias caveat, stated honestly.** Picking the best of five configs on the same
folds is a mild selection bias, so 1.14152 is slightly optimistic as an unbiased estimate.
The paired test against the incumbent (z = 2.48) is the defensible claim; the ranking among
the five losers is not.

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

## Iteration 04 — Heterogeneous part-worths via glmnet (no score gain, real insight)

**Program:** `experiments/iter04_glmnet_pw/run.R`

**Hypothesis:** part-worth coding was worth +0.020 to the plain MNL. With L1/L2 shrinkage,
glmnet can afford part-worths *interacted with demographics* — a heterogeneous part-worth
model the unregularized MNL cannot support — while staying linear in income, which matters
because the test population is twice as wealthy.

**Result:** 1.18118 → **1.16718** (+0.014). Part-worth coding now confirmed to help a
third model family, so the constraint was the level coding, not the estimator. But it
trails `mnl_pw` (1.15686), and when added to the blend it earned **weight 0.001** and moved
the nested score by nothing (1.13883 either way). **Not promoted.** 551 of 958 coefficients
survived the L1 penalty.

**The finding worth keeping.** The strongest surviving interactions are price part-worths
crossed with age, and they are concentrated entirely at the *top* of the price range:

| interaction | coefficient |
|---|---|
| Price_L12 × age | **−0.398** |
| Price_L11 × age | −0.326 |
| Price_L7 × age | −0.280 |
| Price_L9 × age | −0.273 |
| Price_L8 × age | −0.266 |
| Price_L10 × age | −0.244 |

Nothing comparable appears for price levels 1–6. This **refines the earlier linear result**
(Price × age = −0.31): older respondents are not uniformly more price-sensitive — they
specifically reject the *expensive* tiers, while behaving like everyone else at low prices.
A linear price × age term averages that threshold effect into a single slope and hides it.

Positive interactions are smaller but interpretable: gender × certain feature levels
(LB_L3, SC_L4, NS_L2), night-driving × BU_L1, urban × NS_L5.

**Reflection.** A model can be worth running purely for its coefficients. This one will not
appear in the submission, yet it produced the most specific behavioural claim we have —
which serves the report's insights section (5 marks) better than another 0.002 of logloss
would have served the leaderboard. Worth remembering when deciding what to run next:
the two goals are scored separately.

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

---

## Iteration 16 — Bundle-level encoding ❌ STRUCTURALLY IMPOSSIBLE

**Hypothesis as briefed.** Iteration 01 encodes the empirical choice share per
*(design, alternative)*, where a design is a whole 3-bundle choice set: 5,681 designs
with ~3.8 observations each. Thin support is why almost all of the signal had to be
shrunk away for only +0.0046. But 17,043 distinct *bundles* appear across 79,686 slots,
so a bundle should recur in *different* choice sets. Its win rate across varying contexts
would then be a different statistic with better support — "this bundle is attractive"
rather than "this bundle beat those particular rivals."

**The hypothesis is false, and the diagnostics killed it before a model was fitted.**

1. **A bundle never recurs in a different choice set.** There are 5,681 designs and
   exactly 5,681 × 3 = 17,043 distinct real bundles, and the map
   bundle → (design, position) is a **bijection**. Zero bundles appear in more than one
   design; zero appear at more than one position. So a bundle-keyed win rate is not a
   new statistic — it is *the same partition*. Built literally and differenced against
   the existing (design, alt) encoding: **max absolute difference 0.000e+00**, at every
   shrinkage strength. Support per bundle is 4.68 observations, identical to the design
   key's 4.68, because they are the same thing.
2. **The natural repair fails on a softmax.** Backing off to the bundle's *presence
   pattern* (which of the 19 non-price features it carries, ignoring tier) is a property
   of the choice *set*, not the bundle: in **26,562 of 26,562 tasks (100.00%)** all three
   real bundles share one presence pattern — a choice set offers the same 9 features at
   three different tier/price configurations. Constant within a task ⇒ its within-task
   contrast is identically zero ⇒ under a softmax over the four alternatives it **cannot
   move a single prediction**. Confirmed: logloss unchanged to five decimals at every
   weight, for both baselines.
3. **It does not even pool across designs.** 5,511 patterns for 5,681 designs; only 167
   span more than one design; 94.07% of designs own their pattern outright. Tasks per
   pattern 9.64 versus tasks per design 9.35 — a **3% support gain**.

**The conjoint design nests bundles inside choice sets bijectively. There is no
bundle-level statistic between the attribute level and the choice-set level.**

**Reflection.** This is the cheapest experiment in the log: it cost three diagnostic
scripts and zero model fits, because the hypothesis made a *countable* claim ("bundles
recur across choice sets") that could be checked directly. The lesson is to look for the
countable claim inside a modelling idea and check it first. Had this been implemented
before being verified, it would have produced an encoding numerically identical to one
already in the model, and the null result would have been read as "the encoding is
saturated" rather than "I built the same feature twice."

---

## Iteration 17 — Hierarchical Bayes ❌ THE WORST MODEL IN THE REPO

`experiments/iter17_hb/run.R` — `bayesm::rhierMnlRwMixture`, 16,000 MCMC draws,
3-component mixture-of-normals population, demographic channel beta_i = Delta'z_i + u_i.
**~15 hours of compute** (≈180 min × 5 folds + 179 min full refit).

**Hypothesis.** HB is the textbook tool for exactly this data-generating process: panel
conjoint, 19 tasks per respondent, a genuine outside option. Iteration 05's mixed logit
lost (1.17281) but with a *single* normal population distribution, and iteration 11 found
three sharply distinct segments (none-rates 0.049 / 0.228 / 0.648), so the true population
is strongly multimodal and one normal cannot represent it. HB fixes that (mixture) and
adds a demographic channel (the continuous analogue of the membership model carrying 93%
of `lcmnl3`'s gain), with Bayesian shrinkage better behaved than simulated ML at 19
observations per person.

**Result — decisive and in the wrong direction.**

| model | honest OOF |
|---|---|
| `xgb_mono` | 1.13980 |
| `lcmnl3` | 1.14396 |
| `mnl_pw` | 1.15686 |
| **`hbmnl_nod`** — HB, demographic channel ablated (z := 0) | **1.16405** |
| `mixl` | 1.17281 |
| **`hbmnl`** — HB as specified | **1.23703** |

Per fold: 1.25030 / 1.20138 / 1.24167 / 1.24147 / 1.25034. Monte-Carlo noise in the
mixture simulation is 0.00023 (second RNG seed gives 1.23680), so none of this is
simulation error. Blend: `hbmnl` earns weight **0.000** (nested 1.13060), `hbmnl_nod`
earns 0.017 and makes the nested number **worse** (1.13055 vs 1.13044). Rejected.

### Finding 1 — the demographic channel *costs* 0.073, and that is the interesting part

Ablating demographics improves HB from 1.23703 to 1.16405. In `lcmnl3` the demographic
channel is worth 93% of the model's entire gain. **Same information, opposite sign.** The
difference is dimensionality, and the numbers are stark:

| model | how demographics reach tastes | parameters |
|---|---|---|
| `lcmnl3` | → 3-way segment membership softmax → segment part-worths | (3−1) × 29 = **58** |
| `hbmnl` | → shifts all 73 part-worths directly, per demographic | 6 × 73 = **438** |

HB was given *fewer* demographics than `lcmnl3` (6 versus 28 — a measured compute
constraint, since bayesm draws Delta as one joint Gaussian and pays an (nz·nvar)³
Cholesky per draw) and still ended up with 8× the parameters, because it interacts each
demographic with every part-worth instead of with a 3-way membership.

**The segment is a bottleneck, and the bottleneck is the point.** Latent class compresses
"who you are" into three numbers before letting it touch tastes. HB lets demographics move
all 73 part-worths freely. With 1,135 respondents the compressed channel generalises and
the free one memorises. This is a cleaner statement of iteration 14's result (richer
membership, `lcmnl3b`, was also worse) and it belongs in the report: *heterogeneity helps
when it is discrete and low-dimensional; the same heterogeneity hurts when it is
continuous and high-dimensional.*

### Finding 2 — the cold-start gap, quantified

The full refit reports, on identical rows:

| prediction | logloss |
|---|---|
| each respondent's **own** posterior-mean beta_i | **0.35246** |
| the same rows, **population-averaged** (what a new respondent gets) | **1.22914** |

That gap — 0.877 — is the part of HB's fit that is **structurally unavailable at test
time**, because all 263 test respondents are new people. It is the single most compelling
number we have for why panel methods cannot help here, and it explains iteration 05's
mixed-logit failure too. Anyone reading a conjoint textbook will expect HB to win; this
table is the answer.

**Reflection.** The hypothesis header pre-registered "modest at best" and predicted the
mechanism correctly — integrating a smooth F blurs the population-averaged prediction —
but underestimated the size, expecting a small loss rather than a catastrophic one, and
did not anticipate that the demographic channel would flip sign. The pre-registration is
what makes that admission possible: the expectation is on the record, timestamped before
the fit.

**Process failure worth recording.** All six jobs completed and wrote `pred_main_*.rds`,
but the session driving them was interrupted before running the `combine` step, so for
several hours the result looked like a *failure* when it was actually a finished
experiment sitting on disk unassembled. The fix is structural: **a long job should write
its own assembled artifact as its last act**, never leave assembly to a caller that may
not survive. `experiments/iter17_hb/assemble_nod.R` now reconstructs the ablated variant
independently of the original driver.

**Cost of being wrong: ~15 hours of compute for two report findings and a closed idea.**
Expensive in wall-clock, cheap in submissions, and it retires the highest-ceiling
untested item on the list — which is worth something on its own, because it was the idea
most likely to nag at us if left unmeasured.
