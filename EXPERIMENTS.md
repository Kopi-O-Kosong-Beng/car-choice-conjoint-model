# Experiment Log — Analytics Edge Competition

**Purpose:** a running research record. Every iteration keeps its own program under
`experiments/iterNN_<name>/`, and every iteration ends with a written reflection:
*why did it score what it scored, and what would I do differently?*
This file is the index. Anyone (including a future session with no memory of today)
should be able to read it top to bottom and continue the work.

---

# 👉 PICK UP HERE — current state, and why the search is over

> **⛔ The project is FROZEN for modelling.** The default correct action is report work.
> This section is a record of what is settled, **not** a menu of things to try. Twenty-six
> iterations are done and the ⛔ table below almost certainly already contains your idea
> together with the number that killed it. Re-opening the freeze requires the user to say so.
>
> **Read [`STRATEGY_REVIEW.md`](STRATEGY_REVIEW.md) before proposing any plan.** It carries
> two corrections to this file's own doctrine — private-board *ranking* noise is paired and
> ~2–3× smaller than the ±0.02 absolute wobble, and public should be predicted from the
> income-reweighted OOF — plus the measured ~0.8 replication shrinkage, the one remaining
> allowed experiment (EM-start variance of `lcmnl3_both`), the freeze rule, and the dated
> report plan through 10 Aug.

**State as of 27 Jul 2026:** nested blend **1.12819** (income-reweighted 1.13273), public
leaderboard **1.199** from the older 1.13044 blend. Rival reference 1.210, benchmark 1.38629.
`submissions/sub_20260726_2328.csv` is built and **not yet uploaded** — read
`STRATEGY_REVIEW.md` Phase 0 for the pre-registered interpretation of each possible result.

Production blend is **two members**: `xgb_lw2bag` 0.528 + `lcmnl3_both` 0.472.
`model/members.txt` is the source of truth and documents why each dropped member was dropped.

**Three results from 26–27 Jul that change how to read everything above them:**

1. **Iteration 25 — the conditional-logit family had no `Task` term.** Adding two within-task
   position terms to the latent-class utility took it 1.14396 → **1.13863** (z = 6.26), the
   best single model in the repo. The rising decline rate turns out not to be fatigue: it is a
   *consequence* of rising price sensitivity, because the none option is always the cheapest
   alternative present. Replicates on an independent respondent grouping at ~80%.
2. **Iteration 26 — the seed sd is 0.00283, and it invalidates iteration 08.** The monotone
   price constraint's claimed +0.00172 is smaller than the noise it was measured against;
   paired across ten seeds it is −0.00034, CI [−0.00159, +0.00092], winning 5 of 10.
   `xgb_mono` and `xgb_lw2` were the same model carried twice for eighteen iterations.
3. **The blend collapsed from four members to two** at an unchanged score (1.12867 → 1.12819,
   a difference equal to the blend-level seed sd). `xgb_mono` was a duplicate and `mnl_pw`
   contributed −0.00006. The survivors are the two ends of the blend's only real axis of
   disagreement — tree versus logit, 93% of the error variance in one component.

### ⚠️ Read this before optimising anything else

**Know your noise floor before quoting a margin.** Measured in iteration 26:

| level | sd | range across 10 seeds |
|---|---|---|
| single xgboost model | 0.00283 | 0.00896 |
| **nested blend (the decision number)** | **0.00048** | **0.00137** |
| fold-to-fold (within one model) | 0.013 | — |

These are different instruments and they answer different questions. A **member swap** must
clear the model-level sd; a **blend change** need only clear the blend-level sd. Conflating
them is exactly how iteration 08 passed. Anything quoted below 0.003 on a single model, from
any iteration before 26, should be treated as unresolved.

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
| 18 | Structure hunt: 17 diagnostics for untapped signal | — | ✅ one lead found (→ 25), eleven killed |
| 19 | Blend diversity: is a richer combiner worth building? | — | ❌ no — one axis of disagreement, alt-4 weights z = 1.49 |
| 21 | Do results survive an independent fold structure? | in progress | ⏳ running |
| 22 | Per-alternative blend weights | not run | ❌ closed by iteration 19's evidence |
| 24 | Within-task dominance features | not run | ❌ closed by `d10`, R² 0.000415 vs residual |
| 25 | **Task position inside the latent-class utility** | lcmnl3 1.14396 → **1.13863** | ✅ **confirmed, z = 6.26**, in production |
| 26 | Seed-bagging; and is the seed sd bigger than our margins? | mean 1.14303, **sd 0.00283**, bagged 1.13714 | ⚠️ **invalidates iteration 08**; blend gain only +0.00029 |

### Blend progression, continued

| members | nested OOF | submission |
|---|---|---|
| mnl_pw + xgb_lw2 | 1.13556 | `sub_20260726_0116.csv` → public **1.201** |
| + xgb_mono + lcmnl3 (`mnl_pw` falls to weight 0) | **1.13044** | `sub_20260726_0651.csv` → public **1.199** ← current best |
| + `hbmnl` (hierarchical Bayes) | 1.13060 | rejected — weight 0.000 |
| + `hbmnl_nod` (HB, demographic channel ablated) | 1.13055 | rejected — weight 0.017, nested *worse* |
| **`lcmnl3` → `lcmnl3_both`** (task position) | **1.12867** | `sub_20260726_1643.csv` ← ready to upload |

Weights at 1.12867: `lcmnl3_both` 0.504, `xgb_mono` 0.311, `xgb_lw2` 0.185, `mnl_pw` 0.000.
Temperature 0.985, uniform mix 0.28%.

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

---

## Iteration 25 — Task position inside the latent-class model ✅ IN PRODUCTION

`experiments/iter25_taskpos/run.R` — nested blend **1.13044 → 1.12867**.

**Hypothesis.** Respondents answer 19 tasks in sequence and their behaviour drifts.
Measured beforehand in the iteration 18 diagnostics:

| | task 1 | task 19 | slope/task |
|---|---|---|---|
| observed none-rate | 0.237 | 0.337 | +0.00544 (logit z = 9.51) |
| price coefficient (fitted per position, controlling for `lvlsum`) | −0.178 | −0.331 | −0.00552 (z = −6.29) |

Neither is a design confound: mean price by position runs 6.310 to 6.636 with no trend.

**The gap.** The tree models have `Task` as a feature and track the drift (+0.00455,
+0.00469). The conditional-logit family does **not** — `model/02_mnl_partworth.R:45` and
`experiments/iter11_latent_class/run.R:118` both build utilities from
`c("asc2","asc3","asc4", pw, px)` with no `Task` term anywhere. They predict the drift as
essentially flat (+0.00024, +0.00018), and `lcmnl3` carried blend weight 0.447.

So this was not a search for new signal. It was a **known, measured, pre-quantified effect
that one whole model family was structurally unable to express**, sitting inside the blend's
largest member.

**Implementation — two identified within-task terms.** With `Task_c = (Task − 10)/9`:
`none_x_Task = asc4 × Task_c` and `Price_x_Task = Price × Task_c`. Both vary within a
choice set, which is what a conditional logit requires; a bare `Task` main effect would be
constant within a task and the rank-revealing QR would drop it, correctly. Fitted
separately and together so each is attributable.

**Result.**

| variant | OOF | gain vs `lcmnl3` | z | nested blend | shift retention |
|---|---|---|---|---|---|
| `lcmnl3` (previous production) | 1.14396 | — | — | 1.13044 | — |
| `lcmnl3_drift` — none term only | 1.14231 | +0.00165 | 3.47 | 1.12999 | **64%** |
| `lcmnl3_tilt` — price term only | 1.13881 | +0.00515 | 6.28 | 1.12869 | 113% |
| **`lcmnl3_both`** — production | **1.13863** | **+0.00534** | **6.26** | **1.12867** | **119%** |

`lcmnl3_both` is the best single model in the repository, ahead of `xgb_mono` (1.13980).
Blend weight rises from 0.447 to 0.504. Test none-rate 0.2237, unchanged from `lcmnl3`'s
0.223, so it carries no new extrapolation risk.

### The finding: the rising decline rate is not fatigue, it is price sensitivity

The attribution inverts the post-hoc estimate. Correcting `lcmnl3`'s finished predictions
after the fact, the none-drift was worth more than the price tilt (+0.00228 vs +0.00145).
Fitted *inside* the model the order reverses hard: the price tilt is worth +0.00515 and
adding the explicit none term on top buys a further **+0.00019**.

`experiments/iter25_taskpos/mechanism.R` shows why. Fraction of the observed none-rate
slope each model reproduces:

| model | task 1 → task 19 | slope | % of observed |
|---|---|---|---|
| observed | 0.237 → 0.337 | +0.00544 | — |
| `lcmnl3` | 0.299 → 0.311 | +0.00018 | 3.2% |
| `mnl_pw` | 0.297 → 0.309 | +0.00024 | 4.4% |
| `xgb_mono` | 0.251 → 0.338 | +0.00469 | 86.3% |
| `lcmnl3_drift` | 0.255 → 0.358 | +0.00518 | 95.2% |
| **`lcmnl3_tilt`** | 0.249 → 0.362 | +0.00572 | **105.3%** |
| `lcmnl3_both` | 0.252 → 0.356 | +0.00529 | 97.3% |

**`lcmnl3_tilt` was never given a none-specific task term, and it reproduces the entire
none-rate drift** — more of it than the explicit drift term does, and more than the trees.

The mechanism is structural. The none option is the all-zero bundle, so its `Price` is 0 and
it is *always the cheapest alternative in its choice set*. When price sensitivity steepens
through the questionnaire, the cheapest option gains relative utility automatically. The
rising decline rate is a **consequence** of rising price sensitivity, not a separate fatigue
process.

That is why the post-hoc correction needed two knobs and the structural model needs one: a
post-hoc patch operates on outputs and must fix each symptom separately, while a model that
captures the cause gets the second symptom for free. It is also why `lcmnl3_drift` retains
only **64%** under reweighting toward the test population while `lcmnl3_tilt` retains
**113%** — the fatigue parameterisation is fitting a population-specific artefact of the
same underlying effect, and the price parameterisation is fitting the effect.

**Why `both` is in production despite `tilt` being more parsimonious.** They are
statistically indistinguishable (nested 1.12867 vs 1.12869, a 0.00002 difference against a
fold-to-fold SD of 0.013), but `both` wins on plain OOF, on the nested blend, *and* on shift
retention (119% vs 113%). Parsimony does not outrank three measurements. `lcmnl3_tilt` is
kept and documented in `members.txt` as the alternative, and it is the variant the report
should discuss, because it is the one that carries the mechanism.

### Verification

`experiments/iter25_taskpos/verify.R` — the workflow's adversarial verifier agents died on a
billing limit, so the check was done directly. All headline numbers recomputed from the
artifacts on disk and matched to five decimals; 21,565 / 4,997 rows, no NA, rows sum to 1 to
2.2e-16. **The gain appears in every fold** (+0.00456, +0.00830, +0.00528, +0.00592,
+0.00262) — a leak concentrates in one fold, a structural gain does not.

Leakage risk is nil by construction: `Task` is a design variable recorded for every row, all
263 test respondents have all 19 positions, and nothing target-derived enters the model.

**Reflection.** This gain had been sitting in plain sight for eighteen iterations. The
diagnostics that found it (`d3`, `d15`) had *printed* the drift as far back as the structure
hunt, but nobody scored what it was worth, and the models that lacked the term were never
audited against a behaviour the data plainly showed. The lesson is procedural: after
measuring an effect in the data, check which models can actually *express* it. A feature
list is a claim about what the model believes matters, and that claim deserves the same
scrutiny as a hyperparameter. Sixteen of the eighteen prior iterations searched for new
signal; this one found signal the diagnostics had already located and no model had been
given a way to use.

**Cost: about 40 minutes of compute for the largest single-model gain since iteration 11.**

---

## Iteration 26 — The seed is a hyperparameter ⚠️ INVALIDATES ITERATION 08

`experiments/iter26_seedbag/` — ten refits of an identical configuration, changing only the
random seed.

**Hypothesis.** Every xgboost artifact in this repository is a single seed. With
`subsample = 0.8`, `colsample_bytree = 0.8` and a randomly drawn 10% early-stopping holdout,
the fitted predictor carries Monte-Carlo variance that has nothing to do with the data.
Averaging over seeds removes it, and — uniquely among everything in this log — involves **no
model selection**, so it cannot overfit the fold structure and should transfer at ~100%.

**The seed distribution, which this repository had never measured.**

| | |
|---|---|
| mean OOF | 1.14303 |
| **sd across seeds** | **0.00283** |
| range | 1.14061 – 1.14957 (**0.00896**) |
| stored `xgb_mono` (seed 123) | **1.13980** — better than all ten, ≈1.1 sd below the mean |

### Finding 1 — iteration 08 is invalidated

Iteration 08 accepted the monotone price constraint on the grounds that it improved OOF from
1.14152 to 1.13980: a margin of **0.00172, smaller than the seed sd of 0.00283**. It has
carried roughly a third of the blend weight ever since. Its paired z was only 1.35, which
the entry noted, but the conclusion was kept because the model earned blend weight.

Retested by running the **same seeds through both configurations**. The pairing matters:
the seed drives the early-stopping respondent split for both, and their scores correlate
**0.931** across seeds, so pairing removes a large shared component.

| seed | constrained | unconstrained | diff |
|---|---|---|---|
| 1 | 1.14228 | 1.14132 | +0.00096 |
| 2 | 1.14957 | 1.14899 | +0.00058 |
| 3 | 1.14384 | 1.14214 | +0.00170 |
| 4 | 1.14076 | 1.14201 | −0.00125 |
| 5 | 1.14520 | 1.14498 | +0.00022 |
| 6 | 1.14443 | 1.14275 | +0.00168 |

mean(constrained − unconstrained) = **+0.00065**, SE 0.00045, t = +1.45 on 5 df,
95% CI **[−0.00050, +0.00181]**. Iteration 08's claimed −0.00172 falls **outside** that
interval. The constraint wins on **1 of 6** seeds.

**The monotone price constraint does nothing.** `xgb_mono` and `xgb_lw2` have been the same
model carried twice under different names, and the blend has been paying two weights for one
model. The economic argument for the constraint (utility must not increase with price) is
still correct — the model simply already satisfies it without being told, which is itself
worth reporting.

### Finding 2 — bagging works on the model and barely on the blend

| | |
|---|---|
| mean single seed | 1.14303 |
| **bagged, probability space** | **1.13714** |
| bagged, log space (geometric mean) | 1.13721 |

*Pre-registration check:* the header predicted log-space averaging would win, because
logloss is a function of log p. **It did not** — probability space won by 0.00007. Recorded
as stated rather than quietly dropped; the difference is negligible but the prediction was
wrong.

`compare.R` puts `xgb_monobag` +0.00266 ahead of the stored `xgb_mono` at z = 1.77, "not
distinguishable from noise". **That is the wrong estimand**, and `expected_blend.R` exists
to say so. The stored artifact is a lucky draw, and a model's OOF artifact and its *test*
artifact are independent random draws — five fold-fits under one seed versus a separate
refit on all the data. A lucky OOF does not imply a lucky shipped refit, so the expected
quality of a single-seed submission corresponds to the seed **mean**.

Asking the procedure question properly:

| | nested blend |
|---|---|
| E[nested \| member is one random seed], over 10 seeds | 1.12884 (sd 0.00048) |
| stored `xgb_mono` (lucky) | 1.12867 |
| **bagged `xgb_monobag`** | **1.12854** |
| **honest expected gain from bagging** | **+0.00029** |

**The ~0.006 that bagging is worth to the single model is worth ~0.0003 to the blend.** A
blend is *itself* a variance-reduction device, and its other members already absorb most of
the seed noise. Bagging and blending are substitutes, not complements — we had been buying
the benefit without knowing it.

### Finding 3 — the decision number is far more stable than its parts

Seed luck moves the individual model by **0.00896** and the nested blend by only
**0.00137** (sd 0.00048). This is the most useful number the iteration produced, because it
retrospectively licenses other results: iteration 25's +0.00177 blend gain is ≈3.7 blend-level
seed-sd, comfortably real, even though a difference that size would be well inside noise at
the single-model level.

**The precision of an ensemble's score is not the precision of its parts**, and which one
you need depends on which decision you are making. Judging a *member swap* needs the model-level
sd; judging a *blend change* needs the blend-level sd. Conflating them is how iteration 08
happened.

**Reflection.** This should have been the first experiment in the project, not the
twenty-sixth. It costs one extra run to learn that the measurement instrument has a sd of
0.00283, and that single number reprices every margin in the log above it. Instead we ran
eighteen experiments quoting differences of 0.002–0.005 against an unmeasured noise floor,
and one of them was wrong. The general rule: **before quoting a margin, measure the
reproducibility of the thing you are measuring it with.**

The honest accounting of what this bought: +0.00029 on the decision number, one retracted
result, one recalibrated log, and the methodology section of the report.

---

## Iteration 18 — Structure hunt ✅ COMPLETE (one lead found, eleven killed)

`experiments/iter18_structure_hunt/` — 15 diagnostics (`d1`–`d15`) plus two written later
(`d16_liveleads.R`, `d17_disambiguate.R`) to convert effects that had been *printed* into
honest, nested numbers. No model was shipped from this iteration; its output is a decision
about where not to spend time.

**The one live lead**, which became iteration 25: task-position drift, and the fact that the
conditional-logit family had no `Task` term. Observed none-rate 0.237 → 0.337 across
positions (logit slope z = 9.51); price coefficient −0.178 → −0.331 per position
(−0.00552/task, z = −6.29) controlling for `lvlsum`, with mean price by position flat
(6.310–6.636) so it is not a design confound.

**Killed, each with the number that killed it.**

| lead | measured | verdict |
|---|---|---|
| Attribute interactions | +190 pairwise terms → +0.00036 (every pair co-occurs in ≥5,501 tasks, so support is not the issue) | dead |
| Within-task rank / dominance features | R² **0.000415** against the incumbent residual, while the same features explain R² 0.125 of the *outcome* — the model already has the signal. Only 3.29% of real alternatives are dominated | dead |
| Per-task-position temperature | 1.14152 → 1.14080 **in-sample**, so ~0 honest | dead |
| Dispersion-conditional temperature | honest 4-quartile T vs honest global T: **−0.00041**. Conditioning makes it worse; the global temperature is already right | dead |
| Respondent-specific fatigue slopes | slope sd 0.02841 vs 0.02762 expected under pure noise | no real heterogeneity |
| Sequential anchoring / contrast | +0.000029 on xgb_lw2 — and the **placebo (next task, unseen) scores +0.000031**. On the blend the placebo *beats* the real thing | definitively spurious |
| Demographic over-dispersion of the none rate | nested-honest correction +0.00009 (z 0.09) and +0.00020 (z 0.23) | dead |
| Respondent-level shrinkage | +0.00095 on xgb_lw2 but **−0.00068 on the blend** | loses where it counts |
| Transductive edge from design assignment | 19 task-wise ANOVAs are one partition repeated; effective 11 tests, min p = 0.1228, Bonferroni 1.351 | assignment is random w.r.t. demographics |
| Design-share / version-mate encoding | design shares alone score **1.30660** vs the blend's 1.12969 | nothing left to extract |
| Partial-profile / shown-attribute-set structure | already absorbed | dead |

**Headroom, which is the most quotable output of the iteration.** Oracle respondent rates
take the blend from 1.12969 to **0.98565** — a gap of **0.1440** where all remaining loss
lives. Variance decomposition: alternative 4 is **33% between-respondent**, alternatives 1–3
only 5–7%. And demographics reach just **11.2%** of the true none-propensity heterogeneity
(observed var 0.0804, binomial component 0.0111, true 0.0693, R² 0.0968) — **~89% of the
personality driving the outside option is unobservable in this dataset.**

**A self-correction inside the iteration, worth recording.** `d16` reported a
demographic-dispersion gain of −0.003/−0.006. That was a measurement bug: the correction
weight had been fitted where the demographic regression was in-sample. `d17` rebuilt it
nested and the effect vanished to +0.00009. The diagnostic caught its own error only because
someone re-derived it honestly rather than believing the first pass.

---

## Iteration 19 — Blend diversity probe ✅ COMPLETE (read-only; closes iteration 22)

`experiments/iter19_diversity/probe.R` and `probe2_gating.R`. Emitted no artifacts; its
variant-A code reproduces `blend_probe.R` to five decimals (1.13044), so every number sits
on the repository's decision metric.

**1. The blend has ONE axis of disagreement, not three.** Eigenvalues of the 4×4
error-correlation matrix: **3.726 / 0.202 / 0.056 / 0.016** — the first component is 93% of
the error variance, participation ratio 1.15 of 4. Pearson r on the log-odds scale:

| pair | alts 1–3 | alt 4 |
|---|---|---|
| `xgb_lw2` ↔ `xgb_mono` | .985 | .987 |
| `mnl_pw` ↔ `lcmnl3` | .959 | .898 |
| tree ↔ logit (cross-family) | .895–.911 | **.816–.840** |

The axis is **tree vs logit**, and it is widest on the none option. (Iteration 26 later
explained the first row: `xgb_lw2` and `xgb_mono` are the same model, since the monotone
constraint does nothing.)

**2. Only one member is load-bearing.** Leave-one-out, nested, against 1.13044:

| dropped | nested without | contribution | z |
|---|---|---|---|
| `mnl_pw` | 1.13038 | **−0.00006** | −1.89 |
| `xgb_lw2` | 1.13061 | +0.00017 | +0.58 |
| `xgb_mono` | 1.13112 | +0.00067 | +1.60 |
| `lcmnl3` | 1.13477 | **+0.00433** | **+3.20** |

`mnl_pw` is dead freight — removing it is very slightly *beneficial*.

**3. The none option is the EASY class, refuting a standing assumption.** It carries 27.7%
of total loss on a 30.2% row share, i.e. *under*-represented; loss is highest on y = 1
(1.2101 vs 1.0366). And as members disagree more, loss rises on y = 1,2,3 but **falls** on
y = 4 — when they argue, they argue about the none option, and the blend resolves it
correctly. None-calibration is near-perfect across p4 deciles.

**4. A separate weight vector for alternative 4 — tested, and killed.** This was the whole
content of the never-run iteration 22, and it is now answered:

| architecture | nested |
|---|---|
| A production log-pool (M+2 par) | 1.13044 |
| B = A + free none-intercept (M+3) | 1.13054 |
| C = B + split alt-4 weights (2M+3) | 1.12937 |

C over A: **+0.00108, SE 0.00072, z = +1.49** — noise, with one fold in five moving the
wrong way. The fitted structure is beautifully interpretable (trust the tree on bundles,
w = .000/.224/.517/.259; trust the latent-class logit on the none, v = .000/.175/.004/**.821**)
and it still should not be built.

**The decisive argument, which kills the idea rather than merely failing to support it.** On
OOF, *every* member reproduces the true class shares to three decimals — all four predict a
none-rate of 0.302–0.308. The 0.223-vs-0.274 split is a **logit-family vs tree-family**
phenomenon that exists **only on the test set**. A split alt-4 weight would be fitted on data
where the members agree about alternative 4 to within 0.005, then deployed where they
disagree by 5 points. *There is no validation set for the regime it targets.*

**5. Oracle bound, and why the headroom is unreachable.** Per-respondent convex weights,
honest split-half (fit on ~10 of a respondent's tasks, score the other ~9): **1.08965**
against a matched global-pool control of 1.13013 — **+0.04048**, four times the entire gain
from blending. A hostile permutation control (score each respondent with weights fitted on a
*different* random respondent) gives 1.14309 ± 0.00192, *worse* than the global pool, so the
gain is genuinely respondent-specific rather than an artifact. But it needs the respondent's
own labels, and all 263 test respondents have zero observed choices. Two label-free gates:

- tertile of the respondent's mean predicted p4: 1.13068, **−0.00023** (z −1.68)
- demographic `segment`: 1.13514, **−0.00470** (z −4.87, a real loss)

**Zero of the 0.040 is reachable.**

**Verdict and what it directed.** Do not build a richer blend architecture. The actionable
finding was compositional rather than architectural: *the blend's only genuine diversity is
the tree-vs-logit axis, and `lcmnl3` is the only member carrying it* — so improve that member
rather than re-weighting the four. Iteration 25 did exactly that, taking `lcmnl3` from
1.14396 to 1.13863, and it remains the highest-weighted member at 0.504.

---

## Iterations 21, 22, 24 — status after the round-3 billing interruption

Round 3 launched six experiments; the two scouts finished and all six builders were killed
mid-run by a monthly spend limit. They had, however, already written their programs. The
disposition of each:

| iteration | code state | disposition |
|---|---|---|
| **21 fold robustness** | complete: `00_make_folds.R`, `xgb_split.R`, `lcmnl_split.R`, `mnl_split.R`, `blend_probe_split.R`; `folds_b.rds` and `folds_c.rds` built and verified independent (22% agreement vs 20% by chance) | **worth running — in progress.** The one validation that directly addresses selection across 26 iterations on one fold structure |
| **22 blend architecture** | `run.R` written, never executed | **closed without running.** Iteration 19 tested the same three architectures nested and found C-vs-A z = +1.49 with a fold reversal, and showed the motivating signal is absent from the only data it could be fitted on |
| **24 within-task contrasts** | `dom_feats.R`, `probe.R` written, never executed | **closed without running.** `d10` measured the features' R² against the incumbent residual at 0.000415 |
| **16 bundle encoding** | `run.R` written, never executed | **closed on structure.** The briefed idea is impossible (bijection); the surviving marginal-cell variant carries a pre-registered expectation of ~0 and a measured direct-correction gain of 0.00003 |
| 20, 23 | never created — the agents died first | 20 was renumbered to 26 (seed bagging); 23 (importance-weighted training) remains genuinely untested |

**`model/artifacts/folds.rds` was never regenerated or written at any point**, verified
against git.

---

## Iteration 35 — the combiner could not represent a negative weight ✅ ADOPTED (1.12819 → 1.12341)

**This is a combiner change, not a new model.** Nothing is fitted here: no tree is grown, no
likelihood maximised. The script consumes `oof_*.rds` / `test_*.rds` artifacts that earlier
iterations already produced. The only thing that changes is the function that combines them.

### The defect, found by reading the code rather than by searching

`model/06_blend.R` fitted the pool weights as

```r
w <- exp(theta[1:M]); w <- w / sum(w)
```

a softmax onto the **simplex**. Every weight is thereby forced non-negative and forced to sum
to one. The consequence, unexamined for thirty-four iterations, is that a **negative
coefficient is not disfavoured — it is unrepresentable**. A member whose optimal coefficient
is negative can only be pinned at the boundary, `w ≈ 0`, and is then reported as "earns weight
0.000, contributes nothing" and dropped.

That is the exact language `members.txt` uses about `mnl_pw` and `xgb_monobag`, and it is what
iteration 11 recorded when it found that "adding the seven zero-weight members changes the
nested score by 0.00005". It is also why the iteration-30 frontier probe reported its ceiling
as **"+0.00027, at NEGATIVE weight"** — it could see the sign and had no way to act on it.

**"Weight 0.000 under a non-negativity constraint" and "contributes nothing" are different
statements.** In a log-opinion pool a member with a negative coefficient is a **control
variate**: it subtracts a component the retained members share.

### Why it works here specifically

The tree family (`xgb_lw2bag`, `xgb_long`, `xgb_wide`, `xgb_2stage`) is fitted on overlapping
views of the same design with the same algorithm, so its errors share a large common
component. Iteration 19 measured the 4×4 error-correlation matrix's eigenvalues at
3.726 / 0.202 / 0.056 / 0.016 — **93% of the error variance in one direction**. A strictly
worse tree model is a noisy but nearly unbiased *reading* of that shared direction. Subtracting
a multiple of it removes shared tree bias without removing the signal only the good tree has.

Consistent with that mechanism, **only trees price negative**. `mnl_pw` prices at ≈0
(β = −0.02) because the logit direction is already spanned by `lcmnl3_both`.

### Result

| combiner | nested | vs production | clustered z |
|---|---|---|---|
| 2-member **simplex** (production) | 1.12819 | — | — |
| 2-member **free** | 1.12819 | −0.00000 | — |
| 3-member free (+`xgb_long`) | 1.12521 | +0.00298 | +3.03 |
| **5-member free** (+`xgb_wide`, `xgb_2stage`) | **1.12341** | **+0.00478** | **+3.84** |

Per-fold held-out logloss, 5-member: 1.13861 / 1.10686 / 1.11484 / 1.13554 / 1.12121 —
**positive in all five folds**.

Full-data coefficients: `xgb_lw2bag` +1.151, `lcmnl3_both` +0.606, `xgb_long` −0.348,
`xgb_wide` −0.189, `xgb_2stage` −0.237, ε 0.009. Per-fold coefficients are stable
(`xgb_long` −0.30…−0.39).

**The 2-member free row is the implementation check and it matters.** With two members the
simplex can already express the optimum, so free and simplex must agree — and they do, to five
decimals. The entire gain therefore comes from newly admissible negative coefficients, not
from a different optimiser finding a better answer.

Every added member is individually **worse** than both incumbents: `xgb_long` 1.15516,
`xgb_wide` 1.17456, `xgb_2stage` 1.17169, against 1.13682 and 1.13863.

### Verification — the leak question, and why the repo's usual detector is wrong here

Negative weights on out-of-fold predictions is the textbook way stacked generalisation fails.
All members' OOF predictions were generated under the *same* `folds.rds`, so their errors are
correlated **through** the folds, and a negative coefficient could be subtracting fold-specific
structure absent on 263 new respondents.

`CLAUDE.md`'s standing detector — *"a real gain appears in every fold, a leak concentrates in
one"* — **cannot see this failure mode**, because fold-correlated structure appears in every
fold. "Positive in all five folds" is not evidence here.

The right instrument is a **within-fold shuffle placebo**: permute the added members' OOF rows
*inside each fold*, preserving every fold-level property (per-fold mean, variance, calibration,
any fold-specific artefact) while destroying row-level alignment.

| | gain | share of real |
|---|---|---|
| real | **+0.00478** | — |
| within-fold shuffle placebo, 5 reps | **−0.00016** (sd 0.00011) | **−3%** |
| matched-marginal noise placebo, 3 reps | −0.00018 | −4% |

The placebo recovers **nothing**, and is reliably slightly negative — what you expect from
adding uninformative members that cost a little fitted-parameter noise. Two independently
constructed nulls agree.

### The other five checks

| check | result |
|---|---|
| **member-level `folds_b`** | 2-member 1.13248 → 4-member 1.12963, gain +0.00284 (z +3.16) vs matched production gain +0.00251 — **113% retention** (bar is 80%) |
| **segment-reweighted** | 1.19610 → 1.18811, **167% retention** |
| **multiplicity** | 46 candidates scanned; **31 price negative, 15 of those improve the blend**. Bonferroni needs z ≥ 3.27; measured z is +3.84 |
| **deployment / regime asymmetry** | re-applying each member's own OOF→test drift as a shock moves the 5-member blend 0.0055 vs the 2-member's 0.0048 — **~15% amplification**, not catastrophic |
| **guard rails** | mean p4 0.2377, entropy 1.1626, max prob 0.9216, **zero rows > 0.95, zero probabilities < 1e-3** |

The `folds_b` arm is **member-level**, not weight-level: `xgb_long` and `xgb_wide` had their
OOF regenerated under the independent respondent grouping. `xgb_2stage_b` was unavailable, so
that arm is the 4-member subset — which makes it conservative, not optimistic.

The multiplicity result is the answer to "you scanned 30 artifacts and kept the best three."
Thirty-one of forty-six price negative. This is a **property of the tree family**, not three
lucky picks.

### Honest reflection

Three things worth recording.

**The finding came from reading the combiner, not from searching model space.** Eleven
iterations of search the night before produced one adoption worth +0.0003, and a frontier probe
had declared member addition closed at +0.00027. That probe was *correct about what it
measured* and wrong about what it concluded, because it inherited the constraint it was
measuring under. The cheapest remaining gain was a line of code nobody had questioned.

**`model/blendfast.R` has used unconstrained coefficients all along.** The machinery to find
this existed in the repo for a day and was only ever pointed at screening existing member sets,
never at member *addition*. The tool was right there.

**`xgb_pt` ranks first in the marginal table at +0.00444 and is excluded.**
`experiments/iter30_decorr/run.R` states in its own header that the run was killed at fold 3
and any `xgb_pt` artifact from it is misnamed and must not be cited. The top of the candidate
table is a trap; the ⛔ discipline is what keeps it out.

### Status

`members.txt` is **not** changed by this iteration. The 5-member pool is a validated candidate
with a ready submission CSV; production remains the 2-member blend until the board says
otherwise. Local 1.12341 forecasts **~1.189–1.195** public (segment-anchored vs transfer-rate
routes disagree by ~0.005).

**Note for the entropy-floor discussion elsewhere in this file:** 1.12341 sits *below* the
quoted 1.125–1.133 floor. That floor was estimated over *models* and never priced the
*combiner*, so it was too conservative.

---

# ⚠️ ITERATION 43 — THE FREE-SIGN BLEND FAILED ON THE TEST SET (1.12341 local → 1.209 public)

**This is the most important entry in this file.** A result that passed six independent local
checks, replicated at 103% on an independent respondent partition, cleared a Bonferroni
multiplicity bar, and retained 167% under segment reweighting was **wrong on the test set** —
and wrong by more than its own claimed gain, with the sign reversed.

## The numbers

| | local nested | public |
|---|---|---|
| 2-member simplex blend (`sub_20260726_2328.csv`) | 1.12819 | **1.197** |
| 5-member free-sign + calibration (`sub_20260727_2200_free5cal85.csv`) | **1.12341** | **1.209** |

Local said **better by 0.00478**. The board said **worse by 0.012** — a sign flip of roughly
2.5 paired public SE (0.0047), so not noise.

## What was under test

The free-sign combiner (iteration 35): `model/06_blend.R` had fitted weights as a softmax onto
the simplex, making negative coefficients unrepresentable for 34 iterations. Freeing the sign
admitted three strictly *worse* tree models as **control variates** for the tree family's shared
error direction (iteration 19 measured 93% of error variance in one direction):

```
xgb_lw2bag +1.151 | lcmnl3_both +0.606 | xgb_long -0.348 | xgb_wide -0.189 | xgb_2stage -0.237
```

## Why it failed — the mechanism

A control variate cancels bias only if it **drifts with the member it corrects**. Member OOF
predictions come from fits on 80% of respondents (4 of 5 folds); the shipped test predictions
come from 100%-data refits. `xgb_long`, `xgb_wide` and `xgb_2stage` are older artifacts whose
test refits shifted differently from `xgb_lw2bag`'s. Subtracting 0.348 × something that moved
the wrong way does not remove bias — **it adds it.**

The drift was measured beforehand and under-weighted: the tree family's OOF→test mean-p4 shift
spans **0.0157** (`xgb_lw2bag` −0.0346, `xgb_long` −0.0189, `xgb_wide` −0.0280,
`xgb_2stage` −0.0245), and a shock simulation put the amplification at ~15%. The realised cost
was several times that. **A 15% haircut was forecast; a sign flip occurred.**

## THE METHODOLOGICAL FINDING — this is the report's headline

**Six independent checks passed, and all six were blind to the same thing.**

| check | result | why it could not see the failure |
|---|---|---|
| within-fold shuffle placebo | recovers −3% | computed on OOF |
| matched-noise placebo | −4% | computed on OOF |
| member-level `folds_b` replication | +0.00308, z +3.17, **103% retention** | computed on OOF, different partition |
| placebo member (`xgb_mono_b`) | prices positive, gains nothing | computed on OOF |
| multiplicity (46 candidates) | z +3.84 vs Bonferroni bar 3.27 | computed on OOF |
| segment reweighting | **167% retention** | *reweighted* OOF — still OOF |
| margin-neutralised gain | +0.00480, z +3.85 | computed on OOF |

Every one was computed on out-of-fold predictions or a reweighting of them. **The failure lives
in the gap between OOF and test, and no amount of cross-validation reaches across that gap,
because CV never leaves the training population.**

Counting independent *tests* is not the same as counting independent *assumptions*. Seven tests
collapsed to one assumption: that OOF predictions represent test predictions. They do not, when
the two are produced under different fitting regimes.

## Three rules derived from this

1. **Negative weights require matched-regime members.** A control variate must be refit under
   the *same* protocol as the member it corrects, or its correction is meaningless at deploy
   time. Ours were artifacts built at different times under different early-stopping draws.
2. **"Verified N ways" is worthless if the N tests share a blind spot.** Ask what each test
   *cannot* see, and check whether the answers coincide.
3. **The leaderboard is not a scoreboard — it is the only instrument for one specific
   question.** Spending a submission to answer it was correct and cost nothing, because Kaggle
   auto-selects the best public score. The real mistake would have been shipping this at the
   end without ever testing it.

## Decomposition — done WITHOUT spending a second slot

The submission bundled two changes: the free-sign members and the probe-derived none-margin
shift. They can be separated by bounding, because the shift's *entire possible influence* is the
spread of the none-margin cross-entropy across the versions we built:

| file | ships p4 | miss vs measured 0.26651 | none-margin cost | public |
|---|---|---|---|---|
| `sub_20260726_2328.csv` (live, 2-member) | 0.2480 | −0.0185 | 0.00090 | **1.197** |
| `sub_20260727_1420.csv` (5-member, uncalibrated) | 0.2377 | −0.0289 | 0.00224 | not sent |
| `sub_20260727_2100_free5cal.csv` (w = 0.60) | 0.2550 | −0.0115 | 0.00035 | not sent |
| `sub_20260727_2200_free5cal85.csv` (w = 0.85) | 0.2622 | −0.0043 | **0.00005** | **1.209** |

The calibration can move the score by at most **0.00224** in either direction. The observed
damage is **0.012** — more than five times that. **Therefore the free-sign members cost ~0.012
on their own**, and the version we shipped was the best-calibrated of the four.

Corollary: submitting the uncalibrated `sub_20260727_1420.csv` would score **~1.211**, not
better — same five members, dial set further from the measured truth. It was not worth a slot.

Second corollary: an isolating experiment (2-member + calibration alone) would measure a
quantity worth at most 0.0009 — **below Kaggle's three-decimal resolution**. It cannot return a
readable answer and was therefore *not* run, despite being the obvious next step.

## What survives

- **The probe measurement stands.** r = 0.2665 is exact algebra on a returned score and is
  untouched by this failure. It remains the only direct observation of the graded population.
- **The simplex-constraint finding stands as a fact about the code**: negative weights genuinely
  were unrepresentable for 34 iterations, and members genuinely were being recorded as
  "contributes nothing" when their optimal coefficient was negative. What does *not* survive is
  the conclusion that exploiting it improves the shipped model.
- **The 2-member blend at 1.197 is unaffected** and remains the auto-selected private entry.

## Honest reflection

The free-sign result was the most thoroughly verified finding in this project and it was wrong.
It was not wrong because the verification was sloppy — every individual test was sound, and the
mechanism (control variates cancelling a shared error direction) is real and is visible in the
OOF data. It was wrong because **the object being verified was not the object being shipped**.

The gap between "this blend of OOF predictions scores better" and "this blend of test
predictions scores better" is exactly the gap the whole exercise assumed away. Iteration 39 had
already hinted at it: killing the early-stopping carve gained +0.00252 at member level but only
+0.00020 at blend level, because the control variates were absorbing regime-specific variance.
That was a signal that the negative weights were fitting something regime-dependent, and it was
read as a curiosity rather than a warning.

Cost: one submission slot, and the board is unchanged. Bought: the knowledge that ~0.005 of the
apparent local edge was a fitting-regime artifact — discovered *before* it went into a final
submission rather than after.

---

## Iteration 45 — the ensemble's randomness is already optimal ⭕ NULL (both directions)

**A rare shape of result: two opposing hypotheses, both refuted, and the pair is worth more
than either would have been alone.**

### The gap, which was an error rather than an omission

Production ships `xgb_lw2bag`: a 10-seed bag at `subsample = 0.8, colsample_bytree = 0.8`.
Those values were tuned in **iteration 06, for a SINGLE model**, before bagging existed in this
project. Iteration 26 then wrapped ten seeds around them and changed nothing else — its own
header reads *"ONE CHANGE UNDER TEST: the number of seeds averaged."*

So production has been running single-model hyperparameters inside an ensemble, and nobody had
ever checked whether that is right.

### 45a — hypothesis: an ensemble wants MORE randomness. REFUTED.

The reasoning was derived, not guessed:

```
loss(bag) ~ bias^2 + rho * var + (1 - rho) * var / S
```

Raising per-model randomness raises `var` and lowers `rho`. At S = 1 that trade is bad, so
single-model tuning drives randomness down. At S = 10 the `(1-rho)var/S` term is divided by
ten, so cutting correlation should be worth more than the variance it costs — and the ensemble
optimum should sit at MORE randomness than 0.8/0.8.

Pre-registered direction: the winner must have `sub <= 0.8 AND col <= 0.8`, so that a null
could not be re-read as a win.

12 points, 3 seeds each, nested, fixed 540 rounds. **Every point with more randomness was
worse, monotonically.** The winner was the boundary — the incumbent 0.8/0.8 — at 1.13606
against the matched baseline `xgb_lw2fr` of 1.13604, a difference of 0.00002 and pure seed
noise.

### 45b — the opposite hypothesis, registered separately. ALSO REFUTED.

The 45a trend pointed *outside* the grid, toward less randomness. That is a **new hypothesis**,
not a reinterpretation of the old one, so it was registered as such before running, and the
multiplicity was carried forward rather than reset (16 points searched in total, bar z >= 2.98).

4 points at `sub, col ∈ {0.9, 1.0}`. All worse, and increasingly so.

### The combined surface — 16 points

| sub | col | OOF | | sub | col | OOF |
|---|---|---|---|---|---|---|
| 0.5 | 0.4 | 1.14100 | | 0.8 | 0.4 | 1.13829 |
| 0.5 | 0.6 | 1.13924 | | 0.8 | 0.6 | 1.13658 |
| 0.5 | 0.8 | 1.13776 | | **0.8** | **0.8** | **1.13606** ← minimum |
| 0.6 | 0.6 | 1.13780 | | 0.9 | 0.9 | 1.13711 |
| 0.6 | 0.8 | 1.13726 | | 0.9 | 1.0 | 1.13797 |
| 0.7 | 0.6 | 1.13641 | | 1.0 | 0.9 | 1.14091 |
| 0.7 | 0.8 | 1.13707 | | 1.0 | 1.0 | 1.14554 |

**Unimodal, with an interior optimum exactly at production's setting.**

### What this is actually worth

Nothing for the score, and that was the likely outcome. But it converts a live assumption into
a measurement: **the single-model optimum and the ensemble optimum coincide here.** Iteration
06's tuning survives being wrapped in a 10-seed bag, which nobody had verified and which the
bias-variance algebra above says should NOT generally be true.

It also closes the tree's randomness axis in both directions, so no future iteration needs to
revisit it.

### Reflection

Two things worth recording.

**The mechanism was sound and the prediction was still wrong.** `loss(bag) = bias² + ρ·var +
(1−ρ)var/S` is correct; what it does not tell you is where this particular ensemble sits on the
`ρ(randomness)` curve. The 10-seed bag's members are evidently already decorrelated enough that
extra randomness buys no `ρ` reduction worth its variance. A derived hypothesis is still a
hypothesis.

**This was only affordable because of a failed experiment.** Iteration 39 removed the
early-stopping carve; its change did not survive the blend gate, but the harness it built cut
a fit from ~3 minutes to ~11 seconds by eliminating the per-round R callback. A 16-point grid
was not feasible in this project before that. Negative results build infrastructure.

---

## Iteration 46 — temperature on the test-population metric ❌ OVERFITTING, caught by nesting

**A near-miss worth recording, because the in-sample number was convincing.**

**The idea.** Every calibration parameter in this project was fitted on *plain* OOF — the
training population. The graded population is different (two luxury segments are 9.4% of
training respondents and 68.8% of graded rows). So re-fit the temperature on the
segment-reweighted metric, which is the test population's proxy.

**The in-sample result looked strong and had a mechanism:**

| | optimal T | gain |
|---|---|---|
| plain OOF | 1.0022 | +0.00000 |
| segment-reweighted OOF | 1.0812 | **+0.00101** |

and it *survived* correcting the none-margin first (+0.00116 after, slightly larger), so it
was not the margin fix in disguise. The mechanism was plausible: the model has ~1/10 the data
on luxury respondents, so it should be less confident about them, but it applies one global
sharpness everywhere.

**Then the nested test killed it.** Fit T on folds != k, evaluate on fold k, reweighted:

```
T chosen per fold:  1.006, 1.141, 1.186, 1.115, 0.983
base (T = 1)        1.19610
nested T            1.20263
NESTED GAIN         -0.00654      (in-sample: +0.00116)
```

The per-fold instability is the diagnosis: a real effect would have the folds agree; these span
0.98 to 1.19. The metric's own respondent-bootstrap SE is **0.02123**, so the in-sample gain was
**1/18th of the noise in the instrument measuring it**.

**This is iteration 07 again, and iteration 07 called it.** That iteration refuted shift-aware
blend calibration on the *income*-reweighted objective and diagnosed the cause exactly:
*"importance weighting is self-defeating here — reweighting cuts the effective sample, so
fitting parameters against a noisier target produced a worse fit even when judged on that same
noisy target."* Segment reweighting has **ESS 208 of 1,135 respondents**, thinner than income's,
so the effect is stronger rather than weaker.

**What survives.** The segment-reweighted metric remains usable as a *veto* and a rough
forecaster. It must never be used as a *fitting objective* — for any parameter, not just blend
weights. CLAUDE.md already says "never optimise on it (iteration 07)"; this extends that from
income to segment reweighting, which is the version that looks more legitimate and is in fact
worse.

**Reflection.** The failure mode was building the submission CSV before running the nested
check. The in-sample number, the mechanism, and the interaction test with the none-margin all
pointed the same way, and the order of operations is what caught it. A convincing story plus an
unnested number is exactly the shape of the results this log keeps retracting.

---

# ⛔ ITERATION 48 — THE DESIGN-SHARE ENCODING IS LEAKING, AND IT INVALIDATES THE TUNING OF THE WHOLE PROJECT

**This is the most important entry in this file.** It explains three failed submissions, it
retracts the tuning of the tree member, and it means every plain-OOF number recorded above
for an encoded model is inflated. Read it before trusting any earlier figure.

`experiments/iter48_encleak/` — four matched arms plus an independent block construction.

## The defect

`model/encode_design.R` is fold-aware **at the row level** — verified two ways: the code does
`encode_aligned(trl[fold != k], trl[fold == k])`, and 0 of 1,135 respondents span a fold
boundary. The docstring *"a person never contributes to their own encoding"* is literally true.

But `apply_design_encoding()` is called **ONCE, BEFORE any CV loop**
(`model/03_xgb_listwise.R:37`, and in every experiment script that consumes it). So a training
row in fold *j* carries an encoding built from folds ≠ *j* — **a set that includes the scored
fold k**.

For cell `(dkey, alt)` the numerator is exactly `n_ch = C − c_j` (C = cell total over all folds,
`c_j` = own fold's chosen count). Verified numerically: recovering `n_ch` from
`(share_a1, design_n, fold prior)` and subtracting `(C − c_j)` gives max abs error **8.88e-16**.
The four-column tuple is an **exact invertible encoding** of `(C − c_own_fold, N − n_own_fold)`.

**Support is what makes it fatal.** 5,624 designs, 22,496 `(dkey, alt)` cells, mean 3.83 rows
per cell, **median 1 row per (cell, fold)**. **46.3% of training rows sit in a (cell, fold)
block of size 1** — for those, `c_j` *is* the row's own label, so leave-own-fold-out degenerates
to **leave-own-ROW-out**.

Within-cell `cor(share, own chosen)`: **−0.7107 / −0.7110 / −0.6926** for α = 1/5/20. On the
scored rows, `cor(C_U − enc, own label)` is **+0.7442** in production against **+0.1149** under
an honest construction.

## The decomposition — depth 8, 5 seeds per arm, paired and respondent-clustered

| contrast | meaning | delta | SE | z |
|---|---|---|---|---|
| noenc → prod | production's claim | **+0.02179** | 0.00148 | 14.69 |
| honest → leaky | **pure leak**, matched support | **+0.01781** | 0.00139 | 12.79 |
| noenc → honest | **honest value** | **−0.00596** | 0.00169 | **−3.53** |
| leaky → prod | 3→4 fold reference | +0.00995 | 0.00169 | 5.89 |

A second, independent construction (a disjoint one-fold reference block seen identically by
train and score rows, with no complement identity anywhere) also comes out **negative**:
−0.00440 (z −3.45) at depth 8, −0.00287 (z −1.85) at depth 10.

**Two independent honest constructions, both negative. The encoding's genuine value is zero or
worse. All of its apparent +0.0218 is the complement identity.**

## Why it cannot transfer

Test respondents are new people who appear in no cell, so test rows are encoded from all of
`trl` with **nothing subtracted**. `design_n` averages **3.943 on test against 2.876 on
train/OOF** (×1.371), and **7.0% of test rows have `share_a1` strictly above every training
value of the same cell**. The within-cell rule the tree learned cannot fire correctly there.

## THE LEAK SCALES WITH TREE CAPACITY — this is what wrecked the tuning

| config | noenc | prod | leak gain |
|---|---|---|---|
| depth 4 | 1.15640 | 1.15678 | **−0.00038** |
| depth 6 | 1.15259 | 1.14597 | +0.00662 |
| depth 8 | 1.15910 | 1.13673 | **+0.02237** |
| depth 10 | 1.16793 | 1.12855 | +0.03938 |
| d10 / mcw10 / eta .02 / 1400r | 1.19566 | 1.11799 | **+0.07767** |

**And the honest curve runs the other way** — no-encoding gets monotonically *worse* with depth
(d4 1.14994 → d10 1.19566). **The honest optimum is depth 4–5. Production ships depth 8.**

Iteration 01 measured the encoding at +0.00461 (z 2.94) using depth **6**. Production reads
+0.0218 at depth **8**. Nothing changed between them except capacity. Monotone-in-capacity with
no saturation is the leak signature; a real design-share effect must saturate, and the honest
share alone caps out at 1.307.

## What this retracts

1. **Production's nested OOF 1.12819 is inflated by ~0.0077.** The honest figure is **~1.1359**.
2. **The blend weight is wrong.** `w_tree = 0.528` was fitted on leaked OOF; honest wants
   **0.194–0.243**. Forcing 0.528 onto an honest tree costs +0.00437 nested. **We are
   over-weighting the tree by roughly 2.2×, live, in the shipped submission.**
3. **Iteration 47's 27-config sweep is void.** Its ranking is monotone in depth-up and
   mcw-down — exactly the direction that increases cell-isolation power — and the no-encoding
   ranking is the *reverse*.

## ⚠️ IT ALSO BREAKS THIS REPO'S OWN LEAK HEURISTIC

`CLAUDE.md` says *"a real gain appears in every fold; a leak concentrates in one."* **This leak
appears in 5/5 folds at every depth and every seed.** A *structural* leak is uniform across
folds. That heuristic would have missed this entirely, and must be amended.

## What it does NOT explain

**It does not explain the gap to the leaders.** A model without the encoding scores ~0.008
*worse* locally and **essentially identical publicly** (predicted 0.000–0.002 better). Removing
the leak is roughly board-neutral. The leak explains why we *mis-tuned* and *wasted
submissions* — not why we are behind.

## Reflection

The docstring was true and the code was fold-aware, so every reading of it passed. What nobody
checked was **when the function is called relative to the CV loop**. A correct function invoked
at the wrong point is invisible to code review and to per-fold diagnostics alike. The only
thing that caught it was building an isomorphic honest arm with matched support and comparing.

---

# Iteration 49–51 — matrix factorization, and the random-rotation result that wasn't

`experiments/iter49_matfact/`, `iter51_encablation/`

**Hypothesis.** Factorization had never been tried in this repo — no SVD, PCA, NMF anywhere.
Two mechanisms were pre-registered: *rotation* (xgboost splits axis-aligned, so oblique
boundaries are expensive in a sparse one-hot basis) and *dense re-coding* of the 92 attribute
level-dummies.

**Screen (1 seed, plain OOF, base 1.13761):**

| config | OOF | vs base |
|---|---|---|
| demographic PCA, k = 4 | 1.14987 | **+0.0123** |
| demographic PCA, k = 8 | 1.15574 | **+0.0181** |
| PCA replacing raw demographics | 1.16506 | +0.0275 |
| design SVD, k = 8 | 1.13369 | −0.0039 |
| design SVD, k = 32 | 1.12655 | −0.0111 |

**Demographic PCA is actively harmful, and the mechanism is worth recording.** Raw demographics
are coarse ordinals with few levels, which limits memorisation. Rotating them into real-valued
components makes each respondent nearly unique, so the tree **fingerprints individuals** inside
the fold. Folds are respondent-grouped, so that buys nothing on held-out people. It degrades
monotonically with more components — exactly as fingerprinting predicts.

**The design-side SVD looked spectacular.** Iteration 51's ablation found `dsvd72` at 1.11868
against raw one-hot dummies at 1.13663 — **identical column space, rank 71, same information,
different basis** — a gap of 0.018. Iteration 54 then found a **random orthonormal rotation
beat the SVD** (1.11515 vs 1.11861), which killed the factorization interpretation: the
principal directions were not doing the work, any dense rotation was.

**A correctness check I specified wrongly, recorded because the error is instructive.** The
script asserted `dsvd72` must land within one seed sd of `onehot`, reasoning that identical
column spans imply identical models. **That is true for a linear model and false for a tree** —
tree fits are not rotation-invariant, which is the entire mechanism under test. The check
contradicted the hypothesis it was meant to protect, and fired FAIL on a real effect.

---

# ⛔ Iteration 54–59 — the rotation was a LEAK AMPLIFIER. Submitted, scored **1.205**.

`experiments/iter54_rotverify/`, `iter57_foldsb/`, `iter59_rotnoenc/`

**It passed five gates and was still wrong.** Recorded in full because the gates' failure is
more useful than the result.

| gate | bar | result | |
|---|---|---|---|
| 10-seed member gain | > 0.00283 | −0.02192, **10/10 wins**, sd 0.00180 | ✅ |
| nested basis (not transduction) | survives | −0.00007 vs transductive | ✅ |
| segment-reweighted retention | ≥ 90% | **105%** | ✅ |
| blend gate | > 0.00048 | −0.01609 = **33.5 blend sd** | ✅ |
| folds_b replication | ≥ 60% | **91%**, 3/3 | ✅ |

Nested blend **1.11210** against production 1.12819. Predicted public 1.187 (band 1.179–1.188).
**Actual public: 1.205.** A sign flip of 0.008 against a prediction that was 0.018 out.

**The ablation that should have been run first** (`iter59`, 4 arms × 3 seeds, one factor at a
time):

```
            base       rot        gain
WITH  enc   1.13746    1.11561    −0.02185
NO    enc   1.15925    1.17396    +0.01470
SURVIVAL RATIO  −67%
```

**Remove the encoding and the rotation flips sign.** It does not merely stop helping — it hurts
by +0.015 (71 dense columns diluting `colsample_bytree` and fitting noise). So the −0.022 was
**100% leak amplification on top of an intrinsically harmful feature block**. `rot_noenc`
(1.17396) does not even beat the honest production member (~1.1359).

## Why every gate was blind to it

- **folds_b re-runs the same one-shot encoding**, so it replicates the leak rather than testing
  it. 91% retention was 91% retention *of the leak*.
- **Segment reweighting is blind** — the leak is population-independent, so it inflates plain
  and reweighted alike. 105% retention meant nothing.
- **The per-fold heuristic is blind** — structural leaks are uniform across folds.
- **The blend gate is blind** — both members were scored on the same contaminated OOF.

**The rule that would have caught it, and now must be applied:** *a contrast is only valid if
leak exposure is matched on both sides.* The rotation changed **capacity**, which changes leak
exposure, so the contrast was contaminated even though both arms used identical features. By
the same rule, iteration 39's carve removal **is** valid — it changes *data* (recovering 114
respondents), not capacity, and its `ENC_COLS` exposure is byte-identical across arms.

## Reflection

Iteration 49's own header pre-registered *"monotone-in-k ⇒ treat as noise"*, and the gain **was**
monotone in k. That flag was overturned on the strength of iteration 51's encoding ablation —
which tested the wrong alternative hypothesis. The leak hypothesis was never tested until after
the slot was spent. **Ruling out one alternative explanation is not the same as ruling out the
alternative that matters.**

---

# Iterations 50, 53, 61 — three honest nulls

| iter | idea | result |
|---|---|---|
| **50** | reduced-rank demographic × attribute interaction, `u = x'β + z'Cx` with `C = ΓV'` | rank curve unimodal at F = 1 (1.17462 → 1.16649, 2.87 sd), then reverting at F = 2 and F = 4. **Earns weight 0.000 in the blend.** Fully absorbed by the incumbents. |
| **53** | low-df GAM outside-option head on our blend, share chosen by inner CV | nested −0.00074 (1.54 blend sd) on plain OOF, but **+0.00009 on the segment-reweighted metric — ~0% retention.** Same failure mode that killed the design encoding (77%) and a fatigue term (64%). Margin-only property verified at 1.1e-16. |
| **61** | synthesis model: no encoding, one-hot, depth swept, MF cold-start features, segment importance weighting | **fails the blend gate.** SEG 1.20104 vs production 1.19610. |

**Iteration 61's stage detail**, all judged on the segment-reweighted metric (ESS 3,946):

| stage | SEG |
|---|---|
| honest baseline, depth 3 / 4 / 5 / 6 | 1.25077 / 1.23800 / 1.23405 / 1.23172 |
| + MF cold-start features | 1.23138 (**−0.00034 — noise**) |
| + segment importance weighting, τ = 0.25 / 0.50 / 1.00 | 1.23332 / 1.23275 / 1.23156 (**null**) |

**A limitation of that τ sweep, stated so it is not over-read.** The team's second track uses an
importance-weight *exponent* of 0.05–0.10. This sweep started at τ = 0.25 — **2.5× the largest
value in use** — so it tested only heavy reweighting. The correct conclusion is *heavy
reweighting is null on this stack*; light reweighting remains untested.

**Iteration 61's one genuinely useful output:** the blend gave the honest tree weight **0.278**,
against production's 0.528 for the leaky one. Iteration 48 independently predicted the honest
weight at **0.194–0.243**. Two unrelated constructions agreeing that the live blend
over-weights its tree by ~2× is the strongest evidence in this file for acting on it.

---

# Member scores on the metric that actually predicts the board

Segment-reweighted nested OOF (unclipped `p_test/p_train` by `segmentind`). The production
blend reads **1.19610** here against an actual public of **1.197** — it closes **98.7%** of the
+0.069 offset, where plain OOF is off by the full amount.

| member | plain | **segment-reweighted** | encoding |
|---|---|---|---|
| `lcmnl3_both` | 1.13863 | **1.20493** | clean |
| `mnl_pw` | 1.15686 | 1.21199 | clean |
| `xgb_lw2bag` | 1.13682 | 1.21516 | **leaky** |
| `xgb_syn` (iter 61) | 1.15396 | 1.23138 | clean |
| **production blend** | 1.12819 | **1.19610** | — |

**`lcmnl3_both` is our best single component on the graded population** — ahead of the tree it
is out-weighted by. It is also completely clean of the encoding.

**Caveat on the metric.** It has two calibration points: production blend 1.19610 → 1.197
(hit to 0.001), and the rotation blend 1.18596 → 1.205 (**missed by 0.019**, because the leak
inflates plain and reweighted equally). It is the best forecaster available and it is **not** a
law. It is a veto and a rough forecast — **never a fitting objective** (iterations 07, 46).

---

# ⛔ THE ⛔ TABLE, UPDATED

| idea | killed by | number |
|---|---|---|
| design-share encoding as a *source of gain* | iteration 48 | honest value **−0.00596**, z −3.53 |
| depth > 6 on the tree | iteration 48 | honest curve monotonically worse; optimum 4–5 |
| iteration 47's 27-config sweep | iteration 48 | ranked on contaminated OOF; honest ranking reverses |
| random/SVD rotation of the one-hot basis | iteration 59 | survival ratio **−67%** without the encoding |
| demographic PCA as tree features | iteration 49 | +0.0123 to +0.0275 (respondent fingerprinting) |
| reduced-rank demographic × attribute | iteration 50 | blend weight **0.000** |
| GAM outside-option head on our blend | iteration 53 | ~0% retention under reweighting |
| heavy segment importance weighting (τ ≥ 0.25) | iteration 61 | null at every τ tested |
| MF cold-start features on our stack | iteration 61 | −0.00034 against a 0.003 noise floor |
| plain nested OOF as a decision metric | iteration 48 | off by **0.069**; use segment-reweighted |

---

# The submission record these iterations produced

| file | local | public | what it taught |
|---|---|---|---|
| `sub_20260726_2328.csv` | 1.12819 | **1.197** | the 2-member blend baseline |
| `sub_20260727_2200_free5cal85.csv` | 1.12341 | 1.209 | free-sign control variates need matched regimes |
| `sub_20260727_1420.csv` | 1.12341 | 1.211 | predicted 1.211 exactly, pre-registered |
| `sub_20260729_rotblend_cal85.csv` | 1.11210 | **1.205** | the leak, confirmed on the board |
| `sub_20260729_nnblend.csv` | — | **1.194** | best entry; see `experiments/iter62_nnblend/` |

**Since 1.197, every local improvement produced a public regression** — until the entry that
was not selected on local OOF at all. The correlation between our plain local metric and the
board went *negative*, which is the single clearest statement of what iteration 48 found.
