# Report notes — material for the 8-page report (due 10 Aug 2026)

Rubric: (i) best model description **5** · (ii) public vs private fit **2** ·
(iii) insights & limitations **5** · (iv) overall quality **3** — **15 of 30 marks total.**
Full experiment history with reflections: `EXPERIMENTS.md`. Scores: `submissions/log.md`.

---

## (i) The model — what we submitted and why

**A blend of three models, pooled in log-space with a fitted temperature.** Nested
respondent-grouped OOF **1.13044**; public leaderboard 1.201 (benchmark 1.38629).

| member | what it is | OOF | weight |
|---|---|---|---|
| `lcmnl3` | **latent-class conditional logit**, 3 segments, part-worth coded, membership predicted from demographics | 1.14396 | 0.447 |
| `xgb_mono` | gradient-boosted trees, **listwise softmax objective**, monotone price constraint, design-share features | 1.13980 | 0.337 |
| `xgb_lw2` | same but unconstrained | 1.14152 | 0.216 |
| `mnl_pw` | part-worth conditional logit | 1.15686 | **0.000** |

The single-model part-worth logit falls to **zero weight** once the latent-class model is
present. That is exactly right: the latent-class model with one class reproduces it to four
decimal places (verified: 1.15695 vs 1.15686), so it is a strict generalisation and there is
nothing left for the simpler model to contribute.

### The three ideas that produced the score

1. **Part-worth coding of attribute levels** (+0.020, z = 11.8) — the largest single gain.
2. **A listwise softmax training objective** (+0.006, z = 4.3) — making the training loss
   identical to the competition metric rather than a pointwise proxy.
3. **Latent-class heterogeneity** (+0.013, z = 3.8) — discrete taste segments.

### Alternatives tried and rejected (worth a paragraph each)

| approach | result | why it failed |
|---|---|---|
| Mixed logit, continuous random coefficients | 1.17281 | population-averaging costs sharpness; see (iii)2 |
| **Nested logit**, {bundles} vs {none} | 1.15681 | statistically identical to plain MNL — IIA is *not* the binding constraint |
| **Two-stage** none-vs-buy then bundle choice | 1.17169 | z = −11.0; splitting the decision loses the shared utility scale |
| Wide 4-class xgboost | 1.17456 | cannot express within-set comparisons natively |
| Elastic-net logit with demographic interactions | 1.16718 | regularisation cannot recover what part-worths give directly |

The nested-logit result is worth reporting: the textbook fix for IIA made **no difference**,
which says the multinomial logit's restrictive substitution pattern was not what was limiting
us — taste heterogeneity was.

---

## (iii) Insights

### 1. Three distinct buyer segments, and one of them explains the aggregate price curve

The latent-class model splits respondents into three near-equal segments. Price part-worths
(utility relative to the cheapest level, so more negative = more price-averse):

| segment | share | price effect across the range | outside-option constant |
|---|---|---|---|
| **Value-conscious buyers** | 36.6% | **−1.65** (steadily declining) | −1.40 (buys readily) |
| **Price-indifferent buyers** | 32.1% | **+0.50** (flat to slightly positive) | −1.30 (buys readily) |
| **Reluctant non-buyers** | 31.4% | −0.60 (mild) | **+0.17 (inclined to decline)** |

This **reframes the headline finding from the single-model analysis**. Fitting one logit to
everyone gives a concave price curve (−0.52, −0.76, … −1.93, −1.92, saturating at the top),
which invites a Weber–Fechner "people perceive price ratios" reading. The segment model
suggests that curvature is substantially a **mixing artefact**: averaging a steeply negative
price response (37% of people), a flat one (32%), and a mild one (31%) produces an aggregate
curve that bends — even though no individual segment's curve looks like the average.

**Report this carefully.** A positive price coefficient for segment 2 is not behaviourally
standard. Candidate explanations, none yet tested: price acting as a quality signal;
residual confounding between price and bundle richness in the experimental design; or the
segment absorbing respondents whose choices are driven by something we do not observe. State
it as an estimated pattern with these caveats rather than as established consumer psychology.

The third segment is the commercially interesting one: **a third of respondents are inclined
to buy nothing regardless of price**, and no discount reaches them. Marketing effort aimed at
that group is largely wasted; the leverage is in segments 1 and 2.

Segment membership is predicted from demographics only (never from a respondent's own
choices — that would be leakage and is checked for explicitly). The strongest membership
drivers are annual mileage, night-driving share, education and parking habits.

### 2. Taste heterogeneity is real, but only usable in *discrete* form

Mixed logit with continuous random coefficients estimates enormous heterogeneity — price
sensitivity varying roughly six-fold between the 25th and 75th percentile respondent, and an
outside-option constant with sd 2.47. Yet it **lost** to the fixed-coefficient model (1.173
vs 1.157) and earned zero blend weight, while the latent-class model with the same underlying
idea **won** (1.144, z = 3.8).

The difference is what you can condition on. Every test respondent is a stranger, so a
continuous mixture can only be integrated out — you predict the population average, which is
flatter than any individual. A finite mixture instead routes each respondent to a segment
**using their observable demographics**, which survives into the test set. Heterogeneity is
usable exactly to the extent it correlates with something you can measure about a new person.

**Hierarchical Bayes settles this, and the number is the strongest in the report.** HB is
the textbook tool for panel conjoint data, so we fitted it (`bayesm::rhierMnlRwMixture`,
16,000 draws, 3-component mixture population, demographic channel β_i = Δ′z_i + u_i — about
15 hours of compute). It scored **1.237**, the worst model we built. The reason is visible
directly in the fit. On *identical rows*:

| prediction | logloss |
|---|---|
| each respondent's **own** posterior-mean β_i | **0.352** |
| the same rows, **population-averaged** — what a new respondent gets | **1.229** |

That gap of **0.877** is the part of HB's fit that is *structurally unavailable at test
time*. It is not a tuning failure or a convergence problem; it is the cold-start problem
stated numerically. Any method whose power comes from learning individuals — HB, mixed
logit, respondent-level shrinkage — is spending its capacity on a quantity that does not
exist for the 263 people it will be graded on.

**And the demographic channel flips sign with dimensionality.** Ablating demographics from
HB *improves* it, 1.237 → 1.164. In the latent-class model the same demographic information
is worth 93% of the model's entire gain. Same data, same idea, opposite outcome — the
difference is how many parameters the information passes through:

| model | route from demographics to tastes | parameters |
|---|---|---|
| latent class | → 3-way segment membership softmax → segment part-worths | (3−1) × 29 = **58** |
| hierarchical Bayes | → shifts all 73 part-worths, per demographic | 6 × 73 = **438** |

HB was given *fewer* demographics (6 versus 29) and still ended up with roughly eight times
the parameters, because it interacts each one with every part-worth rather than with a
low-dimensional membership. **The segment acts as a bottleneck, and the bottleneck is why it
generalises.** With 1,135 respondents the compressed channel learns something real and the
free one memorises. This also explains our own iteration-14 result, where enriching the
membership model made it worse.

The general statement for the report: *heterogeneity helps when it is discrete and
low-dimensional, and hurts when it is continuous and high-dimensional* — and the binding
constraint is not how much heterogeneity exists in the population, but how much of it can be
recovered from what you observe about a stranger.

### 3. The choice sets are a designed experiment, and that leaks information

Only 299 distinct choice-set designs exist per task position, each shown to ~4.7 respondents,
and **98.5% of test rows reuse a design seen in training**. The empirical choice share within
a design is therefore partly observable — worth +0.005 (z = 2.9).

The size is itself informative. With ~3.8 observations per design the raw shares are so noisy
that at weak shrinkage they score **worse than the 25%-everything benchmark** (1.433 vs
1.386); at moderate shrinkage, 1.307. Most of the signal must be discarded to use any of it.

**Limitation:** this gain is specific to this data collection and would vanish on a new
experimental design. It also retains only 77% of its value when respondents are reweighted
toward the test population, versus ~100% for the structural gains.

### 4. Train and test populations differ materially

Test respondents are about twice as wealthy (median income $60k → $80k, p75 $85k → $125k),
better educated, and drive more. `model/shift_audit.R` reweights training respondents to match
the test income distribution and rechecks every improvement:

| improvement | value retained under reweighting |
|---|---|
| part-worth coding | 101% |
| listwise objective | 112% |
| design-share encoding | 77% |
| latent-class model | 79% |

The two largest gains are **structural** and should hold on the private leaderboard. The two
encoding/membership-driven gains are more fragile — both route through channels (design
identity, demographic membership) whose training support is thinnest exactly where the test
set is concentrated.

### 5. The outside option dominates

The all-zero "none of these" bundle is the most-chosen alternative at 30.2%, against
22.0 / 25.0 / 22.7% for the three real bundles. Two of our failed experiments are informative
about it: a **two-stage** model that predicted "buy at all?" separately from "which bundle?"
scored 1.17169 (z = −11.0), and a **nested logit** isolating it scored no better than plain
MNL. Opting out is the modal behaviour, but it is evidently governed by the *same* utility
scale as the bundles rather than by a separate decision process.

### 6. How much accuracy is left — and why we stopped

Most competition reports end with "future work: try more models." We can do better, because
we measured the ceiling.

**The oracle bound.** Give the model each respondent's *own* observed choice rates — the
best any method could do if it knew every individual perfectly — and the blend's 1.130 falls
to **0.986**. That gap of **0.144** is where *all* remaining loss lives, and essentially none
of it is reachable: every test respondent is a stranger.

**Why it is unreachable, quantified.** Decomposing the variance of choice behaviour into
between-respondent and within-respondent parts:

| alternative | share of variance that is between-respondent |
|---|---|
| alt 4 (the "none" option) | **33%** |
| alts 1–3 (real bundles) | 5–7% |

So the *only* strongly personal decision is whether to buy at all — which bundle you prefer,
conditional on buying, is nearly universal. And of the true between-respondent variation in
propensity to decline, demographics explain only **11.2%** (true variance 0.0693 after
removing the binomial component from an observed 0.0804; demographic R² = 0.0968).
**Roughly 89% of the personality that drives the outside option is simply not observable in
this dataset.** That single number bounds everything: no model, however sophisticated, can
recover taste variation that was never recorded.

**The design side is exhausted too.** The best predictor built purely from choice-set
identity — the empirical design shares that were worth +0.005 as a *feature* — scores
**1.307** on its own, against the blend's 1.130. There is no large untapped design signal.

**What this justified.** Given a 0.144 gap that is ~89% unobservable, a private leaderboard
of ~1,500 rows with SE ≈ ±0.02, and a measured local→public transfer rate that decayed from
58% to about a third, we judged further model search to have a lower expected return than
the analysis itself. We report the ceiling rather than gesturing at future work.

---

## Methodology — the part we would defend hardest

### Validation design

One fixed 5-fold split **grouped by respondent**. Each person answers 19 tasks; splitting
randomly by row lets a model learn "this person is stingy" from 15 of their answers and be
graded on the other 4. The test set is 263 people who appear nowhere in training, so the
local protocol must mirror that. The blend is scored **nested** — weights refit five times,
each excluding the fold it is evaluated on.

Model comparisons use a **paired test with respondent-clustered standard errors**. Fold-to-
fold SD is ±0.013, so a genuine +0.005 improvement is invisible in headline numbers but
clear when paired on identical rows.

### The seed is a hyperparameter nobody tunes, and it invalidated one of our results

Late in the project we asked a question we should have asked at the start: **how much does
a gradient-boosted model's score move if you change nothing but the random seed?** With
`subsample = 0.8`, `colsample_bytree = 0.8` and a randomly drawn 10% early-stopping holdout,
we refitted the identical configuration under ten seeds:

| | |
|---|---|
| mean OOF | 1.14303 |
| **sd across seeds** | **0.00283** |
| range | 1.14061 – 1.14957 (**0.00896**) |

**That sd is larger than most of the improvements in our experiment log.** In particular, we
had accepted a *monotone price constraint* on the grounds that it improved OOF from 1.14152
to 1.13980 — a margin of 0.00172, less than the seed noise it was measured against. It had
carried roughly a third of the blend weight ever since.

Re-testing it properly, **paired by seed** (the same seed drives the early-stopping split for
both configurations, and the two configurations' scores correlate 0.93 across seeds, so
pairing removes a large shared component):

| | |
|---|---|
| mean(constrained − unconstrained) | **+0.00065** — the constraint is slightly *worse* |
| 95% CI | [−0.00050, +0.00181] |
| the original claim (−0.00172) | **outside the CI** |
| seeds where the constraint wins | **1 of 6** |

The constraint does nothing. The original result was a lucky draw, and we had been carrying
two copies of the same model in the blend under different names.

Two lessons we would generalise:

1. **A single-seed comparison cannot resolve a difference smaller than the seed sd**, and
   almost nobody measures the seed sd before quoting a margin. It costs one extra run to
   find out and it recalibrates every number in the log.
2. **Averaging over seeds is the one improvement that cannot overfit the validation set.**
   It involves no model selection — no hyperparameter chosen by looking at a fold, no feature
   kept because it scored well — so unlike every other gain we made it has no reason to decay
   on the leaderboard. Bagging ten seeds moved the model from 1.14303 to 1.13714.

There is a subtlety in how to value that last number, and it is the kind of thing a paired
test will mislead you about. Our stored single-seed artifact happened to score 1.13980,
about 1.1 sd *below* the seed mean — a lucky draw. Comparing the bagged model against it
gives only +0.00266 at z = 1.77, "not significant". But a model's cross-validation artifact
and its **test** artifact come from *independent* random draws: the OOF predictions come from
five fold-fits under one seed, the test predictions from a separate refit on all the data.
Getting a lucky OOF tells you nothing about whether the shipped test refit is lucky. So the
expected quality of a single-seed submission corresponds to the seed **mean**, not to its
observed OOF, and the honest value of bagging is the ~0.006 it buys against that mean. The
significance test answered a question about two artifacts; the decision needed a question
about two *procedures*.

### A leakage postmortem worth including

One experiment scored **1.09962** — nine times the project's entire accumulated gain — from a
single derived feature. It was pure leakage, and the mechanism is subtle enough to be worth
the space:

The feature encoded, per choice-set design, the average *residual* (observed − predicted)
of the respondents who saw it. The fold rule was honoured: the encoding for a fold-*k* row
used only respondents from other folds. **But each of those reference respondents carried a
baseline prediction from a model trained on folds that included fold *k*.** A deep tree
holding design-level features can partially memorise an individual choice set, so the
baseline already embedded fold-*k* choices; subtracting it handed the target its own label
back, sign-flipped.

Three signals exposed it: adding the feature directly to the baseline made predictions
*worse*; its correlation with each row's own held-out residual was *negative* (genuine
signal correlates positively); and the least-shrunk version was the model's second most
important feature.

**First lesson: fold-honest reference *sets* are not sufficient. Any model-derived quantity
attached to a reference observation must also be honest with respect to the fold being
scored.**

**Second lesson, and the more useful one.** We rebuilt the feature on a baseline we argued
was "structurally unable to memorise a choice set" — a conditional logit with ~150 global
coefficients and no design-level features. It scored 1.13721, a plausible +0.0043, and it
*passed* both leak detectors. We then built the fully nested version (20 fits: for every
ordered pair of folds, fit excluding both) and the gain vanished entirely — 1.14151 against
1.14152 for having no such feature at all. The two versions differ only in baseline honesty,
and their paired difference is −0.00430 at **z = −4.54**.

Both detectors passed on the *clean* version, and passed **harder** than on the leaky one
(correlation +0.028 vs +0.025; direct correction +0.00099 vs +0.00078). Yet its gain was
zero.

> **A signal test can establish that a feature contains real information. It cannot
> establish that a flexible learner's gain comes from that information rather than from a
> leak riding alongside it.** Only a construction that makes the leak impossible settles
> the question.

The genuine design signal here is worth about +0.001 and is largely redundant with the
simpler share encoding already in the model. Everything above that was the model reading
back labels it should not have seen. We did not ship it.

---

## (ii) Public vs private leaderboard — to complete after the final submission

Observed calibration across four submissions:

| local nested OOF | public | offset |
|---|---|---|
| 1.17683 (first pipeline) | 1.2230 | +0.046 |
| 1.13878 | 1.201 | +0.062 |
| 1.13556 | 1.201 | +0.065 |
| **1.13044** | **1.199** | **+0.069** |

Three points for the report, all evidenced by this table:

1. **Cross-validated scores are optimistic by a stable ~0.05–0.07, and the gap grows.** Part
   is the documented population shift (test respondents are twice as wealthy); part is the
   ordinary optimism of selecting among eighteen experiments against one fixed fold
   structure. Both mechanisms are worth naming — they are different problems with different
   remedies, and only the first is visible in a shift audit.

2. **Local improvements transfer only partially, and the rate decays.** The first large jump
   transferred at ~58% (local −0.038 → public −0.022). The most recent step transferred at
   ~24–39% (local −0.005 → public −0.002). Later gains are increasingly fitted to the
   particular fold structure rather than to the population.

3. **The public leaderboard has a resolution floor.** Local 1.13878 and 1.13556 both scored
   1.201: a 0.0032 local improvement was *invisible*, since the expected public gain sat
   inside the rounding of a three-decimal display. Below roughly 0.005 local, the public
   board cannot adjudicate a change at all — a genuinely uncomfortable position for anyone
   using it to steer, and a good argument for holding an honest internal metric.

**Expected private performance.** The private leaderboard is a different random 30% of the
same test respondents — ~1,500 rows, SE ≈ ±0.02. We therefore expect private ≈ public within
noise, and we expect that noise to exceed the entire difference between our last three
submissions. We would treat a private-public gap of less than ~0.02 as uninformative about
model quality.

**One prediction we made and can check.** The latent-class model predicted a test "none" rate
of 0.223 against ~0.304 in cross-validation, because its demographic membership channel
routes wealthier respondents toward buying segments. We flagged this before submitting as
either a genuine insight or an over-extrapolation, and noted that only 79% of its gain
survived reweighting toward the test income distribution. The submission improved the public
score (1.201 → 1.199), so the extrapolation was in the right direction: **wealthier
respondents genuinely do decline less often.** This is the cleanest confirmation we have that
the segment structure reflects real behaviour rather than in-sample fitting.

---

## Limitations (for section iii)

1. **Irreducible noise is large.** Per-respondent mean loss ranges 0.59 to 2.71 (median
   1.09). Much of the variation is simply not predictable from the observed attributes and
   demographics.
2. **The private leaderboard is small.** ~1,500 rows implies SE ≈ ±0.02 — comparable to our
   entire improvement after the first day. Ranking differences of that size are not evidence
   of model quality.
3. **Two of our gains are population-specific.** Design-share encoding (77% retained) and the
   latent-class membership channel (79%) both depend on structure that may not hold for the
   wealthier test population. The latent-class model predicts a notably lower "none" rate on
   test (0.223) than the tree models do (~0.275) — an extrapolation we cannot check locally.
4. **Part-worths for rare levels are noisy.** Some attribute levels appear infrequently;
   partial pooling toward a smooth trend would be the principled refinement.
5. **Segment interpretation is model-dependent.** The three-segment split is one of many that
   fit comparably; a 2-class model scores 1.14803 and a 4-class model was not tested at the
   time of writing. The *existence* of heterogeneity is robust; the specific segment
   boundaries are not.
6. **We optimise average logloss.** No subgroup is separately calibrated, and performance on
   the wealthiest respondents — where training support is thinnest and the test set is
   concentrated — is the least reliable part of the prediction.

---

## Reproducibility notes

- xgboost ≥ 3.0 returns a **matrix** from `predict` for `multi:softprob`; the legacy
  flat-vector reshape silently scrambles predictions (produced 1.54, worse than benchmark,
  with no error raised).
- Conditional-logit identification: `Price` is 0 only on the none-option, so using 0 as the
  part-worth reference makes the price dummies collinear with the none-constant (singular
  Hessian). The reference must be the lowest level occurring on a real bundle, and a
  rank-revealing QR on the **task-demeaned** design matrix catches the rest — it found a
  second dependency (`HU_L2`) that hand-reasoning missed.
- Anything constant within a choice set is unidentified in a conditional logit; the demeaned
  rank check is the correct general test.
