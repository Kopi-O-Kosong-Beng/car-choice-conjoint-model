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

Observed calibration:

| local nested OOF | public |
|---|---|
| 1.17683 (team's first pipeline) | 1.2230 |
| 1.13878 | 1.201 |
| 1.13556 | 1.201 |
| 1.13044 | (pending) |

Two points to make in the report:

1. **Local gains transfer at roughly 58%.** Our local score improved 0.038 while the public
   score improved 0.022. The gap is partly the population shift documented above and partly
   the ordinary optimism of selecting among many experiments against one fixed CV split.
2. **The public leaderboard has a resolution floor.** Local 1.13878 and 1.13556 both scored
   1.201 — a 0.0032 local improvement was invisible, because at 58% transfer the expected
   public gain (0.0019) sits inside the rounding of a three-decimal display. Beyond a point,
   further tuning cannot be validated publicly at all. The private leaderboard is ~1,500 rows
   with SE ≈ ±0.02, larger than everything gained after the first day.

This is the honest frame for discussing public-versus-private fit: we expect them to be
similar, we expect both to sit ~0.06 above our cross-validated estimate, and we expect the
difference between our last few submissions to be indistinguishable on either board.

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
