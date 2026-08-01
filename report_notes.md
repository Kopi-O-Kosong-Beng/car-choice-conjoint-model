# Report notes — material for the 8-page report (due 10 Aug 2026)

Rubric: (i) best model description **5** · (ii) public vs private fit **2** ·
(iii) insights & limitations **5** · (iv) overall quality **3** — **15 of 30 marks total.**
Full experiment history with reflections: `EXPERIMENTS.md`. Scores: `submissions/log.md`.

> # ✅ THE COMPETITION IS OVER — THE GRADED FILE IS KNOWN
>
> Kaggle closed 1 Aug 2026. **Selected submission: `submissions/cand_pool5050_final00.csv`.**
>
> | | logloss | rank |
> |---|---|---|
> | **Public** (~70% of test rows) | **1.185** | **3rd** |
> | **Private** (~30%, the graded set) | **1.185** | **4th** |
>
> **Section (ii) of the rubric is now answerable with a measurement rather than an argument,
> and it is the strongest single fact this project owns:** the public→private drift was
> **zero to three decimal places**, against a private-draw sd of ~0.011. Write (ii) around
> that. The full post-mortem — including why the rank moved 3rd→4th on an unchanged score —
> is at the end of `submissions/log.md`.

> ## ⚠️ CORRECTIONS REQUIRED BEFORE DRAFTING (this file predates iteration 48)
>
> Everything below was written in the 1.199/1.201 era. Five things changed and **must not be
> copied into the report as written**:
>
> 1. **Section (i) describes the wrong model.** The graded submission is
>    `cand_pool5050_final00.csv` at **1.185** — a log-opinion pool at fixed `w = 0.5` of the
>    main track's `sub_20260730_final00.csv` (1.193) and the second track's
>    `sub_20260729_nnblend.csv` (1.194) anchored to `r*`. It superseded both. Rewrite (i)
>    around it; see "The graded model, described" immediately below. The three-ideas list
>    (part-worths, listwise objective, latent class) survives and still describes the main
>    track's members.
> 2. **Insight 3 (design-share encoding, "+0.005, z 2.9") is RETRACTED** — iteration 48
>    proved it a structural CV leak; honest value −0.00596 (z −3.53), and the honest
>    design-level signal caps at ~+0.001. Do not present it as a gain. Present it as the
>    project's central methodological finding instead (see EXPERIMENTS.md iterations 48 and
>    54–59): a *structural* leak is uniform across folds, scales with capacity, and passed
>    five independent gates including folds_b replication and segment reweighting.
> 3. **The shift-audit row "design-share encoding 77%" describes a leak decaying, not a gain
>    decaying.** Reframe or cut.
> 4. **Section (ii)'s calibration table is 4 points; there are now 8**, including two public
>    *sign flips* (1.12341 → 1.209, 1.11210 → 1.205), the probe measurement (r* = 0.26648,
>    exact algebra on a returned score), and the anchor prediction that landed (predicted
>    0.00879, board returned ~0.010). See `submissions/log.md` 28–30 Jul. Also fix
>    Limitation 2: ±0.02 is *absolute* noise; ranking noise is paired, ~0.006–0.012
>    (STRATEGY_REVIEW Part II.1).
> 5. **The monotone-constraint numbers here are the interim 6-seed run.** The final
>    10-seed numbers (iteration 26, `members.txt`): −0.00034, 95% CI [−0.00159, +0.00092],
>    wins 5 of 10. Use those.
>
> The consolidated evidence table at the **end of this file** is the verified backbone for
> "what failed and why that is informative".

---

## The graded model, described — drafting material for rubric item (i)

**`cand_pool5050_final00.csv` = a log-opinion pool, at fixed `w = 0.5`, of two independently
built modelling tracks, both first anchored to a leaderboard-measured constant.**

### Parent A — the main track (public 1.193)

`model/run_all.R`, then `experiments/iter82_provenance/build_candidates.R`:

1. **A two-member nested blend** — `xgb_lw2bag` (listwise-softmax gradient boosting, seed-bagged)
   at weight 0.528 and `lcmnl3_both` (a 3-class latent-class multinomial logit) at 0.472.
   Weights fitted *inside* each CV fold, never on the full OOF.
2. **A nested 6-coefficient residual logit** on (relative price, outside constant,
   total-vs-best) × (global, luxury deviation), ridge penalty 0.03, uniform observation
   weights. The single survivor of ~160 arms tested on 30 Jul: +0.00300 segment-reweighted on
   production folds (z +0.77) and +0.00628 on the independent `folds_b` (z +1.60) — same sign
   on two independent fold structures, combined ≈1.7σ. Never cleared the project's own z ≥ 2
   bar; shipped as the best-supported remaining gain, not a proven one.
3. **The probe anchor** — see below.

### Parent B — the second track (public 1.194), anchored

`experiments/iter62_nnblend/`, the team's second modelling track, built by a teammate with no
shared code with `model/`: a listwise tree arm (xgboost, depth 3, 20 seeds × 5 folds) blended
geometrically at `w = 0.25` with a listwise MLP arm on the same design matrix, then a
calibration tower (per-segment Ch4 shift, luxury temperature, residual logit) and a low-df
binomial GAM supplying an independent outside-option probability mixed into the margin at 25%.

Iteration 82 found it had **never been anchored to `r*`** — it shipped at mean p4 = 0.21086
against a measured 0.26648. Correcting that was worth ~0.0097 and is where most of the
1.193 → 1.185 gain came from.

### The one measured constant, and why it is not fitting

A constant submission of `(1/6, 1/6, 1/6, 1/2)` (`submissions/probe_alt4.csv`) returned 1.499.
For a constant prediction the logloss is algebraically determined by the true none-rate, so
one returned score inverts exactly:

> `r* = (log 6 − 1.499) / log 3 = 0.266481153`

This is **measured on the test set, not fitted on our folds**, which is why it transferred at
~100% where local gains transferred at ~⅓ or worse. Both parents were tilted to mean p4 = `r*`
by a single global log-odds multiplier — zero fitted parameters, within-choice conditionals
preserved to machine precision — and the pool re-anchored after pooling.

### Why pool, and why `w` was never tuned

`w = 0.5` is **fixed, not fitted, and could not honestly have been fitted**: the second track
has no OOF on `folds.rds` and can never have one — its fold constructor differs, with adjusted
Rand index ≈ 0.002 against `folds.rds`, i.e. statistically independent partitions. With no
honest objective to tune on, 0.5 is the only choice that spends no selection budget.

The justification is a bound, not a score. From
`loss(pool_w) = (1−w)L_A + w·L_B + E[log Z_w]` with `Z_w = Σ_k A_k^(1−w) B_k^w`, Hölder's
inequality gives **`loss(pool) ≤ max(L_A, L_B)` on every row set**, including the private
1,499. When exactly one submission counts and public rank is nearly uninformative about
private rank, a bounded worst case was worth more than a marginally better point forecast.

**And pooling only pays if the parents differ.** `experiments/iter82_provenance/track_distances.R`
measures the conditional (calibration-removed) distance between the two tracks at **0.01291**,
against a same-model-reseeded floor of **0.00552** and a widest-axis-we-own of **0.04837**.
2.3× above the floor — two genuinely distinct models, not one model twice. The pool lands
nearly equidistant from both parents (0.00310 / 0.00336, ratio 0.92), and the tracks disagree
on 96.6% of rows while agreeing to ~0.005 on average: no shared bias to inherit, plenty of
independent error to cancel.

### What the result confirms

| forecast | pre-registered | returned |
|---|---|---|
| `r*` from the alt-4 probe | 0.266481153, by algebra | anchor gained as predicted |
| `final00` | 1.1930 | 1.193 |
| `cand_pool5050_final00` | 1.186, band 1.184–1.189 | **1.185** |

All three were committed to git **before** upload. Public 1.185 / private 1.185.

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
observed OOF. The significance test answered a question about two artifacts; the decision
needed a question about two *procedures*.

**And when we asked the procedure question properly, the answer deflated.** We reran the
whole nested blend ten times, once per seed, to get the quantity that actually governs the
submission:

| | |
|---|---|
| E[nested blend \| member is one random seed] | 1.12884 (sd 0.00048) |
| nested blend with the bagged member | **1.12854** |
| **expected gain from bagging, at the blend level** | **+0.00029** |

The ~0.006 that bagging is worth to the *single model* is worth only ~0.0003 to the
*blend* — because **a blend is itself a variance-reduction device**, and its other members
already absorb most of the seed noise. Bagging and blending are substitutes, not
complements, and we had unknowingly been buying the benefit already.

This is the most useful thing the exercise produced, and it cuts both ways. It says the
decision number is far more stable than the single-model numbers feeding it: the blend moves
only 0.00137 across seeds that move the individual model by 0.00896. That in turn is what
licenses us to believe a 0.00177 blend-level improvement elsewhere in the project — it is
roughly 3.7 seed-sd at the blend level, even though it would be well inside noise at the
model level. **The precision of an ensemble's score is not the precision of its parts**, and
which one you need depends on which decision you are making.

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

---

# Consolidated evidence table (30 Jul review — verified against artifacts)

Everything this project actually knows, with the number and where it came from. This is the
backbone for sections (ii) and (iii). ✅ = re-verified from artifacts on 30 Jul.

## What worked (all pre-registered, all replicated or board-confirmed)

| finding | number | provenance |
|---|---|---|
| Part-worth coding of attribute levels | +0.020, z 11.8; 101% shift retention | iter02 |
| Listwise softmax objective (train the metric) | +0.006, z 4.3; 112% retention | iter03 |
| Latent-class heterogeneity, demographic membership | 1.15686 → 1.14396, z 3.77; demographics = 93% of the gain | iter11 |
| Task-position terms in the LC utility (price-tilt mechanism) | +0.00534 z 6.26; folds_b +0.00422 z 4.70; 119% retention; tilt alone reproduces 105% of the none-rate drift while pure scale drift is ~0 (Swait–Louviere identification) | iter25 |
| Seed bagging | model-level −0.006; **blend-level only +0.00029** — bagging and blending are substitutes | iter26 |
| **The probe**: constant `(1/6,1/6,1/6,1/2)` inverted to the graded none-rate | r* = 0.266481153 exactly; killed a candidate that would have cost ~0.010; anchor confirmed on the board (predicted 0.00879 recoverable, returned ~0.010) ✅ | 27–30 Jul, `submissions/log.md` |
| Residual-logit correction (6 coef, nested) | +0.00300 seg-rw (z 0.77) and +0.00628 on folds_b (z 1.60); combined ≈1.7σ — sole survivor of ~160 arms ✅ | iter67/68 |

## The measured instruments (quote these, not folklore)

| instrument | number | provenance |
|---|---|---|
| single-model seed sd / blend seed sd / fold SD | 0.00283 / 0.00048 / 0.013 | iter26 |
| replication on an independent fold structure | member 79%, blend 81% | iter21 |
| segment-reweighted OOF | ESS **208 of 1,135 respondents**; tracked the board to 0.001 when leak-free; missed by 0.019 on the leaky rotation ✅ | predict_lb.R, iter48 |
| public transfer of local gains | 58% (early) → ~⅓ → **negative** during the leak era | `submissions/log.md` |

## What failed, and why that is informative (the insight marks live here)

| failure | number | what it teaches | provenance |
|---|---|---|---|
| Hierarchical Bayes | 1.23703; own-β 0.352 vs population-averaged 1.229 | the 0.877 gap is the cold-start problem stated numerically — individual-level learning is structurally unavailable for 263 strangers | iter17 |
| Mixed logit (continuous heterogeneity) | 1.17281, weight 0 | heterogeneity is only usable in discrete, demographic-routable form | iter05 |
| Two-stage none-vs-buy — **refuted twice, independently** | 1.17169 (z −11) and hurdle+rank 1.17478 ✅ | the decomposition is wrong, not the stage-2 model class: opting out shares the bundles' utility scale | iter09; 30 Jul |
| Nested logit on {bundles} vs {none} | 1.15681 ≈ MNL | IIA is not the binding constraint; heterogeneity is | iter10 |
| Bundle-level encoding | max diff 0.000e+00 | killed by counting (bijection) before any fit — structure-first triage | iter16 |
| Residual design encoding | 1.09962 fake; nested double-OOF gain −0.00430, z −4.54 | fold-honest reference *sets* are insufficient; every derived quantity must be honest w.r.t. the scored fold | iter12/15 |
| **Design-share encoding** — in production 47 iterations | apparent +0.0218; honest **−0.00596** (z −3.53); leak scales d4 −0.0004 → d10 +0.078 | a *structural* leak is uniform across folds and monotone in capacity; per-fold consistency proves nothing | iter48 |
| Random-rotation member — passed **five** pre-registered gates, submitted | local −0.02192 (10/10 seeds); public **1.205**; survival without the encoding **−67%** | every gate was computed on (reweightings of) contaminated OOF; a contrast is only valid if leak exposure is matched | iter54–59 |
| Free-sign blend — passed six checks, submitted | local −0.00478; public **1.209** | N tests sharing one assumption are one test; OOF fits and shipped refits are different regimes | iter35/43 |
| Fitting anything on the reweighted objective — refuted **three times** | worse on its own metric each time | reweighting collapses ESS (208 respondents); variance beats bias | iter07, 46, 68 |
| Ensemble-randomness retune, both directions | 16-point surface unimodal at the incumbent 0.8/0.8 | derived hypotheses are still hypotheses | iter45 |
| Demographic PCA features | +0.0123 to +0.0275 (worse) | dense rotations of coarse ordinals let trees fingerprint respondents | iter49 |
| "Refit w_tree honestly" (the former standing item) | flat: z ≤ +0.97 at every w in [0.10, 0.528] ✅ | down-weighting a leaky member ≠ replacing it with an honest one | iter67 |
| The ceiling | oracle 0.986 vs blend 1.130; ~89% of none-propensity unobservable; design-only predictor 1.307 | the remaining loss is measured, and it is not reachable | iter17/18 |

**The one-line synthesis for the report:** the project's accuracy came from three structural
choices made in week one; everything afterwards that *looked* like a gain and wasn't was
caught by exactly one habit — building the honest twin of every construction and comparing —
and the two times that habit was skipped, the leaderboard caught it instead (1.209, 1.205).
