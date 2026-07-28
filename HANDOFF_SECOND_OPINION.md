# Second-opinion brief — TAE_R-izzlers, Kaggle discrete-choice competition

Paste this whole file as your first message. Attach the files listed in **§9**.

---

## 0. What I want from you

I am 4 days from the close of a Kaggle competition and I want an **independent second
opinion**, not encouragement. Specifically:

1. **Where is the remaining accuracy?** A rival is ~0.006–0.008 ahead of us on a
   like-for-like local metric (derivation in §6). We have closed the model-combination axis
   five independent ways, so the gap is almost certainly in the *base models*. Tell me which
   base model you would build and why, derived from the structure of the problem.
2. **Attack our two load-bearing findings** (§4). If either is wrong, most of the last two
   days collapses. I would rather learn that from you than from the private leaderboard.
3. **Is our hierarchical-Bayes failure a real result or a bug?** (§7) It is the single
   most suspicious number in the project.
4. **Should we keep modelling at all**, or freeze and write the report? Answer with numbers.

Please mark each recommendation as **DERIVED** (follows from the problem structure) or
**GUESSED**. I have had enough plausible-sounding ideas; I need ones that survive arithmetic.

---

## 1. The problem

Predict which of **4 car safety-feature bundles** a respondent chooses. Metric: **mean
multiclass logloss**, lower is better. Graded coursework, so **R only** — a hard competition
rule, please do not propose Python.

**Data shape**
- Train: **1,135 respondents × 19 choice tasks** = 21,565 tasks (86,260 rows, 4 per task).
- Test: **263 entirely different respondents × 19 tasks** = 4,997 tasks.
- Each task shows 3 real bundles + **alternative 4 = the all-zero "none of these" option**,
  chosen 30.2% of the time in training. Its Price is 0, so it is always the cheapest option
  present.
- Attributes are **ordinal tiers**, 3–7 levels each; Price has 12. Coding them as
  part-worths (one dummy per level) rather than as numbers was worth **0.020** — the single
  largest win in the project, because price response turned out concave and saturating.
- It is a **designed partial-profile conjoint**: exactly 9 of 19 non-price attributes are
  shown per task, and the shown-set is common to all 3 real bundles. Only 299 distinct
  choice sets per task position, each seen by ~4.7 people, and **98.5% of test rows reuse a
  design that appears in training**.
- Respondent demographics are available for train **and test** (segment, income, age,
  region, etc.). The test population is much wealthier: two luxury segments are **9.4% of
  training but 68.8% of the graded rows**, and one segment that is 27% of training is
  **entirely absent** from test.

**Benchmark** (uniform 0.25 everywhere): 1.38629.

---

## 2. Methodology rules we hold ourselves to

These exist because violating them has produced retracted results here before.

1. **Folds are grouped by respondent**, fixed seed, never regenerated. Test respondents are
   new people; a row-wise split silently inflates every score.
2. **The decision number is the nested blend OOF**: for each of 5 folds, fit the combiner on
   the other four and score the held-out fold. Never a plain OOF, never a single fold.
3. **Judge with paired, respondent-clustered standard errors**, never by comparing headline
   numbers.
4. **One change per experiment.**
5. **Nest everything that is fitted** — weights, temperatures, encodings, and the baselines
   encodings are built on. Both of our leakage incidents were un-nested quantities riding on
   honest-looking reference sets.
6. **Noise floors are measured, and there is more than one.** Comparing two single models
   needs the model-level seed sd **0.00283**; comparing two blends needs the blend-level seed
   sd **0.00048**. Conflating them is how one accepted result survived eighteen iterations
   before being retracted.

---

## 3. Current state

| | nested OOF | public |
|---|---|---|
| live on Kaggle (2-member blend) | 1.12819 | **1.197** |
| **best candidate, built, not uploaded** | **1.12341** + calibration | forecast 1.187–1.195 |

Public leaderboard: parinwaris **1.186** | jarren_ng 1.190 | lianne 1.193 | **us 1.197** |
Zhi_Heng 1.197. Paired team-vs-team SE is **0.0047** public (~3,500 rows) and **0.0072**
private (~1,500 rows). Kaggle auto-selects your best *public* submission for private scoring,
so a worse submission costs a slot and nothing else.

**The candidate blend** is a log-opinion pool — weight each member's log-probabilities, sum,
softmax, add a small uniform floor:

| member | what it is | its own OOF | coefficient |
|---|---|---|---|
| `xgb_lw2bag` | listwise-softmax xgboost, 10 seeds averaged | 1.13682 | **+1.151** |
| `lcmnl3_both` | 3-class latent-class conditional logit + task-position terms | 1.13863 | **+0.606** |
| `xgb_long` | early xgboost variant | 1.15516 | **−0.348** |
| `xgb_wide` | early xgboost variant | 1.17456 | **−0.189** |
| `xgb_2stage` | two-stage xgboost | 1.17169 | **−0.237** |

Note the last three are **worse models entering at negative weight** — see §4A.

---

## 4. The two findings that carry the current state. Attack these first.

### A. The combiner could not represent a negative weight

For 34 iterations the pool fitted weights as `w <- exp(theta)/sum(exp(theta))` — a softmax
onto the simplex, so every weight was forced non-negative and to sum to 1. **A negative
coefficient was unrepresentable.** Members whose optimal coefficient was negative were pinned
at zero and recorded as "contributes nothing", then dropped.

Freeing the sign lets three *strictly worse* tree models enter as **control variates** for the
tree family's shared error. The mechanism: those trees are fitted on overlapping views of the
same design with the same algorithm, and the 4×4 error-correlation matrix has eigenvalues
3.726 / 0.202 / 0.056 / 0.016 — **93% of error variance in one direction**. A worse tree is a
noisy but nearly unbiased reading of that shared direction; subtracting a multiple of it
cancels the bias without touching the signal. Consistent with this, **only trees price
negative** — the part-worth logit prices at ≈0, because that direction is already spanned.

**Result: 1.12819 → 1.12341, +0.00478, respondent-clustered z +3.84, positive in all 5 folds.**

Verification performed:
- **Within-fold shuffle placebo** (permute the added members' OOF rows *inside* each fold,
  preserving fold-level structure, destroying row-level alignment): recovers **−3%**. A
  matched-marginal noise placebo agrees at −4%.
- **Member-level replication on an independent respondent partition**, with the member OOFs
  independently regenerated: +0.00308 vs +0.00298 on the production folds = **103%
  retention**; coefficients negative and stable in all 5 folds.
- **Placebo member**: a near-duplicate of the main tree prices *positive* and adds nothing,
  exactly as the control-variate story predicts.
- **Multiplicity**: 46 candidates scanned, 31 price negative, 15 improve. Bonferroni bar
  z ≥ 3.27; measured 3.84.
- **Segment reweighting** to the test population: gain *retains 167%*.

**Known residual risk we cannot test locally:** member OOF predictions come from fits on 80%
of respondents, but the shipped test predictions come from 100%-data refits. A negative
coefficient amplifies any differential drift. Measured amplification ~15%; bounded haircut
10–20% of the gain.

**Question for you:** is there a failure mode these detectors still cannot see?

### B. We measured a test-set quantity directly with a throwaway submission

We submitted a CSV where **every row is the constant (1/6, 1/6, 1/6, 1/2)**. For a constant
prediction the logloss is exact algebra:

```
score = -(r·log(1/2) + (1-r)·log(1/6)) = 1.7918 - 1.0986·r
  =>   r = (1.7918 - score) / 1.0986
```

where `r` is the fraction of scored rows on which "none" was chosen. **Kaggle returned 1.499,
so r = 0.2665** (3-dp rounding band 0.2661–0.2670).

This rejected both prior beliefs: our latent-class model predicted 0.2237, a teammate's
estimator said ~0.30, and the tree was nearly exact at 0.2726. It also killed a candidate that
would have pushed the shipped rate to 0.2064 — roughly 0.010 of damage avoided.

We then applied a **logit shift** to alternative 4 so the shipped rate moves toward the
measured truth, shrunk by weight `w`. Getting `w` right required a correction: the naive
`w* = e²/(e²+s²)` minimises MSE of the *rate*, but expected logloss `E[KL(r‖t)]` is minimised
at `t = E[r]` — the variance adds a constant and does not move the optimum. Shrinkage below 1
is justified only by prior pull on the mean (the private respondents may be different people).
We settled on **w = 0.85**, worth **+0.00219**.

**Question for you:** is w = 0.85 defensible? And what *other* scalar would you buy with a
submission slot, given ~8 slots and 4 days? Give the exact CSV construction and the inversion
algebra.

---

## 5. What is already dead — please do not re-propose these

Each was measured, not assumed.

| idea | verdict |
|---|---|
| adding a better *positive*-weight member | frontier probe over 32 artifacts, 7 families: best possible **+0.00027** |
| more combiner search | **five** independent methods (greedy, forward, backward, ridge over 41 members, class intercepts) all land at 1.1234 |
| monotone price constraint | claimed +0.00172, retracted — paired over 10 seeds it is −0.00034 |
| bundle-level encoding | structurally impossible (bijection with the design) |
| residual/design encoding off a model baseline | **100% label leakage**, proven by nested double-OOF |
| bagging the latent-class EM starts | −0.00480; the starts are genuinely multimodal and training LL picks the good one |
| ordinal smoothing of part-worths | +0.00007; the part-worths are not noise-limited |
| killing the early-stopping carve | member-level +0.00252 (z 2.45) but blend +0.00020 — the control variates already absorb it |
| temperature / calibration maps | OOF-optimal global T = 1.003; ceiling +0.00018 |
| pseudo-labelling / self-training | a new respondent has zero labels, so the Bayes-optimal prediction *is* the population mixture |
| focal loss, gated pooling, adversarial-validation weights | all null or negative |

**Also measured:** the cold-start gap is **0.877 nats** — per-respondent posterior 0.352 vs
population-averaged 1.229. Since test respondents are disjoint, that gap is structurally
unavailable. Demographics reach only 11.2% of the between-respondent variance.

---

## 6. Why we think the rival is at ~1.117 on our metric

Our own five (local → public) pairs:

```
1.17683 -> 1.233   offset +0.0562
1.13878 -> 1.201   offset +0.0622
1.13556 -> 1.201   offset +0.0654
1.13044 -> 1.199   offset +0.0686
1.12819 -> 1.197   offset +0.0688
```

The offset **grows as local improves** (slope −0.25 per unit local, R² 0.89). Inverting for
public = 1.186 gives local ≈ **1.115–1.130**, centre ~1.117. Caveat: this applies *our* offset
to *their* model, and a team whose model natively lands near r = 0.2665 would earn a smaller
offset for free — so 1.117 may be a lower bound on their local.

Since our combiner is provably at its optimum given our members, **their advantage must be in
the base models.**

---

## 7. The most suspicious number we own

Our **hierarchical Bayes** model scored **1.23703** — the worst model in the repository, worse
than a plain conditional logit at ~1.157 and barely better than the 1.386 benchmark.

HB is the textbook-correct model for this data structure: repeated choices from the same
respondent, heterogeneous tastes, a designed conjoint. A number that bad is more consistent
with a specification or convergence failure than with "HB does not work here". Our mixed logit
was also weak.

**If you think there is a better base model to be had, this is where I would look first.**
The script is attached (`experiments/iter17_hb/run.R`). Tell me what is wrong with it, or tell
me the parameterisation you would use instead — priors, number of draws, burn-in,
identification constraints, how you would handle the outside option, and how you would get the
per-respondent posterior to help when the test respondents are disjoint (or whether it
provably cannot).

---

## 8. Constraints on any answer

- **R only.** Any package is fine (`mlogit`, `bayesm`, `xgboost`, `glmnet`, `Stan` via `rstan`,
  etc.), no other language.
- **4 days** to competition close; then a report worth **15 of 30 marks** with 9 clear days.
- A single model fit currently costs 1–40 minutes; we can afford a few hours of compute per
  idea, not days.
- **~8 submission slots left**, 2 per UTC day.
- Any proposal must state **how it would be falsified** and what noise floor it must clear.

---

## 9. Files to attach

**Essential (start here):**
1. `CLAUDE.md` — the project's own rules, traps, and corrections (~2k tokens)
2. `EXPERIMENTS.md` — every hypothesis, result, and reflection including failures (~19k tokens)
3. `submissions/log.md` — every score, the offset calibration, the probe (~3k tokens)

**For the base-model question (§7):**
4. `experiments/iter17_hb/run.R` — the hierarchical Bayes failure
5. `model/02_mnl_partworth.R` — the part-worth conditional logit that works
6. `experiments/iter25_taskpos/run.R` — the latent-class model in production

**For the combiner question (§4A):**
7. `model/06_blend.R` — the combiner, including the simplex constraint that was the bug
8. `experiments/iter35_freepool5/run.R` — the free-sign experiment

**Supporting:**
9. `model/03_xgb_listwise.R` — the listwise-objective tree
10. `model/encode_design.R` — the design-share encoder
11. `model/compare.R` — how we compute paired clustered SEs
12. `report_notes.md` — findings written up so far

If attachment limits bite, files 1–4 carry most of the signal.
