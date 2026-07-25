# Report notes — running log of decisions, results, and insights

Feeds the 8-page report due 10 Aug 2026. Sections mirror the rubric:
(i) best model description ×5, (ii) public vs private fit ×2, (iii) insights & limitations ×5, quality ×3.
Full experiment history with reflections: `EXPERIMENTS.md`.

---

## (i) The model — what we built and why

**Final form: a two-member blend, pooled in log-space with a fitted temperature and a
small uniform mixture.**

| member | what it is | OOF |
|---|---|---|
| `xgb_lw` (weight ~0.62) | gradient-boosted trees on long-format data (one row per alternative), trained with a **custom listwise softmax objective** so the training loss equals the competition metric, plus design-level empirical-share features | 1.14477 |
| `mnl_pw` (weight ~0.38) | conditional logit with **part-worth coded** attribute levels and price×demographic interactions | 1.15686 |

Blend combines log-probabilities with weights, temperature and uniform-mix fraction all
fitted on out-of-fold predictions; nested evaluation gives **1.13888**.

Six further models were fitted and given zero weight (linear-coded MNL, elastic-net logit,
wide 4-class xgboost, pointwise long xgboost, mixed logit v1 and v2). The optimizer keeps
exactly one member from each family — trees and linear utility — and discards the rest as
redundant, which is itself worth reporting: the two families make genuinely different errors.

### Validation design (the part we'd defend hardest)

Every model uses one fixed 5-fold split **grouped by respondent** (`Case`). This matters
because the test set is 263 respondents who appear nowhere in training. Splitting by row
instead would let a model learn "respondent #47 is stingy" from 15 of their tasks and be
graded on the other 4 — inflating local scores while teaching nothing that transfers.
The blend is scored **nested** (weights refit five times, each excluding the fold it is
evaluated on), so no number we act on has ever seen its own tuning data.

Model comparisons use a **paired test with respondent-clustered standard errors**
(`model/compare.R`). Fold-to-fold SD is ±0.013, so a genuine +0.005 improvement is
invisible in headline numbers; paired on identical rows it is z = 4.3. Three improvements
were accepted this way and one was rejected.

---

## (iii) Insights from the model

### 1. Price sensitivity is concave — people perceive ratios, not differences

Part-worth utilities for the 12 price levels (relative to level 1):

| level | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| utility | −0.52 | −0.76 | −0.90 | −1.08 | −1.11 | −1.22 | −1.38 | −1.47 | −1.71 | −1.93 | −1.92 |

Each successive price step costs *less* utility than the one before, and levels 11–12 are
statistically indistinguishable — sensitivity saturates entirely at the top. This is a
Weber–Fechner (log) response. Coding price linearly, as we and most teams did initially,
fits a straight line through a clearly curved response; freeing the shape was worth
**+0.020 logloss (z = 11.8)**, the single largest gain of the project.

### 1b. Older respondents reject *expensive* bundles, not price in general

The plain MNL gives a single Price × age interaction of −0.31 (z = −18), which reads as
"older respondents are more price-sensitive". A regularized model carrying price
part-worths *interacted* with demographics shows that reading is too coarse. The age
interaction is concentrated entirely at the top of the price range:

| interaction | coefficient |
|---|---|
| Price_L12 × age | **−0.398** |
| Price_L11 × age | −0.326 |
| Price_L7 × age | −0.280 |
| Price_L8–L10 × age | −0.24 to −0.27 |
| Price_L1–L6 × age | (nothing comparable) |

So it is a **threshold effect, not a slope**: at low and middling prices older respondents
behave much like everyone else, then drop away sharply once bundles reach the expensive
tiers. A linear Price × age term averages this into one number and conceals it.

**Commercial reading:** premium bundles lose older buyers disproportionately, so the
relevant question is not "are older customers price-sensitive" but "where is the cliff".

### 2. Taste heterogeneity is large — but not usable at prediction time

Mixed logit with a log-normal price coefficient estimates σ = 1.31, meaning price
sensitivity differs roughly **six-fold between the 25th and 75th percentile respondent**.
The none-option constant has mean −1.34 with sd **2.47**: some respondents will not buy
any bundle at almost any price, while others are easy sells.

And yet the mixed logit *lost* to the fixed-coefficient part-worth model (1.173 vs 1.157)
and earned zero blend weight. The reason is structural and worth stating plainly: with
every test respondent unseen, we can only ever predict the population-*averaged*
probability. Integrating over the taste distribution buys calibration but costs sharpness,
and the fixed model already captures the average response well. **Heterogeneity is real
but not conditionable** — a limitation of the prediction task, not of the estimator.

### 3. The choice sets are a designed experiment, and that leaks information

Only 299 distinct choice-set designs exist per task position, each shown to ~4.7
respondents; **98.5% of test rows reuse a design that appears in training**. The empirical
choice share within a design is a nonparametric estimate of the population choice
probability, available for almost every test row. Shrunk appropriately it is worth
**+0.005 (z = 2.9)**.

The size of that gain is itself informative. Raw shares from ~3 observations are so noisy
that at weak shrinkage (α = 1) they score *worse than the 25%-everything benchmark*
(1.433 vs 1.386), while at α = 5 they score 1.307. The signal is real but thin; most of it
must be shrunk away to be usable.

### 4. The outside option dominates

The all-zero "none" bundle is the most-chosen alternative (30.2% vs 22.0/25.0/22.7%).
Any model that treats this as just a fourth product misses that opting out is the modal
behaviour and is driven by different considerations (price level, total bundle richness)
than choosing between bundles.

### 5. Train and test populations differ materially

Test respondents are roughly twice as wealthy (median income $60k → $80k, p75 $85k →
$125k, means $75k vs $151k), better educated, and drive more. This is a genuine
distribution shift, not sampling noise, and it explains part of the persistent gap between
local scores and leaderboard scores that our team observed from the first submission.
We responded by adding explicit price×income terms (linear-in-income models extrapolate;
trees cannot) and by tracking an income-reweighted OOF as a secondary diagnostic
(1.14297 vs 1.13888 — the shift costs roughly 0.004).

---

## Limitations (for section iii)

1. **Irreducible noise is large.** Per-respondent mean loss ranges from 0.59 to 2.71
   (median 1.09). A substantial share of choices is simply not predictable from the
   observed attributes and demographics — no model will recover it.
2. **The private leaderboard is small.** ~1,500 rows from ~80 respondents implies a
   standard error near ±0.02 on the reported score, comparable to the *entire* improvement
   we achieved tonight. Public-private divergence should be expected and is not evidence
   of overfitting per se.
3. **Design-share features depend on design reuse.** They would contribute nothing on a
   genuinely new experimental design — this component is specific to this data collection,
   not a transferable modelling insight.
4. **Part-worths for rare levels are noisy.** Some attribute levels appear infrequently;
   their estimated utilities carry wide intervals. Partial pooling toward a smooth trend
   would be the principled refinement.
5. **We optimize average logloss.** The model is not calibrated to any particular
   subgroup, and performance on the wealthiest respondents — where training support is
   thinnest and the test set is concentrated — is the least reliable part of the prediction.

---

## (ii) Public vs private fit — to complete after submissions

Observed calibration points (local → public): 1.17683 → 1.2230 (+0.046); a rival team's
1.161 → 1.210 (+0.049). Our nested 1.13888 therefore projects to ≈1.19 public.
Record each actual result in `submissions/log.md` and discuss the pattern here.

---

## Bug log / pitfalls (ours, worth a sentence in the report on reproducibility)

- xgboost ≥3.0 returns a **matrix** from `predict` for `multi:softprob`; the legacy
  flat-vector reshape silently scrambles predictions (produced OOF 1.54, worse than
  benchmark, with no error raised).
- Conditional-logit identification: `Price` is 0 *only* on the none-option, so taking 0 as
  the part-worth reference makes the price dummies collinear with the none-constant
  (singular Hessian). Reference must be the lowest level occurring on a real bundle. A
  rank-revealing QR on the **task-demeaned** design matrix caught a second dependency
  (`HU_L2`) that hand-reasoning missed.
- Anything constant within a choice set is unidentified in a conditional logit — the
  demeaned rank check is the right general test.
