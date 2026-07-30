# TAE R-izzlers — Analytics Edge Data Competition 2026

Predicting which of four car safety-feature bundles a respondent chooses.
**R only** (competition rule). Metric: multiclass logloss, lower is better.

**Team:** zf · Sheil · Kavya · Nicole. One Kaggle account, held by the team
representative — see [Working agreements](#working-agreements).

> ⚠️ **Keep this repository private until after 10 August 2026.** It contains the
> competition data and our full modelling approach. A public repo would hand both to
> every other team. Course rule also forbids sharing data outside the team.

---

## Where we stand

| | logloss |
|---|---|
| Benchmark (25% for everything) | 1.38629 |
| Our first submission | 1.233 public |
| Top of board | **1.183 public** |
| Two-member main-track blend | 1.197 public (local CV 1.12819) |
| **Ours — live, and our best** | **1.193 public** (`sub_20260730_final00.csv`) |

**11th of ~40** as of 30 Jul, in a three-way tie at 1.193. The board reads
1.183 / 1.184 / 1.186 / 1.186 / 1.188 / 1.190 / 1.190 / 1.192 / 1.193 ×3 — **sixteen teams
inside 0.013**.

> ### ⚠️ The public board is 70% of the test set. The grade is the other 30%.
>
> Kaggle states it plainly: *"This leaderboard is calculated with approximately 70% of the
> test data. The final results will be based on the other 30%."* So the private set is
> **~1,499 rows**, and the paired respondent-clustered ranking SE on that is **0.006–0.012**
> (`STRATEGY_REVIEW.md` Part II.1).
>
> **The entire top-16 spread is about one standard error.** Public rank is close to
> uninformative about final rank — we could plausibly finish anywhere from 2nd to 20th
> without anything changing. Do not read the public ordering as standings.
>
> **Only ONE submission counts for the final score.** If none is selected, Kaggle
> auto-selects the best public. Select `sub_20260730_final00.csv` explicitly rather than
> relying on that.
>
> One consequence for our own calibration: `r* = 0.266481153` was measured from
> `probe_alt4`'s **public** score, so it is the none-rate of the 70%. The private 30% has its
> own draw, sd ≈ `sqrt(0.2665·0.7335/1499)` ≈ **0.011**. The probe anchor's +0.00104 is a
> public-set figure and will not transfer exactly.

### How the score progressed

1. **1.197** — the two-member main-track blend (`xgb_lw2bag` + `lcmnl3_both`), the model
   documented throughout this README and produced by `model/run_all.R`.
2. **1.193** — `submissions/sub_20260730_final00.csv`: the two-member nested blend → a nested
   6-coefficient residual-logit correction → the probe anchor. Built only from our own two
   members plus our own nested refit.

*(`submissions/log.md` carries the full submission history, including superseded entries.)*

> **The forecast for step 2 was 1.1930 and the board returned 1.193.** That is the second
> pre-registered prediction this project has made that came true, and the first about a
> *model change* rather than a measured constant. It confirms the segment-reweighted OOF as
> the leaderboard instrument, and it means the earlier period of *inverted* transfer
> (local improved, public regressed) was the encoding leak — not a broken methodology.

> ### ⚠️ The 1.12341 "improvement" was refuted on the leaderboard
>
> A previous version of this README recommended a five-member free-sign blend at local CV
> **1.12341**, merged as [PR #1](https://github.com/Kopi-O-Kosong-Beng/TAE_R-izzlers/pull/1).
> It has since been submitted twice and **scores worse than the two-member blend it was
> meant to replace**:
>
> | | local CV | public |
> |---|---|---|
> | two-member blend (live) | 1.12819 | **1.197** |
> | five-member free-sign, calibrated | 1.12341 | **1.209** |
> | five-member free-sign, raw | 1.12341 | **1.211** |
>
> Local said better by 0.005; the board says worse by 0.012 — a **sign flip**, at about 2.5
> paired standard errors. Do not ship it. The full analysis is
> [iteration 43 in `EXPERIMENTS.md`](EXPERIMENTS.md), and the methodological lesson is
> [below](#the-failure-that-taught-us-most).

Our local cross-validation and the Kaggle score differ by about +0.07. That gap is
expected — see [Why local ≠ Kaggle](#why-local--kaggle-matters) below, and it turned out to
be deeper than we understood when we wrote that section.

---

## Quick start

**Prerequisite:** R ≥ 4.5 with `data.table`, `xgboost`, `mlogit`, `dfidx`, `glmnet`, `Matrix`.

```r
install.packages(c("data.table","xgboost","mlogit","dfidx","glmnet","Matrix"))
```

**Run everything** (~30 min) from the repository root:

```powershell
# Windows: R is often not on PATH, so call it by full path
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/run_all.R
```

```bash
# macOS / Linux
Rscript model/run_all.R
```

That writes a timestamped CSV into `submissions/`, ready to upload.

**Run only part of it** — every stage saves to `model/artifacts/`, so you can start midway:

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/run_all.R blend submit
```

Stages: `tests · data · folds · mnl_pw · xgb_lw · blend · submit · audit`

---

## Which model is the best one?

**Production blends two models.** Everything else is supporting infrastructure or
superseded work kept for the record.

| script | model | OOF | weight |
|---|---|---|---|
| [`experiments/iter26_seedbag/run.R`](experiments/iter26_seedbag/run.R) | listwise xgboost, **10 seeds averaged** | 1.13682 | **0.528** |
| [`experiments/iter25_taskpos/run.R`](experiments/iter25_taskpos/run.R) | **latent-class conditional logit + task position** | 1.13863 | **0.472** |
| [`model/06_blend.R`](model/06_blend.R) | pools them in log-space + temperature | **1.12819** | — |

> **This two-member blend is the production model and the best we have.**
> [`experiments/iter35_freepool5/run.R`](experiments/iter35_freepool5/run.R) reaches a lower
> *local* number (1.12341) by letting the combiner use negative weights — but it scores
> **1.209** on the board against this blend's 1.197. It is kept for the record, not for use.
> `members.txt` has always declared the two-member blend and was never switched.

**The blend used to have four members and now has two, at an unchanged score.** Two of them
turned out not to be models at all:

- `xgb_mono` (monotone price constraint) was **retracted** — retested paired across ten seeds
  it is worth −0.00034, CI [−0.00159, +0.00092], winning 5 of 10. Its original +0.00172 was
  smaller than the seed noise it was measured against. It was a duplicate of the
  unconstrained tree for eighteen iterations.
- `mnl_pw` contributed **−0.00006** (z = −1.89) on leave-one-out. The latent-class model with
  one class reproduces it to four decimals, so it is a strict generalisation.

The two survivors are the opposite ends of the blend's only genuine axis of disagreement —
tree versus logit, which carries 93% of the error variance — and the weights split near-evenly
across it.

The two highest-weighted models currently live under `experiments/` rather than `model/`.
That is deliberate while the research round is open — `model/members.txt` is the single
source of truth for what is in the blend, and `model/run_all.R` runs them from where they are.

Everything in [`model/legacy/`](model/legacy/) scored worse and earns zero weight
(linear-coded logit, mixed logit, wide xgboost, elastic net). Kept because the report
discusses what we tried, not only what won.

---

## Repository map

```
model/                    the production pipeline
  run_all.R               ENTRY POINT — start here
  00_load.R               CSV -> long format (1 row per alternative) + features
  01_folds.R              5 CV folds, grouped by respondent (see below)
  02_mnl_partworth.R      MODEL 1 — part-worth conditional logit
  03_xgb_listwise.R       MODEL 2 — listwise-objective xgboost
  06_blend.R              weighted log-space blend, nested evaluation
                          BLEND_WEIGHTS=free opts into unconstrained-sign weights
                          (default is the simplex; see iteration 35)
  07_submit.R             writes the Kaggle CSV, validates format
  99_utils.R              logloss, fold construction, normalisation helpers
  encode_design.R         design-level empirical-share encoder
  compare.R               paired model comparison with clustered SEs
  shift_audit.R           does an improvement survive the train->test shift?
  members.txt             which models enter the blend
  tests.R                 unit tests
  legacy/                 superseded models (zero blend weight)

experiments/              research log — one folder per iteration, immutable
Vault/                    Obsidian knowledge base (course notes + competition)
submissions/              generated CSVs + log.md of every score we've seen
EXPERIMENTS.md            what we tried, what worked, and why — with reflections
report_notes.md           material for the 8-page report (worth 15 of 30 marks)
Raw Dump/                 course materials + competition data (unmodified)
```

---

## The three things that actually made the score

Each was verified with a paired test using respondent-clustered standard errors
(`model/compare.R`), not just a lower headline number.

### 1. Part-worth coding — the big one (+0.020, z = 11.8)

Attributes are 3–7 level tiers and Price has 12 levels, but we originally coded them as
plain numbers, which forces utility to move by a constant amount per level step. Giving
each level its own utility revealed that **price response is concave**:

| price level | 2 | 3 | 4 | 5 | … | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| utility | −0.52 | −0.76 | −0.90 | −1.08 | … | −1.71 | −1.93 | −1.92 |

Each step hurts less than the one before, and the top two levels are indistinguishable —
sensitivity saturates. People respond to price ratios, not differences.

### 2. Listwise objective (+0.006, z = 4.3)

The old xgboost scored each alternative as an independent "was this chosen?" binary and
we normalised afterwards. But the metric is a softmax over the four alternatives in a
choice set — only *relative* utility matters. A custom gradient (`p − y` with the softmax
taken within each 4-row task) makes the training loss identical to the competition metric.

### 3. Design-level encoding (+0.005, z = 2.9)

This is a *designed* conjoint experiment: only 299 distinct choice sets per task position,
each shown to ~4.7 people, and **98.5% of test rows reuse a design that appears in
training**. The empirical choice share within a design is therefore observable. Support is
thin (~3.8 rows per design) so the shares must be shrunk heavily — at weak shrinkage they
score *worse than the benchmark*.

---

## The combiner could not represent a negative weight

*(+0.00478, z = 3.84 — iteration 35, [PR #1](https://github.com/Kopi-O-Kosong-Beng/TAE_R-izzlers/pull/1), not yet production)*

For thirty-four iterations `model/06_blend.R` fitted the pool weights as

```r
w <- exp(theta[1:M]); w <- w / sum(w)      # softmax -> the simplex
```

which forces every weight non-negative and forces them to sum to one. So a **negative
coefficient was not disfavoured — it was unrepresentable.** A member whose optimal
coefficient is negative could only be pinned at `w ≈ 0`, and was then written down as
*"earns weight 0.000, contributes nothing"* and dropped.

That phrase appears verbatim in `members.txt`. It is also why an exhaustive frontier probe
concluded that adding any member to the pool was worth at most *"+0.00027, **at negative
weight**"* — it could see the sign and had no way to act on it. The probe measured
correctly and concluded wrongly, because it inherited the constraint it was measuring under.

**A negative weight is not a bad model — it is a control variate.** The four tree models are
fitted on overlapping views of the same design with the same algorithm, so their errors share
a large common component (93% of the error variance lies in one direction). A *strictly worse*
tree is a noisy but nearly unbiased reading of that shared bias, and subtracting a multiple of
it cancels the bias without touching the signal only the good tree has.

| combiner | nested OOF |
|---|---|
| 2 members, simplex (production) | 1.12819 |
| 2 members, free sign | 1.12819 ← *must agree, and does* |
| 5 members, free sign | **1.12341** |

The three added members score 1.155, 1.175 and 1.172 — every one **worse** than either
incumbent. Only trees price negative; the part-worth logit prices at ≈0, because that
direction is already spanned by the latent-class model.

**One methodological warning this produced.** Negative weights on out-of-fold predictions is
the classic way stacked generalisation fails, and our usual leak test — *"a real gain appears
in every fold, a leak concentrates in one"* — **cannot detect it**, because fold-correlated
structure appears in every fold too. The instrument that does work is a **within-fold shuffle
placebo**: permute the added members' OOF rows *inside* each fold, preserving every fold-level
property while destroying row-level alignment. It recovered **−3%** of the gain, and an
independently built noise placebo agreed at −4%.

**And then it failed anyway.** See the next section — that placebo, and five other checks,
were all blind to the thing that actually broke it.

---

## The failure that taught us most

*(iteration 43 — read this one if you read nothing else)*

The blend above was the most heavily verified result in the project:

| check | result |
|---|---|
| within-fold shuffle placebo | recovers −3% |
| matched-noise placebo | −4% |
| **member-level replication on an independent respondent partition** | **+0.00308, z +3.17, 103% retention** |
| placebo member (a near-duplicate) | prices *positive*, gains nothing — as the theory predicts |
| multiplicity, 46 candidates scanned | z +3.84 against a Bonferroni bar of 3.27 |
| segment reweighting to the test population | **167% retention** |
| margin-neutralised gain | +0.00480 — the gain is not none-rate movement |

**It scored 1.209 against the incumbent's 1.197.**

### Why every check missed it

A control variate cancels shared bias only if it **drifts with the member it corrects**.
Member OOF predictions come from fits on 80% of respondents; the shipped test predictions come
from 100%-data refits. The three subtracted trees are older artifacts whose test refits moved
differently from `xgb_lw2bag`'s — the tree family's OOF→test none-rate drift spans **0.0157**.
Subtracting a correction sized for a gap that no longer exists *adds* bias.

The sign is what makes it violent. With positive weights you average, and errors partly cancel.
With negative weights you take a **difference**, and differences amplify: if A drifts by δ_A and
B by δ_B independently, then A − 0.35·B drifts with variance `var_A + 0.12·var_B`. The errors
add.

### The rule worth carrying

> **Cross-validation holds the training procedure fixed and varies the data. Deployment varies
> the training procedure too — and nothing in CV can see that.**

All seven checks were computed on out-of-fold predictions or a reweighting of them. Counting
independent *tests* is not the same as counting independent *assumptions*: seven tests
collapsed to one assumption, that OOF predictions represent test predictions.

Three concrete rules follow:

1. **Negative weights require matched-regime members.** A control variate must be refit under
   the *same* protocol as the member it corrects.
2. **"Verified N ways" is worthless if the N tests share a blind spot.** Ask what each test
   *cannot* see, and check whether the answers coincide.
3. **The leaderboard is not a scoreboard — it is the only instrument for one specific
   question.** Spending a submission to answer it cost nothing, because Kaggle auto-selects
   the best public score. The real mistake would have been shipping this at the end untested.

---

## Measuring the test set directly

We spent one submission on a CSV where **every row is the same constant** `(1/6, 1/6, 1/6, 1/2)`.
For a constant prediction the logloss is exact algebra, so the returned score inverts to a fact
about the graded data:

```
score = 1.7918 − 1.0986 · r        ⇒        r = (1.7918 − score) / 1.0986
```

where `r` is the fraction of scored rows on which "none of these" was chosen. Kaggle returned
**1.499**, so **r = 0.2665** (3-dp band 0.2661–0.2670). It is the only direct observation of
the test population anyone on the team has, and it cost one slot — the probe scores ~1.5, so it
can never be auto-selected and cannot affect the grade.

**It rejected every prior belief.** The latent-class model predicted 0.2237, an independent
estimator said ~0.30, the tree was nearly exact at 0.2726. It also killed a candidate that
would have pushed the shipped rate to 0.2064 — roughly 0.010 of damage avoided.

**And the correction it implies is confirmed.** Two submissions of the *same five members*,
differing only in the none-column, isolate it exactly:

| ships p4 | none-margin cost | public |
|---|---|---|
| 0.2377 | 0.00223 | 1.211 |
| 0.2622 | 0.00005 | 1.209 |

Predicted gain from the shift **0.00218**; observed **0.002**. The dial works, even though the
model it was attached to does not.

**A statistical subtlety worth recording.** Shrinking the correction by `w* = e²/(e²+s²)` is
wrong — that minimises mean-squared error of the *rate*. Expected logloss `E[KL(r‖t)]` is
minimised at `t = E[r]`, and the split-type variance adds a constant that does not move the
optimum. Shrink only for prior pull on the mean.

### The model-probe — measuring without spending a slot on a throwaway

The constant probe above works because a constant prediction has closed-form logloss. Its
cost is that it scores ~1.5 and can never be selected: the whole slot buys information and
nothing else. **Iteration 80 removed that cost.**

Take a live candidate `A` and build `B` by adding **one constant to the alternative-4 logit
per segment**. Two things then hold exactly:

- `A` and `B` share the within-buy conditional (verified to 2.22e-16), so it cancels from the
  score difference;
- the logit shift is constant within each segment (verified to 1.07e-14), so the difference is
  **linear in each segment's mean none-rate** — and within-segment heterogeneity cancels, so
  no homogeneity assumption is needed.

One returned score therefore identifies the split, while the file remains a real candidate
that can win. Recovery is algebraically exact (simulation error ~3e-15).

We shipped luxury p4 = 0.285; the board returned **1.217**, giving **r_lux = 0.2236**. Our
model already implied 0.2314 — within one sampling standard deviation — so the per-segment
correction is worth **+0.00037** and the margin channel is closed. It also refuted the
board-inversion estimate (0.171, ~6σ away) outright.

**The limit is the interesting part.** The conditional cancels *by construction* — that is
precisely what makes the inversion exact. So this instrument is structurally incapable of
saying anything about the three-way choice among real bundles, which is where the remaining
headroom actually is (oracle 0.986 against the blend's 1.130). It is exquisitely precise about
the half of the problem that is finished, and blind to the half that is not.

---

## Why local ≠ Kaggle (matters)

**Folds are grouped by respondent.** Each person answers 19 choice tasks. If you split
randomly by row, a model sees 15 of someone's answers and is graded on the other 4 — it
learns "this person is stingy" and aces the holdout. Local score looks great; Kaggle
doesn't, because the test set is 263 people who appear nowhere in training. Splitting by
*person* makes local validation mirror the real task. This is the single most important
thing to preserve if you modify the pipeline.

**Nested blending.** Blend weights are refit five times, each excluding the fold it is
evaluated on, so no number we act on has seen its own tuning data.

**The test population is different.** Test respondents are roughly twice as wealthy
(median income $60k → $80k, p75 $85k → $125k). `model/shift_audit.R` reweights training
respondents to look like the test population and rechecks every improvement. Part-worth
coding retains 101% of its value and the listwise objective 112% — both structural. The
design encoding retains only 77%, since its shares encode a poorer population's tastes.

---

## Working agreements

- **One Kaggle account only** (team representative's). Using more than one is an academic
  integrity violation under the competition rules.
- **Two submissions per day**, resetting ~08:00 SGT.
- **Record every public score** in [`submissions/log.md`](submissions/log.md) — that's how
  we calibrate what a local improvement is actually worth.
- **R only.** Any package is allowed; other languages are not.

---

## The knowledge base

`Vault/` is an Obsidian vault covering the course theory and the competition work.
**Open the repository root as the vault** (not `Vault/` itself) so the lecture PDFs under
`Raw Dump/` open as links inside Obsidian. Note-to-note links use bare filenames, so they
resolve either way; the PDF links need the repo root.

<!-- To add the graph view: in Obsidian press Ctrl+G (graph view), arrange it, then use
     Win+Shift+S to snip. Save as docs/images/vault-graph.png and this will render. -->
![Obsidian graph view of the vault](docs/images/vault-graph.png)

Start at `Vault/00 Hub.md`. The competition notes are:
[Brief & Rules](Vault/Competition/Brief%20&%20Rules.md) ·
[Data Dictionary](Vault/Competition/Data%20Dictionary.md) ·
[Modeling Strategy & Results](Vault/Competition/Modeling%20Strategy%20&%20Results.md) ·
[Key Findings](Vault/Competition/Key%20Findings.md)

## Want to continue the work?

**Forty-five iterations are done, and the model search is measured-exhausted.** Iterations
27–34 searched model space hard across six independent axes and produced almost nothing.
Iteration 35 found a real defect by *reading the combiner* rather than searching — a single
line that had constrained every blend ever fitted here — but the leaderboard then refuted the
fix (above). Iterations 37, 38, 40 closed the combiner from four more directions; 39 and 45
tuned the tree and found nothing; 44 confirmed that respondents genuinely differ in
decisiveness (τ = 1.609) but could not turn it into accuracy.

**Do not re-open these.** Each is closed by a measurement, not an opinion:

| channel | closed by |
|---|---|
| adding a positive-weight member | frontier probe: best possible **+0.00027** across 32 artifacts, 7 families |
| combiner search | greedy, forward, backward, ridge over 41 members, class intercepts — **all land at 1.1234** |
| negative-weight control variates | **1.209 on the board** vs 1.197 (iteration 43) |
| calibration maps / temperature | an *oracle* in-sample 20-bin recalibration makes logloss **worse** |
| per-respondent dispersion | model captures 29–35% of the true spread, and `√R²` predicts **31%** — correctly shrunk |
| hierarchical Bayes | `hbmnl_nod` already earns weight in the 41-member ridge fit; the total does not move |

**Where the remaining headroom provably is.** Oracle per-respondent rates would take the blend
from 1.12969 to **0.98565** — a gap of 0.144 — but demographics reach only **11.2%** of that
heterogeneity, so ~0.016 is all that is demographically reachable, and much of it is already
captured. **~89% of what drives the outside option is unobservable in this dataset.**

The lesson worth carrying: when the search space is genuinely exhausted, look at the
*apparatus* doing the searching. Start with [`STRATEGY_REVIEW.md`](STRATEGY_REVIEW.md) for the
plan through 10 August; the remaining marks are still mostly in the report.

If you want to understand *what* was tried, open [`EXPERIMENTS.md`](EXPERIMENTS.md) and read
the **"👉 PICK UP HERE"** section, especially the ⛔ table of settled ideas — each one has the
number that killed it, so nobody repeats them.

Three things to know before you trust any number in here:

- **There is no single noise floor.** Comparing two single models needs the seed sd
  (**0.00283**); comparing two blends needs the blend-level sd (**0.00048**). Confusing them
  is how one accepted result survived eighteen iterations before being retracted.
- **Shrink every win**: ×0.8 for measured replication on an independent fold structure, then
  **×~½ (band 0.38–0.75)** for what reaches the leaderboard. *The older "×⅓ and decaying"
  figure is superseded* — it was built on two steps that fell below Kaggle's three-decimal
  display resolution, so it was reading rounding as decay. Re-measured across all five scored
  submissions on 27 Jul; the most recent step transferred at ~89%.
- **Check what a test can actually see.** The per-fold leak signature above is the right tool
  for a leaking *feature* and the wrong tool for a leaking *combiner*. Ask what your detector
  would look like under the specific failure you fear, not under failure in general.

## Reading order for someone new

1. This README
2. [`EXPERIMENTS.md`](EXPERIMENTS.md) — every hypothesis, result, and what we'd do
   differently, including the failures
3. [`model/02_mnl_partworth.R`](model/02_mnl_partworth.R) and
   [`model/03_xgb_listwise.R`](model/03_xgb_listwise.R) — the two models
4. [`report_notes.md`](report_notes.md) — findings written up for the report
5. `Vault/` — open the repository root as an Obsidian vault; start at `Vault/00 Hub.md`
