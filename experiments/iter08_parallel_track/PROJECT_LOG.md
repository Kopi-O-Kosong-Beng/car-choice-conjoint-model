> # ⚠️ SUPERSEDED — READ THIS FIRST
>
> **This file was written on 25 July 2026 and its score claims are WRONG.**
> It is kept only for its experiment history and methodology. Corrections:
>
> | this file says | actual (verified 26 July, read-only Kaggle CLI) |
> |---|---|
> | "team is graded on **1.233**" | **1.201** — a third submission was made 25 Jul 15:57 UTC from the same team account |
> | "v7 is NOT submitted, est. real ~1.214" | true, but v7 at ~1.214 is **WORSE** than the graded 1.201 |
> | "class leader 1.202" | the team is **rank 1 at 1.201**; two teams sit at 1.202 |
>
> **Do not act on the submission advice in this file.** v7 and v9 are both worse than
> what is already on the leaderboard and must not be submitted. The current plan is
> [`PLAN.md`](PLAN.md) in this folder.
>
> Everything else here — the version history, the ~25 rejected experiments, the
> validation methodology, the R gotchas — remains accurate and useful.

# Analytics Edge Competition 2026 — Project Log & Handover

Everything needed to resume work in a fresh session. Companion files: `MODELS_LEDGER.md`
(version table), `NOTES.md` (original course/brief research), `lockbox_setup.R` (validation harness).

---

## 1. Where we stand

| | LogLoss |
|---|---|
| **Best model (v7), local OOF** | **1.1529** |
| **Best REAL Kaggle score (submitted)** | **1.233** |
| Class leader (public LB) | 1.202 |
| Uniform benchmark | 1.38629 |

**The single most important fact: v7 is NOT submitted.** The team is graded on 1.233 from a
much older model. `submission_best.csv` holds v7's predictions and is validated and ready.
Estimated real score ~1.214 (local + the ~0.06 local→real gap measured from our 2 submissions).

Submissions used: 2 of 2 on 2026-07-24 (1.233, then 1.249). Fresh ones available since.
**Never submit without explicit user consent.**

- Competition ends **1 Aug 2026, 12:00 SGT**. Report due 10 Aug 2026.
- Report work is explicitly DEFERRED until the user asks (after the competition).
- Team: `Sheil_Mistry_Team_3`. Kaggle slug: `the-analytics-edge-competition-2026`.
- Kaggle CLI at `~/.local/bin/kaggle`, auth via `~/.kaggle/access_token` (works).
  Read-only: `kaggle competitions submissions -c <slug>` / `... leaderboard -c <slug> --show`.

---

## 2. The data (verified facts, not assumptions)

- `train2024.csv` / `test2024.csv` **are the real current-year files** despite the 2024 names.
- 1,135 train respondents × 19 tasks = 21,565 rows. 263 test respondents × 19 = 4,997 rows.
  **Train and test respondents are completely disjoint** — this is a new-people generalisation
  task, which is why all validation splits by `Case` (respondent), never by row.
- Each task shows 4 bundles. **Alternative 4 is ALWAYS the outside option** ("buy nothing"):
  all 20 of its attributes are 0 in every row of both files. It is chosen **30%** of the time.
  Giving it an alternative-specific constant was one of the biggest early wins (its fitted
  intercept ≈ −2.9).
- **Partial-profile conjoint design**: each real bundle shows exactly **10 of the 19** non-price
  features; the rest are 0 meaning "not shown", not "level zero". Every feature appears in ~47%
  of bundles. This creates a linear dependency among attribute dummies (only 9 vary freely).
- Attributes are **ordinal level indices**, not real units (CC 0–3, NS 0–5, BU 0–6, Price 0–12).
  Price 0 occurs ONLY on alt 4, making `factor(Price)` exactly collinear with the alt-4 ASC.
- **Severe train/test population shift in `segment`**: "Small Car" is 27% of train and **0%** of
  test; "Prestige Luxury Sedan" is 5% of train vs 37% of test. Other demographics are comparable.
- Task-order fatigue is real but weak: outside-option rate rises 24% (task 1) → 34% (task 19).
- 98.5% of test tasks reuse a bundle-design seen in train — but exploiting this is a **dead end**
  (tested; see rejected list).

---

## 3. Current best model (v7) — `model_v7_enriched_shared_utility.R` == `best_model.R`

A geometric blend of two very different architectures, `p_ours^0.30 × p_mate^0.70`:

**Our side** (weight 0.30) — itself `p_MNL^0.2 × p_xgb^0.8`:
- MNL (`mlogit`): `Choice ~ <20 attrs> + Price:ppark`, **no `-1`** so alternative-specific
  constants are fitted. Uses **segment importance weights** (test segment share ÷ train share).
- xgboost `multi:softprob`, wide format (all 4 alternatives in one row), heavily regularised
  (depth 3, eta 0.03, subsample 0.85, colsample 0.5, lambda 4, 600 rounds), **6-seed averaged**.
  Features: raw attrs, alt1-vs-others diffs, within-task vs-mean/vs-max, bundle loadedness,
  price rank, task number, 11 demographics, **Price×income and Price×age**.

**Teammate's side** (weight 0.70) — equal blend of 2 models, then **temperature 0.9 (SHARPENING)**:
- Two **shared-utility long-format** xgboost models. This is the key architecture: one row per
  **alternative** (4 rows/task) holding only that alternative's features plus a slot one-hot,
  trained `binary:logistic` ("was this chosen?"), then renormalised within each task. It forces
  ONE shared scoring function across alternatives — the random-utility structure a conditional
  logit has, which a wide 4-class model structurally cannot express.
- Both configs are **our retuned ones** (depth 5, eta 0.04, 500 rounds; A colsample 0.5/mcw 30,
  B colsample 0.9/mcw 10), each **3-seed averaged**, on **enriched** long features.

Run it: `Rscript best_model.R` (~18 min). Prints in-sample and OOF LogLoss, writes
`submission_best.csv` (4,997 rows, probabilities sum to 1).

---

## 4. Validation methodology — the hard-won discipline

**A change is only ADOPTED if a paired bootstrap 95% CI EXCLUDES ZERO**, resampling
**respondents** (not rows — a respondent's 19 tasks are correlated).

Why this exists: early on, the MNL's interaction terms were chosen by searching for whichever
minimised CV LogLoss, and then that *same* CV score was reported as the quality estimate. The
folds both picked the winner and graded it. It read 1.177; the real Kaggle score was 1.233.

- `source("lockbox_setup.R")` recreates the exact split used throughout: 908 development
  respondents + 227 untouched lockbox, plus `fold_id_sel`, `competition_logloss()`,
  `paired_bootstrap()`. Verified to reproduce the original split exactly.
- For borderline results (P ≈ 0.90–0.97, or a CI touching zero), **repeat across ~10 independent
  splits and t-test the per-split differences**. This mattered: a single split said our model and
  the teammate's were tied (P=0.70); repeating 6× showed theirs won 6/6 (p=0.028).
- **Cross-fit anything learned on OOF** (e.g. blend weights) or the score is optimistic.
- Local→real gap is ~0.06 and is **largely irreducible** — it is genuine population shift, proven
  by a segment-shift stress test in which *every* model degraded equally (~0.03), and by
  propensity reweighting failing to help.
- **Do NOT use "lockbox mean" reporting** (user's explicit instruction). Report plain OOF LogLoss.
- Scripts should print only the LogLoss (in-sample + OOF). Grade A/B/C labels were removed per
  user request.

---

## 5. Version history

| Ver | File | Change | OOF | Kaggle |
|---|---|---|---|---|
| v1 | (submitted) | MNL(4 interactions) + xgb + multinom stacker | 1.177 ⚠ leaky | **1.233** |
| v2 | (submitted) | MNL(ASC, Price:ppark) + xgb, geometric mean, segment weights | 1.1807 | **1.249** |
| v3 | `model_v3_xgb_enriched_seedavg.R` | enriched features + tuned + 6-seed xgb, blend w=0.2 | 1.1657 | — |
| v4 | `model_v4_price_demo_interactions.R` | + Price×income/age interactions | 1.1644 | — |
| — | (teammate's, `xgb_choice_ensemble_R_only.R`) | shared-utility long format ×2 + wide, learned blend + temp | 1.1586 | — |
| v5 | `model_v5_combined_architectures.R` | v4 × teammate's, geometric 0.4/0.6 | 1.1574 | — |
| v6 | `model_v6_tuned_shared_utility.R` | + retuned & seed-averaged shared-utility models | 1.1555 | — |
| **v7** | `model_v7_enriched_shared_utility.R` | + enriched long features, **wide model dropped** | **1.1529** | — |

⚠ v1's local number came from the leaky procedure and is not comparable.

---

## 6. What was ADOPTED (and why)

1. **ASC for the outside option** — removing `-1` from the MNL formula. −0.029, P=1.000.
2. **Segment importance weighting on the MNL** — corrects the train/test segment shift.
   Confirmed across 10 repeated splits (9/10, p=0.025). **MNL only** — see rejected list.
3. **Enriched wide features + tuned xgboost + blend reweighted to 0.2** (v3) — biggest single win.
4. **Seed-averaging** stochastic xgboost fits — small but free variance reduction.
5. **Price×income / Price×age** on our wide model (v4) — WTP heterogeneity. CI [−0.0073,−0.0002].
6. **Combining the two architectures** (v5) — they disagree on ~10% of top-1 picks. −0.0074, 6/6 splits.
7. **Retuning the shared-utility hyperparameters** (v6) — never searched by anyone. Their d6
   1.1618 → tuned 1.1577. Winners prefer depth 5, eta 0.04, heavy colsample.
8. **Enriching the long format** (v7) — value-vs-task-best, bundle loadedness, price rank, task.
   ~0.003 on *every* config.
9. **Dropping the wide multiclass model** (v7) — once the shared-utility models were enriched it
   was dragging the blend down. −0.0017, CI [−0.0026,−0.0007].

## 7. What was REJECTED (do not re-try without a new angle)

**Model classes:** LightGBM (weaker + impractically slow, >35 min/search), MLP/nnet (1.2203 alone,
no ensemble help), random forest (1.49–2.49; vote probabilities are hopeless for LogLoss even with
Laplace smoothing), xgboost DART (pathologically slow, >24 min), G-MNL (converges but ~17 min and
no `predict` for new data), latent class Q=3/4 (worse than Q=2), nested logit (identical to MNL).

**Features/encoding:** `factor(Price)` (collinear with alt-4 ASC; still worse after fixing),
all-attributes-as-factors, feature pruning (top-k of 324), value/dominance features
(price-per-feature, dominance counts), full pairwise diffs, quadratic price, prank×income,
Price×miles/region/urbanicity/night/parking, NV×night, PP×ppark, WTP features on the long format.

**Structural:** two-stage outside-option model (flat 4-class already captures it), MNL-as-xgboost
base-margin (residual boosting), feature-subspace diversity, task-fatigue MNL term.

**Blending/calibration:** multinom stacker (lost to plain geometric mean), learned blend weights +
temperature over all 5 base models (better in-sample 1.1564, identical cross-fitted 1.1574 — it
concentrated weight and destroyed diversity), temperature **softening** T=1.1 (adopted then
REVERTED when it raised OOF by 0.003 — note the optimum is *sharpening*, T≈0.9).

**Covariate shift:** propensity reweighting on all demographics (no better than segment-only),
**segment weights on any tree model** (badly harmful: 1.2006 vs 1.1618 on d6; same on our xgb).

**Data exploitation:** design-lookup / target encoding. 98.5% of test designs appear in train, but
pure lookup scores 1.32 and an empirical-Bayes blend with the model gains nothing — each design is
seen by only a handful of respondents, too noisy to beat a model that pools across designs.

---

## 8. Untried ideas (ranked)

1. **Custom softmax objective** — the long format approximates a conditional logit via
   `binary:logistic` + renormalise. The exact version is a softmax over each task's 4 alternatives,
   implementable as a custom xgboost objective. Highest ceiling, highest implementation risk.
2. More shared-utility configs / deeper search now that enrichment changed the feature space
   (the 22-config search predates enrichment).
3. Re-tune **our** wide xgboost on the enriched vocabulary (its tuning also predates it).
4. Per-alternative or outside-option-specific calibration (alt 4 is structurally different).

---

## 9. Gotchas that cost real time

- `predict.mlogit()` requires the `Choice` column (and any `weights` column) to be **physically
  present in `newdata`**, even with dummy values.
- `mlogit(..., weights = X)` needs `X` to be a genuine column in the data, not an external vector;
  otherwise `predict` on new data fails.
- xgboost 3.x `predict()` already returns an **n × 4 matrix** — reshaping with
  `matrix(..., byrow=TRUE)` silently scrambles rows and wrecks the score. `best_iteration` must be
  read via `xgb.attributes(bst)`, not `bst$best_iteration`. `watchlist` is now `evals`.
- Long-format row order must be task1-alt1..4, task2-alt1..4 … to line up with the label vector.
- `as.data.frame(dfidx(...))` does not expose `No` as a column — it lives in `$idx$No`.
- Passing `R = R` into `mlogit` for mixed logit fails ("object 'R' not found"); use `do.call`.
- Background jobs launched with bare `nohup ... &` are not harness-tracked (no completion
  notification). Use the tool's background flag instead.
- `test` has fewer `segment` levels than `train`; force union factor levels or interaction terms
  break at predict time.

---

## 10. File map

```
best_model.R                          # == v7, the deliverable. Run this.
model_v3..v7_*.R                      # version history, each self-contained
xgb_choice_ensemble_R_only.R          # teammate's original script (reads train.csv/test.csv)
lockbox_setup.R                       # validation harness — source() before testing changes
analysis.Rmd / analysis.html          # original 9-model comparison (report material)
MODELS_LEDGER.md                      # version table + evidence
PROJECT_LOG.md                        # this file
NOTES.md                              # course methods survey, brief summary
Brief.pdf                             # competition brief
train2024.csv / test2024.csv / sample_submission2024.csv
submission_best.csv                   # v7 predictions, validated, READY TO SUBMIT
submission_xgb_choice_ensemble.csv    # teammate's submission file
```
