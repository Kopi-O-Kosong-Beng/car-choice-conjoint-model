# Analytics Edge Data Competition 2026 — Modeling Strategy & Design

**Date:** 2026-07-25 · **Team:** 3 (Zhi Feng, Sheil, Kavya, Nicole) · **Deadline:** Kaggle 1 Aug 2026 12:00 SGT, report 10 Aug 2026 12:00 SGT

## 1. Problem

Predict which of 4 car-safety-feature bundles each respondent chooses (alternative 4 = all-zero "none" option). Metric: mean multiclass logloss (benchmark 1.38629). Train: 21,565 rows = 1,135 respondents (`Case` 1–1135) × ~19 tasks. Test: 4,997 rows from **new respondents** (`Case` 1136+). Public LB = random 70% of test rows; private = 30%; best public submission is auto-scored on private. Max 2 submissions/day. **R only** (any package; AI assist allowed).

Team state at design time: Sheil's v7 stack (MNL + wide xgboost + multinom stacker) OOF 1.1527 local; observed local→public offset ≈ +0.05 (1.17683 → 1.2230). Suspected OOF optimism: stepwise interaction search on full data; stacker scored on its own training OOF. To be verified when her script arrives.

## 2. Goal & success criteria

- Primary: minimize private-LB logloss. Practical target: honest (nested) OOF ≈ 1.11–1.13 → public ≈ 1.17–1.19, beating the 1.210 rival reference.
- Every submission decision is made on a **trustworthy local number** (nested OOF on fixed grouped folds).
- Preserve interpretability artifacts (MNL/mixed-logit coefficients, WTP) for the 15-mark report.

## 3. Architecture

Folder `model/` beside `Raw Dump/`, numbered R scripts, each runnable via `Rscript`:

| Script | Purpose |
|---|---|
| `00_load.R` | Read CSVs, wide→long conversion (1 row per alternative), feature engineering, saved as RDS. Sourced by all models. |
| `01_folds.R` | One fixed 5-fold assignment **grouped by `Case`**, seeded, saved to `artifacts/folds.rds`. All models must use it. |
| `02_mnl.R` | Conditional logit (mlogit) with ASCs + demographic×price interactions; any selection done inside CV. |
| `03_mixl.R` | Mixed logit — random coefficients on Price + key attributes, panel structure; population-averaged predictions for new respondents. Primary package `logitr` (fast), fallback `mlogit`. |
| `04_xgb_long.R` | xgboost, long format: binary target "this alternative chosen", per-task softmax normalization; cross-alternative features. |
| `05_xgb_wide.R` | Wide 4-class xgboost (Sheil-style, own implementation until her script arrives; hers slots in as an extra member). |
| `06_blend.R` | Log-space weighted blend + temperature, weights by `optim` on OOF; **nested evaluation** (weights refit leaving each fold out) is the reported number. |
| `07_submit.R` | Refit members on full train, predict test, clip probs (1e-6), renormalize to sum 1, write `submissions/sub_YYYYMMDD_desc.csv`, append to `submissions/log.md`. |
| `99_utils.R` | logloss, per-fold SD, train-vs-test covariate shift check. |

Artifacts (RDS, OOF matrices) in `model/artifacts/`, gitignored-style disposable.

## 4. Features (long format)

- Alternative attributes as numeric levels: CC GN NS BU FA LD BZ FC FP RP PP KA SC TS NV MA LB AF HU Price.
- Alternative-specific constants (alt id, esp. alt 4 = none).
- Cross-alternative (within task): attribute minus task-mean, price rank, price minus min rival price, bundle richness (# nonzero attributes), richness rank, best-rival richness/price for the none-option row.
- Respondent: numeric forms (`agea`, `incomea`, `milesa`, `nighta`, `*ind` codes), key categoricals one-hot for trees; demographics enter MNL only via interactions.
- `Task` (1–19) fatigue term.

## 5. Validation protocol

- Fixed seed, 5 folds grouped by `Case` (test = unseen respondents, so folds must be too).
- Each model: OOF predictions on identical folds; report mean logloss ± per-fold SD.
- Blend: weights/temperature fit on 4 folds' OOF, evaluated on 5th, rotated → nested OOF is THE decision number.
- Submission log records (nested OOF, public score) pairs to calibrate the local→public offset (~4 points known already from team history).

## 6. Submission plan

≈13 slots remain (2/day, resets ~08:00 SGT). Rule: submit only if nested OOF beats current best by > 1 per-fold SD, or to establish the offset for a new model family. Coordinate with Sheil (account holder). Keep 1 slot/day in reserve until evening.

## 7. Risks & mitigations

- **R env on this machine unknown** → check first; install packages as needed.
- **Mixed logit slow/finicky** → logitr with Halton draws, few random coefficients first (Price, then top attributes); cap fitting time; mlogit fallback; HB (`bayesm`) only as overnight stretch goal.
- **OOF→LB gap persists** → honest nested numbers + offset tracking; optional shrink-toward-uniform ε tuned on held-out fold.
- **Teammate scripts delayed** → pipeline is self-sufficient; their models are additive blend members.

## 8. Out of scope (for now)

Neural nets in R (torch), importance-weighting for covariate shift (check first, act only if shift is material), Python anything (rule).

## 9. Companion deliverables

- Obsidian vault generated from `Raw Dump/` (per-topic notes + competition hub), after first honest OOF exists.
- `report_notes.md` accumulating decisions, coefficients, WTP calculations, and public/private fit observations for the 8-page report (due 10 Aug).

*Self-review: no placeholders; consistent with competition rules (R-only, 2/day, one account); scoped to one implementation plan. Not committed — folder is not a git repo (course data dump; init deliberately skipped).*
