# Competition Brief & Rules (distilled)

Source: [[Raw Dump/The Analytics Edge Data Competition 2026.pdf|official brief PDF]]

## Timeline
- Kaggle: **24 Jul – 1 Aug 2026, 12:00 SGT** · Report due **10 Aug 2026, 12:00 SGT**

## Task & metric
- Predict the probability each of **4 safety-feature bundles** is chosen (bundle 4 = "none").
- Metric: mean multiclass **logloss** (lower = better). Uniform ¼ benchmark = **1.38629**.
- Rows not summing to 1 are auto-rescaled — we still normalize ourselves (numerical safety).

## Leaderboard mechanics (exploit correctly!)
- Test = 4,997 rows → **70% public / 30% private**, random split, redrawn each year.
- Final ranking = **private** LB. Your **best public** submission is auto-selected for private scoring → submitting an improvement is essentially risk-free.
- Private has only ~1,500 rows → expect ±0.02 public↔private wobble ([[Vault/Competition/Modeling Strategy & Results|see strategy]]).

## Rules
- **2 submissions per team per day** (resets ~08:00 SGT for us). ONE Kaggle account per team (ours: Sheil's, team name `sheil_mistry_team_3`). Extra accounts = ethics violation.
- **R only**, any package. AI assistance explicitly allowed. No help from people outside the team; no sharing data outside the team.

## Grading (30 marks)
| Component | Marks | Floor |
|---|---|---|
| Private LB | 7 | ≥3 if we beat benchmark |
| Public LB | 8 | ≥4 if we beat benchmark |
| Report (≤8 pages) | 15 | model description 5 · public-vs-private fit 2 · insights & limitations 5 · quality 3 |

**Takeaway: the report is worth as much as both leaderboards combined** — keep `report_notes.md` updated as we model, and preserve interpretable artifacts (coefficients, WTP) from every choice model.
