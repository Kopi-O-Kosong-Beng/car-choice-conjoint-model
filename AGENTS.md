# Guide for AI assistants

See [`CLAUDE.md`](CLAUDE.md) — identical content, duplicated here so tools that look for
`AGENTS.md` (Codex, Cursor, Copilot, Gemini CLI, and others) find it too.

Human readers want [`README.md`](README.md) instead.

## The 60-second version

- SUTD Analytics Edge Kaggle competition. Predict choice among 4 car safety bundles.
  Metric: multiclass logloss. **R only** — hard rule.
- Current: local nested CV **1.13556**, public **1.201** (benchmark 1.38629).
- Entry point: `model/run_all.R`. The submission is a blend of two models:
  `model/03_xgb_listwise.R` (weight 0.64) and `model/02_mnl_partworth.R` (weight 0.36).
- **Never** regenerate `model/artifacts/folds.rds`, and never split CV by row — folds are
  grouped by respondent because the test set is 263 entirely new people.
- Judge every change with `model/compare.R` (paired test, respondent-clustered SEs), not
  by comparing headline scores.
- Log every experiment in `EXPERIMENTS.md`, failures included.

Full detail, traps, and open ideas: [`CLAUDE.md`](CLAUDE.md).
