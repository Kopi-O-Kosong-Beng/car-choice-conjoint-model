# Guide for AI assistants

See [`CLAUDE.md`](CLAUDE.md) — identical content, duplicated here so tools that look for
`AGENTS.md` (Codex, Cursor, Copilot, Gemini CLI, and others) find it too.

Human readers want [`README.md`](README.md) instead.

## The 60-second version

- SUTD Analytics Edge Kaggle competition. Predict choice among 4 car safety bundles.
  Metric: multiclass logloss. **R only** — hard rule.
- Current: local nested CV **1.13044**, public **1.199** (benchmark 1.38629).
- Entry point: `model/run_all.R`. The submission blends three models — latent-class
  conditional logit (`experiments/iter11_latent_class/`, weight 0.447), listwise xgboost with
  a monotone price constraint (`experiments/iter08_mono_tuned/`, 0.337), and unconstrained
  listwise xgboost (`model/03_xgb_listwise.R`, 0.216). `model/members.txt` is the source of truth.
- **Leakage lesson that cost us a fake 1.09962:** fold-honest reference *sets* are not
  enough — any model-derived quantity attached to a reference observation must also be honest
  w.r.t. the fold being scored. See `experiments/iter12_residual_encoding/run.R`.
- **Never** regenerate `model/artifacts/folds.rds`, and never split CV by row — folds are
  grouped by respondent because the test set is 263 entirely new people.
- Judge every change with `model/compare.R` (paired test, respondent-clustered SEs), not
  by comparing headline scores.
- Log every experiment in `EXPERIMENTS.md`, failures included.

Full detail, traps, and open ideas: [`CLAUDE.md`](CLAUDE.md).
