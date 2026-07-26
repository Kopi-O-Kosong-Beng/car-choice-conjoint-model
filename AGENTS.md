# Guide for AI assistants

See [`CLAUDE.md`](CLAUDE.md) — identical guidance, duplicated here so tools that look for
`AGENTS.md` (Codex, Cursor, Copilot, Gemini CLI, and others) find it too.

Human readers want [`README.md`](README.md) instead.

## The 60-second version

- SUTD Analytics Edge Kaggle competition. Predict choice among 4 car safety bundles.
  Metric: multiclass logloss. **R only** — hard rule, never propose Python.
- Current: nested blend **1.12819**, public **1.199** (benchmark 1.38629, rival 1.210).
  Kaggle closes **1 Aug**, report due **10 Aug**.

### ⛔ The project is FROZEN for modelling

**The default correct action is report work, not model improvement.** Twenty-six iterations
have been run and the search space is measured-exhausted; `EXPERIMENTS.md`'s ⛔ table
probably already contains your idea along with the number that killed it. Proposing a new
model, feature, or blend member requires the user to explicitly re-open the freeze — say so
and ask. Read [`STRATEGY_REVIEW.md`](STRATEGY_REVIEW.md) before proposing any plan; it holds
the reasoning, the endgame timetable, and the report skeleton.

### The five things most likely to make you wrong

1. **There is no single noise floor.** Model-level seed sd **0.00283**, blend-level
   **0.00048**, fold-to-fold SD 0.013. Conflating the first two is how a false positive
   survived eighteen iterations. Anything under 0.003 on a single model, from before
   iteration 26, is unresolved.
2. **Shrink every win**: ×0.8 for measured off-split replication, then ×~⅓ for transfer to
   the public board, where only ~0.001 is even visible at three decimals.
3. **Never regenerate `model/artifacts/folds.rds`**, and never split CV by row — folds are
   grouped by respondent because the test set is 263 unseen people. Validation splits
   `folds_b.rds` / `folds_c.rds` already exist for replication.
4. **Nest everything fitted**, including the baselines that derived features are built on.
   A design-residual encoding scored a fake 1.09962; the "fixed" version was still 100%
   leakage, proven only by a nested double-OOF. A real gain shows in *every* fold; a leak
   concentrates in one.
5. **Artifact vs procedure.** `compare.R` compares two artifacts; decisions usually need two
   procedures. A lucky seed is not a good method.

### Non-negotiables

- Judge every change with `model/compare.R` (paired, respondent-clustered), never by
  comparing headline numbers. One change per experiment.
- Blend membership is explicit in `model/members.txt` — no auto-discovery.
- Two Kaggle submissions per day, one team account.
- Log every experiment in `EXPERIMENTS.md`, **failures included** — they are the most useful
  entries and carry most of the report's insight marks.

Full detail, traps, and the freeze protocol: [`CLAUDE.md`](CLAUDE.md).
