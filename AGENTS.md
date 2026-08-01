# Guide for AI assistants

See [`CLAUDE.md`](CLAUDE.md) — identical guidance, duplicated here so tools that look for
`AGENTS.md` (Codex, Cursor, Copilot, Gemini CLI, and others) find it too.

Human readers want [`README.md`](README.md) instead.

## The 60-second version

- SUTD Analytics Edge Kaggle competition. Predict choice among 4 car safety bundles.
  Metric: multiclass logloss. **R only** — hard rule, never propose Python.
- **✅ FINISHED.** Kaggle closed 1 Aug 2026. Selected submission
  `submissions/cand_pool5050_final00.csv`: **public 1.185 (3rd), private 1.185 (4th)**.
  Benchmark 1.38629. Report due **10 Aug** — 15 of 30 marks, and the only thing left.

### ⛔ The competition is over — the freeze is permanent

**The only correct action is report work.** There is no score left to move and no submission
left to make. `EXPERIMENTS.md`'s ⛔ table probably already contains any modelling idea you are
about to have, along with the number that killed it. If the user asks for a run anyway, it
should be to produce a figure or number the report needs — and even then, prefer reading an
existing artifact.

Start from [`report_notes.md`](report_notes.md); it now carries a drafted "The graded model,
described" section for rubric item (i). The post-mortem for item (ii) is the final section of
[`submissions/log.md`](submissions/log.md).

**The headline finding, and the answer to rubric item (ii):** public 1.185, private 1.185 —
**zero drift to three decimals**, against a ~1,499-row private draw whose own sampling sd is
~0.011. The model did not overfit the leaderboard, and `r*` — a constant calibrated entirely
against public rows — transferred intact to rows it had never seen. The rank still slipped
3rd → 4th on an unchanged score, which is the paired ranking noise (SE 0.006–0.012) made
visible. **The rank moved; the model did not.**

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
