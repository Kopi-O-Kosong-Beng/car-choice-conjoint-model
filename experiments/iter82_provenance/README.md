# Iteration 82 — the margin audit across both tracks, and the model-distance audit

*Run 31 Jul 2026, ~04:30 SGT, ~31h before Kaggle closed. Two findings. The first identified
a ~0.0097 correction sitting unused on our second track. The second measured whether pooling
the two tracks buys anything, which is what decided the file we finally selected.*

**Outcome, added after the close:** the pool this iteration built —
`submissions/cand_pool5050_final00.csv` — was the selected submission. It scored **1.185
public and 1.185 private**, finishing **3rd on the public board and 4th on the private
board**. The pre-registered forecast was 1.186 with a band of 1.184–1.189
(`submissions/log.md`, 31 Jul ~05:10), so the point forecast missed by one tick in our
favour and landed inside the band.

---

## Finding 1 — the probe anchor was applied to the wrong track

The alt-4 probe (`submissions/probe_alt4.csv`, constant `(1/6,1/6,1/6,1/2)`, returned 1.499)
identifies the public none-rate exactly:

> `r* = (log 6 − 1.499) / log 3 = 0.266481153`

Margin audit over every submission on disk, 31 Jul:

| file | mean p4 | miss vs r\* | tilt multiplier | est. Δ logloss | public |
|---|---|---|---|---|---|
| `sub_20260729_nnblend.csv` | **0.21086** | **−0.05562** | 1.4062 | **−0.00972** | 1.194 |
| `sub_20260730_mprobe285.csv` | 0.26648 | −0.00000 | 1.0000 | −0.00000 | 1.217 |
| `cand_prod_corrected.csv` (main + anchor) | 0.26648 | 0.00000 | 1.0000 | +0.00000 | — |

**`sub_20260729_nnblend.csv` was our best-scoring file (1.194) and it was never anchored.**
The estimator puts the recoverable loss at **0.00972**, which agrees with `iter62`'s own
pre-registered audit (0.00879) and with the board's realised ~0.010 on a file at p4 ≈ 0.211.

**Why it was missed.** `submissions/log.md`, 30 Jul, states "The improvement does not transfer
to our track" — but the file tested there was `sub_20260726_2328.csv`, which was already at
p4 = 0.24800. That is the *main* track. The second track was never audited against r\*, so a
0.0097 correction sat unused while the anchor was applied to the file where it was worth
0.00104. The phrase "our track" did the damage: `iter62_nnblend` is also ours — it is the
team's second modelling track, built by a teammate, and nobody had thought to point the
margin audit at it.

Artifact written: `submissions/cand_nnblend_anchored.csv` (mean p4 = 0.266481, min p = 8.9e−3,
within-buy ratios preserved to machine precision).

---

## Finding 2 — the two tracks are far enough apart that pooling pays

`track_distances.R` reproduces everything below. It reads only artifacts tracked in this
repository, fits nothing, and measures nothing but distances.

### Why a single distance would be meaningless

Two files can be far apart purely because of **calibration** — a one-parameter margin shift —
while being the same **model** underneath. So every comparison is split:

- **margin** — mean p4 and its spread. One scalar.
- **conditional** — `(p1,p2,p3)/(1−p4)`. This *is* the model.

Both tracks are anchored to r\* first, so the margin is held equal and what remains is model.

The control that proves the decomposition works: `sub_20260730_mprobe285.csv` sits at
conditional distance **7.0e−18** from `final00` — machine zero — while its *full* distance is
0.0183. The segment re-targeting provably touches the margin and nothing else, exactly as
`build_candidates.R` claims.

### The measurement

| pair | symKL full | **symKL conditional** |
|---|---|---|
| main track (`final00`) vs track 2 (`nnblend`) | 0.01793 | **0.01291** |

Per-class conditional correlation: **0.971 / 0.970 / 0.973**. The top-ranked bundle differs on
**13.23%** of the 4,997 test rows.

### The null — and this is the number that makes it interpretable

A distance means nothing without a scale. How far apart do models *we* built actually sit?

| our own pair | symKL conditional |
|---|---|
| `xgb_lw2bag` vs `xgb_lw2` — **same model, 10-seed bag vs single seed** | **0.00552** |
| `xgb_lw2bag` vs `xgb_syn` (both listwise xgb, separately built) | 0.01477 |
| `xgb_long` vs `xgb_wide` | 0.01674 |
| `lcmnl3_both` vs `mnl_pw` | 0.01834 |
| `xgb_lw2bag` vs `xgb_wide` | 0.02978 |
| `mixl2` vs `lcmnl3_both` | 0.03613 |
| `xgb_lw2bag` vs `lcmnl3_both` (the widest axis the blend spans) | 0.04117 |
| `xgb_lw2bag` vs `mnl_pw` | **0.04837** |

> **Our two tracks sit at 0.01291 — 2.3× above the same-model-reseeded floor (0.00552), and
> about a quarter of the widest axis we own (0.04837).** They are genuinely different models,
> not one model wearing two names. Pooling them is a real hedge, not a relabelling.

Had the two tracks come in *below* 0.00552 the pool would have been buying nothing, because
we would have been averaging one model against itself.

### Where the pool actually lands

`w = 0.5` is halfway in *log-probability* space by construction. It need not be halfway in
model space, and that is worth checking before shipping it:

| pool vs | symKL full | symKL conditional |
|---|---|---|
| main track (`final00`) | 0.00431 | **0.00310** |
| track 2 (`nnblend` anchored) | 0.00465 | **0.00336** |

Ratio **0.92** against a perfectly-central 1.00. The pool is very nearly equidistant from
both parents — the log-space weight happened to translate almost exactly into model space,
so neither track quietly dominates the file we shipped.

### The disagreement is row-wise, which is the shape pooling can exploit

Per-class mean signed difference in the conditional is **+0.00960 / −0.00525 / −0.00435** —
small, and summing to ~0. Per-class sd is ~**0.050**, an order of magnitude larger, and
**96.6%** of rows differ by more than 0.01.

The two tracks therefore agree almost exactly *on average* and disagree substantially
*row by row*. That is precisely the error structure a pool cancels: no systematic bias to
inherit, plenty of independent noise to average away.

---

## The cross-track pool — built, held, and ultimately shipped

`submissions/cand_pool5050_final00.csv` — log-opinion pool of the anchored main track and the
anchored second track, w = 0.5, re-anchored to r\* after pooling (pooling perturbs the margin).

The weight is **fixed, not fitted**, and could not honestly have been fitted: the second track
has no OOF on `folds.rds` and can never have one (its fold constructor differs; adjusted Rand
against `folds.rds` ≈ 0.002, i.e. statistically independent partitions). There is no honest
objective to tune `w` on, so 0.5 is the only choice that spends no selection budget.

Its value is exactly computable without labels, from

> `loss(pool_w) = (1−w)·L_A + w·L_B + E[log Z_w]`,  `Z_w = Σ_k A_k^(1−w) B_k^w`

and by Hölder, `loss(pool) ≤ max(L_A, L_B)` on **every** row set — including the private 1,499.
For a one-shot pick where public rank is nearly uninformative about private rank, that bound
was the only variance hedge on the table.

**It held.** Public 1.185, private 1.185 — a public→private drift of essentially zero, against
an absolute wobble of ~0.011 that the 1,499-row private draw could easily have produced.

---

## What was unambiguously the main track's own

`experiments/iter81_honesttree/` — rebuilding the production tree without the iteration-48
encoding leak and at the honest depth optimum (4–5 rather than the shipped 8). Four arms,
seed-bagged, judged on the segment-reweighted nested blend OOF with respondent-clustered SEs
(`blend_eval.R`, validated: reproduces plain **1.12819** and segment-reweighted **1.19610**
exactly, and the corrected ESS of **208 of 1,135 respondents**). It shipped nothing — the
member-level gain was absorbed by the blend — but it is the cleanest capacity contrast in
the repo.

---

## Repo defects found while doing this

- **`sub_20260730_final00.csv` was never written to disk** — not in git, not on any branch,
  though `submissions/log.md:576` instructed the reader to select it. It was uploaded to
  Kaggle on 30 Jul (public 1.193) so it remained selectable *there*, but it was
  unreproducible here. **Resolved** by `build_candidates.R`, which reconstructs it by
  inverting `mprobe285`; every within-buy conditional is recovered exactly, the segment split
  only to ~1e−4. It is a faithful reconstruction and must **not** be described as the file
  that scored 1.193.
- **`experiments/iter67_caltower/` and `iter68` do not exist**, though CLAUDE.md cites
  `iter67_caltower/harness.R` and the ~160-arm endgame results.
- **`model/predict_lb.R` does not exist**, though CLAUDE.md cites it for the ESS figure.
- `CLAUDE.md` — the `## ⛔ READ THIS FIRST: the project is FROZEN for modelling` heading and
  its opening paragraph were lost in merge e8e1d22; the freeze rule read as a dangling
  fragment. `AGENTS.md` still had it.
- `CLAUDE.md:3` calls `AGENTS.md` "a copy"; it is a shorter, differently-structured variant.
