# Iteration 81 — rebuilding the production tree honestly

**Verdict: the production tree stands. Ship nothing from this. It is report material.**
**And the pre-registered decision rule was itself wrong — that is the most useful part.**

## What ran

Four arms, each 4 seeds, geometric-mean bagged, folds and features byte-identical to
`model/03_xgb_listwise.R` except where stated. ~4 h wall clock, 4 arms in parallel.

| arm | design encoding | max_depth | bagged plain OOF | single-seed sd | test mean p4 |
|---|---|---|---|---|---|
| `ctrl` | **leaky** (as production) | 8 | **1.13794** | 0.00499 | 0.26937 |
| `noenc8` | none | 8 | 1.15599 | 0.00304 | 0.27138 |
| `noenc5` | none | 5 | **1.15012** | 0.00066 | 0.27224 |
| `noenc4` | none | 4 | **1.15002** | 0.00158 | 0.27392 |

**Control validated.** `ctrl` reproduces production `xgb_lw2bag` (1.13682) at 1.13794 — 4 seeds
against 10, a gap of 0.00112 well inside the model-level seed sd of 0.00283. Blended against
`lcmnl3_both` it reads segment-reweighted Δ **+0.00047, z +0.53** versus the production blend,
i.e. statistically indistinguishable. The harness is faithful.

## Result 1 — the honest depth optimum is confirmed, and this contrast IS valid

`noenc8` → `noenc5` / `noenc4` changes capacity with **leak exposure matched at zero on both
sides**. That is precisely the condition iterations 54–59 violated, and paid 1.205 for.

> depth 8 **1.15599** → depth 5 **1.15012** → depth 4 **1.15002**

Depth 4–5 beats depth 8 by **0.0060**, independently confirming iteration 48's honest depth
optimum of 4–5 against the shipped 8, by a different route (iteration 48 inferred it from how
the leak's apparent value scaled with capacity; this measures it directly with the leak absent).

A second, unlooked-for finding: **single-seed variance collapses with depth** — sd 0.00499 at
leaky depth 8, 0.00304 at honest depth 8, **0.00066** at depth 5. The deep leaky arm is roughly
**7.5× noisier across seeds** than the shallow honest one. Anything ever accepted on a
single-seed depth-8 margin below ~0.005 is unresolved, which is a stronger statement than the
0.00283 figure in `CLAUDE.md` and should replace it for that configuration.

## Result 2 — the leak-removal contrast is UNDECIDABLE with any instrument we own

Blending each arm with `lcmnl3_both`, nested, against the production blend:

| challenger | plain Δ | segment-reweighted Δ | paired z | per-fold |
|---|---|---|---|---|
| `xgbh_ctrl` | +0.00042 | +0.00047 | +0.53 | mixed |
| `xgbh_noenc8` | +0.00852 | +0.00601 | +1.38 | +0.012 +0.008 +0.005 +0.008 +0.010 |
| `xgbh_noenc5` | +0.00773 | +0.00604 | +1.60 | +0.011 +0.008 +0.004 +0.007 +0.008 |
| `xgbh_noenc4` | +0.00774 | +0.00599 | +1.59 | +0.011 +0.007 +0.004 +0.008 +0.009 |

Removing the encoding looks **0.006 worse**, uniformly across every arm and every fold.

**This does not mean the encoding helps.** `CLAUDE.md:147-148` already says the
segment-reweighted metric "is **blind to the encoding leak**, which is population-independent
and inflates plain and reweighted alike." Both of our OOF metrics are inflated on the `ctrl`
side by construction. A contrast whose two sides have different leak exposure is contaminated
**on both metrics**, so this comparison cannot adjudicate leak removal in either direction. The
uniform-across-all-five-folds signature is exactly the structural-leak fingerprint `CLAUDE.md`
warns about — it is what a leak looks like, not what a real effect looks like.

**The pre-registered decision rule in `run.R` was therefore the wrong instrument, and it was
wrong in a way this repo had already written down.** The header named the segment-reweighted
nested blend as the decision measure for *all four* arms; it is only valid for the three that
share zero leak exposure. Recording this rather than quietly re-cutting the analysis, because
it is the fourth instance in this project of a confidently-designed test that could not see
what it was pointed at — after iteration 27's non-nested estimators, iteration 30's circular
residual regression, and the AUC contrast that "refuted" this very leak.

**The only instrument that could decide leak removal is the leaderboard**, at one slot. Given
~160 arms on 30 Jul yielded one 1.7σ survivor, and the freeze's validation chain (compare.R →
shift_audit.R → `folds_b`) cannot be cleared before 1 Aug 12:00 SGT, that slot is not worth
spending. **Production stands.**

## Result 3 — the depth gain does not survive blending

`noenc4`, `noenc5` and `noenc8` differ by 0.0060 as single models but land at segment-reweighted
1.20210 / 1.20214 / 1.20212 in the blend — a spread of **0.00004**. Blending with `lcmnl3_both`
absorbs the entire depth effect. This is the same mechanism iteration 26 recorded for bagging
("bagging and blending are substitutes") and iteration 19's finding that 93% of blend error
variance sits on the tree-vs-logit axis: improvements *along* the tree axis are largely
projected out by a blend already fitted across it. A member-level gain of 0.006 is worth ~0.0000
at blend level here.

## What this is worth

Nothing on the board, by design. For the report it supplies: an independent confirmation of the
honest depth optimum; a measured depth–variance relationship that recalibrates the noise floor;
a demonstration that member-level gains do not survive a blend fitted across the dominant error
axis; and a fourth documented case of an instrument that could not see the thing it was built
to measure — with the failure diagnosed rather than hidden.

Artifacts `model/artifacts/{oof,test}_xgbh_{ctrl,noenc8,noenc5,noenc4}.rds` are written and are
**deliberately absent from `model/members.txt`**. Per rule 5 they are not blend candidates.
