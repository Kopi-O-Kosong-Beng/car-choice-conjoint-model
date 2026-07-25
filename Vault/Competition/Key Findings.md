---
title: Key Findings
type: findings
updated: 2026-07-26
tags: [competition, insights, report]
---

# Key Findings

The four results worth putting in the report. Each was verified with a paired test using
respondent-clustered standard errors (`model/compare.R`), not just a lower headline score.

---

## 1. Price sensitivity is concave — people perceive ratios, not differences

Part-worth utilities for the 12 price levels, relative to level 1:

| level | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| utility | −0.52 | −0.76 | −0.90 | −1.08 | −1.11 | −1.22 | −1.38 | −1.47 | −1.71 | −1.93 | −1.92 |

Each price step costs *less* utility than the one before, and levels 11–12 are
statistically indistinguishable — sensitivity **saturates**. This is a Weber–Fechner
(logarithmic) response.

Coding price as a plain number forces a straight line through this curve. Freeing the
shape was worth **+0.020 logloss (z = 11.8)**, the largest single gain of the project.

**Report angle:** a manufacturer pricing at the top of the range gains almost nothing by
shaving one level off — the perceived difference has flattened out.

---

## 2. Taste heterogeneity is large, but not usable for prediction

Mixed logit with a log-normal price coefficient estimates σ = 1.31: price sensitivity
differs roughly **six-fold** between the 25th and 75th percentile respondent. The
none-option constant has mean −1.34 and **sd 2.47** — some people won't buy any bundle at
almost any price, others are easy sells.

And yet the mixed logit *lost* to the simpler fixed-coefficient model (1.173 vs 1.157) and
earned zero blend weight.

**Why:** every test respondent is someone we've never seen, so we can only predict the
population-*averaged* probability. Integrating over a taste distribution buys calibration
but costs sharpness. **Heterogeneity is real but not conditionable** — a limitation of the
prediction task, not of the estimator. This is the strongest "limitations" material we have.

---

## 3. The choice sets are a designed experiment

Only 299 distinct choice-set designs exist per task position, each shown to ~4.7
respondents, and **98.5% of test rows reuse a design that appears in training**. So the
empirical choice share within a design is partly observable.

Worth **+0.005 (z = 2.9)** — real, but small, and the size is itself informative. With
only ~3.8 observations per design the raw shares are so noisy that at weak shrinkage
(α = 1) they score **worse than the 25%-everything benchmark** (1.433 vs 1.386). At α = 5
they score 1.307. Most of the signal must be shrunk away to be usable.

**Caveat for the report:** this gain is specific to this data collection. It would vanish
on a genuinely new experimental design, and it retains only 77% of its value when
respondents are reweighted toward the test population.

---

## 4. Train and test populations genuinely differ

Test respondents are roughly twice as wealthy (median income $60k → $80k, p75 $85k →
$125k, mean $75k → $151k), better educated, and drive more.

This is real distribution shift, not sampling noise, and it explains part of the gap
between our local score (1.13878) and the leaderboard (1.201). We responded by adding
explicit price×income terms — linear-in-income models extrapolate, trees cannot — and by
tracking an income-reweighted OOF as a secondary diagnostic.

`model/shift_audit.R` reweights training respondents to match the test income
distribution and rechecks each improvement:

| improvement | retained under reweighting |
|---|---|
| part-worth coding | 101% |
| listwise objective | 112% |
| design encoding | 77% |

The two big wins are **structural** and should hold on the private leaderboard.

---

## The outside option dominates

Worth stating plainly: the all-zero "none of these" bundle is the **most-chosen
alternative** at 30.2%, versus 22.0 / 25.0 / 22.7% for the three real bundles. Any model
treating it as just a fourth product misses that opting out is the modal behaviour, driven
by different considerations (price level, total bundle richness) than choosing *between*
bundles.

Related: [[Vault/Topics/Topic 3 - Discrete Choice]] · [[Vault/Competition/Modeling Strategy & Results]]
