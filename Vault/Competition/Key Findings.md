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

## 1b. Three buyer segments — and they may explain the concave curve above

A latent-class conditional logit (3 segments, membership predicted from demographics) splits
respondents into three near-equal groups:

| segment | share | price effect across the range | outside-option constant |
|---|---|---|---|
| **Value-conscious buyers** | 36.6% | **−1.65** steadily declining | −1.40 (buys readily) |
| **Price-indifferent buyers** | 32.1% | **+0.50** flat to slightly positive | −1.30 (buys readily) |
| **Reluctant non-buyers** | 31.4% | −0.60 mild | **+0.17 — inclined to decline** |

This **reframes finding 1**. The aggregate concave price curve may be substantially a
*mixing artefact*: average a steep response (37%), a flat one (32%) and a mild one (31%) and
the average bends, even though no segment's own curve looks like it.

⚠️ A positive price coefficient (segment 2) is not behaviourally standard. Candidate
explanations, none tested: price as a quality signal; residual price/richness confounding in
the design; or the segment absorbing respondents driven by something unobserved. Present it
as an estimated pattern with caveats, not as established consumer psychology.

**Commercially:** a third of respondents decline regardless of price — no discount reaches
them. The leverage is in segments 1 and 2.

Worth **+0.013 (z = 3.8)**; it takes the largest blend weight (0.447) and drives the plain
part-worth logit to zero weight, being a strict generalisation of it.

---

## 2. Taste heterogeneity is large, but only usable in *discrete* form

Mixed logit with a log-normal price coefficient estimates σ = 1.31: price sensitivity
differs roughly **six-fold** between the 25th and 75th percentile respondent. The
none-option constant has mean −1.34 and **sd 2.47** — some people won't buy any bundle at
almost any price, others are easy sells.

And yet the mixed logit *lost* to the simpler fixed-coefficient model (1.173 vs 1.157) and
earned zero blend weight.

**Why the continuous version fails but the discrete one wins:** every test respondent is a
stranger, so a continuous mixture can only be *integrated out* — you predict the population
average, flatter than any individual. A finite mixture instead routes each respondent to a
segment **using observable demographics**, which survives into the test set.

**Heterogeneity is usable exactly to the extent it correlates with something measurable about
a new person.** That is the sharpest methodological point we have for the report — and it is
why [[Key Findings#1b. Three buyer segments — and they may explain the concave curve above|the latent-class model]]
succeeded where mixed logit did not.

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

---

## 6. A leakage postmortem (methodology material, worth its own paragraph)

One experiment scored **1.09962** — nine times the project's entire accumulated gain — from a
single derived feature. It was pure leakage.

The feature encoded, per choice-set design, the average *residual* (observed − predicted) of
the respondents who saw it. The fold rule was honoured: the encoding for a fold-*k* row used
only respondents from other folds. **But each reference respondent carried a baseline
prediction from a model trained on folds that included fold *k*.** A deep tree holding
design-level features can partially memorise a choice set, so the baseline already embedded
fold-*k* choices; subtracting it handed the target its own label back, sign-flipped.

Three signals exposed it before any control was run: adding the feature directly to the
baseline made predictions *worse*; its correlation with each row's own held-out residual was
*negative* (genuine signal correlates positively); and the least-shrunk variant was the
model's second most important feature.

**General lesson: fold-honest reference *sets* are not sufficient — any model-derived
quantity attached to a reference observation must also be honest with respect to the fold
being scored.** Rebuilt with a baseline structurally incapable of memorising a design, the
same idea is worth ~+0.004.

Related: [[Topic 3 - Discrete Choice]] · [[Modeling Strategy & Results]]
