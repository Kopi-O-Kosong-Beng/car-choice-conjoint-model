---
title: Topic 3 — Discrete Choice
type: topic
tags: [course, topic-3]
updated: 2026-07-26
---

# Topic 3 — Discrete Choice ⭐ (the competition topic)

Sources: [[Raw Dump/Topic 3/discrete_choice.pdf|lecture PDF]] · `Raw Dump/Topic 3/Transportation analytics/automobiles.Rmd` · `Raw Dump/Topic 3/move analysis/oscars.Rmd`

## Core idea

A person facing alternatives $j = 1..J$ gets **utility** $U_{ij} = V_{ij} + \varepsilon_{ij}$, with observable part $V_{ij} = \beta' x_{ij}$ and iid Gumbel noise. They pick the max-utility alternative, which yields the **conditional/multinomial logit (MNL)** choice probability — a softmax:

$$P_{ij} = \frac{e^{V_{ij}}}{\sum_k e^{V_{ik}}}$$

- **Alternative attributes** (Price, feature levels) get one shared coefficient each.
- **Individual characteristics** (age, income…) can't drive choice alone (they cancel in the softmax) — they enter via **interactions** with attributes or as alternative-specific coefficients.
- **ASCs** (alternative-specific constants) absorb the average appeal of each option — crucial for our all-zero "none" bundle (its ASC ≈ −2.9 yet it wins 30% of choices via price).
- Estimation = maximum likelihood; in R: `mlogit` (data via `dfidx`, long format: one row per alternative).

## Key properties & limitations

- **IIA** (independence of irrelevant alternatives): relative odds between two options ignore all others — red-bus/blue-bus problem. Violated when alternatives share unobserved traits.
- Coefficient ratios are **willingness-to-pay**: WTP for feature X = $-\beta_X / \beta_{Price}$ (in $ per feature level). Great report material.

## Mixed logit (the upgrade that matters for the competition)

Let coefficients vary across people: $\beta_i \sim N(\mu, \Sigma)$. Fit with simulated ML (Halton draws; `mlogit(..., rpar=, panel=TRUE)` — panel because each person answers 19 tasks).
- Breaks IIA, captures **taste heterogeneity**.
- For a **new** person (our whole test set!) predict the population-averaged probability $\int P(\text{choice}|\beta) f(\beta) d\beta$ — flatter, better-calibrated probabilities than MNL → lower logloss.
- Our fit found sd(Price-coef) ≈ 0.47 and sd(none-ASC) ≈ 2.07 — people differ *hugely* in opt-out propensity. See [[Modeling Strategy & Results]].

## In our pipeline

- `model/02_mnl.R` — MNL + Price×(age, income, ppark, segment, region) interactions.
- `model/03_mixl.R` — mixed logit, random Price + none-ASC, unconditional simulation predictions.
- Trees (`model/04_xgb_long.R`) approximate the same comparisons via *engineered* cross-alternative features (price rank, price − cheapest rival) since they don't share parameters across alternatives natively.
