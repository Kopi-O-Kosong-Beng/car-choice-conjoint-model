---
title: Topic 4 — Model Selection
type: topic
tags: [course, topic-4]
updated: 2026-07-26
---

# Topic 4 — Model Selection

Sources: [[Raw Dump/Topic 4/model_selection.pdf|lecture PDF]] · [[Raw Dump/Topic 4/hitters/hitters notebook.pdf|Hitters notebook]] · [[Raw Dump/Topic 4/Economic_Growth/economic growth notebook.pdf|Growth notebook]] · `hitters.Rmd`, `economicgrowth.Rmd`

- Bias–variance tradeoff; train vs test error; **cross-validation** as the honest estimator (our whole pipeline rests on this — grouped by respondent, see [[Modeling Strategy & Results]]).
- Subset selection (best/forward/backward via `leaps`), then **regularization**: ridge (L2) and **lasso** (L1) with `glmnet`; `cv.glmnet` picks λ. Lasso = selection + shrinkage.
- AIC/BIC/adjusted-R² as penalized fit criteria.
- Hitters (salary) & economic growth (many weak predictors, p ≈ n): regularization beats stepwise — the very reason we distrust Sheil's full-data stepwise interactions and select inside CV instead.
