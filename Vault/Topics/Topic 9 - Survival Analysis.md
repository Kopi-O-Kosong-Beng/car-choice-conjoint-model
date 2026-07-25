# Topic 9 — Censored Data & Survival Analysis

Sources: [[Raw Dump/Topic 9/Lecture Slides_CensoredDataAnalysis.pdf|lecture slides]] · `1DataCensoring.Rmd`, `2SurvivalAnalyses.Rmd` · `heart.csv`, `extramarital.csv`

- **Censoring**: outcome only partially observed (right-censored durations; Tobit-style censored regression for bounded outcomes). Ignoring it biases everything.
- **Kaplan–Meier** survival curves (`survival::survfit`), log-rank tests; **Cox proportional hazards** (`coxph`): hazard ratios, no baseline distribution assumed; parametric alternatives (`survreg`).
- Stanford heart transplant & extramarital data as cases.
- Not directly used in the competition, but the likelihood-with-partial-information mindset mirrors how the mixed logit integrates over unobserved tastes ([[Vault/Topics/Topic 3 - Discrete Choice]]).
