# Topic 2 — Logistic Regression

Sources: [[Raw Dump/Topic 2/logistic_regression(1).pdf|lecture PDF]] · [[Raw Dump/Topic 2/Health Analysis/healthcareanalytics(1).pdf|Framingham case]] · [[Raw Dump/Topic 2/Space Analysis/1784612.pdf|Challenger case]] · `fhs.Rmd`, `challenger.Rmd`

- **Model**: $P(y=1) = \frac{1}{1+e^{-\beta'x}}$ (`glm(..., family=binomial)`). Coefficients = log-odds; $e^\beta$ = odds ratio.
- Evaluation: confusion matrix & threshold choice, ROC / AUC, **calibration** — and logloss, which is exactly our competition metric generalized to 4 classes.
- **Framingham**: 10-year CHD risk from age, smoking, BP, cholesterol — risk scores as decision tools.
- **Challenger O-rings**: P(failure) vs launch temperature; the disaster of extrapolating (31°F far below all test data) and of discarding "no-failure" observations — selection bias.
- Discrete choice ([[Vault/Topics/Topic 3 - Discrete Choice]]) is the multi-alternative generalization: logistic = 2-alternative logit.
