# Topic 1 — Linear Regression

Sources: [[Raw Dump/Topic 1/linear_regression (1).pdf|lecture PDF]] · [[Raw Dump/Topic 1/Wine_Analytics/wineanalytics.pdf|wine case PDF]] · [[Raw Dump/Topic 1/Sports Analysis/sportsanalytics.pdf|moneyball case PDF]] · `wine.Rmd`, `moneyball.Rmd`

- **Model**: $y = \beta_0 + \beta'x + \varepsilon$, fit by least squares (`lm`). Read: coefficients, standard errors, t/p-values, $R^2$ vs adjusted $R^2$, residual plots.
- **Wine case (Ashenfelter)**: predicting Bordeaux vintage quality from weather (age, harvest rain, growing-season temp) — regression beating expert tasters; lesson: simple models + right features can beat domain intuition.
- **Moneyball case**: runs scored ~ OBP + SLG (OBP undervalued by scouts); two-stage logic (runs → wins) — lesson: decompose the outcome you actually control.
- Watch-outs: multicollinearity (VIF), out-of-sample validation (train/test split), extrapolation beyond data support — the same extrapolation issue behind our [[Vault/Competition/Data Dictionary|test-set income shift]].
