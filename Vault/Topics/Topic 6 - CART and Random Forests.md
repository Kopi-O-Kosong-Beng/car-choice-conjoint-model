# Topic 6 — CART & Random Forests

Sources: [[Raw Dump/Topic 6/Lecture Slides_CART&RF(2).pdf|lecture slides]] · `Cart(1).Rmd`, `Bagging and Random Forests(2).Rmd` · `supreme(1).csv`

- **CART** (`rpart`): recursive binary splits minimizing impurity (Gini/deviance); pruning via complexity parameter `cp` + CV; human-readable rules.
- **Bagging**: bootstrap + average many trees → variance reduction. **Random forest** (`randomForest`): bagging + random feature subsets per split → decorrelated trees; OOB error ≈ free CV; importance plots.
- Supreme Court case: predicting justice votes — interpretability vs accuracy tradeoff.
- Gradient boosting (xgboost — used heavily in [[Vault/Competition/Modeling Strategy & Results|our pipeline]]) is the sequential cousin: trees fit to residuals, learning-rate-shrunk. Trees can't extrapolate — why we pair them with linear-utility choice models given the test income shift.
