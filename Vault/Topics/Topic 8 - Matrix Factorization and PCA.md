---
title: Topic 8 — Matrix Factorization and PCA
type: topic
tags: [course, topic-8]
updated: 2026-07-26
---

# Topic 8 — Matrix Factorization, SVD & PCA

Sources: [[Raw Dump/Topic 8/Lecture Slides_MatrixFactorization(3).pdf|lecture slides]] · `ImagecompressionSVD(2).Rmd`, `SocialProgressIndexwithPCA(1).Rmd` · `socprog2020.csv`

- **SVD**: $X = U\Sigma V'$; truncating to top-k singular values = best rank-k approximation → image compression demo; also the engine behind latent-factor recommenders (ties to [[Topic 7 - Clustering and RecSys]]).
- **PCA** (`prcomp`, scale!): principal components = directions of max variance; scree plots, loadings, biplots; Social Progress Index case: composite country indices from correlated indicators.
- Uses: dimensionality reduction before regression/clustering; multicollinearity fix; visualization.
- Competition angle: the 19 feature-level columns are low-cardinality and few — no dimensionality pressure, so PCA is not on our ideas queue.
