# Topic 7 — Clustering & Recommendation Systems

Sources: [[Raw Dump/Topic 7/Lecture Slides_Clustering&RS(2).pdf|lecture slides]] · `Clustering(1).Rmd`, `Recommendation_systems(1).Rmd`, `process_genre_data.R` · MovieLens data (`movies.csv`, `ratings(1).csv`, `genres.csv`)

- **k-means** (`kmeans`): minimize within-cluster SS; choose k via elbow/scree; scale features first. **Hierarchical** (`hclust` + `dist`): dendrograms, linkage choices.
- **RecSys**: content-based (item features) vs **collaborative filtering** (user-user / item-item similarity on the ratings matrix); cold-start problem.
- MovieLens: cluster movies by genre; recommend via cluster membership + ratings.
- Competition angle: could cluster respondents by demographics/choice patterns as segments — but mixed logit ([[Vault/Topics/Topic 3 - Discrete Choice]]) handles heterogeneity continuously, which we found works better than hard segments.
