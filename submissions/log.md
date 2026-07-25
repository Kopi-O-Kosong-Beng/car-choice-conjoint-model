# Submission log — record the public score here after every Kaggle upload

Known history (from team chat, for offset calibration):

| when | file | members | local OOF | expectation | public |
|---|---|---|---|---|---|
| 2026-07-24 ~22:00 | (Sheil v1) | MNL+xgb+multinom stack | 1.17683 | — | 1.2230 |
| 2026-07-24 ~23:39 | (Sheil, 2nd try) | fable variant | — | — | worse than 1.2230 |
| (rival team ref) | (somesh) | 4-model version | 1.161 | — | 1.210 |
| 2026-07-25 22:04 | sub_20260725_2204.csv | mixl+mnl+xgb_long+xgb_wide | 1.15294 | expect public ~1.203 | (pending) |
| 2026-07-25 22:53 | sub_20260725_2253.csv | glmnet+mixl+mnl+xgb_long+xgb_wide | 1.15304 | expect public ~1.203 | (pending) |
| 2026-07-25 23:26 | sub_20260725_2326.csv | mnl_pw+mnl+glmnet+mixl+xgb_de+xgb_long+xgb_wide | 1.14211 | expect public ~1.192 | (pending) |
| 2026-07-25 23:34 | sub_20260725_2334.csv | mnl_pw+mnl+glmnet+mixl+xgb_lw+xgb_de+xgb_long+xgb_wide | 1.13888 | expect public ~1.189 | (pending) |
| 2026-07-25 23:49 | sub_20260725_2349.csv | mnl_pw+mnl+glmnet+mixl2+mixl+xgb_lw+xgb_de+xgb_long+xgb_wide | 1.13878 | expect public ~1.189 | **1.201** ✅ |

*(rows 2204/2253/2326/2334 were superseded before upload and never submitted — only 2349 went to Kaggle)*

## Calibration — what the two data points tell us

| | local (honest OOF) | public | offset |
|---|---|---|---|
| Sheil v1 (24 Jul) | 1.17683 | 1.2230 | +0.0462 |
| rival "somesh" | 1.161 | 1.210 | +0.0490 |
| **ours (25 Jul)** | **1.13878** | **1.201** | **+0.0622** |

**The offset is not constant — it grows as the local score improves.**
Local improved by 0.0381 (1.17683 → 1.13878); public improved by 0.0220 (1.2230 → 1.201).
**Transfer rate ≈ 58%**: roughly six-tenths of each local gain reaches the leaderboard.

Implications for the remaining days:
1. Local gains are still worth chasing, but discount them by ~40% when predicting the LB.
2. Predicting the next public score: `public ≈ 1.201 − 0.58 × (1.13878 − new_local)`.
3. A growing offset is the signature of mild CV-specific overfitting and/or the test
   population being genuinely harder (richer respondents, thinner training support).
   Prefer changes that are *structurally* justified over changes that only lower OOF.
4. Caveat: the public LB is ~3,500 rows, SE ≈ ±0.013, so the difference between a +0.046
   and a +0.062 offset is not decisively outside noise. Two of our own points is thin
   evidence — treat the 58% as a working estimate, not a law.

**Standing: 1.201 beats the rival reference (1.210) and our own previous best (1.2230).**
| 2026-07-26 01:16 | sub_20260726_0116.csv | mnl_pw+xgb_lw2 | 1.13556 | expect public ~1.186 | (pending) |
