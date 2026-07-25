---
title: Data Dictionary
type: reference
tags: [competition, data]
updated: 2026-07-26
---

# Data Dictionary

Files: `Raw Dump/Competition Data/train2024.csv` (21,565 rows), `test2024.csv` (4,997), `sample_submission2024.csv` (submit on `No` + `Ch1..Ch4`).

## Structure

- **`Case`** — respondent id. Train: 1–1135. Test: 1136–1398 (**263 brand-new people** — nobody overlaps train).
- **`Task`** — choice-task number within respondent, 1–19 (every respondent answers exactly 19).
- **`No`** — global row id (train 1–21565, test 21566–26562). Submission key.
- Each row shows **4 bundles**; exactly one is chosen (`Ch1..Ch4` one-hot; empty in test).

## The 20 bundle attributes (suffix 1–4 per bundle)

`CC GN NS BU FA LD BZ FC FP RP PP KA SC TS NV MA LB AF HU` (safety-feature levels, 0 = absent) + `Price`.
**Bundle 4 is all-zero in every single row → it's the "none of these / keep my car" opt-out option.**

- Train choice shares: bundle1 22.0% · bundle2 25.0% · bundle3 22.7% · **none 30.2%** (largest!)

## Respondent demographics (constant across a person's 19 tasks)

Each in up to 3 encodings: label (`segment`), integer code (`segmentind`), numeric (`agea` etc.):
`segment(5 car classes) · year(2000–2004 wave) · miles/milesa · night/nighta (% night driving) · ppark (parking frequency) · gender · age/agea · educ · region (NE/…) · Urb (urban/suburban/rural) · income/incomea`

## ⚠️ Train → test distribution shift (checked 25 Jul)

| variable | train mean | test mean |
|---|---|---|
| incomea | **$75,473** | **$150,963** |
| incomeind | 5.64 | 8.41 |
| milesa | 200 | 249 |
| educind | 3.46 | 3.13 (more educated) |
| agea | 39.9 | 39.6 (≈same) |

Test respondents are ~2× richer. Consequences: (1) local CV understates leaderboard loss; (2) price-sensitivity×income must extrapolate correctly → we added an explicit `Price×income` term to the choice models; (3) tree models can't extrapolate beyond train income support — blending with linear-in-income models covers that.

Related: [[Modeling Strategy & Results]] · [[Topic 3 - Discrete Choice]]
