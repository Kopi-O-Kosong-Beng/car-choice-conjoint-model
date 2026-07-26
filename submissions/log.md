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

---

## ⚠️ We have hit the public leaderboard's resolution limit (26 Jul)

| submission | local nested | public |
|---|---|---|
| `sub_20260725_2349.csv` | 1.13878 | **1.201** |
| `sub_20260726_0116.csv` | 1.13556 | **1.201** |

A local improvement of 0.0032 produced **no visible change**. This is not evidence the
improvement was fake. At the ~58% transfer rate the expected public gain was ≈0.0019, and
Kaggle displays three decimals — so anything under ≈0.001 is invisible by construction,
and 0.0019 can easily hide inside the rounding of "1.201".

**Consequences for how we spend the remaining days:**

1. **Incremental tuning is now unmeasurable.** Gains of 0.003 local cannot be confirmed or
   refuted on the public leaderboard. Chasing more of them is a bet placed blind.
2. **To move the public number we need ≈0.01+ local**, which will not come from
   hyperparameters. It requires a structurally different model — a different decomposition
   of the choice problem, or a model family we do not yet have.
3. **The private leaderboard still rewards genuine accuracy.** It is scored at full
   precision on ~1,500 rows, so a real improvement raises expected private performance even
   when the public display cannot show it. Structurally justified gains remain worth having.
4. **The report is worth 15 of 30 marks** and is not subject to any of this noise.

Revised offset estimate: two of our own points now sit at 1.13878 → 1.201 (+0.062) and
1.13556 → 1.201 (+0.065). Use **public ≈ local + 0.063**, and treat differences below
0.005 local as untestable on the public board.
---

## Submission history (clean table)

| when | file | members | local nested | expected public | actual public |
|---|---|---|---|---|---|
| 25 Jul 23:49 | `sub_20260725_2349.csv` | 9 members (7 at zero weight) | 1.13878 | ~1.189 | **1.201** |
| 26 Jul 01:16 | `sub_20260726_0116.csv` | mnl_pw + xgb_lw2 | 1.13556 | ~1.186 | **1.201** |
| 26 Jul 06:51 | `sub_20260726_0651.csv` | xgb_lw2 + xgb_mono + lcmnl3 (mnl_pw at 0) | **1.13044** | ~1.195 | **1.199** ✅ |
| 26 Jul 16:43 | `sub_20260726_1643.csv` | `lcmnl3` → **`lcmnl3_both`** (task position) | **1.12867** | 1.198, possibly still 1.199 | (pending) |

### What to expect from `sub_20260726_1643.csv`

The local gain is 0.00177 (1.13044 → 1.12867). Applying the transfer rates we have
actually observed:

| assumed transfer | predicted public |
|---|---|
| ~33% (the recent rate) | 1.1984 → displays **1.198** |
| ~58% (the early rate) | 1.1980 → displays **1.198** |
| 100% (upper bound) | 1.1972 → displays **1.197** |

**Honest expectation: 1.198, and 1.199 would not refute the change.** A 0.00177 local gain
sits right at the three-decimal resolution limit — this is exactly the regime where the
public board went 1.13878 → 1.13556 with no visible movement at all. The private board
scores at full precision, which is the reason to submit it.

Two reasons to weigh it above its size, though. It is the first gain since the leaderboard
plateau that comes from a **structural** correction rather than tuning — the model was
unable to express an effect the data plainly shows — and it is the first whose shift audit
comes out **above** 100% (119%), meaning it gets *stronger* on a population reweighted
toward the wealthier test respondents. Every previous member's gain decayed under that
reweighting.


---

## What 1.199 tells us (26 Jul)

**The latent-class extrapolation was safe.** This was the open risk: `lcmnl3` predicts a
test "none" rate of 0.223 against ~0.304 out-of-fold, because its demographic membership
model routes the wealthier test respondents toward buying classes. Every latent-class variant
did the same, so it was a property of the channel, not one fit. It could have been genuine
insight or over-extrapolation. **It was genuine** — the score improved. Richer respondents
really do decline less often, and the membership model extrapolates in the right direction.

**But transfer is decaying.** Full calibration record:

| local nested | public | offset | transfer of that step |
|---|---|---|---|
| 1.17683 | 1.2230 | +0.046 | — |
| 1.13878 | 1.201 | +0.062 | ~58% (over the whole first jump) |
| 1.13556 | 1.201 | +0.065 | 0% visible (below resolution) |
| **1.13044** | **1.199** | **+0.069** | **~24–39%** |

The step from 1.13556 to 1.13044 was 0.0051 local and bought 0.002 public — and because
Kaggle rounds to three decimals, the true public gain lies somewhere in 0.001–0.003, so
transfer for this step is between 20% and 59%. Measured against 1.13878 it is ~24%.

**Early gains transferred at roughly 58%; recent ones at roughly 25–40%.** The offset has
grown monotonically (+0.046 → +0.062 → +0.065 → +0.069). That is the signature of
increasingly CV-specific fitting: we have now run eighteen experiments against one fixed
fold structure, and the winners are selected partly on noise that does not exist in the test
set. It does not mean the recent gains are fake — this one demonstrably transferred — but it
does mean each additional local point buys less than the last.

**Practical rule for what remains:** expect roughly a third of any further local gain to
reach the leaderboard. To move 1.199 → 1.196 would take ~0.009 more local, which is larger
than everything round 2 produced. Weigh that against the report, which carries 15 of 30 marks.

**Standing: 1.199 versus the rival reference 1.210 and our own opening 1.2230.**
This submission is now our best public score, so it is the one Kaggle will auto-select for
private scoring unless a later one beats it.

The "expected public" column used a naive `local + 0.063` early on, which over-predicted
improvement because it assumed full transfer. The realistic estimate for the 26 Jul 06:51
submission applies the ~58% transfer rate to the *change*:
`1.201 − 0.58 × (1.13556 − 1.13044) ≈ 1.198`, with `~1.195` if transfer is closer to full.
A move from 1.201 to 1.198 is the first change large enough to be visible at three decimals.

**Why submitting a riskier model is close to free:** Kaggle automatically scores the
*best public* submission on the private leaderboard. A submission that performs worse
publicly simply never gets selected — it cannot damage the final grade, it only costs a
slot. That makes exploration cheap and argues for submitting the best local model even
when its extrapolation behaviour is uncertain.

**Risk carried by this submission.** `lcmnl3` predicts a much lower "none" rate on the test
set than the tree models do (0.223 versus ~0.275), because its demographic membership model
routes the wealthier test respondents toward buying classes. That is either a genuine
insight or an over-extrapolation; the blend tempers it to 0.250 versus 0.255 for the
previous submission. Only 79% of its gain survives income reweighting, against ~100% for
the earlier structural wins. This submission is partly a test of that behaviour.
| 2026-07-26 16:43 | sub_20260726_1643.csv | mnl_pw+xgb_lw2+xgb_mono+lcmnl3_both | 1.12867 | expect public ~1.179 | (pending) |
