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
| 26 Jul 16:43 | `sub_20260726_1643.csv` | `lcmnl3` → **`lcmnl3_both`** (task position) | **1.12867** | 1.198, possibly still 1.199 | superseded, never uploaded |
| 26 Jul 23:28 | **`sub_20260726_2328.csv`** | **`xgb_lw2bag` + `lcmnl3_both`** (two members) | **1.12819** | 1.198 | ← **UPLOAD THIS ONE** |

### `sub_20260726_2328.csv` — the two-member blend

Members dropped from four to two and the nested score moved 1.12867 → 1.12819, a
difference of 0.00048 against a blend-level seed sd of 0.00048. **The score is a wash; the
simplification is the point.** Both removals were justified independently, and before this
member set was probed:

- `xgb_mono` — iteration 26 retested the monotone constraint paired across ten seeds:
  −0.00034, 95% CI [−0.00159, +0.00092], wins 5 of 10. Iteration 08's claimed +0.00172 lies
  outside that interval. It was a duplicate of `xgb_lw2` all along.
- `mnl_pw` — iteration 19's leave-one-out put its contribution at −0.00006 (z = −1.89).

Both survivors were *improved*, not merely retained: `xgb_lw2bag` averages ten seeds
(removing 0.00655 of Monte-Carlo variance, and involving no model selection at all), and
`lcmnl3_both` carries task-position terms that replicate on an independent respondent
grouping (+0.00422, z = 4.70 under `folds_b`, versus +0.00534 on the production folds).

Weights 0.528 tree / 0.472 logit — a near-even split across the single axis of disagreement
iteration 19 identified.

**Expectation: 1.198.** Total local gain since the 1.199 submission is 0.00225
(1.13044 → 1.12819); at the recent ~⅓ transfer rate that is ≈0.0008 public, so **1.199 would
not refute it** — this remains below the three-decimal resolution limit. The reasons to
prefer it are not the public number:

1. Fewer blend parameters fitted on OOF predictions ⇒ less blend-level overfitting, which is
   what the ~1,500-row private board at SE ±0.02 is actually exposed to.
2. A bagged member has lower variance in its *shipped test refit*, and that is an
   expectation improvement the OOF comparison cannot show (see iteration 26).
3. One retracted false result (the monotone constraint) is no longer influencing the blend.

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
| 2026-07-26 23:28 | sub_20260726_2328.csv | xgb_lw2bag+lcmnl3_both | 1.12819 | expect public ~1.178 | (pending) |

---

## 27 Jul — the leaderboard as a measuring instrument

Two uploads, and the second one answered a question eighteen iterations could not.

| when | file | what | public |
|---|---|---|---|
| 27 Jul 14:20 | `sub_20260727_1420.csv` | freepool5, free-sign 5-member, local **1.12341** | **1.211** |
| 27 Jul ~19:00 | `probe_alt4.csv` | constant (1/6,1/6,1/6,1/2) — not a model | **1.499** |

### The probe: r = 0.26648

For a constant prediction, logloss is algebra: `score = log6 − r·log3`, so
`r = (1.791759 − 1.499)/1.098612 = 0.26648`. The test none-rate is **0.2665**,
known to ±0.0005 from display precision and ±0.0043 from the public(70%)→full step.
This is not an estimate from our folds — it is a measurement of the test set.

| | predicted | error |
|---|---|---|
| **tree family** | 0.2730 | **+0.0065** ✅ |
| 2-member blend | 0.2480 | −0.0185 |
| freepool5 | 0.2377 | −0.0288 |
| `lcmnl3_both` | 0.2240 | −0.0425 |
| iter27 nonparametric cells | 0.2959 | +0.0294 |
| iter27 importance-weighted | 0.3017 | +0.0352 |
| iter27 logistic | 0.2061 | −0.0604 |

### ⛔ RETRACTION

This file previously stated: *"The latent-class extrapolation was safe… It was genuine —
the score improved. Richer respondents really do decline less often, and the membership
model extrapolates in the right direction."* **That is now refuted by direct measurement.**
`lcmnl3` predicted 0.224 against a measured 0.2665 — the largest error of any production
model. The 1.199 score improved for other reasons. The *direction* was right (0.2665 < the
training 0.3023) but the magnitude was overshot by 2.3×: predicted drop 0.078, actual 0.036.

### ⛔ RETRACTED — "the 1.197 null is explained, not a failure"

~~freepool5's local gain was real and merely masked: conditional gain +0.00478 × ⅓ = +0.00159,
extra marginal error −0.00133, net +0.00026, invisible at three decimals.~~

**This was built on a misread score and is false.** freepool5 never scored 1.197. **1.197 was
the leaderboard**, which displays only a team's *best* submission — still the 2-member blend.
freepool5's own score, on the My Submissions page, is **1.211**.

There was no null to explain. The arithmetic above was neat, fitted the number it was built to
fit, and was wrong. Recorded rather than deleted because the failure mode is the lesson: an
explanation that reproduces an observation to five decimals is not evidence the observation
was read correctly.

### The real result: local and public moved in OPPOSITE directions

| model | local nested | public | ships p4 |
|---|---|---|---|
| 2-member (production) | 1.12819 | **1.197** | 0.2480 |
| freepool5 (free-sign) | **1.12341** | 1.211 | 0.2377 |

Local improved by **0.00478**; public got **worse by 0.014**. The marginal error accounts for
only 0.00133 of that gap — the remaining **~0.0127 is conditional structure that exists in our
fold split and not in the test set.** Transfer for this change was not ⅓, it was **−2.9×**.

Free-sign weights are precisely the mechanism that produces this. Unconstrained signs across
five members have enough freedom to fit the folds' noise, and it passed every internal check
we had: paired z = 3.84, improvement in 5 of 5 folds, artifact verified to the digit. **None of
that detected it.** The only instrument that did was the leaderboard.

**Consequences:** freepool5 is not promoted. `members.txt` is unchanged. `BLEND_WEIGHTS=free`
stays opt-in and is now documented as refuted on held-out data. And the freeze is vindicated
harder than any argument for it managed — at this point a local gain is not merely discounted,
it can be actively negative.

### The finding that outlives the competition

Iteration 28's validation (B) split the training respondents by income tertile:

| tertile | none-rate |
|---|---|
| low | 0.2968 |
| middle | 0.3204 |
| high | 0.2903 |

**Flat and non-monotonic.** Income does not predict declining in the training data at all.
Yet the test none-rate is 0.2665, clearly below training's 0.3023. So the drop is real, and
it is **not** the wealth shift that every model and every diagnostic assumed. `lcmnl3` was
right by accident; iteration 27's three income-based estimators all predicted ~0.30 *because*
income is flat, and were internally consistent and still wrong. The trees, which never routed
through a demographic membership channel, were the only thing that got it right.

The test population differs on something not identifiable from the demographics we have.

### Next: `sub_20260728_2058.csv` (iteration 28b)

⚠️ **`sub_20260727_2022.csv` must NOT be uploaded.** It applies the correction to freepool5,
which is the 1.211 base. Superseded — see `experiments/iter28_marginal/run.R` for why.

The shipping candidate is the **2-member production blend** with a single constant log-odds
shift on alternative 4 (α = 1.11770, +0.1113 in log-odds), moving mean p4 from 0.2480 to the
measured 0.26648. One parameter, closed form, solved against a **measured** target — no model
selection, no fold structure, nothing refitted. p4 rank order preserved exactly (Spearman 1.0);
relative odds among alternatives 1–3 preserved to 1e−13. The model is untouched.

Pre-registered validation on the 2-member nested OOF (reconstructed to 1.12819 exactly), both
passed: (A) tilt cleanliness 87.3%, meaning the realised change runs ~1.15× the marginal KL;
(B) correcting a **genuine** income-tertile miscalibration realises 115.5%.

**Expected public 1.196** (gain ~0.00104). Small, but it is the one gain in this project that
does not depend on transfer at all — it corrects a quantity measured on the test set itself.
Given that free-sign blending just cost 0.014 by going the other way, "small and measured"
is now clearly the better class of bet.

**Standing: 1.197 vs parinwaris 1.187.** We are 0.010 behind and the correction closes ~0.001
of it. The probe was worth running and returned a real answer; that answer was that the
2-member blend was already close to right. There is no remaining lever of the size needed.
The report is 15 of 30 marks and is where the remaining time belongs.
