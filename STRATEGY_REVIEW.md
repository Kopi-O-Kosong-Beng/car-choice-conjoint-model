# Strategy review — an expert panel reads this project (27 Jul 2026)

*Four readers: a statistician who lives in noise decompositions, a discrete-choice
econometrician, a quant who converts everything into decision value, and a competition
strategist. Feedback first, then the refined process, then the dated endgame plan.
No new modelling is proposed here except one pure-variance measurement.*

**State when written:** nested blend **1.12819** (`xgb_lw2bag` 0.528 + `lcmnl3_both`
0.472), public **1.199** (best), rival reference 1.210, benchmark 1.38629.
`sub_20260726_2328.csv` built, not yet uploaded. Kaggle closes 1 Aug; report due 10 Aug.

---

## Verdict in one paragraph

The methodology is already stronger than the score. Respondent-grouped folds, nested blend
evaluation, paired clustered tests, pre-registered hypotheses, two caught leaks, one
retracted false positive, and replication on an independent fold structure — this is how
professionals work, and almost no student team does it. The remaining risks are not in any
single change; they are (a) one doctrine error about private-leaderboard noise that
understates the value of what you already have, (b) accumulated selection across 26
experiments, which you have now *measured* (≈80% replication) rather than guessed, and
(c) time allocation, because the report is 15 deterministic marks against a private board
whose *ranking* is more resolvable than you think but whose remaining reachable gains are
nearly zero. The winning move from here is defence plus writing, not search.

---

## Part I — What survives expert scrutiny (do not change)

1. **Respondent-grouped, fixed folds.** The single most important design decision in the
   repo. Everything else inherits its validity from this.
2. **The nested blend as the one decision number.** No self-tuned number is ever quoted.
3. **Paired tests with clustered SEs.** Headline differences of 0.002 are meaningless at a
   fold-to-fold SD of 0.013; the pairing is what makes them measurable.
4. **Pre-registration in script headers.** Iteration 17's "modest at best" and iteration
   16's countable claim were both on the record before the compute ran. That is what makes
   the failures publishable instead of embarrassing.
5. **The failure log.** Nine of the strongest findings in this project are negative results
   with numbers attached. The report's insight marks will come disproportionately from
   these.
6. **Structure-first triage.** Iteration 16 was killed by counting (a bijection) before any
   model was fitted. Cheapest experiment in the log; the habit generalises.

---

## Part II — Where your own doctrine is wrong (corrections)

### 1. "Private SE ±0.02 swamps everything" conflates two different noises — the quant's point

The ±0.02 is the sampling SE of one team's **absolute** private score on ~1,500 rows. But
the competition is decided by **ranking**, and every team is scored on the *same* 1,500
rows. Rank differences are a **paired comparison**: the row-difficulty shocks that dominate
the ±0.02 hit every team identically and cancel in the difference. What matters is the SE
of the *loss difference*, driven only by where the teams' predictions disagree.

Fermi estimate with stated assumptions: your own cross-family paired SE (tree vs logit,
maximally different models you possess) is ≈0.0014 at 1,135 respondents / 21,565 tasks.
Scaling to ~1,500 private rows (√(21565/1500) ≈ 3.8×) gives ≈0.005; between two *teams*
(more disagreement than between your own members) call it **0.006–0.012**. Your public
lead over the known rival is 0.011. Under these assumptions that lead is **~1–2 paired SE**
— a probable but not safe private win against that team, order 80%.

**Two implications, one comforting and one not:**
- Genuine local gains buy real private-rank probability — roughly +5pp win-probability per
  +0.002 *true* gain against a nearby team. The doctrine that "nothing matters below 0.02"
  was wrong and would have justified stopping too early.
- Conversely, the gains still have to be *true* (survive the ~0.8 replication factor and
  shift audit), and the search space is measured-exhausted (Part III). So the way to buy
  win probability now is **variance reduction and not blowing up**, not new members.

### 2. The plain nested OOF is the wrong leaderboard predictor

The income-reweighted nested OOF (currently 1.13273 vs plain 1.12819) prices in the one
distribution shift you can measure. The offset story (+0.046 → +0.069) mixes three causes —
population shift, accumulated selection, resolution floor — and the reweighted number
removes the first. Going forward, quote **both** numbers for any candidate submission and
predict public from the reweighted one. (Do *not* re-tune the blend on the reweighted
objective — iteration 07 already refuted that; ESS collapses to 13,689 rows and the fit
loses even on its own metric. Diagnose with it, never optimise on it.)

### 3. Every win needs a shrinkage factor, and you have measured yours

Member-level replication on an independent fold structure: 79%. Blend-level: 81%. So the
standing rule is now empirical, not a guess: **multiply any measured gain by 0.8 before
believing it, and by a further ~⅓ before predicting the public display.** Selection across
26 experiments is real but modest — you looked, you measured, it is ~20%, and iteration 25
survives it comfortably.

### 4. The task-position finding needs its identification defence written down — the econometrician's point

A referee's first objection to "price sensitivity rises with task position" is the classic
scale-vs-taste confound (Swait–Louviere): a rising *scale* λ(task) — respondents becoming
more deterministic — mimics rising sensitivity on every coefficient at once, including the
none-ASC. **You have already run the discriminating test and not noticed its role:** a
per-task-position temperature (= pure scale drift) was worth ~0 honest (d3/d16), while the
price-specific tilt was worth +0.00515 and reproduces 105% of the none-rate drift on its
own. Scale drift is refuted; taste drift in the price dimension is confirmed. Write this
into the report as a deliberate identification argument — it is one paragraph and it is the
difference between "we added a feature" and "we distinguished two mechanisms."

---

## Part III — Gaps the panel found

| # | gap | severity | action |
|---|---|---|---|
| 1 | **EM-start variance of `lcmnl3_both` never measured.** You measured the tree's seed sd (0.00283) and bagged it — but the other half of the blend is an EM fit with screened random starts, and its analogous "start sd" is unknown. By your own logic this should have been measured before trusting any lcmnl margin. | medium | The **one remaining allowed experiment**: refit `lcmnl3_both` under ~5 different screening seeds; if start-sd is material, average them (pure variance reduction, no selection). Otherwise close the question. |
| 2 | Seed-0 anomaly: the production RNG stream makes the monotone constraint look good on *both* fold structures (+0.0017, +0.0021), while 10 explicit seeds say null. Almost certainly one shared lucky stream observed twice, within 1 sd of the paired null. | low | Logged. Optional 25-min check (run seed 0 explicitly through both bagging configs); does not change any decision — the shipped member is the bagged unconstrained tree either way. |
| 3 | `06_blend.R` has no provenance guard — `blend.rds` disagreed with `members.txt` once already. | low, cheap | 5-line chore: refuse to write artifacts if the stored member list ≠ declared list. |
| 4 | You are calibrated against **one** rival score (1.210), but "a lot more people are improving." Decisions are being made against a stale picture of the field. | medium | Zero-cost: next Kaggle visit, record the top-10 public scores and dates into `submissions/log.md`. The defence-vs-offence balance depends on the actual gap above and below you. |
| 5 | Report drafting has not started; report_notes.md is material, not prose. 15 deterministic marks vs ~7 noisy private marks. | high | Part IV timetable. |

Resolved while reviewing: final-submission mechanics are **best-public auto-select**
(Brief & Rules), so a worse-public upload can never hurt the grade — exploration costs a
slot and nothing else. No manual selection decision exists to plan for.

---

## Part IV — The endgame plan

Kaggle closes **1 Aug**; report due **10 Aug**. Two submissions/day, reset ~08:00 SGT →
about **10 slots** remain. You will not use most of them, and that is correct: four
submissions total is why public-overfitting risk is negligible here.

### Phase 0 — today (27 Jul)

1. **Upload `sub_20260726_2328.csv`** (two-member blend, nested 1.12819, reweighted
   1.13273). Pre-registered reading of the result — decided now so the number cannot be
   rationalised after the fact:

   | public shows | interpretation | action |
   |---|---|---|
   | 1.197 | transfer ~full; reweighted predictor validated | freeze, write report |
   | **1.198** | **expected case** (~⅓–½ transfer) | freeze, write report |
   | 1.199 | below resolution; change unconfirmed but not refuted | freeze, write report |
   | ≥1.200 | unexpected — bagged refit or none-rate 0.248 suspect | keep 1.199 incumbent as auto-selected best; investigate before any further upload |

2. Record the current **top-10 public leaderboard** into `submissions/log.md` (gap #4).
3. `git push` — 30+ commits are local-only; a disk failure currently loses the project.

### Phase 1 — 27–28 Jul: close the variance question, then freeze

- Run the **EM-start sensitivity** measurement (gap #1). Decision rule, pre-registered: if
  start-sd ≥ 0.001, average 5 starts into `lcmnl3_bothbag` and swap it in only if the
  nested blend improves under the *same* comparison discipline as iteration 26 (procedure
  vs procedure, not artifact vs artifact); if start-sd < 0.001, do nothing.
- Do the `06_blend.R` provenance guard (gap #3).
- **Then freeze the model.** No new members, no new features, no retuning, regardless of
  what anyone on the leaderboard does. The freeze is what protects the private score: every
  additional selection event spends a little of the ~0.8 replication factor.
- Allowed after the freeze: uploading an already-built artifact; reverting to 1.199's
  incumbent if pre-registered triggers fire. Nothing else.

### Phase 2 — 29 Jul–5 Aug: the report sprint (the 15-mark asset)

Eight pages, mapped to the rubric (model 5 · public-vs-private 2 · insights 5 · quality 3):

| pages | section | key exhibits (already computed, in EXPERIMENTS.md / report_notes.md) |
|---|---|---|
| 0.5 | Problem & data | design structure: 299 designs/position, 98.5% test-design reuse |
| 1.0 | Validation design | respondent grouping rationale; **noise-floor table** (seed sd 0.00283 model / 0.00048 blend / fold SD 0.013); nested weights |
| 1.5 | The model | part-worth coding (+0.020, concave price curve), listwise objective (+0.006), latent class w/ demographic membership, task-position terms, 2-member blend at 0.528/0.472 |
| 1.5 | What failed, and why that is informative | HB cold-start table (0.352 vs 1.229); leakage postmortem (1.09962 → nested double-OOF); the retracted monotone constraint (seed lesson) |
| 2.0 | Insights | 3 buyer segments; discrete-vs-continuous heterogeneity (58 vs 438 params); task-position mechanism **with the scale-vs-taste identification paragraph** (Part II.4); ~89% of none-propensity unobservable, oracle 0.986 |
| 1.0 | Public vs private fit | calibration table (4 points), transfer decay 58%→⅓, offset decomposition, **paired-vs-absolute noise argument** (Part II.1) — this is exactly what rubric item (ii) wants and no other team will have it |
| 0.5 | Limitations & reproducibility | design-specific gains; one-population caveat; run_all.R |

Drafting order: insights first (highest marks per hour, material is ready), then
public-vs-private, then model, then intro last.

### Phase 3 — 6–9 Aug: polish, figures, teammate review

- Two figures maximum: the concave price part-worth curve; the none-rate-by-task-position
  plot (observed vs each model family) — it tells the mechanism story in one image.
- One full read by Sheil for the "quality 3" marks; check the 8-page limit with the
  segment tables in an appendix if the course allows one.

### What is deliberately NOT in the plan

- Importance-weighted *training* (the never-run iter23): expected ≤0.002 before the 0.8
  shrinkage, on an ESS of 13,689; refuted-adjacent (iter07) and it would re-open selection
  after the freeze. Cut.
- Any new model family, encoding, or blend architecture: the search space is
  measured-exhausted (iterations 09, 10, 12, 14–19, 22, 24 all closed with numbers;
  headroom ~89% unobservable).
- Chasing 1.197 public. It needs ~0.009 true local; round 2 + round 3 combined produced
  0.00225. The remaining marks are in the report and in not damaging what exists.

---

## Part V — The working mechanism, distilled (keep this beyond the competition)

1. **Measure the instrument before the effect.** The seed sd (0.00283) invalidated an
   accepted result 18 iterations after the fact. One calibration run first — always.
2. **Pre-register hypothesis *and* decision rule** in the script header before compute.
3. **One change per experiment.** Attribution dies otherwise (iteration 05's lesson).
4. **Paired, clustered comparisons only.** Never compare headline numbers.
5. **Nest everything that is fitted** — weights, temperatures, encodings, baselines. The
   two leakage incidents were both un-nested quantities riding on honest-looking sets.
6. **Replicate accepted changes on a second fold structure** before they enter production.
   `folds_b` exists now; the cost is one refit.
7. **Artifact vs procedure.** A lucky artifact is not a good procedure (the bagging
   estimand lesson). Ask which one the decision needs before running the test.
8. **Shrink every win by the measured replication factor** (here ~0.8), then by measured
   transfer (~⅓) for out-of-distribution predictions.
9. **One artifact, one producing script**, and the consumer refuses on mismatch.
10. **Long jobs write their own assembled outputs** — two "failed" experiments were
    actually finished work nobody assembled.
11. **Log failures with the number that killed them.** The ⛔ table has prevented at least
    three repeated experiments.
12. **Convert gains into decision value before spending time on them.** +0.002 local ≈
    +5pp private-win probability against a nearby team ≈ far less than one report page.

---

## Appendix — assumptions behind the private-board arithmetic

- Private = 30% of 4,997 ≈ 1,500 rows, same rows for all teams, redrawn annually.
- Cross-team prediction disagreement assumed ≥ your own tree-vs-logit disagreement
  (paired SE 0.0014 at full n); scaled by √(21565/1500) ≈ 3.8 → 0.005, inflated to
  0.006–0.012 for inter-team variation. If private rows are sampled per-respondent rather
  than per-row, clustering pushes toward the high end.
- Win probability vs a single team at true gap g: Φ(g / SE_paired). At g = 0.011,
  SE = 0.008 → ≈ 0.83. These are Fermi numbers for prioritisation, not forecasts; their
  role is to show the *ranking* noise is ~2–3× smaller than the ±0.02 absolute-score
  wobble, which changes what is worth doing.
