# Merge plan — how the two tracks combine, and what to do with the remaining days

Written 26 July 2026, after the two workstreams were compared for the first time.
Companion to [`FINDINGS.md`](FINDINGS.md) (what the parallel track found) — this file is
what to *do* about it.

**Base is this repo.** It has fixed shared folds, separable per-model OOF/test artifacts,
`compare.R` clustered paired tests, a nested blend, `members.txt` discipline, and it is
git-tracked and visible to the team. The parallel track was a monolith with no artifact
layer; nothing about its *architecture* is worth migrating. Its value is four findings.

---

## 0. The score facts (verified 26 July, read-only Kaggle CLI — no submission made)

Account `Sheil_Mistry_Team_3` (teamId 16593953) has **three** submissions:

| date (UTC) | file | public |
|---|---|---|
| 2026-07-24 14:00 | submission_best.csv (v1 stacker) | 1.233 |
| 2026-07-24 15:39 | submission_best.csv (v2) | 1.249 |
| **2026-07-25 15:57** | **sub_20260725_2349.csv** | **1.201** |

One account, both members submitting from it. **The team is currently rank 1 on the public
leaderboard at 1.201 — by 0.001.** Two teams sit at 1.202 and four are within 0.004, which
is well inside leaderboard sampling noise (±0.02–0.04 on 263 respondents). The README's
"rival team 1.210" is stale; the field has caught up. Defending the 8 public marks is a
live concern, not a settled one.

**Consequences:**
- The parallel track's v7 (~1.214 est.) and v9 (~1.211 est.) are **worse than what is
  already graded**. They are dead as submission candidates. Do not submit them.
- Two slots were burned within 100 minutes on 24 July on near-identical models.
  **Appoint one submission owner** for the final week.
- Any document claiming the team is graded on 1.233 is out of date — see the correction
  header on `PROJECT_LOG.md` in this folder.

---

## 1. Ports, in priority order

| # | Port | Into | Acceptance test |
|---|---|---|---|
| 1 | **Segment shift audit** | new `model/segment_audit.R`, modelled on `shift_audit.R` | Reproduces from raw CSVs: Small Car 27%→0%, Prestige Luxury Sedan 5%→37%, ESS ≈ 208 of 1,135. Then rescores every existing `oof_*.rds` under segment weights and prints per-segment logloss plus alt-4 observed-vs-predicted for the current blend. **No refitting — minutes on existing artifacts.** |
| 2 | **Tuned listwise hyperparameters** (depth 5, mcw 80, eta 0.04, colsample 0.30, alpha 1, lambda 10) | `experiments/iterNN_lw_retune/` → `oof_xgb_lw2.rds` | `compare.R xgb_lw xgb_lw2` z ≥ 2, **and** wins on ≥3 independently seeded respondent partitions, **and** retention ≈ 100% under both income and segment reweighting, **and** the nested blend OOF improves. Do not blind-swap — see §2. |
| 3 | **Fresh-partition confirmation protocol** | a paragraph in `EXPERIMENTS.md` + a small helper | Any adoption with z between 2 and ~3 must also win on 3–5 independently seeded partitions. This repo has exactly ONE fixed partition with 10+ experiments selected on it; its own adversarial review flagged this and nobody built the fix. The parallel track did. |
| 4 | **MCS / SPA machinery** | `model/mcs.R`, reading `oof_*.rds` | Reproduces a known `compare.R` verdict (e.g. xgb_de→xgb_lw, z=4.3) through the SPA lens. Used twice: to price in the week's search before the final submission set, and as report material. |
| 5 | **Gradient unit test** | `model/tests.R` | The listwise gradient matches numerical differentiation to < 1e-6. **Note:** this repo's hessian is `2·p(1−p)`, the parallel track's is `p(1−p)`. The *gradient* is identical; the 2× is step-size damping absorbed by eta/early stopping. Test the gradient; leave the hessian alone — changing it needs revalidation for zero expected gain. |

---

## 2. Will the tuned config transfer? Directionally, not literally

**First, confirm a duplicate:** this repo's listwise objective and the parallel track's "C2"
are the same thing — both compute a within-task softmax with `grad = p − y`, differing only
in hessian scaling. **C2 must not be ported as a new model.**

The useful corollary: the parallel track's ablations showed the *hyperparameters* were the
active ingredient, and they helped **even under the exact softmax objective**. That is direct
evidence the regularisation result is objective-robust.

Three things still differ here: (i) this repo's feature set adds design-encoding columns and
`price_min_rival`, so `colsample 0.30` samples a different column mix; (ii) this repo uses
early stopping on a 10% respondent holdout, so fixed 800/1200 rounds do not map; (iii) current
params (depth 6, mcw 10, colsample 0.8) are far from the winner. So the transferable claim is
the **direction** — much heavier regularisation — backed by 5/5 fresh partitions,
95% CI [−0.00246, −0.00182].

**Re-validate with a bracketed search under this harness:** 8–12 configs centred on the winner,
depth {5,6} × mcw {30,80} × colsample {0.3,0.5} × lambda {3,10}, keeping early stopping and the
fixed folds. ~15 min/config ≈ one afternoon. Expected ~−0.002 local, ~−0.001 public after the
transfer discount. Small, but the highest-confidence gain available — and it is already item 4
on this repo's own open-ideas list, with the expensive part done.

---

## 3. Segment shift vs income shift — additive, and cheap to prove

They are correlated (luxury respondents are richer) but **cannot be the same phenomenon**, for a
structural reason: this repo's income weights are capped at [0.2, 5]. "Small Car = 27% of train,
0% of test" requires weight **exactly zero** — categorical exclusion, which a capped income tilt
cannot express even in principle. The ESS numbers agree: income reweighting keeps most of the
sample; segment reweighting leaves ~208 of 1,135. The income audit is a mild tilt; the real shift
is a population replacement.

Three tests, all on existing OOF artifacts, ~1–2 hours total, no fitting:

1. **Correlate the two weight vectors** per respondent. If they were the same phenomenon, r ≈ 1
   and retentions would match. The segment weights have mass at exactly 0, so they cannot.
2. **Run the audit both ways** on every blend member. If segment-weighted retention reorders what
   income-weighted retention passed, the two audits measure different things — that is the
   additivity proof. Prime suspect: the **design encoding**, already weakest at 77% retention
   under the mild income tilt, whose empirical shares encode the tastes of a training population
   that is 27% a segment absent from test.
3. **Per-segment calibration of this repo's blend.** The parallel track proved, for its own model
   family, that the two segments making up 69% of test have the worst logloss *and* a systematic
   **+0.06 over-prediction of the outside option** (observed 0.16, predicted 0.22). Nobody has
   checked whether this repo's blend shows the same. **If it does, this is the largest remaining
   gain available** — potentially 0.005–0.015 on the graded population versus ~0.001 for the
   retune — because it improves exactly the rows that are graded.

**If additive**, three pipeline changes: (a) `segment_audit.R` becomes a **veto gate** alongside
the 1-SD rule — mechanism-gated, not significance-gated, since the segment metric only resolves
~0.012; (b) ablate the design-encoding features under the segment metric; (c) if test 3 confirms
the alt-4 bias, one targeted experiment — e.g. a segment-interacted none-constant in `mnl_pw`,
which already has `Price_x_seg` terms but a none-constant that does not vary by segment.
Respect the parallel track's hard-won evidence that segment *importance weights* help MNLs and
**wreck trees**.

---

## 4. Do not port; abandon outright

- **v7 and v9 as submissions** — both dominated by the graded 1.201.
- **The parallel track's C2 softmax model** — duplicate of `03_xgb_listwise.R`. Port only the test.
- **Its shared-utility models as extra blend members** — this repo's blend optimiser already zeroes
  within-family redundancy. Only C1's *hyperparameters* transfer.
- **Its MNL** (1.221) — strictly dominated by `mnl_pw` (1.157). And its wide 4-class xgboost, which
  died independently in both codebases (a nice report sentence in itself).
- **Its lockbox harness** — running two parallel validation systems in one repo is how teams fool
  themselves. Fixed folds + `compare.R` is the system; take only the fresh-partition protocol.
- **Its blend machinery** (geometric pools, bagged EM weights) — the nested log-space blend is
  equivalent or better. Bagged EM is a report footnote.
- **Any post-hoc recalibration** — refuted independently by both codebases (iteration 07 here; the
  rejected T=1.1 there).

---

## 5. Submission schedule (~14 slots left; 2/day, reset ~08:00 SGT)

**Governing logic:** auto-select means *the set of things you submit IS your final-model choice*.
Two consequences — never submit anything you would not accept as the graded model (a lucky public
draw on a weak model hijacks the private score), and since unused slots expire worthless while the
8 public marks reward the max of noisy draws, the endgame should submit **several near-equal,
structurally distinct** variants, each within ~0.003 local of the best.

- **Jul 26:** no submission. Run §3's diagnostics; start the retune. Nothing beats 1.201 yet.
- **Jul 27:** if the retune passes its gates and the reblend beats 1.13883 by > 1 SD, submit
  (expected ~1.199–1.200, which also defends rank against the 1.202s). Hold the second slot.
- **Jul 28–30:** one slot per *adopted* change only — the segment/alt-4 fix if confirmed, then the
  queued residual/bundle design-encoding work. Hold second slots. Log every score immediately.
- **Jul 31 – Aug 1 morning:** endgame. Run the MCS over the week's candidates and submit the 2–4
  members of the surviving indistinguishable set not yet on the board. **Nothing new or
  experimental on Aug 1** — only pre-validated CSVs generated the night before.
- Throughout: explicit consent before every submission; one designated submitter.

---

## 6. The biggest risk nobody is tracking

**Every decision gate in this repo is scored on the wrong population, and the income audit is a
false comfort.** Fixed-fold OOF, `compare.R`, the 1-SD rule, the nested blend weights, even
`shift_audit.R` — all computed on a training population that is 27% a segment absent from test,
while 69% of the graded population comes from segments that are ~9% of training.

This is not theoretical. The parallel track has direct proof: a component (its wide xgboost) that
looked fine on every plain metric was the **worst** model on the graded mix (1.2477 vs 1.1659), and
dropping it was worth −0.0079 there while costing nothing locally.

The sharpest instance here: **the design-encoding features are baked into `xgb_lw` as features, not
a separable blend member.** If their empirical shares — already only 77% income-retention — are net
harmful under the true segment mix, the blend optimiser **cannot remove them**, and the +0.06
outside-option over-prediction on the two dominant test segments would be the visible symptom. This
also cleanly explains the growing local→public offset that is currently attributed to a vague
"harder test set."

Checkable in about two hours on existing artifacts (§3, tests 2–3). That is why it is day-1 work.

**Secondary untracked risks:** single-submitter coordination on one shared account (nearly bitten on
24 July), and stale documentation misinforming whoever works next — the 1.233/1.201 confusion cost a
full planning cycle before it was caught.

---

## Today, in order

1. Update both logs with the resolved score facts; appoint a submission owner.
2. Merge the parallel track's rejected-experiment ledger into `EXPERIMENTS.md` so nobody re-runs it.
3. Run the segment audit + per-segment alt-4 calibration on this repo's blend (§3, tests 2–3).
4. Launch the bracketed retune search overnight (§2).

**No submission until something beats the 1.201 model locally by more than 1 SD.**
