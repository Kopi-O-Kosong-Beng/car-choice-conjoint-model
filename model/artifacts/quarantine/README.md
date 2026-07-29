# Quarantined artifacts — DO NOT USE, DO NOT BLEND, DO NOT CITE

Everything here is **contaminated by label leakage**. Kept rather than deleted so the leak
stays reproducible for the report.

## Leaky — the score is fake

| artifact | OOF | why |
|---|---|---|
| `xgb_resenc` | **1.09962** | residual design encoding off a TREE baseline. Iteration 12. Fake. |
| `xgb_resenc2` | 1.13721 | residual encoding off a single-OOF `mnl_pw` baseline. Iteration 15 proved the +0.0043 was **100% leak** (z = -4.54) by rebuilding it on a nested double-OOF baseline: `xgb_resenc3` scores 1.14151 vs 1.14152 with no encoding at all. |
| `xgb_bagRes` | 1.13900 | same leak, bagged, built 26 Jul before the iteration-15 proof landed. |
| `xgb_bagResB` | 1.13776 | same, on the deep config. |

## Not quarantined, and here is why — read before moving anything

**`xgb_lw2bag3` / `xgb_lw2bag3_b`** were briefly quarantined on 27 Jul as "undeclared
provenance" and then **restored — that call was wrong.** They are the deliberate matched
control for iteration 21's fold-robustness check: only 3 seeds were run under `folds_b`, so
a **3-seed** bag on the production folds is required to compare bag-with-bag rather than
3-seed-vs-10-seed. `xgb_lw2bag3` (1.13856) is the seed-42 arm, `xgb_lw2bag3_b` the `folds_b`
arm. They are *supposed* to be worse than the 10-seed `xgb_lw2bag` (1.13682); that is the
point, not a defect. Neither is a blend candidate, and neither belongs here.

The lesson worth keeping: "no producing script" was a **premature** inference drawn while
the producing job was still running and had not yet written its assembly step. Check whether
anything is still running before concluding an artifact is orphaned.

**Note on `xgb_resenc3`:** it stays in `model/artifacts/` and is **not** quarantined. It is
the *honest control* — the nested double-OOF rebuild that proved the leak — and at 1.14151
it is worse than production, so it cannot win any ranking by accident.

**The hazard this creates:** `model/shift_audit.R` scans every `oof_*.rds` and sorts by
plain score, so `xgb_resenc` topped its table on every run. Anyone ranking artifacts by
score finds a fake best model. That is why these live here and not beside production.

Mechanism, for the report: even ~150 global coefficients with no design-level features
absorb enough fold-k choice information to be worth 0.0043 when subtracted. Both leak
detectors *passed* on the honest version and passed *harder* than on the leaky one — the
detectors were right about the sign and useless about the size. Only a nested double-OOF
baseline separated signal from leak.
