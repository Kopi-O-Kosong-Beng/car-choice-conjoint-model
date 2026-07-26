# =============================================================================
# ITERATION 21 — IS THE MODEL RANKING AN ARTEFACT OF ONE FOLD STRUCTURE?
#
# HYPOTHESIS (stated before running)
# ---------------------------------
# Eighteen experiments have been selected against a single seed-42 respondent-grouped
# split. The local->public offset has grown monotonically across those experiments
# (+0.046 -> +0.062 -> +0.065 -> +0.069), which is the signature of selection on
# fold-specific noise rather than on real structure.
#
# If that diagnosis is right, then on an INDEPENDENT respondent grouping:
#   (a) the margins between our models should SHRINK -- in particular xgb_mono's
#       +0.0017 over xgb_lw2 (z = 2.2 on split 42) should weaken or reverse;
#   (b) the blend weights should MOVE materially away from
#       xgb_lw2 0.216 / xgb_mono 0.337 / lcmnl3 0.447;
#   (c) the nested blend OOF should come out materially WORSE than 1.13044,
#       and the gap is a direct estimate of how much of 1.13044 is fold-specific fitting.
#
# The counter-hypothesis is equally informative: if everything is stable, the growing
# local->public offset is POPULATION SHIFT (test respondents are ~2x wealthier), not
# selection bias, and we should stop discounting local gains as overfitting and start
# discounting them as non-transportable.
#
# PAYOFF IF (b) HOLDS. If the weights move a lot, then averaging the weight vectors
# across fold structures is more robust for the private leaderboard than any single
# fold-42-optimal vector, with a clean justification.
#
# WHAT IS HELD FIXED. Everything except the fold assignment. Same features, same
# hyperparameters, same seeds, same objectives, same early-stopping protocol, same
# design-share encoder (which is fold-aware and therefore automatically re-derived
# under the new split -- see model/encode_design.R).
#
# GUARDS
#   * model/artifacts/folds.rds is NEVER read for fitting here and never written.
#     New splits are model/artifacts/folds_b.rds (seed 43) and folds_c.rds (seed 44),
#     built by 00_make_folds.R with the same make_case_folds() as production.
#   * All artifacts get a _b / _c suffix. No production artifact is overwritten.
#   * OOF under a different split is NOT comparable row-for-row to production OOF.
#     Every number is scored within its own split.
#
# HOW TO RUN (each step is a separate process; the xgb ones take ~15-25 min)
#   Rscript experiments/iter21_foldrobust/00_make_folds.R
#   Rscript experiments/iter21_foldrobust/xgb_split.R lw2  b
#   Rscript experiments/iter21_foldrobust/xgb_split.R mono b
#   Rscript experiments/iter21_foldrobust/lcmnl_split.R 3 b
#   Rscript experiments/iter21_foldrobust/mnl_split.R b
#   Rscript experiments/iter21_foldrobust/blend_probe_split.R b mnl_pw_b,xgb_lw2_b,xgb_mono_b,lcmnl3_b
#   Rscript experiments/iter21_foldrobust/analyse.R
#
# This file is the header/index for the iteration; the runnable pieces are the
# scripts named above.
# =============================================================================
cat("iteration 21 driver: see the header. Run the individual scripts listed there.\n")
