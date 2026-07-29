# Segment-aware evaluation + importance weighting.
#
# THE PROBLEM WE DISCOVERED
# The test set is a completely different market segment mix from training:
#
#   segment                    train%   test%
#   Prestige Luxury Sedan        5.1    36.9
#   Midsize Luxury Utility       4.3    31.9
#   Full-size Pickup            11.3    21.7
#   Midsize Utility             18.6     6.1
#   Midsize Car                 33.7     3.4
#   Small Car                   27.0     0.0   <- absent from test entirely
#
# Luxury respondents behave differently (they reject all bundles ~16% of the
# time vs ~31% for everyone else) and our model is WORST on them
# (val log-loss 1.32 for Prestige Luxury Sedan vs 1.11-1.16 elsewhere).
# Test is 69% luxury; training is 9% luxury.
#
# Consequence: plain validation log-loss (1.152) is measuring the wrong
# population. Reweighting validation to the test segment mix gives 1.2256,
# which matches our actual Kaggle score of 1.212. The "mysterious" gap was
# never noise -- it was segment mix.
#
# THIS FILE PROVIDES
#  1. TEST_SEGMENT_SHARE            -- the test mix
#  2. testmix_log_loss()            -- the metric we should have been using
#  3. segment_importance_weights()  -- training weights that tilt the model
#                                      toward test-like respondents
#  4. oof_predictions()             -- out-of-fold predictions over ALL
#                                      training respondents, so per-segment
#                                      estimates use all 107 luxury
#                                      respondents instead of the ~30 that
#                                      land in any single hold-out.

TEST_SEGMENT_SHARE <- c(
  "Prestige Luxury Sedan"            = 0.3688,
  "Midsize Luxury Utility segements" = 0.3194,
  "Full-size Pickup"                 = 0.2167,
  "Midsize Utility"                  = 0.0608,
  "Midsize Car"                      = 0.0342,
  "Small Car"                        = 0.0000
)

# Weighted log-loss where each segment contributes its TEST share.
testmix_log_loss <- function(y, probs, segment, eps = 1e-15) {
  ll <- -log(pmin(pmax(probs[cbind(seq_along(y), y + 1L)], eps), 1 - eps))
  w <- numeric(length(ll))
  for (s in names(TEST_SEGMENT_SHARE)) {
    idx <- which(segment == s)
    if (length(idx) && TEST_SEGMENT_SHARE[[s]] > 0)
      w[idx] <- TEST_SEGMENT_SHARE[[s]] / length(idx)
  }
  sum(w * ll) / sum(w)
}

# Per-segment breakdown (unweighted within segment).
segment_log_loss <- function(y, probs, segment, eps = 1e-15) {
  ll <- -log(pmin(pmax(probs[cbind(seq_along(y), y + 1L)], eps), 1 - eps))
  tapply(ll, segment, mean)
}

# Training weights = (test share / train share)^alpha.
#   alpha = 0   -> uniform (what we've done so far)
#   alpha = 1   -> full covariate-shift correction
# Small Car has test share 0; `floor_w` keeps a little of its signal rather
# than discarding 27% of the training data outright.
segment_importance_weights <- function(segment, alpha = 1, floor_w = 0.05) {
  train_share <- table(segment) / length(segment)
  w <- numeric(length(segment))
  for (s in names(train_share)) {
    ts <- if (s %in% names(TEST_SEGMENT_SHARE)) TEST_SEGMENT_SHARE[[s]] else 0
    ratio <- max(ts, 1e-9) / as.numeric(train_share[[s]])
    w[segment == s] <- max(ratio^alpha, floor_w)
  }
  w * length(w) / sum(w)     # mean weight 1
}

# Out-of-fold predictions over the WHOLE training set, case-aware.
# Averaged across seeds. Optional per-row training weights.
oof_predictions <- function(X, y, case, seeds, n_folds = 5,
                            params = SENIOR_PARAMS, nrounds = SENIOR_NROUNDS,
                            weights = NULL, verbose = FALSE) {
  acc <- matrix(0, nrow = nrow(X), ncol = 4)
  for (s in seeds) {
    folds <- case_kfold(case, n_folds, s)
    for (i in seq_along(folds)) {
      tr <- folds[[i]]$train_idx; va <- folds[[i]]$val_idx
      dtr <- if (is.null(weights))
        xgb.DMatrix(data = X[tr, , drop = FALSE], label = y[tr])
      else
        xgb.DMatrix(data = X[tr, , drop = FALSE], label = y[tr], weight = weights[tr])
      m <- xgb.train(params = params, data = dtr, nrounds = nrounds, verbose = 0)
      pr <- predict(m, xgb.DMatrix(data = X[va, , drop = FALSE]))
      if (!is.matrix(pr)) pr <- matrix(pr, ncol = 4, byrow = TRUE)
      acc[va, ] <- acc[va, ] + pr
      if (verbose) cat("    seed", s, "fold", i, "\n")
    }
  }
  acc / length(seeds)
}
