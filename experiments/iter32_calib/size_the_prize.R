# =============================================================================
# ITERATION 32 — IS THE MODEL MISCALIBRATED IN *SHAPE*, NOT JUST IN MEAN?
#
# THE STRATEGIC POINT. Everything fitted on our folds transfers at ~37% or worse
# (freepool5 transferred at -290%). The ONE thing that transferred at 100% was
# the alt-4 probe: a quantity MEASURED on the test set itself. We have ~8 slots
# left and the leaderboard is the only unbiased sample of the test distribution
# that exists. No other team is using it as an instrument.
#
# The probe measured one number -- the MEAN of p4 -- and correcting it was worth
# 0.001 because we were only 0.018 out. But a mean is one number. The full
# CALIBRATION CURVE of p4 is a whole function, and there is a strong prior it is
# wrong in shape: iteration 30 measured respondent heterogeneity at sd 0.96 in
# log-odds, so the true spread of decline propensity is enormous, and a model
# that cannot see it will produce p4 that is systematically compressed toward the
# middle -- too high where the truth is low, too low where the truth is high.
# That error is invisible in the mean. It is exactly what a calibration curve
# shows.
#
# THE CAMPAIGN THIS WOULD JUSTIFY. Bin test rows by predicted p4. One probe per
# bin -- constant (1/6,1/6,1/6,1/2) inside the bin, flat 1/4 everywhere else --
# and the returned score solves algebraically for that bin's true decline rate.
# Three probes plus the r we already measured pins a four-bin curve exactly. Then
# apply the measured curve. Like the marginal correction, it is measurement, not
# fitting, so it transfers at ~100%.
#
# THE QUESTION THIS SCRIPT ANSWERS, BEFORE SPENDING A SINGLE SLOT. Is the shape
# error big enough to be worth 3 of our remaining 8 submissions? OOF cannot tell
# us -- the model is calibrated there by construction. So measure calibration on
# an OUT-OF-POPULATION holdout, which is the honest proxy for the test set:
#   * hold out the richer half of respondents (the axis the test set moved on)
#   * hold out a random half (control)
# and compute what PERFECT recalibration would have been worth on each.
#
# DECISION RULE, fixed now: run the probe campaign only if perfect recalibration
# is worth >= 0.004 on the out-of-population holdout. Below that it cannot cover
# the 0.011 we need and the slots are better spent elsewhere.
#
# DIAGNOSTIC -- emits nothing, spends no slot.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R"); set.seed(42)

long <- readRDS("model/artifacts/long.rds")
tk <- unique(long[, .(No, Case, is_test, y)], by = "No"); setorder(tk, No)
tr <- tk[is_test == FALSE]; y <- tr$y; cases <- tr$Case

memb <- readRDS("model/artifacts/blend.rds")$members
OOF <- lapply(memb, function(m) { x <- readRDS(sprintf("model/artifacts/oof_%s.rds", m))
                                  setorder(x, No); as.matrix(x[, .(p1,p2,p3,p4)]) })
L <- 0.528*log(pmax(OOF[[1]],1e-12)) + 0.472*log(pmax(OOF[[2]],1e-12))
P <- exp(L - apply(L,1,max)); P <- P/rowSums(P)
cat(sprintf("blend OOF %.5f | mean p4 %.5f | truth %.5f\n\n",
            logloss(y,P), mean(P[,4]), mean(y==4)))

# recalibrate p4 within bins to the bin's true rate; rescale 1..3 proportionally
recal_p4 <- function(rows, brk) {
  Q <- P[rows, , drop = FALSE]; yy <- y[rows]
  b <- cut(Q[,4], breaks = brk, include.lowest = TRUE, labels = FALSE)
  for (j in sort(unique(b))) {
    i <- which(b == j); tgt <- mean(yy[i] == 4)
    s <- (1 - tgt) / (1 - Q[i,4])
    Q[i,1:3] <- Q[i,1:3] * s; Q[i,4] <- tgt
  }
  Q / rowSums(Q)
}

u <- unique(cases)
dem <- unique(long[, .(Case, incomea)]); inc <- dem[match(u, dem$Case), incomea]
H <- list(rich   = which(cases %in% u[inc > median(inc, na.rm=TRUE)]),
          random = which(cases %in% sample(u, length(u) %/% 2)),
          all    = seq_along(y))

for (h in names(H)) {
  rows <- H[[h]]; Q <- P[rows,,drop=FALSE]; yy <- y[rows]
  cat(sprintf("=== holdout: %s  (n=%d rows) ===\n", h, length(rows)))
  cat(sprintf("  as shipped: %.5f | mean p4 pred %.4f actual %.4f\n",
              logloss(yy, Q), mean(Q[,4]), mean(yy==4)))
  # the shape: predicted vs actual within quintiles of predicted p4
  br <- unique(quantile(Q[,4], seq(0,1,by=0.2)))
  b  <- cut(Q[,4], breaks = br, include.lowest = TRUE, labels = FALSE)
  cat("  quintile   pred p4   actual p4      gap\n")
  for (j in sort(unique(b))) {
    i <- which(b == j)
    cat(sprintf("     %d      %.4f     %.4f    %+.4f\n",
                j, mean(Q[i,4]), mean(yy[i]==4), mean(yy[i]==4) - mean(Q[i,4])))
  }
  m_only <- { s <- (1-mean(yy==4))/(1-Q[,4]); R <- cbind(Q[,1:3]*s, mean(yy==4)); R/rowSums(R) }
  cat(sprintf("  MEAN-only correction (what we already ship): %.5f  gain %+.5f\n",
              logloss(yy, m_only), logloss(yy,Q) - logloss(yy, m_only)))
  for (K in c(4, 5, 10)) {
    brk <- unique(quantile(Q[,4], seq(0, 1, length.out = K + 1)))
    g <- logloss(yy, Q) - logloss(yy, recal_p4(rows, brk))
    cat(sprintf("  FULL %2d-bin curve (needs %d probes):            gain %+.5f\n", K, K-1, g))
  }
  cat("\n")
}
cat("DECISION RULE: run the probe campaign only if the RICH holdout shows >= 0.004.\n")
cat("NOTE: these bin gains are in-sample within the holdout, so they are an\n")
cat("      OPTIMISTIC ceiling -- the real probe measures the same bins with noise.\n")
