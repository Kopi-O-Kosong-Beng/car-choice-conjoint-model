# =============================================================================
# ITERATION 30 — HOW MUCH SIGNAL IS LEFT?  (read-only oracle, emits nothing)
#
# THE QUESTION NOBODY HAS ASKED IN THIRTY ITERATIONS. We are at public 1.197,
# first place is 1.186, the uniform benchmark is 1.38629. We keep asking "how do
# we get better" without ever asking "how good is it possible to be".
#
# WHY THAT MATTERS MORE THAN ANY MODEL IDEA. Discrete choice is not image
# classification. A cat is always a cat, so image models can drive logloss toward
# zero. A person facing the same two bundles twice may genuinely choose
# differently -- random utility theory says so explicitly, and the Gumbel error
# term is not measurement noise to be modelled away, it IS the phenomenon. So
# this problem has a hard, non-zero floor, and *that* is why nobody here scores
# 0.3 the way they do on other Kaggle competitions. The interesting question is
# where the floor sits and how much of the distance to it we have already taken.
#
# THE ORACLE. Take our own blend's predictions and grant them information no
# model could ever have for a test respondent: that respondent's OTHER choices.
# For each person, fit a small personal adjustment on 15 of their tasks and score
# the 4 held out. Three parameters, each one an axis the discrete-choice
# literature names explicitly:
#
#     scale  T_i   -- how deterministic this person is (Swait-Louviere scale
#                     heterogeneity, routinely confounded with taste)
#     none   a_i   -- this person's standing propensity to decline
#     price  b_i   -- this person's price sensitivity
#
#     log q_ijt  =  log p_blend(ijt) / T_i  +  a_i * 1[j = none]  +  b_i * Price
#
# Because it STARTS from the blend, the gap it opens is exactly the respondent-
# level heterogeneity our population model cannot reach -- no more, no less.
#
# HOW TO READ THE RESULT.
#   * If the oracle lands near 1.19, we are already at the floor, the entire
#     field is fighting over noise, and no model will produce 0.011.
#   * If it lands near 1.05, there is real heterogeneity, and the question
#     becomes whether ANY of it is reachable from demographics alone -- which the
#     alt-4 probe already suggests it is not, since the wealth channel that every
#     model leaned on turned out to be flat.
#   * Whichever component dominates names the axis worth modelling. If it is
#     scale, that is a genuinely under-exploited direction: our members model
#     taste heterogeneity (latent classes) but NOTHING in the blend models scale
#     heterogeneity, and the two are famously confounded.
#
# This is an UPPER BOUND ON ACHIEVABLE PERFORMANCE, not a model. It cheats by
# construction. It emits nothing and needs no submission slot.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
set.seed(42)

long <- readRDS("model/artifacts/long.rds")
setorder(long, No, alt)
tk <- unique(long[, .(No, Case, is_test, y)], by = "No"); setorder(tk, No)
tr <- tk[is_test == FALSE]; y <- tr$y

memb <- readRDS("model/artifacts/blend.rds")$members
OOF  <- lapply(memb, function(m) {
  x <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(x, No)
  as.matrix(x[, .(p1, p2, p3, p4)])
})
W <- c(0.528, 0.472)
L <- W[1] * log(pmax(OOF[[1]], 1e-12)) + W[2] * log(pmax(OOF[[2]], 1e-12))
P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
cat(sprintf("population blend on training tasks: %.5f   (public 1.197)\n", logloss(y, P)))
cat(sprintf("uniform benchmark: %.5f\n\n", log(4)))

# per-task Price vector and the none indicator, aligned to the task ordering
pr <- matrix(long[is_test == FALSE, Price], ncol = 4, byrow = TRUE)
pr <- scale(pr)[, , drop = FALSE]
none <- matrix(rep(c(0, 0, 0, 1), nrow(P)), ncol = 4, byrow = TRUE)
LP <- log(pmax(P, 1e-12))

adjust <- function(th, rows, use) {
  M <- LP[rows, , drop = FALSE]
  if (use["T"]) M <- M / exp(th[1])
  if (use["a"]) M <- M + th[2] * none[rows, , drop = FALSE]
  if (use["b"]) M <- M + th[3] * pr[rows, , drop = FALSE]
  Q <- exp(M - apply(M, 1, max)); Q / rowSums(Q)
}

run_oracle <- function(use, label) {
  cases <- tr$Case; u <- unique(cases)
  Pout <- matrix(NA_real_, nrow(P), 4)
  np <- sum(use)
  for (cs in u) {
    idx <- which(cases == cs); k <- length(idx)
    grp <- sample(rep_len(1:5, k))
    for (g in 1:5) {
      te <- idx[grp == g]; trn <- idx[grp != g]
      if (!length(te)) next
      o <- optim(c(0, 0, 0), function(th)
                   logloss(y[trn], adjust(th, trn, use)) + 0.02 * sum(th^2),
                 method = "BFGS", control = list(maxit = 60))
      Pout[te, ] <- adjust(o$par, te, use)
    }
  }
  l <- logloss(y, Pout)
  cat(sprintf("  %-34s %d par   %.5f   gain %+.5f\n", label, np, l, logloss(y, P) - l))
  l
}

cat("=== ORACLE: what if we knew the person? (their own held-out tasks) ===\n")
l_T   <- run_oracle(c(T = TRUE,  a = FALSE, b = FALSE), "scale only  (how random they are)")
l_a   <- run_oracle(c(T = FALSE, a = TRUE,  b = FALSE), "none-propensity only")
l_b   <- run_oracle(c(T = FALSE, a = FALSE, b = TRUE),  "price sensitivity only")
l_all <- run_oracle(c(T = TRUE,  a = TRUE,  b = TRUE),  "all three")

base <- logloss(y, P)
cat(sprintf("\n=== THE MAP ===\n"))
cat(sprintf("  uniform benchmark                  %.5f\n", log(4)))
cat(sprintf("  our population model (OOF)         %.5f\n", base))
cat(sprintf("  ORACLE floor (knows the person)    %.5f\n", l_all))
cat(sprintf("\n  signal captured so far:            %.1f%% of (benchmark -> floor)\n",
            100 * (log(4) - base) / (log(4) - l_all)))
cat(sprintf("  respondent heterogeneity we cannot see: %.5f\n", base - l_all))
cat(sprintf("  gap we need to close on the board:      %.5f (1.197 -> 1.186)\n", 0.011))
cat(sprintf("\n  => the prize is %.1f%% of the remaining headroom.\n",
            100 * 0.011 / (base - l_all)))
