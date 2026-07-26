# =============================================================================
# What is the EXPECTED blend score if we ship a single-seed member?
#
# WHY compare.R's z = 1.77 IS THE WRONG ESTIMAND HERE.
#
# model/compare.R says xgb_monobag beats the stored xgb_mono by +0.00266 with
# z = 1.77 -- "not distinguishable from noise". That is a true statement about
# those two artifacts and a misleading guide to the decision, because the stored
# xgb_mono is a LUCKY DRAW: its OOF of 1.13980 sits 1.14 seed-sd BELOW the
# 10-seed mean of 1.14303.
#
# The key structural point is that a model's OOF artifact and its TEST artifact
# come from INDEPENDENT random draws:
#   oof_xgb_mono.rds  <- five fold-fits, R seed 123, xgboost seed 0
#   test_xgb_mono.rds <- one refit on all training data, R seed 7, xgboost seed 0
# Getting a lucky OOF tells you nothing about whether the test refit is lucky.
# So the expected quality of the shipped test predictions corresponds to the MEAN
# single-seed performance (1.14303), not to the observed OOF (1.13980).
#
# Bagging changes both sides symmetrically: the bagged OOF averages ten draws of
# the 5-fold procedure and the bagged test artifact averages ten full refits.
#
# THE RIGHT COMPARISON is therefore:
#   E[nested blend | member is ONE random seed]   -- averaged over the 10 seeds
#   vs
#   nested blend with the bagged member
# rather than "lucky incumbent vs bagged".
#
# This script computes the first quantity by rerunning the nested blend ten
# times, once per seed, and reports the spread as well as the mean -- the spread
# is itself the answer to "how much does the decision number move because of a
# choice nobody made deliberately?"
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
y     <- ymap$y
fmap  <- folds[order(No), fold]

getP <- function(nm) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", nm)); setorder(d, No)
  stopifnot(identical(d$No, ymap$No))
  as.matrix(d[, .(p1, p2, p3, p4)])
}
FIXED <- list(mnl_pw = getP("mnl_pw"), xgb_lw2 = getP("xgb_lw2"),
              lcmnl3_both = getP("lcmnl3_both"))

# --- the blend objective, identical to model/06_blend.R -----------------------
nested_blend <- function(Ps) {
  M <- length(Ps); eps0 <- 1e-12
  blend <- function(theta, rows) {
    w <- exp(theta[1:M]); w <- w / sum(w)
    Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
    L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, Ps))
    L <- L / Tt
    P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
    (1 - eA) * P + eA * 0.25
  }
  obj <- function(theta, rows) logloss(y[rows], blend(theta, rows))
  nested <- numeric(5)
  for (k in 1:5) {
    o <- optim(c(rep(0, M), 0, -3), obj, rows = (fmap != k),
               method = "Nelder-Mead", control = list(maxit = 3000))
    nested[k] <- obj(o$par, fmap == k)
  }
  mean(nested)
}

fs <- sort(Sys.glob("experiments/iter26_seedbag/seed_[0-9][0-9][0-9].rds"))
cat("seeds found:", length(fs), "\n\n")

cat("--- nested blend with ONE seed as the third member ---\n")
res <- sapply(fs, function(f) {
  p <- readRDS(f)
  nb <- nested_blend(list(FIXED$mnl_pw, FIXED$xgb_lw2, p$oof, FIXED$lcmnl3_both))
  cat(sprintf("  seed %-3d single %.5f   nested blend %.5f\n",
              p$seed, logloss(y, p$oof), nb))
  nb
})
cat(sprintf("\n  E[nested | one random seed] = %.5f   sd %.5f   min %.5f   max %.5f\n",
            mean(res), sd(res), min(res), max(res)))

cat("\n--- reference points ---\n")
nb_stored <- nested_blend(list(FIXED$mnl_pw, FIXED$xgb_lw2, getP("xgb_mono"), FIXED$lcmnl3_both))
nb_bag    <- nested_blend(list(FIXED$mnl_pw, FIXED$xgb_lw2, getP("xgb_monobag"), FIXED$lcmnl3_both))
cat(sprintf("  stored xgb_mono (a lucky single seed): %.5f\n", nb_stored))
cat(sprintf("  bagged xgb_monobag (10 seeds):         %.5f\n", nb_bag))
cat(sprintf("\n  >>> honest expected gain from bagging = %.5f\n", mean(res) - nb_bag))
cat(sprintf("      (the compare.R contrast against the lucky incumbent, %.5f, understates it)\n",
            nb_stored - nb_bag))
cat(sprintf("\n  How much does the decision number move on seed luck alone? %.5f\n",
            max(res) - min(res)))
cat("  For scale: the fold-to-fold SD of the decision number is 0.013, and the\n")
cat("  entire local gain from iteration 25 was 0.00177.\n")
