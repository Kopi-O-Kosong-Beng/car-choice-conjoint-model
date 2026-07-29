# =============================================================================
# ITERATION 57 -- PART C: interrogate the ONE thing that was not a flat zero.
#
# B1 found that a FIXED ceiling at 0.65-0.75 is worth +0.00008 to +0.00015, and
# that it repeats on the fr10 base and on an INDEPENDENT respondent grouping.
# Part A found that CHOOSING that ceiling per fold loses (-0.00007). Both can be
# true: the effect is real and smaller than the cost of estimating where it is.
#
# This part decides which, with:
#   C1  paired respondent-clustered z, and the per-fold breakdown (leak signature:
#       a real gain appears in EVERY fold, a leak concentrates in one).
#   C2  per-alternative reliability -- is the top-end overconfidence a property of
#       the none option (which the probe can anchor) or of the buy options (which
#       it cannot)?
#   C3  RESTRICTED isotonic: fit the monotone map only above a threshold, identity
#       below. This interpolates between "ceiling clip" (maximally constrained) and
#       "full isotonic" (unconstrained, already refuted at -0.00155) and finds the
#       best honestly-nested member of the whole monotone class.
#   C4  what the surviving transform would do to the shipped test predictions, and
#       whether it fights the anchored none-margin correction.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
DIR <- "experiments/iter57_deploytrim"
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

pk <- readRDS(file.path(DIR, "dt57_pack_prod.rds"))
P0 <- pk$Poof; y <- pk$y; fold <- pk$fold; Case <- pk$Case; BASE <- pk$nested
Pt <- pk$Ptest; n <- length(y)
nrm <- function(P) P / rowSums(P)
liv <- function(P) -log(pmax(P[cbind(seq_len(nrow(P)), y)], 1e-15))
pair <- function(la, lb, B = 3000) {           # positive = b better
  d <- la - lb; u <- unique(Case)
  num <- tapply(d, Case, sum); den <- tapply(rep(1, n), Case, sum)
  set.seed(57)
  bs <- replicate(B, { s <- sample(u, length(u), TRUE)
                       sum(num[as.character(s)]) / sum(den[as.character(s)]) })
  c(gain = mean(d), se = sd(bs), z = mean(d) / sd(bs))
}
l0 <- liv(P0)

rule("C1 -- IS THE CEILING GAIN REAL? paired z and the per-fold signature")
cat(sprintf("%-8s %9s %10s %8s %s\n", "ceiling", "gain", "se", "z", "per-fold gain"))
for (ce in c(0.85, 0.80, 0.75, 0.70, 0.65, 0.60)) {
  l1 <- liv(nrm(pmin(P0, ce))); st <- pair(l0, l1)
  pf <- sapply(1:5, function(k) mean(l0[fold == k]) - mean(l1[fold == k]))
  cat(sprintf("%-8.2f %+9.5f %10.5f %+8.2f  %s   (%d of 5 positive)\n",
              ce, st["gain"], st["se"], st["z"],
              paste(sprintf("%+.5f", pf), collapse = " "), sum(pf > 0)))
}
cat("\nscale for these numbers:\n")
cat("  blend-level seed sd 0.00048 | model-level seed sd 0.00283 | fold sd 0.013\n")
cat("  pre-registered adoption gate for a LOCALLY FITTED transform: 0.00100\n")

rule("C2 -- PER-ALTERNATIVE RELIABILITY at the top of the range")
Z <- matrix(0, n, 4); Z[cbind(seq_len(n), y)] <- 1
for (j in 1:4) {
  p <- P0[, j]; z <- Z[, j]
  br <- c(0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1)
  b <- cut(p, br, include.lowest = TRUE)
  d <- data.table(bin = levels(b), n = as.integer(table(b)),
                  pred = as.numeric(tapply(p, b, mean)),
                  obs  = as.numeric(tapply(z, b, mean)))
  d[, resid := obs - pred]
  cat(sprintf("\n  alternative %d%s\n", j, if (j == 4) { "  (the 'none' option)" } else { "" }))
  print(d[n > 30], digits = 4, row.names = FALSE)
}

rule("C3 -- RESTRICTED ISOTONIC: best honestly-nested member of the monotone class")
cat("  map acts only on cells above the threshold; identity below.\n")
cat("  thr=0 is the full isotonic map (refuted); thr high -> ceiling clip.\n\n")
iso_fit <- function(p, z) { o <- order(p); ir <- isoreg(p[o], z[o])
  k <- !duplicated(ir$x, fromLast = TRUE); approxfun(ir$x[k], ir$yf[k], rule = 2) }
cat(sprintf("  %-6s %10s %10s %8s %s\n", "thr", "nested", "gain", "z", "per-fold"))
for (thr in c(0, 0.20, 0.35, 0.50, 0.60, 0.70, 0.80)) {
  Pn <- matrix(NA_real_, n, 4)
  for (k in 1:5) {
    tr <- which(fold != k); te <- which(fold == k)
    Ztr <- matrix(0, length(tr), 4); Ztr[cbind(seq_along(tr), y[tr])] <- 1
    sel <- as.vector(P0[tr, ]) >= thr
    g <- iso_fit(as.vector(P0[tr, ])[sel], as.vector(Ztr)[sel])
    v <- as.vector(P0[te, ]); w <- v
    hit <- v >= thr; w[hit] <- g(v[hit])
    Q <- pmax(matrix(w, length(te), 4), 1e-6); Pn[te, ] <- Q / rowSums(Q)
  }
  l1 <- liv(Pn); st <- pair(l0, l1)
  pf <- sapply(1:5, function(k) mean(l0[fold == k]) - mean(l1[fold == k]))
  cat(sprintf("  %-6.2f %10.5f %+10.5f %+8.2f  %s\n", thr, mean(l1), st["gain"], st["z"],
              paste(sprintf("%+.5f", pf), collapse = " ")))
}

rule("C4 -- THE SURVIVING TRANSFORM ON THE SHIPPED TEST PREDICTIONS")
R <- (1.7918 - 1.499) / 1.0986
ent <- function(M) mean(-rowSums(M * log(pmax(M, 1e-15))))
apply_delta <- function(P, d) { L <- log(pmax(P, 1e-15)); L[, 4] <- L[, 4] + d
                                E <- exp(L - apply(L, 1, max)); E / rowSums(E) }
solve_delta <- function(P, tgt) uniroot(function(d) mean(apply_delta(P, d)[, 4]) - tgt,
                                        c(-5, 5), tol = 1e-12)$root
kl <- function(r, q) r * log(r / q) + (1 - r) * log((1 - r) / (1 - q))
cat(sprintf("  measured test none-rate %.5f | shipped %.5f\n\n", R, mean(Pt[, 4])))
cat(sprintf("  %-24s %9s %9s %9s %9s\n", "variant", "cells hit", "ship p4", "max p", "entropy"))
cat(sprintf("  %-24s %9s %9.5f %9.4f %9.5f\n", "shipped (none)", "-", mean(Pt[, 4]),
            max(Pt), ent(Pt)))
for (ce in c(0.80, 0.75, 0.70, 0.65)) {
  Q <- nrm(pmin(Pt, ce))
  cat(sprintf("  %-24s %9d %9.5f %9.4f %9.5f\n", sprintf("ceiling %.2f", ce),
              sum(Pt > ce), mean(Q[, 4]), max(Q), ent(Q)))
}
cat("\n  interaction with the ANCHORED none-margin shift (order matters?):\n")
d0 <- solve_delta(Pt, R)
A  <- apply_delta(Pt, d0)                       # anchored only
B1 <- nrm(pmin(A, 0.70))                        # anchor then clip
B2 <- apply_delta(nrm(pmin(Pt, 0.70)), solve_delta(nrm(pmin(Pt, 0.70)), R))  # clip then anchor
cat(sprintf("    anchored only        ship p4 %.5f  margin gain %+.5f\n",
            mean(A[, 4]), kl(R, mean(Pt[,4])) - kl(R, mean(A[,4]))))
cat(sprintf("    anchor -> clip 0.70  ship p4 %.5f  (target now MISSED by %+.5f)\n",
            mean(B1[, 4]), mean(B1[, 4]) - R))
cat(sprintf("    clip 0.70 -> anchor  ship p4 %.5f  (target hit, %d cells clipped)\n",
            mean(B2[, 4]), sum(Pt > 0.70)))
cat("\n  A ceiling clip MOVES the none margin, so it cannot be applied after the\n")
cat("  anchored shift without breaking the one guarantee we actually have.\n")
cat("  Correct order is clip THEN anchor -- which is another reason not to bother\n")
cat("  for a gain of 0.0001.\n")
cat("\nOK\n")
