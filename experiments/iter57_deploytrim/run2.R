# =============================================================================
# ITERATION 57 -- PART B
#
# Part A closed the class with an isotonic upper bound. Part B does four things
# the upper bound alone does not:
#   1. FIXED-VALUE CURVES. A clip at a FIXED floor/ceiling has zero fitted
#      parameters, so evaluating it on the nested OOF is honest at every grid
#      point (only picking the argmin across the grid is biased). The whole
#      curve is the direct answer to "does trimming help", with no nesting
#      argument required at all.
#   2. THE MECHANICAL REASON. Count how much probability mass and how much
#      logloss actually lives in the region a clip could touch.
#   3. REPLICATION on the fr10 base and on an INDEPENDENT respondent grouping
#      (folds_b, seed 43) -- CLAUDE.md step 4.
#   4. The "2 system" reading: arithmetic vs log-opinion pooling at the
#      production weights, zero fitted parameters.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
DIR <- "experiments/iter57_deploytrim"
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

pk <- readRDS(file.path(DIR, "dt57_pack_prod.rds"))
P0 <- pk$Poof; y <- pk$y; fold <- pk$fold; Case <- pk$Case; BASE <- pk$nested
n  <- length(y)
nrm <- function(P) P / rowSums(P)
LL  <- function(P) -mean(log(pmax(P[cbind(seq_len(nrow(P)), y)], 1e-15)))

rule("B1 -- FIXED CLIP CURVES  (zero fitted parameters: every point is honest)")
cat(sprintf("baseline nested OOF %.5f | OOF P range [%.5f, %.5f]\n",
            BASE, min(P0), max(P0)))
cat(sprintf("the blend's own uniform-eps term is %.5f, i.e. a HARD FLOOR of %.5f\n\n",
            0.00511, 0.00511 / 4))

cat("FLOOR  (clip up, then renormalise)\n")
cat(sprintf("  %-9s %9s %10s %10s\n", "floor", "cells hit", "nested", "delta"))
for (f in c(0.001, 0.002, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08)) {
  Q <- nrm(pmax(P0, f)); v <- LL(Q)
  cat(sprintf("  %-9.4f %9d %10.5f %+10.5f\n", f, sum(P0 < f), v, BASE - v))
}
cat("\nCEILING  (winsorise down, then renormalise)\n")
cat(sprintf("  %-9s %9s %10s %10s\n", "ceiling", "cells hit", "nested", "delta"))
for (ce in c(0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50)) {
  Q <- nrm(pmin(P0, ce)); v <- LL(Q)
  cat(sprintf("  %-9.4f %9d %10.5f %+10.5f\n", ce, sum(P0 > ce), v, BASE - v))
}
cat("\nEXTRA UNIFORM MIX   P' = (1-a)P + a/4\n")
for (a in c(0.005, 0.01, 0.02, 0.05, 0.10, 0.20)) {
  v <- LL((1 - a) * P0 + a * 0.25); cat(sprintf("  a=%-6.3f %10.5f %+10.5f\n", a, v, BASE - v))
}
cat("\nTEMPERATURE   P' propto P^(1/T)\n")
for (Tt in c(0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20)) {
  L <- log(pmax(P0, 1e-15)) / Tt; E <- exp(L - pmax(L[,1],L[,2],L[,3],L[,4]))
  v <- LL(nrm(E)); cat(sprintf("  T=%-6.2f %10.5f %+10.5f\n", Tt, v, BASE - v))
}
MG <- matrix(as.numeric(prop.table(table(y))), n, 4, byrow = TRUE)
cat("\nSHRINK TOWARD THE TRAIN MARGINAL   P' = (1-a)P + a*m\n")
for (a in c(0.01, 0.02, 0.05, 0.10, 0.20, 0.40)) {
  v <- LL((1 - a) * P0 + a * MG); cat(sprintf("  a=%-6.3f %10.5f %+10.5f\n", a, v, BASE - v))
}

rule("B2 -- WHY THE CLIP CURVES ARE FLAT: there is nothing in the tails")
py <- P0[cbind(seq_len(n), y)]; l <- -log(py)
cells <- as.vector(P0)
cat(sprintf("  86,260 (row,alt) cells.  below 0.005: %d   below 0.01: %d   below 0.02: %d\n",
            sum(cells < 0.005), sum(cells < 0.01), sum(cells < 0.02)))
cat(sprintf("                          above 0.80 : %d   above 0.85: %d   above 0.90: %d\n",
            sum(cells > 0.80), sum(cells > 0.85), sum(cells > 0.90)))
cat(sprintf("  rows whose CHOSEN alternative got < 0.02: %d of %d, carrying %.2f%% of total loss\n",
            sum(py < 0.02), n, 100 * sum(l[py < 0.02]) / sum(l)))
cat(sprintf("  rows whose CHOSEN alternative got > 0.80: %d of %d, carrying %.2f%% of total loss\n",
            sum(py > 0.80), n, 100 * sum(l[py > 0.80]) / sum(l)))
cat("\n  ORACLE BOUND on trimming the low tail: replace every chosen-prob below t by\n")
cat("  the value that a clairvoyant would choose. Even with perfect hindsight:\n")
for (t in c(0.01, 0.02, 0.05)) {
  idx <- which(py < t)
  best <- sum(l) - sum(l[idx]) + length(idx) * (-log(t))   # cannot do better than t
  cat(sprintf("    t=%.2f  %d rows  oracle nested >= %.5f  (max possible gain %+.5f)\n",
              t, length(idx), best / n, BASE - best / n))
}

rule("B3 -- THE 'TWO SYSTEM' READING: arithmetic vs log-opinion pool, 0 fitted par")
memb <- c("xgb_lw2bag", "lcmnl3_both")
Ms <- lapply(memb, function(m) { d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m))
                                 setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
w <- c(0.528, 0.472)
lin <- nrm(w[1] * Ms[[1]] + w[2] * Ms[[2]])
cat(sprintf("  member  %-13s plain OOF %.5f\n", memb[1], LL(Ms[[1]])))
cat(sprintf("  member  %-13s plain OOF %.5f\n", memb[2], LL(Ms[[2]])))
cat(sprintf("  ARITHMETIC pool at production weights, plain OOF   %.5f\n", LL(lin)))
Lg <- w[1] * log(pmax(Ms[[1]], 1e-12)) + w[2] * log(pmax(Ms[[2]], 1e-12))
gp <- nrm(exp(Lg - pmax(Lg[,1],Lg[,2],Lg[,3],Lg[,4])))
cat(sprintf("  LOG-OPINION pool at production weights, plain OOF  %.5f\n", LL(gp)))
cat(sprintf("  (nested production number, for scale: %.5f)\n", BASE))
cat("\n  A trimmed/arithmetic mean of the two systems is BLUNTER than the log pool.\n")
cat("  Whichever wins here is the same question the temperature curve answers.\n")

rule("B4 -- REPLICATION 1: the fr10 base (the best built-but-unsent blend)")
p2 <- readRDS(file.path(DIR, "dt57_pack_fr10.rds"))
Q0 <- p2$Poof; y2 <- p2$y; f2 <- p2$fold; B2 <- p2$nested
LL2 <- function(P) -mean(log(pmax(P[cbind(seq_len(nrow(P)), y2)], 1e-15)))
cat(sprintf("  baseline nested %.5f\n", B2))
for (f in c(0.005, 0.01, 0.02)) cat(sprintf("  floor %.3f   %+.5f\n", f, B2 - LL2(nrm(pmax(Q0, f)))))
for (ce in c(0.85, 0.75, 0.65)) cat(sprintf("  ceil  %.3f   %+.5f\n", ce, B2 - LL2(nrm(pmin(Q0, ce)))))
for (a in c(0.01, 0.05)) cat(sprintf("  unif  %.3f   %+.5f\n", a, B2 - LL2((1-a)*Q0 + a*0.25)))

iso_fit <- function(p, z) { o <- order(p); ir <- isoreg(p[o], z[o])
  k <- !duplicated(ir$x, fromLast = TRUE); approxfun(ir$x[k], ir$yf[k], rule = 2) }
iso_nested <- function(P, yy, ff, per_alt) {
  N <- nrow(P); Pn <- matrix(NA_real_, N, 4)
  for (k in sort(unique(ff))) {
    tr <- which(ff != k); te <- which(ff == k)
    Z <- matrix(0, length(tr), 4); Z[cbind(seq_along(tr), yy[tr])] <- 1
    if (per_alt) {
      gs <- lapply(1:4, function(j) iso_fit(P[tr, j], Z[, j]))
      Q  <- sapply(1:4, function(j) gs[[j]](P[te, j]))
    } else {
      g <- iso_fit(as.vector(P[tr, ]), as.vector(Z))
      Q <- matrix(g(as.vector(P[te, ])), length(te), 4)
    }
    Q <- pmax(Q, 1e-6); Pn[te, ] <- Q / rowSums(Q)
  }
  -mean(log(pmax(Pn[cbind(seq_len(N), yy)], 1e-15)))
}
cat(sprintf("  pooled  isotonic nested %.5f  (%+.5f)\n",
            iso_nested(Q0, y2, f2, FALSE), B2 - iso_nested(Q0, y2, f2, FALSE)))
cat(sprintf("  per-alt isotonic nested %.5f  (%+.5f)\n",
            iso_nested(Q0, y2, f2, TRUE),  B2 - iso_nested(Q0, y2, f2, TRUE)))

rule("B5 -- REPLICATION 2: an INDEPENDENT respondent grouping (folds_b, seed 43)")
fb <- readRDS("model/artifacts/folds_b.rds")
fmb <- fb[order(No), fold]
mb <- c("xgb_lw2bag3_b", "lcmnl3_both_b")
Mb <- lapply(mb, function(m) { d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m))
                               setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
Mn <- length(Mb)
blend_b <- function(theta, rows) {
  ww <- exp(theta[1:Mn]); ww <- ww / sum(ww)
  Tt <- exp(theta[Mn+1]); eA <- plogis(theta[Mn+2]) * 0.10
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], 1e-12)), ww, Mb))
  L <- L / Tt; E <- exp(L - apply(L, 1, max)); P <- E / rowSums(E)
  (1 - eA) * P + eA * 0.25
}
objb <- function(th, rows) logloss(y[rows], blend_b(th, rows))
Pb <- matrix(NA_real_, n, 4); nb <- numeric(5)
for (k in 1:5) {
  o <- optim(c(rep(0, Mn), 0, -3), objb, rows = (fmb != k),
             method = "Nelder-Mead", control = list(maxit = 3000))
  te <- which(fmb == k); Pb[te, ] <- blend_b(o$par, te); nb[k] <- logloss(y[te], Pb[te, ])
}
Bb <- mean(nb)
cat(sprintf("  members %s | nested OOF on folds_b = %.5f\n", paste(mb, collapse = " + "), Bb))
LLb <- function(P) -mean(log(pmax(P[cbind(seq_len(n), y)], 1e-15)))
for (f in c(0.005, 0.01, 0.02)) cat(sprintf("  floor %.3f   %+.5f\n", f, Bb - LLb(nrm(pmax(Pb, f)))))
for (ce in c(0.85, 0.75, 0.65)) cat(sprintf("  ceil  %.3f   %+.5f\n", ce, Bb - LLb(nrm(pmin(Pb, ce)))))
for (a in c(0.01, 0.05)) cat(sprintf("  unif  %.3f   %+.5f\n", a, Bb - LLb((1-a)*Pb + a*0.25)))
cat(sprintf("  pooled  isotonic nested %+.5f\n", Bb - iso_nested(Pb, y, fmb, FALSE)))
cat(sprintf("  per-alt isotonic nested %+.5f\n", Bb - iso_nested(Pb, y, fmb, TRUE)))

rule("B6 -- SHRINK-TO-MARGINAL CANNOT REACH THE ANCHORED TARGET")
Pt <- pk$Ptest; R <- (1.7918 - 1.499) / 1.0986
q0 <- mean(Pt[, 4])
cat(sprintf("  shipped p4 %.5f, measured target %.5f\n", q0, R))
cat("  P' = (1-a)P + a*m has mean p4 = (1-a)*q0 + a*m4, so hitting the target needs\n")
cat(sprintf("  a*(m4 - %.5f) = %.5f. With m4 = the test marginal itself (%.5f) that\n",
            q0, R - q0, R))
cat("  requires a = 1.000 -- every row replaced by the marginal, all information gone.\n\n")
cat(sprintf("  %-8s %10s %12s\n", "a", "needed m4", "comment"))
for (a in c(1.00, 0.50, 0.20, 0.10, 0.05)) {
  m4 <- q0 + (R - q0) / a
  cat(sprintf("  %-8.2f %10.5f %12s\n", a, m4,
              if (m4 > 1) { "IMPOSSIBLE" } else { "blunts as a side effect" }))
}
cat("\n  The alt-4 utility shift reaches the SAME anchored target while leaving the\n")
cat("  within-buy conditional distribution exactly unchanged (iteration 36 gate 4).\n")
cat("  Shrink-to-marginal is therefore strictly dominated for this purpose: it buys\n")
cat("  the same margin correction and pays for it with blunting the curve above\n")
cat("  says is worthless.\n")
cat("\nOK\n")
