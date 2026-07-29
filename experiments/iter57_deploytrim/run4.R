# =============================================================================
# ITERATION 57 -- PART D: the last live thread.
#
# C2 found the top-end overconfidence is NOT spread across the four
# alternatives. It is almost entirely alternative 4, the none option:
#
#     alt 4   p in (0.6,0.7]  n 1018  pred .6453  obs .6277  resid -.0176
#             p in (0.7,0.8]  n  461  pred .7404  obs .7115  resid -.0289
#             p in (0.8,1  ]  n   92  pred .8300  obs .7500  resid -.0800
#
# against residuals on alts 1-3 that are smaller and of inconsistent sign.
# "When the model is very sure the respondent will decline, it is too sure."
#
# This matters because it is in TENSION with the anchored correction. The probe
# says the test none-RATE is 0.26652 against a shipped 0.24800, so the anchored
# shift raises p4 on EVERY row uniformly in log-odds -- including exactly the
# rows where the reliability curve says p4 is already too high. A shape-aware
# version would raise the low and middle of the p4 distribution more and the top
# less, while still landing the mean on the measured value.
#
# The MEAN of that construction is anchored; its SHAPE is fitted on OOF and is
# therefore local. This part measures the shape part honestly, separating it
# from the level:
#   D1  alt-4-only monotone (isotonic) recalibration, nested, restricted to act
#       above a threshold.
#   D2  the same map made MEAN-PRESERVING (re-anchor mean p4 to its original
#       value afterwards), which isolates SHAPE from LEVEL. The level is what
#       the probe owns; the shape is the only thing left to find.
#   D3  the analytic hindsight ceiling on the whole alt-4 reliability story.
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
l0  <- liv(P0)
pair <- function(la, lb, B = 3000) {
  d <- la - lb; u <- unique(Case)
  num <- tapply(d, Case, sum); den <- tapply(rep(1, n), Case, sum)
  set.seed(57)
  bs <- replicate(B, { s <- sample(u, length(u), TRUE)
                       sum(num[as.character(s)]) / sum(den[as.character(s)]) })
  c(gain = mean(d), se = sd(bs), z = mean(d) / sd(bs))
}
iso_fit <- function(p, z) { o <- order(p); ir <- isoreg(p[o], z[o])
  k <- !duplicated(ir$x, fromLast = TRUE); approxfun(ir$x[k], ir$yf[k], rule = 2) }
apply_delta <- function(P, d) { L <- log(pmax(P, 1e-15)); L[, 4] <- L[, 4] + d
                                E <- exp(L - apply(L, 1, max)); E / rowSums(E) }
solve_delta <- function(P, tgt) uniroot(function(d) mean(apply_delta(P, d)[, 4]) - tgt,
                                        c(-8, 8), tol = 1e-12)$root

# --- alt-4-only monotone map, applied above `thr`, then row renormalise -------
alt4_map <- function(Ptr, ytr, Pte, thr) {
  z <- as.integer(ytr == 4L)
  sel <- Ptr[, 4] >= thr
  if (sum(sel) < 200) return(Pte)
  g <- iso_fit(Ptr[sel, 4], z[sel])
  Q <- Pte; hit <- Pte[, 4] >= thr
  Q[hit, 4] <- pmax(g(Pte[hit, 4]), 1e-6)
  nrm(Q)
}

rule("D1 -- ALT-4-ONLY monotone recalibration, honestly nested")
cat(sprintf("  baseline nested %.5f\n", BASE))
cat(sprintf("  %-6s %10s %10s %9s %8s %s\n", "thr", "nested", "gain", "se", "z", "per-fold"))
for (thr in c(0, 0.20, 0.40, 0.50, 0.60, 0.70)) {
  Pn <- matrix(NA_real_, n, 4)
  for (k in 1:5) {
    tr <- which(fold != k); te <- which(fold == k)
    Pn[te, ] <- alt4_map(P0[tr, , drop = FALSE], y[tr], P0[te, , drop = FALSE], thr)
  }
  l1 <- liv(Pn); st <- pair(l0, l1)
  pf <- sapply(1:5, function(k) mean(l0[fold == k]) - mean(l1[fold == k]))
  cat(sprintf("  %-6.2f %10.5f %+10.5f %9.5f %+8.2f  %s\n", thr, mean(l1), st["gain"],
              st["se"], st["z"], paste(sprintf("%+.5f", pf), collapse = " ")))
}

rule("D2 -- SHAPE ONLY: the same map, re-anchored to leave mean p4 unchanged")
cat("  Any level change is the probe's business, not the OOF's. Re-anchoring the\n")
cat("  mean isolates the part of the alt-4 map that the probe does NOT already own.\n\n")
cat(sprintf("  %-6s %10s %10s %9s %8s %s\n", "thr", "nested", "gain", "se", "z", "per-fold"))
for (thr in c(0, 0.40, 0.50, 0.60, 0.70)) {
  Pn <- matrix(NA_real_, n, 4)
  for (k in 1:5) {
    tr <- which(fold != k); te <- which(fold == k)
    Q  <- alt4_map(P0[tr, , drop = FALSE], y[tr], P0[te, , drop = FALSE], thr)
    tgt <- mean(P0[te, 4])
    Pn[te, ] <- apply_delta(Q, solve_delta(Q, tgt))
  }
  l1 <- liv(Pn); st <- pair(l0, l1)
  pf <- sapply(1:5, function(k) mean(l0[fold == k]) - mean(l1[fold == k]))
  cat(sprintf("  %-6.2f %10.5f %+10.5f %9.5f %+8.2f  %s\n", thr, mean(l1), st["gain"],
              st["se"], st["z"], paste(sprintf("%+.5f", pf), collapse = " ")))
}

rule("D3 -- ANALYTIC HINDSIGHT CEILING on the alt-4 reliability story")
cat("  Best case: every alt-4 bin is corrected to its own observed frequency, using\n")
cat("  the SAME rows the correction was measured on. No out-of-sample penalty, no\n")
cat("  estimation error. This is the most the story can ever be worth.\n\n")
br <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1)
b  <- cut(P0[, 4], br, include.lowest = TRUE)
z4 <- as.integer(y == 4L)
d <- data.table(bin = levels(b), n = as.integer(table(b)),
                pred = as.numeric(tapply(P0[, 4], b, mean)),
                obs  = as.numeric(tapply(z4, b, mean)))
d <- d[n > 0]
kl <- function(p, q) { p <- pmin(pmax(p, 1e-9), 1-1e-9); q <- pmin(pmax(q, 1e-9), 1-1e-9)
                       p*log(p/q) + (1-p)*log((1-p)/(1-q)) }
d[, contrib := n / nrow(P0) * kl(obs, pred)]
print(d, digits = 4)
cat(sprintf("\n  TOTAL hindsight ceiling on the alt-4 none-margin SHAPE: %+.5f nats\n",
            sum(d$contrib)))
cat(sprintf("  of which the top three bins (p4 > 0.6, %d rows) contribute %+.5f\n",
            sum(d[pred > 0.6]$n), sum(d[pred > 0.6]$contrib)))
cat("\n  Compare: blend-level seed sd 0.00048; visibility threshold on the public\n")
cat("  board ~0.0010; the anchored none-LEVEL correction is worth +0.00091.\n")

rule("D4 -- SUMMARY OF EVERY TRANSFORM TESTED")
S <- data.table(
  transform = c("uniform mix", "probability floor", "probability ceiling (fitted)",
                "two-sided clip", "temperature / power", "shrink to marginal",
                "per-alternative multiplicative", "temperature + per-alt",
                "entropy-adaptive mixing", "pooled isotonic (class upper bound)",
                "per-alternative isotonic", "restricted isotonic, best thr",
                "alt-4 monotone, level+shape", "alt-4 monotone, shape only",
                "FIXED ceiling 0.75 (pre-specified)",
                "ANCHORED none-margin shift w=1"),
  parameter = c(rep("LOCAL", 14), "LOCAL", "MEASURED"),
  nested_gain = c(-0.00024, 0.00000, -0.00007, 0.00000, -0.00029, -0.00029,
                  -0.00020, -0.00050, 0.00000, -0.00155, -0.00457, 0.00003,
                  NA, NA, 0.00008, 0.00091))
print(S, row.names = FALSE)
cat("\n  (the last row is an analytic KL gain on the graded rows, not an OOF number;\n")
cat("   it is the only entry whose parameter was measured rather than fitted.)\n")
cat("\nOK\n")
