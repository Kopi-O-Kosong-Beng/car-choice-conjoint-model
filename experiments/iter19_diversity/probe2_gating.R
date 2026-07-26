# =============================================================================
# Iteration 19, ADDENDUM -- falsification + reachability of the oracle headroom.
# Read-only. Emits NO oof_/test_ artifacts.
#
# probe.R found that a per-respondent convex combination of the 4 members,
# fitted on half a respondent's tasks and scored on the other half, is worth
# 0.040 logloss. That number uses the respondent's OWN LABELS, which do not
# exist for the 263 test respondents, so it is a ceiling, not a method.
# Two questions follow, and this script answers both.
#
#   H4 (falsification). Is that 0.040 genuinely respondent-SPECIFIC, or would
#       any random weight vector do as well? Control: score respondent r with
#       the weights fitted on a DIFFERENT, randomly chosen respondent. If the
#       gain survives the permutation it was never about respondent identity.
#
#   H5 (reachability). Can a LABEL-FREE gate recover any of it? Split
#       respondents into tertiles by a quantity computable at test time -- the
#       respondent's mean predicted none-probability across their 19 tasks,
#       averaged over members -- and fit a separate production-style weight
#       vector inside each tertile. Nested by fold. Also tried: a demographic
#       gate on `segment`.
#
# RESULT (written after the run; hypotheses above left as they were).
#   H4: the gain is real and respondent-specific. Correct matching 1.08965 vs
#       global-pool control 1.13013; permuted matching 1.14309 +- 0.00192, i.e.
#       WORSE than the global pool. Respondent-specific component +0.05344.
#   H5: unreachable. The label-free p4-tertile gate scores 1.13068 (-0.00023 vs
#       ungated, z = -1.68) and the `segment` gate 1.13514 (-0.00470, z = -4.87,
#       a real loss). None of the 0.040 of headroom is recoverable from these
#       observables. The oracle needs the respondent's own past choices, and
#       every test respondent has zero observed choices.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
set.seed(42)

MEMB <- c("mnl_pw", "xgb_lw2", "xgb_mono", "lcmnl3")
M <- length(MEMB); eps0 <- 1e-12

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds"); setorder(folds, No)
ymap  <- unique(long[is_test == FALSE, .(No, y, Case, Task, segment)]); setorder(ymap, No)
stopifnot(identical(folds$No, ymap$No))
y <- ymap$y; Case <- ymap$Case; fmap <- folds$fold; N <- length(y)

readP <- function(nm) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", nm)); setorder(d, No)
  as.matrix(d[, .(p1, p2, p3, p4)])
}
OOF <- lapply(MEMB, readP); names(OOF) <- MEMB
LP  <- lapply(OOF, function(P) log(pmax(P, eps0)))
Q   <- sapply(OOF, function(P) P[cbind(1:N, y)])

# ---------------------------------------------------------------- H4 control
em_w <- function(rows, grp, iters = 400) {
  q <- Q[rows, , drop = FALSE]; g <- as.integer(factor(grp[rows])); G <- max(g)
  W <- matrix(1 / M, G, M)
  for (it in 1:iters) {
    num <- q * W[g, , drop = FALSE]; R <- num / rowSums(num)
    W <- rowsum(R, g) / tabulate(g, nbins = G)
    W <- pmax(W, 1e-10); W <- W / rowSums(W)
  }
  list(W = W, g = g)
}
half <- (ymap$Task %% 2L) == 1L
lev  <- sort(unique(Case)); g_all <- as.integer(factor(Case, levels = lev))
h1 <- em_w(half, Case)$W       # fitted on odd tasks
h2 <- em_w(!half, Case)$W      # fitted on even tasks
sc <- function(rows, W, idx) -log(pmax(rowSums(Q[rows, , drop = FALSE] * W[idx, , drop = FALSE]), 1e-15))
real <- mean(c(sc(!half, h1, g_all[!half]), sc(half, h2, g_all[half])))
gl1 <- em_w(half, rep(1L, N))$W[1, ]; gl2 <- em_w(!half, rep(1L, N))$W[1, ]
ctrl <- mean(c(-log(pmax(as.vector(Q[!half, ] %*% gl1), 1e-15)),
               -log(pmax(as.vector(Q[half, ]  %*% gl2), 1e-15))))
cat("\n########## H4. IS THE PER-RESPONDENT GAIN RESPONDENT-SPECIFIC? ##########\n")
cat(sprintf("global pool, split-half protocol         : %.5f\n", ctrl))
cat(sprintf("per-respondent weights, correct matching : %.5f  (gain %+.5f)\n", real, ctrl - real))
pv <- numeric(10)
for (b in 1:10) {
  p1 <- sample(nrow(h1)); p2 <- sample(nrow(h2))
  pv[b] <- mean(c(sc(!half, h1, p1[g_all[!half]]), sc(half, h2, p2[g_all[half]])))
}
cat(sprintf("per-respondent weights, PERMUTED matching: %.5f +- %.5f  (gain %+.5f)\n",
            mean(pv), sd(pv), ctrl - mean(pv)))
cat(sprintf("=> respondent-specific component: %+.5f\n", mean(pv) - real))

# ------------------------------------------------------- shared blend machinery
blendA <- function(theta, Ls, rows) {
  a <- theta[1:M]; w <- exp(a - max(a)); w <- w / sum(w)
  Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
  L <- Reduce(`+`, Map(function(wi, Lm) wi * Lm[rows, , drop = FALSE], w, Ls)) / Tt
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
fitA <- function(rows, start = c(rep(0, M), 0, -3)) {
  obj <- function(th) logloss(y[rows], blendA(th, LP, rows))
  o <- list(par = start, value = obj(start))
  for (r in 1:5) {
    oo <- optim(o$par, obj, method = "Nelder-Mead", control = list(maxit = 6000, reltol = 1e-12))
    if (oo$value > o$value - 1e-10) { o <- oo; break }; o <- oo
  }
  o
}

# ------------------------------------------------------------- H5 gated blends
# label-free respondent covariate: mean predicted p4 over the respondent's tasks
p4bar <- rowMeans(sapply(OOF, function(P) P[, 4]))
rcov  <- data.table(Case = Case, p4 = p4bar)[, .(m = mean(p4)), by = Case]
setkey(rcov, Case)
covv  <- rcov[.(Case), m]                       # per-row respondent covariate

run_gate <- function(gvec, label) {
  rl <- numeric(N)
  for (k in 1:5) {
    tr <- fmap != k; te <- fmap == k
    for (g in sort(unique(gvec))) {
      rtr <- tr & gvec == g; rte <- te & gvec == g
      if (!any(rte)) next
      o <- fitA(rtr)
      P <- blendA(o$par, LP, rte)
      rl[rte] <- -log(pmax(P[cbind(1:sum(rte), y[rte])], 1e-15))
    }
  }
  cat(sprintf("%-42s nested %.5f\n", label, mean(rl)))
  rl
}
cat("\n########## H5. CAN A LABEL-FREE GATE REACH ANY OF IT? ##########\n")
base <- run_gate(rep(1L, N), "ungated (variant A, reference)")
# Tertiles of the respondent covariate. The cut points use every respondent's
# PREDICTIONS but no respondent's LABEL, so this partition is computable on the
# test set and carries no target information -- the gate is label-free.
cuts_all <- quantile(rcov$m, c(1/3, 2/3))
gate_full <- as.integer(cut(covv, c(-Inf, cuts_all, Inf), labels = 1:3))
g1 <- run_gate(gate_full, "gate = tertile of respondent mean pred p4")
g2 <- run_gate(as.integer(factor(ymap$segment)), "gate = demographic `segment`")
cat("\nsegment sizes:\n"); print(table(ymap$segment))

paired <- function(la, lb, lab) {
  d <- data.table(Case = Case, d = la - lb)[, .(dm = mean(d)), by = Case]
  est <- mean(d$dm); se <- sd(d$dm) / sqrt(nrow(d))
  cat(sprintf("%-42s %+.5f  SE %.5f  z = %+5.2f  %s\n", lab, est, se, est / se,
              if (abs(est / se) < 2) "(noise)" else if (est > 0) "(REAL gain)" else "(REAL loss)"))
}
cat("\npaired, respondent-clustered (positive = gated better):\n")
paired(base, g1, "p4-tertile gate over ungated")
paired(base, g2, "segment gate over ungated")
cat("\nDONE2\n")
