# =============================================================================
# ITERATION 26 — covariate-gated log-opinion pool. HYPOTHESIS BEFORE RESULTS.
#
# THE MATHEMATICS. The production combiner is a log-opinion pool with CONSTANT
# weights:            s_j(x) = sum_m beta_m * log p_mj(x)
# then softmax and an eps-uniform mix. Nothing justifies constancy. The members
# are structurally different estimators -- a latent-class MNL (weight 0.504), a
# monotone-constrained tree (0.311) and an unconstrained tree (0.185) -- and
# their relative reliability should vary over covariate space. The latent-class
# model carries an explicit taste-segment mixture, so it ought to dominate where
# segment structure is informative; the trees ought to dominate where the design
# encoding and nonlinear attribute interactions carry the signal.
#
# Let the weights be affine in a few gating covariates:
#         beta_m(x) = beta_m0 + sum_k gamma_mk * z_k(x)
#   =>    s_j(x) = sum_m beta_m0 * log p_mj + sum_{m,k} gamma_mk * (z_k * log p_mj)
#
# This is STILL LINEAR IN THE PARAMETERS. So it is exactly multinomial logistic
# regression on the augmented feature set {log p_m} U {z_k * log p_m}, hence
# CONVEX with a unique optimum at fixed eps -- a certificate, not a search. It is
# a mixture-of-experts gate on top of the pool, and it requires NO refitting of
# any member: pure algebra on existing OOF artifacts.
#
# WHY THIS IS NOT A DEAD IDEA. The refuted blend work was: free simplex weights
# (still constant), bagged EM weights (still constant), arithmetic instead of
# geometric pooling (a different pool, still constant), and temperature/eps
# recalibration (a global monotone transform). Every one of them re-parameterises
# a CONSTANT combiner. None makes the weights a function of x. That degree of
# freedom has never been tried here.
#
# GATING VARIABLES, chosen structurally and fixed before running -- not searched:
#   z1  task position (centred, scaled)  -- iteration 25 showed position enters
#                                           the latent-class utility, so the
#                                           members' relative accuracy plausibly
#                                           drifts across the 19 tasks.
#   z2  luxury-segment indicator (segments 3 and 5) -- these are 9.4% of train
#                                           and 68.8% of test; if any gate
#                                           matters for the GRADE it is this one.
#   z3  log income (centred, scaled)     -- the documented train/test shift axis.
# K = 3 gates, M = 4 members => 16 parameters, on 1,135 independent respondents.
#
# OVERFITTING CONTROL. Ridge penalty lambda on the INTERACTION terms only (never
# on beta_m0, so the model nests the incumbent exactly at lambda -> Inf). lambda
# is chosen by an INNER cross-validation over respondents inside each outer
# training fold, so the outer nested number never sees its own tuning. Pre-
# registered grid: lambda in {1e-2, 1e-1, 1, 10, 100}. All results reported.
#
# HYPOTHESIS: a real but small gain, -0.000 to -0.002. Gating is a 12-parameter
# refinement of a combiner that is already near-optimal on a plateau; the honest
# prior is that most of the value is in z2 (segment), because that is the only
# gate with a documented mechanism behind it.
# FALSIFIED IF: nested OOF does not beat the incumbent, or the fitted gammas
# shrink to zero at the CV-selected lambda.
#
# Run: Rscript experiments/iter26_gated_pool/run.R      Runtime: ~5-10 min.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
wide  <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")
memb  <- trimws(sub("#.*$", "", readLines("model/members.txt")))
memb  <- memb[nzchar(memb)]
cat("members:", paste(memb, collapse = ", "), "\n")

ymap <- unique(long[is_test == FALSE, .(No, y, Case, Task, segmentind)]); setorder(ymap, No)
y <- ymap$y; n <- length(y)
fm <- folds[order(No), fold]
cs <- ymap$Case

LP <- lapply(memb, function(m) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  log(pmax(as.matrix(d[, .(p1, p2, p3, p4)]), 1e-12))
})
M <- length(LP)

# ---- gating covariates (fixed before running) --------------------------------
inc <- unique(wide[is_test == FALSE, .(No, incomea)])[order(No), incomea]
z1 <- as.numeric(scale(ymap$Task))
z2 <- as.numeric(ymap$segmentind %in% c(3L, 5L))          # luxury indicator
z3 <- as.numeric(scale(log1p(pmax(inc, 0))))
Z  <- cbind(z1 = z1, z2 = z2, z3 = z3)
K  <- ncol(Z)
cat(sprintf("gates: K=%d  (task position, luxury indicator, log income)\n", K))
cat(sprintf("params: M*(1+K) = %d  on %d respondents\n\n", M * (1 + K), uniqueN(cs)))

sm <- function(S) { E <- exp(S - apply(S, 1, max)); E / rowSums(E) }

# theta = c(beta_0 [M], gamma [M*K])   -- score is linear in theta
scores <- function(theta, rows) {
  b0 <- theta[1:M]; G <- matrix(theta[(M + 1):(M * (1 + K))], nrow = M, ncol = K)
  S <- matrix(0, length(rows), 4)
  for (m in seq_len(M)) {
    w <- b0[m] + as.vector(Z[rows, , drop = FALSE] %*% G[m, ])
    S <- S + w * LP[[m]][rows, , drop = FALSE]
  }
  S
}
# loss + analytic gradient, ridge on gamma only
fg <- function(theta, rows, eps, lam) {
  S <- scores(theta, rows); P <- sm(S)
  yr <- y[rows]; idx <- cbind(seq_along(rows), yr)
  q <- (1 - eps) * P[idx] + eps * 0.25
  gam <- theta[(M + 1):(M * (1 + K))]
  val <- -mean(log(pmax(q, 1e-300))) + lam * sum(gam^2) / length(rows)
  Ind <- matrix(0, length(rows), 4); Ind[idx] <- 1
  Gm <- -(1 - eps) * (P[idx] / q) * (Ind - P)              # dL/ds, n x 4
  gb <- numeric(M); gg <- matrix(0, M, K)
  for (m in seq_len(M)) {
    Lm <- LP[[m]][rows, , drop = FALSE]
    rowdot <- rowSums(Gm * Lm)                              # per-row dL/dw_m
    gb[m] <- sum(rowdot)
    gg[m, ] <- as.vector(crossprod(Z[rows, , drop = FALSE], rowdot))
  }
  gr <- c(gb, as.vector(gg)) / length(rows) +
        c(rep(0, M), 2 * lam * gam / length(rows))
  list(value = val, grad = gr)
}
fit <- function(rows, eps, lam, init = NULL) {
  th0 <- if (is.null(init)) c(rep(1 / M, M), rep(0, M * K)) else init
  optim(th0, fn = function(t) fg(t, rows, eps, lam)$value,
        gr = function(t) fg(t, rows, eps, lam)$grad,
        method = "BFGS", control = list(maxit = 600, reltol = 1e-11))
}
loss_at <- function(theta, rows, eps) {
  P <- sm(scores(theta, rows)); idx <- cbind(seq_along(rows), y[rows])
  -mean(log(pmax((1 - eps) * P[idx] + eps * 0.25, 1e-15)))
}

# ---- gradient self-test ------------------------------------------------------
set.seed(2); rt <- sample(n, 500); tt <- c(runif(M, .1, .5), runif(M * K, -.05, .05))
an <- fg(tt, rt, 0.02, 1)$grad
num <- vapply(seq_along(tt), function(i) { h <- 1e-6; a <- tt; b <- tt
  a[i] <- a[i] + h; b[i] <- b[i] - h
  (fg(a, rt, 0.02, 1)$value - fg(b, rt, 0.02, 1)$value) / (2 * h) }, numeric(1))
cat("gradient check  max|analytic - numeric| =", format(max(abs(an - num)), digits = 3), "\n")
stopifnot(max(abs(an - num)) < 1e-6)

EPS  <- seq(0, 0.04, by = 0.01)
LAMS <- c(1e-2, 1e-1, 1, 10, 100)

# ---- nested outer loop, with INNER CV to pick lambda --------------------------
inner_pick <- function(tr_rows) {
  tr_cases <- unique(cs[tr_rows])
  set.seed(99); ifold <- sample(rep_len(1:3, length(tr_cases)))
  names(ifold) <- as.character(tr_cases)
  sc <- matrix(NA_real_, length(LAMS), 3)
  for (j in 1:3) {
    itr <- tr_rows[ifold[as.character(cs[tr_rows])] != j]
    iva <- tr_rows[ifold[as.character(cs[tr_rows])] == j]
    for (li in seq_along(LAMS)) {
      best <- Inf
      for (e in EPS) { o <- fit(itr, e, LAMS[li]); v <- loss_at(o$par, iva, e)
                       if (v < best) best <- v }
      sc[li, j] <- best
    }
  }
  LAMS[which.min(rowMeans(sc))]
}

nest_inc <- nest_gat <- numeric(5); lam_sel <- numeric(5)
for (k in 1:5) {
  tr <- which(fm != k); te <- which(fm == k)
  # incumbent: gamma forced to zero (constant weights) -- exact nesting
  bi <- Inf; bp <- NULL; be <- NA
  for (e in EPS) {
    o <- optim(rep(1 / M, M),
               fn = function(b) fg(c(b, rep(0, M * K)), tr, e, 0)$value,
               gr = function(b) fg(c(b, rep(0, M * K)), tr, e, 0)$grad[1:M],
               method = "BFGS", control = list(maxit = 400))
    if (o$value < bi) { bi <- o$value; bp <- c(o$par, rep(0, M * K)); be <- e }
  }
  nest_inc[k] <- loss_at(bp, te, be)
  lam <- inner_pick(tr); lam_sel[k] <- lam
  bg <- Inf; gp <- NULL; ge <- NA
  for (e in EPS) { o <- fit(tr, e, lam, init = bp)
                   if (o$value < bg) { bg <- o$value; gp <- o$par; ge <- e } }
  nest_gat[k] <- loss_at(gp, te, ge)
  cat(sprintf("fold %d  lambda %-6g  incumbent %.5f  gated %.5f  delta %+.5f\n",
              k, lam, nest_inc[k], nest_gat[k], nest_gat[k] - nest_inc[k]))
  flush.console()
}
cat(sprintf("\nNESTED incumbent (constant weights) : %.5f\n", mean(nest_inc)))
cat(sprintf("NESTED gated pool                   : %.5f\n", mean(nest_gat)))
cat(sprintf(">>> delta %+.5f   (negative = gating helps)\n", mean(nest_gat) - mean(nest_inc)))

# paired respondent-clustered test on the nested predictions
li_i <- li_g <- numeric(n)
for (k in 1:5) {
  tr <- which(fm != k); te <- which(fm == k)
  bi <- Inf; bp <- NULL; be <- NA
  for (e in EPS) {
    o <- optim(rep(1/M, M), fn = function(b) fg(c(b, rep(0, M*K)), tr, e, 0)$value,
               gr = function(b) fg(c(b, rep(0, M*K)), tr, e, 0)$grad[1:M],
               method = "BFGS", control = list(maxit = 400))
    if (o$value < bi) { bi <- o$value; bp <- c(o$par, rep(0, M*K)); be <- e } }
  o <- fit(tr, be, lam_sel[k], init = bp)
  Pi <- sm(scores(bp, te)); Pg <- sm(scores(o$par, te))
  Pi <- (1-be)*Pi + be*0.25; Pg <- (1-be)*Pg + be*0.25
  li_i[te] <- -log(pmax(Pi[cbind(seq_along(te), y[te])], 1e-15))
  li_g[te] <- -log(pmax(Pg[cbind(seq_along(te), y[te])], 1e-15))
}
d  <- li_i - li_g
dr <- tapply(d, cs, mean); se <- sd(dr) / sqrt(length(dr))
cat(sprintf("\npaired clustered  delta %+.5f  SE %.5f  z %+.2f  (respondents better %.1f%%)\n",
            mean(dr), se, mean(dr) / se, 100 * mean(dr > 0)))

# ---- what did the gate actually learn? --------------------------------------
lam <- inner_pick(seq_len(n))
bf <- Inf; fp <- NULL
for (e in EPS) { o <- fit(seq_len(n), e, lam); if (o$value < bf) { bf <- o$value; fp <- o$par } }
G <- matrix(fp[(M + 1):(M * (1 + K))], nrow = M, ncol = K,
            dimnames = list(memb, c("task_pos", "luxury", "log_income")))
cat(sprintf("\nfull-data fit (lambda %g)  base weights: %s\n", lam,
            paste(sprintf("%s %.3f", memb, fp[1:M]), collapse = " | ")))
cat("gate coefficients (how each member's weight shifts):\n"); print(round(G, 4))
saveRDS(list(nested_inc = mean(nest_inc), nested_gated = mean(nest_gat),
             paired = c(delta = mean(dr), se = se), gates = G, lambda = lam_sel),
        "experiments/iter26_gated_pool/results.rds")
cat("\nNOT promoted. Adoption requires the paired z and no segment-metric regression.\nOK\n")
