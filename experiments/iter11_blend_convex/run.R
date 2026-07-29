# =============================================================================
# ITERATION 11 — reliable blend optimisation. HYPOTHESIS WRITTEN BEFORE RUNNING.
#
# HYPOTHESIS: `model/06_blend.R` optimises 11 free parameters (9 log-weights +
# log-T + logit-eps) with SINGLE-START Nelder-Mead. That parameterisation is
# REDUNDANT -- softmax(log-weights) has M-1 degrees of freedom but uses M
# parameters, and w/T is scale-degenerate -- so the objective has flat ridges
# that a derivative-free simplex stalls on. A probe on the 9-member artifact set
# found 4 of 13 random restarts reaching 1.137639 versus the production
# 1.137778, with mixl2 at weight 0.076 rather than 0; the worst restart landed at
# 1.1421. So at least one member is zeroed by OPTIMISER FAILURE, not because the
# optimum puts it at zero.
#
# THE FIX IS A REPARAMETERISATION, NOT A SEARCH. Write the pooled score as
#     s_ij = sum_m beta_m * log P_m[i,j],    beta_m = w_m / T   (unconstrained)
# Then, at eps = 0, the loss is EXACTLY multinomial logistic regression on the
# fixed features log P_m, hence CONVEX in beta with a unique optimum. Solving it
# with an analytic gradient is a certificate, not a lottery. (w, T) are recovered
# afterwards as T = 1/sum(beta), w = beta*T -- and note the recovered w need not
# be simplex-constrained, which is itself a degree of freedom 06_blend.R forbids.
#
# The eps floor breaks strict convexity, so eps is handled by an outer 1-D grid
# with beta re-solved convexly at each eps. That is a 1-D scan over a shallow
# well-behaved bowl, not an 11-D simplex crawl.
#
# WHAT THIS SCRIPT DOES NOT DO: it does not overwrite any production artifact and
# does not touch model/06_blend.R. It reports the incumbent optimiser and the
# convex one side by side on the SAME data so the difference is attributable.
# Promotion is a separate, deliberate step.
#
# EXPECTATION: a small objective improvement (~0.0001-0.0003, below the ~0.0015
# paired clustered SE) and materially different weights. The point is a
# TRUSTWORTHY decision number, not a score win -- every later adoption in the
# queue is judged against it.
#
# Run: Rscript experiments/iter11_blend_convex/run.R
# Runtime: ~2 min. Requires the oof_*.rds named in model/members.txt.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

avail <- gsub("^oof_|\\.rds$", "", list.files("model/artifacts", pattern = "^oof_.*\\.rds$"))
avail <- avail[file.exists(sprintf("model/artifacts/test_%s.rds", avail))]
want  <- trimws(sub("#.*$", "", readLines("model/members.txt")))
want  <- want[nzchar(want)]
memb  <- want[want %in% avail]
if (length(memb) < length(want))
  cat("WARNING: missing artifacts for:", paste(setdiff(want, avail), collapse = ", "),
      "\n         proceeding with:", paste(memb, collapse = ", "), "\n")
stopifnot(length(memb) >= 2)
cat("members:", paste(memb, collapse = ", "), "\n\n")

OOF <- lapply(memb, function(m) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  as.matrix(d[, .(p1, p2, p3, p4)])
})
names(OOF) <- memb
M <- length(memb)
stopifnot(all(sapply(OOF, nrow) == 21565))

ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
y <- ymap$y
fmap <- folds[order(No), fold]
n <- length(y)

# log-probability features: LP[[m]] is n x 4
LP <- lapply(OOF, function(P) log(pmax(P, 1e-12)))
Yid <- cbind(seq_len(n), y)

# ---- incumbent: exactly 06_blend.R's parameterisation and optimiser ----------
blend_inc <- function(theta, rows) {
  w <- exp(theta[1:M]); w <- w / sum(w)
  Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
  L <- Reduce(`+`, Map(function(wi, lp) wi * lp[rows, , drop = FALSE], w, LP)) / Tt
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
obj_inc <- function(theta, rows) logloss(y[rows], blend_inc(theta, rows))
fit_inc <- function(rows) optim(c(rep(0, M), 0, -3), obj_inc, rows = rows,
                                method = "Nelder-Mead", control = list(maxit = 3000))

# ---- convex reparameterisation: s = sum_m beta_m * logP_m --------------------
# At eps = 0 this is multinomial logistic on fixed features => convex in beta.
scores <- function(beta, rows) {
  S <- matrix(0, length(rows), 4)
  for (m in seq_len(M)) S <- S + beta[m] * LP[[m]][rows, , drop = FALSE]
  S
}
softmax_rows <- function(S) { E <- exp(S - apply(S, 1, max)); E / rowSums(E) }

# loss and ANALYTIC gradient in beta at fixed eps
fg <- function(beta, rows, eps) {
  S <- scores(beta, rows); P <- softmax_rows(S)
  yr <- y[rows]; idx <- cbind(seq_along(rows), yr)
  q <- (1 - eps) * P[idx] + eps * 0.25
  val <- -mean(log(pmax(q, 1e-300)))
  # dL_i/ds_ij = -(1-eps) * p_iy * (delta_{yj} - p_ij) / q_i
  Ind <- matrix(0, length(rows), 4); Ind[idx] <- 1
  G <- -(1 - eps) * (P[idx] / q) * (Ind - P)          # n x 4
  gr <- vapply(seq_len(M), function(m) sum(G * LP[[m]][rows, , drop = FALSE]),
               numeric(1)) / length(rows)
  list(value = val, grad = gr)
}
fit_convex <- function(rows, eps_grid = seq(0, 0.09, by = 0.005), beta0 = NULL) {
  best <- NULL
  b0 <- if (is.null(beta0)) rep(1 / M, M) else beta0
  for (eps in eps_grid) {
    o <- optim(b0, fn = function(b) fg(b, rows, eps)$value,
               gr = function(b) fg(b, rows, eps)$grad,
               method = "BFGS", control = list(maxit = 500, reltol = 1e-12))
    if (is.null(best) || o$value < best$value)
      best <- list(value = o$value, beta = o$par, eps = eps)
    b0 <- o$par                                        # warm start along the grid
  }
  best
}
predict_convex <- function(par, rows) {
  P <- softmax_rows(scores(par$beta, rows))
  (1 - par$eps) * P + par$eps * 0.25
}

# ---- gradient self-test (must pass before any number is believed) -----------
set.seed(1)
rows_t <- sample(n, 400); b_t <- runif(M, 0.2, 0.8); e_t <- 0.03
an <- fg(b_t, rows_t, e_t)$grad
num <- vapply(seq_len(M), function(m) {
  h <- 1e-6; bp <- b_t; bm <- b_t; bp[m] <- bp[m] + h; bm[m] <- bm[m] - h
  (fg(bp, rows_t, e_t)$value - fg(bm, rows_t, e_t)$value) / (2 * h)
}, numeric(1))
cat("gradient check  max|analytic - numeric| =", format(max(abs(an - num)), digits = 3), "\n")
stopifnot(max(abs(an - num)) < 1e-6)

# ---- convexity certificate: many random starts must reach the SAME optimum --
cat("\n--- convexity check at eps = 0 (10 random starts, full data) ---\n")
allrows <- seq_len(n)
vals <- vapply(1:10, function(s) {
  set.seed(100 + s)
  o <- optim(runif(M, -1, 2), fn = function(b) fg(b, allrows, 0)$value,
             gr = function(b) fg(b, allrows, 0)$grad, method = "BFGS",
             control = list(maxit = 500, reltol = 1e-12))
  o$value
}, numeric(1))
cat("  range over starts:", format(diff(range(vals)), digits = 3),
    " (convex => ~0)   best:", round(min(vals), 6), "\n")

# ---- nested comparison: identical folds, identical data, two optimisers -----
cat("\n--- NESTED evaluation (fit weights on folds != k, score fold k) ---\n")
nest_inc <- nest_cvx <- numeric(5)
for (k in 1:5) {
  tr <- which(fmap != k); te <- which(fmap == k)
  oi <- fit_inc(tr)
  nest_inc[k] <- obj_inc(oi$par, te)
  pc <- fit_convex(tr)
  nest_cvx[k] <- logloss(y[te], predict_convex(pc, te))
  cat(sprintf("  fold %d   incumbent %.5f   convex %.5f   delta %+.5f\n",
              k, nest_inc[k], nest_cvx[k], nest_cvx[k] - nest_inc[k]))
}
cat(sprintf("\n  NESTED incumbent : %.5f  +- %.5f\n", mean(nest_inc), sd(nest_inc)))
cat(sprintf("  NESTED convex    : %.5f  +- %.5f\n", mean(nest_cvx), sd(nest_cvx)))
cat(sprintf("  >>> nested delta : %+.5f  (negative = convex better)\n",
            mean(nest_cvx) - mean(nest_inc)))

# paired, respondent-clustered test on the nested predictions
resp <- unique(long[is_test == FALSE, .(No, Case)])[order(No), Case]
li_i <- li_c <- numeric(n)
for (k in 1:5) {
  tr <- which(fmap != k); te <- which(fmap == k)
  oi <- fit_inc(tr); pc <- fit_convex(tr)
  Pi <- blend_inc(oi$par, te); Pc <- predict_convex(pc, te)
  li_i[te] <- -log(pmax(Pi[cbind(seq_along(te), y[te])], 1e-15))
  li_c[te] <- -log(pmax(Pc[cbind(seq_along(te), y[te])], 1e-15))
}
d  <- li_i - li_c                                    # positive => convex better
dr <- tapply(d, resp, mean)
se <- sd(dr) / sqrt(length(dr))
cat(sprintf("\n  paired clustered delta %+.5f  SE %.5f  z %+.2f  (respondents better: %.1f%%)\n",
            mean(dr), se, mean(dr) / se, 100 * mean(dr > 0)))

# ---- full-data fits: what do the weights actually look like? ----------------
cat("\n--- full-data fit, weights side by side ---\n")
oi <- fit_inc(allrows); pc <- fit_convex(allrows)
w_i <- exp(oi$par[1:M]); w_i <- w_i / sum(w_i)
Tt_i <- exp(oi$par[M + 1]); e_i <- plogis(oi$par[M + 2]) * 0.10
Tt_c <- 1 / sum(pc$beta); w_c <- pc$beta * Tt_c
cat(sprintf("  incumbent  obj %.6f   T %.3f  eps %.4f\n", oi$value, Tt_i, e_i))
cat(sprintf("  convex     obj %.6f   T %.3f  eps %.4f  (sum beta %.3f)\n",
            pc$value, Tt_c, pc$eps, sum(pc$beta)))
cat("\n  member          w_incumbent   w_convex     beta\n")
for (m in seq_len(M))
  cat(sprintf("  %-14s %9.4f   %9.4f  %8.4f\n", memb[m], w_i[m], w_c[m], pc$beta[m]))

saveRDS(list(members = memb, incumbent = list(par = oi$par, obj = oi$value, nested = mean(nest_inc)),
             convex = list(beta = pc$beta, eps = pc$eps, obj = pc$value, nested = mean(nest_cvx)),
             paired = list(delta = mean(dr), se = se, z = mean(dr) / se)),
        "experiments/iter11_blend_convex/results.rds")
cat("\nNOT promoted. model/06_blend.R untouched.\nOK\n")
