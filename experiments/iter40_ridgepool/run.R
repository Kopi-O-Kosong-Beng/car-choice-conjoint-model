# =============================================================================
# ITERATION 40 -- RIDGE-PENALISED FREE-SIGN POOL OVER *ALL* MEMBERS
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# WHY -- this is the diagnosed failure of iterations 37 and 38, not a re-run
# -----------------------------------------------------------------------------
# Iteration 37 (forward, 115 nested evals) and iteration 38 (backward, 62 evals)
# both searched the free-sign member space and both came back at or below
# iteration 35's greedy 1.12341. But BOTH are greedy DISCRETE searches, and
# iteration 37's own trace shows the pathology:
#
#   it never picked xgb_wide, because once xgb_long and xgb_2stage are in,
#   xgb_wide's MARGINAL value falls below the step gate -- yet all three
#   together beat any two.
#
# That is non-additivity, and stepwise selection is structurally blind to it: it
# evaluates each member conditional on what it has already taken, and can never
# reach a diffuse optimum spread across many individually-weak members.
#
# The escape is to stop SELECTING and start SHRINKING. Fit every candidate
# simultaneously with an L2 penalty on the coefficient vector. A ridge fit is a
# continuous relaxation of subset selection: it can place small non-zero weight
# on many members at once, which is exactly the configuration greedy search
# cannot represent. If the control-variate story from iteration 35 is right --
# that these members span a shared tree-error direction -- then the optimum is
# plausibly a broad, small-coefficient combination rather than three big ones.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# A ridge-penalised free-sign pool over ALL admissible artifacts will beat
# iteration 35's 5-member 1.12341, because the penalty makes it safe to include
# members whose individual marginal value is below any sensible selection gate
# but whose joint contribution to spanning the shared error direction is real.
#
# -----------------------------------------------------------------------------
# WHY THE PENALTY IS WHAT MAKES THIS ADMISSIBLE
# -----------------------------------------------------------------------------
# Fitting ~38 free coefficients on 21,565 OOF rows from 1,135 respondents is
# exactly the blend-level overfitting the ~1,500-row private board is exposed to.
# The penalty is not a tuning nicety here, it is the thing that makes the fit
# legitimate. So lambda is chosen by an INNER cross-validation inside each outer
# fold -- the outer nested number NEVER sees its own tuning. This is CLAUDE.md
# rule 7 (nest everything fitted) applied to the penalty itself.
#
# Note this is a strictly LARGER model class than iteration 35: at lambda -> 0
# with only 5 members it reproduces the free-sign pool, and at lambda -> Inf it
# reproduces the uniform pool. If the inner CV picks a large lambda, that is
# itself the finding -- it would say the 5-member solution was already the
# right amount of complexity.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
# ADOPT over iteration 35's 1.12341 only if ALL of:
#   1. nested OOF improves by >= 0.00100 (2x the blend-level seed sd 0.00048);
#   2. paired respondent-clustered z >= 2 against the 5-member pool;
#   3. the gain is positive in at least 4 of the 5 folds;
#   4. shipped test none-rate does not move FURTHER from the measured 0.2665
#      than the 5-member pool's 0.2377;
#   5. the selected lambda is INTERIOR to the grid in a majority of folds -- if
#      it pins at either end the grid was wrong and the result is not trustworthy.
# REJECT otherwise. A rejection with a large selected lambda is a POSITIVE
# finding: it would say the free-sign frontier really is a small set, closing
# this axis for good.
#
# EXCLUSIONS, identical to iterations 37/38 and equally load-bearing:
#   xgb_pt (withdrawn -- iter30_decorr killed at fold 3; ranks FIRST on marginal
#   value, so this exclusion matters), xgb_resenc* (100% leakage, iteration 15),
#   *_b / *_c (validation-split artifacts), blend* (blends of blends),
#   *_cal (already-calibrated test-side objects).
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

DIR <- "experiments/iter40_ridgepool"
dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
LAMBDAS <- c(0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1)
INNER_K <- 3L
TRUE_NONE <- 0.26651
REF5 <- c("xgb_lw2bag", "lcmnl3_both", "xgb_long", "xgb_wide", "xgb_2stage")

rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
y <- ymap$y; case <- ymap$Case; n <- length(y)
fmap <- folds[order(No), fold]

LG <- function(m, pre = "oof") {
  d <- readRDS(sprintf("model/artifacts/%s_%s.rds", pre, m)); setorder(d, No)
  log(pmax(as.matrix(d[, .(p1, p2, p3, p4)]), 1e-12))
}

avail <- gsub("^oof_|\\.rds$", "", list.files("model/artifacts", pattern = "^oof_.*\\.rds$"))
MEMB <- avail[!grepl("^blend|resenc|_b$|_c$|_cal$|^xgb_pt$", avail)]
MEMB <- MEMB[sapply(MEMB, function(m) tryCatch(nrow(LG(m)) == n, error = function(e) FALSE))]
M <- length(MEMB)
L <- lapply(MEMB, LG); names(L) <- MEMB
# stack into an n x 4 x M array once; the fit is called ~150 times so this matters
A <- array(0, c(n, 4L, M)); for (i in seq_len(M)) A[, , i] <- L[[i]]

# ---- penalised objective with ANALYTIC gradient ------------------------------
# S = sum_m beta_m * L_m ; P = softmax(S) rowwise ; q = (1-e) P[,y] + e/4
# loss = -mean(log q) + lam * ||beta||^2
# dloss/dS_ij = -(1/n) (1-e) P_iy (1{j=y} - P_ij) / q_i
# dloss/dbeta_m = sum_ij dloss/dS_ij * L_m[i,j]
mk <- function(rows) {
  Ar <- A[rows, , , drop = FALSE]; yr <- y[rows]; nr <- length(yr)
  idx <- cbind(seq_len(nr), yr)
  list(
    fn = function(th, lam) {
      b <- th[1:M]; e <- plogis(th[M + 1]) * 0.10
      S <- matrix(0, nr, 4L); for (i in seq_len(M)) S <- S + b[i] * Ar[, , i]
      S <- S - do.call(pmax, lapply(1:4, function(j) S[, j]))
      E <- exp(S); P <- E / rowSums(E)
      q <- (1 - e) * P[idx] + e * 0.25
      -mean(log(pmax(q, 1e-300))) + lam * sum(b^2)
    },
    gr = function(th, lam) {
      b <- th[1:M]; ee <- plogis(th[M + 1]); e <- ee * 0.10
      S <- matrix(0, nr, 4L); for (i in seq_len(M)) S <- S + b[i] * Ar[, , i]
      S <- S - do.call(pmax, lapply(1:4, function(j) S[, j]))
      E <- exp(S); P <- E / rowSums(E)
      Py <- P[idx]; q <- pmax((1 - e) * Py + e * 0.25, 1e-300)
      Ind <- matrix(0, nr, 4L); Ind[idx] <- 1
      G <- -(1 / nr) * (1 - e) * (Py / q) * (Ind - P)     # nr x 4
      gb <- vapply(seq_len(M), function(i) sum(G * Ar[, , i]), 0) + 2 * lam * b
      # d/dtheta_{M+1}: e = 0.10*plogis(t) -> de/dt = 0.10*ee*(1-ee)
      ge <- -mean((0.25 - Py) / q) * 0.10 * ee * (1 - ee)
      c(gb, ge)
    })
}
fit_ridge <- function(rows, lam, start = NULL) {
  o <- mk(rows)
  th0 <- if (is.null(start)) c(rep(1 / M, M), -3) else start
  r <- optim(th0, function(t) o$fn(t, lam), function(t) o$gr(t, lam),
             method = "BFGS", control = list(maxit = 400, reltol = 1e-11))
  r$par
}
score_rows <- function(th, rows) {
  b <- th[1:M]; e <- plogis(th[M + 1]) * 0.10
  Ar <- A[rows, , , drop = FALSE]; yr <- y[rows]; nr <- length(yr)
  S <- matrix(0, nr, 4L); for (i in seq_len(M)) S <- S + b[i] * Ar[, , i]
  S <- S - do.call(pmax, lapply(1:4, function(j) S[, j]))
  E <- exp(S); P <- E / rowSums(E)
  -log(pmax((1 - e) * P[cbind(seq_len(nr), yr)] + e * 0.25, 1e-300))
}

rule("SETUP")
cat(sprintf("  members in the pool: %d\n", M))
cat(sprintf("  lambda grid: %s\n", paste(LAMBDAS, collapse = ", ")))
cat(sprintf("  inner CV folds (respondent-grouped, inside each outer fold): %d\n", INNER_K))

# ---- nested outer loop with inner-CV lambda selection ------------------------
rule("NESTED FIT (lambda chosen by inner CV, never by the outer number)")
set.seed(40)
li <- numeric(n); pf <- numeric(5); lam_sel <- numeric(5); nb <- list()
ucase <- unique(case)
for (k in 1:5) {
  tr <- which(fmap != k); te <- which(fmap == k)
  tr_case <- unique(case[tr])
  inner <- setNames(sample(rep_len(1:INNER_K, length(tr_case))), tr_case)
  icv <- inner[as.character(case[tr])]
  sc <- numeric(length(LAMBDAS))
  for (li_ in seq_along(LAMBDAS)) {
    s <- 0
    for (j in 1:INNER_K) {
      itr <- tr[icv != j]; ite <- tr[icv == j]
      th <- fit_ridge(itr, LAMBDAS[li_])
      s <- s + sum(score_rows(th, ite))
    }
    sc[li_] <- s / length(tr)
  }
  best <- which.min(sc); lam_sel[k] <- LAMBDAS[best]
  th <- fit_ridge(tr, LAMBDAS[best])
  li[te] <- score_rows(th, te); pf[k] <- mean(li[te]); nb[[k]] <- th[1:M]
  cat(sprintf("  fold %d: lambda %-7g  held-out %.5f   (inner curve min at position %d of %d)\n",
              k, LAMBDAS[best], pf[k], best, length(LAMBDAS)))
}
nested <- mean(pf)

rule("RESULT")
cat(sprintf("  ridge pool nested        %.5f\n", nested))
cat(sprintf("  iteration 35 5-member    %.5f\n", 1.12341))
cat(sprintf("  delta                    %+.5f\n", 1.12341 - nested))
cat(sprintf("  lambdas selected         %s\n", paste(lam_sel, collapse = ", ")))
interior <- mean(lam_sel > min(LAMBDAS) & lam_sel < max(LAMBDAS))
cat(sprintf("  gate 5 (lambda interior in a majority of folds): %s (%.0f%%)\n",
            if (interior > 0.5) { "PASS" } else { "FAIL" }, 100 * interior))

saveRDS(list(members = MEMB, nested = nested, perfold = pf, li = li,
             lambda = lam_sel, betas = nb), file.path(DIR, "result.rds"))
cat("\n  wrote result.rds\n")
