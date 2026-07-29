# =============================================================================
# ITERATION 33 — Smooth (ordinal) part-worths inside the latent-class utility.
#
# WHAT THIS FILE IS. A byte-for-byte copy of experiments/iter25_taskpos/run.R
# with ONE change: the penalty on the class-specific taste vector. Folds, design
# matrices, membership model, convergence budget, starts and seeds are the
# incumbent's. Its control is `lcmnl3_both` (1.13863) and NOTHING else.
#
# THE DEFECT
# ----------
# Coding attribute levels as part-worths rather than as numbers was the single
# largest win in this project, worth 0.020 logloss. But it goes one step too far:
# the attributes are ORDINAL TIERS (3-7 levels; Price has 12), and part-worth
# coding throws the ordering away entirely. Each level is estimated as if it had
# no relationship to the level below it. Thin levels therefore get noisy,
# non-monotone estimates that no amount of data on the neighbouring tiers can
# stabilise. report_notes.md:333 already names this as the outstanding refinement:
# "Part-worths for rare levels are noisy... partial pooling toward a smooth trend
# would be the principled refinement."
#
# So this is the second-order correction to the biggest first-order win in the
# log. It is not a search for new signal; the signal is already in the model and
# is being spent on estimating 87 free parameters per class from 21,565 tasks.
#
# THE CHANGE
# ----------
#   Q = LAMBDA_B * I  +  LAMBDA_S * D'D
#
# replacing the scalar ridge `LAMBDA_B * I`. D holds one row per interior knot of
# each ordinal attribute: the second difference (+1, -2, +1) across three
# consecutive surviving levels. Penalising ||D beta||^2 shrinks the part-worth
# profile toward a straight line in the tier index -- toward the LINEAR coding
# that part-worths beat -- without ever forcing it there. The result is partial
# pooling between neighbouring tiers, which is exactly what a noisy rare level
# needs and what an unordered dummy cannot receive.
#
# WHY IT GOES IN THE LATENT-CLASS MODEL AND NOT IN mnl_pw
# -------------------------------------------------------
# The obvious home is model/02_mnl_partworth.R. But `mnl_pw` currently carries
# blend weight 0.000 -- `lcmnl3_both` strictly generalises it and displaced it --
# so a better `mnl_pw` would plausibly change nothing at all downstream. The
# latent-class model carries 0.504, and it is MORE exposed to the defect, not
# less: it estimates a separate 87-vector per class from soft-weighted subsets of
# the respondents, so its effective sample size per part-worth is roughly a third
# of the pooled model's. If noisy rare levels hurt anywhere, they hurt here.
#
# RANK-INDEX SPACING, NOT THE NUMERIC LEVEL VALUES -- stated because it is the
# one real modelling choice in this file. The second differences are taken over
# the RANK of each level within its attribute, so every adjacent pair of tiers is
# treated as one step apart. Using the numeric level codes as the x-axis instead
# would reintroduce, through the prior, precisely the cardinality assumption that
# part-worth coding earned 0.020 by rejecting. Penalising curvature in the tier
# SEQUENCE is the correct ordinal statement; penalising curvature in the tier
# VALUES is not.
#
# LAMBDA_S IS CHOSEN BY NESTED INNER CV, SO THE DECISION NUMBER NEVER SEES ITS
# OWN TUNING. Within each outer fold's training respondents (and again for the
# full refit), respondents are split into two inner folds -- grouped by Case,
# seeded independently, never touching model/artifacts/folds.rds -- each LAMBDA_S
# on a pre-declared grid is fitted and scored on the held-out inner half, and the
# winner is refitted on the whole outer-training set. The outer fold is never
# consulted. Compare this with LAMBDA_B and LAMBDA_G, which iteration 11 fixed a
# priori at 2.0 precisely to avoid tuning on OOF; nested selection is the other
# legitimate way to spend a hyperparameter, and it is what the extra structure
# here needs.
#
# THE NULL IS REACHABLE FROM INSIDE THE PROCEDURE. The grid INCLUDES 0. At
# LAMBDA_S = 0, Q collapses to LAMBDA_B * I and the model is arithmetically
# identical to `lcmnl3_both`. So if smoothing does not help, the inner CV selects
# 0 and this experiment reproduces its own control. That is deliberate: it makes
# the experiment self-falsifying rather than merely comparable, and "0 in every
# fold" is a clean, publishable negative result rather than an ambiguous wash.
#
# Inner fits run at a reduced EM budget (LC_INNER, default 10 iterations, det
# start only). Iteration 11's budget note applies unchanged: under-converging can
# only HURT a fit, never flatter it, so a LAMBDA_S chosen under a smaller budget
# is a conservative choice, not an optimistic one.
#
# EXPECTATION. -0.001 to -0.002 at member level and plausibly ~0 at blend level:
# three separate demonstrations on 26 Jul (bagging, gating, carriers) showed the
# ensemble absorbing member-level repairs. A real possibility, pre-registered, is
# that the inner CV picks 0 everywhere -- LAMBDA_B = 2 may already be doing enough
# shrinking, and 21,565 tasks may simply be enough data for 87 parameters.
#
# PRE-REGISTERED ADOPTION GATES (all three must pass; declared before running)
#   1. MEMBER   Rscript model/compare.R lcmnl3_both lcmnl3_sm  ->  z >= 2,
#      paired with respondent-clustered SE.
#   2. BLEND    swap the name into model/members.txt, Rscript model/06_blend.R,
#      nested must improve on 1.12867. A non-regression check, not a paired test.
#   3. SHIFT    Rscript model/shift_audit.R, retention >= ~80%.
#   Fail any one -> revert members.txt and log it as a failure.
#
# ATTRIBUTION. The control is `lcmnl3_both`, NOT iteration 32's bag. One change
# per experiment. If both iterations win, combining them is a THIRD experiment --
# iteration 08 is the standing reminder that two separately-winning changes had
# never been run together and could not be assumed additive.
#
# LEAKAGE: none. A penalty matrix is a function of the DESIGN only -- which
# attribute levels exist and in what order. No outcome touches it. The inner CV
# splits by respondent, so no respondent is ever scored by a fit that saw them.
#
#   Rscript experiments/iter33_lcsmooth/run.R
# =============================================================================

suppressMessages({ library(data.table) })
source("model/99_utils.R")

if (length(commandArgs(trailingOnly = TRUE))) stop("usage: run.R   (no arguments)")

# Fixed at the incumbent's configuration. Nothing here is a free choice: the
# point of this iteration is that exactly one thing differs from lcmnl3_both.
C        <- 3L
TASKMODE <- "both"
NAME     <- "lcmnl3_sm"

LAMBDA_B <- 2.0     # ridge on class-specific taste parameters
LAMBDA_G <- 2.0     # ridge on membership-model coefficients

# The smoothing grid, PRE-DECLARED. 0 is in it deliberately: at LAMBDA_S = 0 the
# penalty matrix collapses to LAMBDA_B * I and this script is arithmetically
# identical to the incumbent, so the inner CV can reject smoothing outright and
# the experiment reproduces its own control. "0 in every fold" is the clean
# negative result.
LAMBDA_S_GRID <- c(0, 4, 16, 64)
# N_SCREEN / N_MAX are overridable by env var purely so the EM path can be smoke
# tested cheaply. The reported runs use the defaults below.
#
# Budget note, stated up front so the numbers are not over-claimed: this machine is
# shared and saturated, and a fully-converged EM (tol 1e-6, ~60 iterations) costs
# hours per configuration. The schedule below stops at a RELATIVE log-likelihood
# tolerance of 1e-5 -- about 0.2 nats over 21,565 tasks, i.e. 1e-5 of a logloss unit
# -- or 25 iterations, whichever comes first. Under-converging can only HURT the
# model, so the reported OOF is conservative, never flattered. The identical budget
# is applied to every fold, to the full refit, and to both C = 2 and C = 3.
N_SCREEN <- as.integer(Sys.getenv("LC_SCREEN", "3"))   # EM iters used to screen starts
N_MAX    <- as.integer(Sys.getenv("LC_MAX", "25"))     # further EM iters for the winner
TOL      <- as.numeric(Sys.getenv("LC_TOL", "1e-5"))   # relative tol on the mixture LL
# Budget for the INNER CV fits that choose LAMBDA_S. Smaller than the outer
# budget on purpose: iteration 11's note applies unchanged -- under-converging can
# only hurt a fit, never flatter it -- so a LAMBDA_S selected under a reduced
# budget is a conservative choice. The outer fit that actually produces the OOF
# prediction always gets the full N_SCREEN + N_MAX.
N_INNER  <- as.integer(Sys.getenv("LC_INNER", "10"))
# Seed for the inner respondent split. NOT 42, and it never touches
# model/artifacts/folds.rds -- the outer partition is fixed and sacred.
INNER_SEED <- 907L

OUTDIR <- "experiments/iter33_lcsmooth"

# =============================================================================
# 1. Design matrices
# =============================================================================
build_data <- function() {
  long <- readRDS("model/artifacts/long.rds")
  setorder(long, No, alt)

  # every task must be exactly four contiguous rows, alt = 1,2,3,4 in order --
  # the fast reshape below depends on it
  stopifnot(nrow(long) %% 4 == 0)
  stopifnot(all(long$alt == rep(1:4, nrow(long) / 4)))
  stopifnot(all(long$No == rep(long$No[seq(1, nrow(long), 4)], each = 4)))

  # ---- part-worth coding -----------------------------------------------------
  # Reference level = lowest level that occurs on a REAL bundle (alt 1-3).
  # Price is 0 only on the all-zero none-option, so 0 as reference would make the
  # price dummies collinear with the none-constant (singular Hessian).
  pw <- character(0)
  for (a in ATTRS) {
    lv  <- sort(unique(long[alt != 4][[a]]))
    ref <- lv[1]
    for (l in setdiff(lv, ref)) {
      nm <- sprintf("%s_L%s", a, l)
      long[, (nm) := as.numeric(get(a) == l)]
      pw <- c(pw, nm)
    }
  }
  px <- c("Price_x_age", "Price_x_ppark", "Price_x_inc",
          grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))

  # ---- task-position terms (the one change this iteration makes) -------------
  # Task_c is centred on the middle task and scaled to [-1, 1] so its coefficient
  # sits on the same footing as the part-worths and the ridge penalty means the
  # same thing for it as for everything else.
  #
  # Both terms vary WITHIN a choice set, which is what a conditional logit needs:
  # asc4 is 1 only on the none option, and Price differs across the alternatives.
  # A bare Task main effect would be constant within a task and therefore
  # unidentified -- the rank-revealing QR below would drop it, correctly.
  long[, Task_c := (as.numeric(Task) - 10) / 9]
  tp <- character(0)
  if (TASKMODE %in% c("drift", "both")) {
    long[, none_x_Task := asc4 * Task_c]
    tp <- c(tp, "none_x_Task")
  }
  if (TASKMODE %in% c("tilt", "both")) {
    long[, Price_x_Task := Price * Task_c]
    tp <- c(tp, "Price_x_Task")
  }
  cat("task-position terms added:", if (length(tp)) paste(tp, collapse = ", ") else "none", "\n")

  cand <- c("asc2", "asc3", "asc4", pw, px, tp)

  # ---- identification: rank-revealing QR on the TASK-DEMEANED design ---------
  # anything constant within a choice set cannot be identified by a conditional
  # logit. Done on training rows; ASCs listed first so pivoting keeps them.
  Xall  <- as.matrix(long[, ..cand])
  i1    <- seq(1L, nrow(long), by = 4L)
  tr_ti <- which(long$is_test[i1] == FALSE)
  ridx  <- as.vector(rbind(4L * tr_ti - 3L, 4L * tr_ti - 2L, 4L * tr_ti - 1L, 4L * tr_ti))
  Xt    <- Xall[ridx, , drop = FALSE]
  j1    <- seq(1L, nrow(Xt), by = 4L)
  M     <- (Xt[j1, ] + Xt[j1 + 1L, ] + Xt[j1 + 2L, ] + Xt[j1 + 3L, ]) / 4
  Xc    <- Xt - M[rep(seq_len(nrow(M)), each = 4L), ]
  rm(Xt, M, Xall); gc(verbose = FALSE)
  q    <- qr(Xc, tol = 1e-7)
  keep <- cand[sort(q$pivot[seq_len(q$rank)])]
  rm(Xc); gc(verbose = FALSE)
  drp  <- setdiff(cand, keep)
  if (length(drp)) cat("dropped as unidentified:", paste(drp, collapse = ", "), "\n")
  cat("taste parameters per class:", length(keep), "\n")

  # ---- membership design (one row per respondent) ----------------------------
  DEMO_CAT <- c("segmentind", "pparkind", "genderind", "educind", "regionind", "Urbind")
  DEMO_NUM <- c("agea", "incomea", "milesa", "nighta", "yearind", "milesind", "nightind")
  dem <- unique(long[, c("Case", "is_test", DEMO_CAT, DEMO_NUM), with = FALSE])
  setorder(dem, Case)
  stopifnot(nrow(dem) == uniqueN(long$Case))

  Zl <- list(Intercept = rep(1, nrow(dem)))
  for (v in DEMO_CAT) {                      # dummy code, first level = reference
    lv <- sort(unique(dem[[v]]))
    for (l in lv[-1]) Zl[[sprintf("%s_%s", v, l)]] <- as.numeric(dem[[v]] == l)
  }
  # heavy-tailed money/mileage counts get a log; then standardise so the ridge
  # penalty means the same thing for every covariate. Uses FEATURES only (no
  # outcomes), so computing it over all 1398 respondents is not leakage.
  tf <- function(v, x) if (v %in% c("incomea", "milesa", "nighta")) log1p(x) else x
  for (v in DEMO_NUM) {
    x <- tf(v, as.numeric(dem[[v]]))
    Zl[[v]] <- (x - mean(x)) / sd(x)
  }
  Z <- do.call(cbind, Zl)
  rownames(Z) <- as.character(dem$Case)
  cat("membership covariates:", ncol(Z), "\n")

  list(long = long, keep = keep, Z = Z, dem = dem)
}

# =============================================================================
# 1b. The smoothing operator D  (new in this iteration)
# =============================================================================
# One row per INTERIOR knot of each ordinal attribute: the second difference
# (+1, -2, +1) across three consecutive levels, expressed in the `keep` basis.
#
# Three details that are easy to get wrong and are asserted rather than assumed:
#
#  * The REFERENCE level is a knot, not an absent one. build_data() codes levels
#    against the lowest level occurring on a real bundle, so that level's
#    part-worth is fixed at exactly 0 -- a genuine constraint, and the anchor the
#    rest of the profile bends around. It contributes a coefficient of 0 to any
#    row it appears in, i.e. it is simply omitted from that row.
#  * Levels dropped by the rank-revealing QR (HU_L2 is the known case) have no
#    column at all. Their knot is removed and the second difference is taken
#    across the SURVIVING neighbours, which is the best available statement.
#  * Spacing is by RANK, not by the numeric level code -- see the header. Every
#    adjacent pair of surviving tiers counts as one step.
#
# The operator is a function of the DESIGN only. No outcome enters it.
build_smoother <- function(long, keep) {
  rows <- list(); grid <- character(0)
  for (a in ATTRS) {
    lv  <- sort(unique(long[alt != 4][[a]]))       # same construction as build_data
    ref <- lv[1]
    # position of each level's coefficient in the beta vector; NA = fixed at 0
    pos <- vapply(lv, function(l) {
      if (identical(l, ref)) return(NA_integer_)
      j <- match(sprintf("%s_L%s", a, l), keep)
      if (is.na(j)) NA_integer_ else j
    }, integer(1))
    # the reference survives as a fixed-zero knot; a QR-dropped level does not
    alive <- !is.na(pos) | seq_along(lv) == 1L
    p <- pos[alive]
    grid <- c(grid, sprintf("%s: %d of %d levels (%s)", a, sum(alive), length(lv),
                            paste(lv[alive], collapse = ",")))
    if (length(p) < 3L) next
    for (j in 2:(length(p) - 1L)) {
      r <- numeric(length(keep))
      if (!is.na(p[j - 1L])) r[p[j - 1L]] <- r[p[j - 1L]] + 1
      if (!is.na(p[j]))      r[p[j]]      <- r[p[j]]      - 2
      if (!is.na(p[j + 1L])) r[p[j + 1L]] <- r[p[j + 1L]] + 1
      rows[[length(rows) + 1L]] <- r
    }
  }
  cat("--- smoothing knots per attribute ---\n")
  for (g in grid) cat("   ", g, "\n")
  D <- if (length(rows)) do.call(rbind, rows) else matrix(0, 0, length(keep))
  cat("second-difference rows:", nrow(D), "over", length(keep), "taste parameters\n")
  # Sanity: the penalty must touch part-worth columns ONLY. ASCs, the price x
  # demographic interactions and the task-position terms keep the plain ridge.
  touched <- keep[colSums(abs(D)) > 0]
  stopifnot(all(grepl("_L", touched, fixed = TRUE)))
  stopifnot(!any(c("asc2", "asc3", "asc4") %in% touched))
  cat("columns penalised:", length(touched), "(all part-worths; ASCs, Price_x_*",
      "and task terms untouched)\n")
  D
}

# =============================================================================
# 2. Weighted conditional logit  (Newton-Raphson, analytic Hessian)
# =============================================================================
# X     : (4N) x P design, four contiguous rows per task
# yrow  : length-N chosen alternative (1..4)
# wtask : length-N task weight (the respondent's posterior for this class)
# ridge : lambda; objective is  sum_t w_t * ll_t - 0.5*lambda*||beta||^2

cl_prob <- function(X, beta) {
  Vm <- matrix(as.vector(X %*% beta), ncol = 4, byrow = TRUE)
  mx <- pmax(Vm[, 1], Vm[, 2], Vm[, 3], Vm[, 4])
  E  <- exp(Vm - mx)
  S  <- rowSums(E)
  list(P = E / S, lse = mx + log(S), Vm = Vm)
}

cl_ll_task <- function(X, beta, yrow) {
  cp <- cl_prob(X, beta)
  cp$Vm[cbind(seq_along(yrow), yrow)] - cp$lse
}

# ---------------------------------------------------------------------------
# THE ONE CHANGE vs experiments/iter25_taskpos/run.R.
#
# `ridge` was a SCALAR lambda, entering the objective as -0.5*lambda*||beta||^2.
# It may now also be a P x P penalty MATRIX Q, entering as -0.5*beta'Q beta. The
# scalar path is kept and is byte-identical to the incumbent's, so passing a
# scalar (as the C = 1 branch still does) changes nothing whatsoever.
#
# Everything else in this section -- the Newton step, the analytic Hessian, the
# step halving, the tolerance -- is untouched.
# ---------------------------------------------------------------------------
pen_quad <- function(beta, Q)
  if (is.matrix(Q)) as.numeric(crossprod(beta, Q %*% beta)) else Q * sum(beta^2)
pen_grad <- function(beta, Q)
  if (is.matrix(Q)) as.vector(Q %*% beta) else Q * beta

cl_obj <- function(X, beta, yrow, wtask, ridge) {
  sum(wtask * cl_ll_task(X, beta, yrow)) - 0.5 * pen_quad(beta, ridge)
}

cl_newton <- function(X, yrow, wtask, beta, ridge, nsteps = 2, tol = 1e-8) {
  N   <- length(yrow)
  i1  <- seq(1L, 4L * N, by = 4L)
  wr  <- rep(wtask, each = 4L)
  Xch <- X[i1 - 1L + yrow, , drop = FALSE]
  f   <- cl_obj(X, beta, yrow, wtask, ridge)
  for (s in seq_len(nsteps)) {
    cp <- cl_prob(X, beta)
    pv <- as.vector(t(cp$P))
    Xp <- X * pv
    Xb <- Xp[i1, , drop = FALSE] + Xp[i1 + 1L, , drop = FALSE] +
          Xp[i1 + 2L, , drop = FALSE] + Xp[i1 + 3L, , drop = FALSE]
    g  <- colSums(wtask * (Xch - Xb)) - pen_grad(beta, ridge)
    A  <- crossprod(X, X * (wr * pv))
    B  <- crossprod(Xb, Xb * wtask)
    H  <- A - B                                   # = -(Hessian of the loglik)
    if (is.matrix(ridge)) H <- H + ridge else diag(H) <- diag(H) + ridge
    step <- tryCatch(solve(H, g), error = function(e) {
      diag(H) <- diag(H) + 1e-4 * max(diag(H)); solve(H, g)
    })
    # step halving: never accept a move that lowers the penalised objective
    ok <- FALSE
    for (h in 0:12) {
      bnew <- beta + step / 2^h
      fnew <- cl_obj(X, bnew, yrow, wtask, ridge)
      if (is.finite(fnew) && fnew >= f) { ok <- TRUE; break }
    }
    if (!ok) break
    rel  <- abs(fnew - f) / (abs(f) + 1)
    beta <- bnew; f <- fnew
    if (rel < tol) break
  }
  list(beta = beta, obj = f)
}

# =============================================================================
# 3. Membership model: multinomial logit on soft targets
# =============================================================================
row_softmax <- function(eta) {
  mx <- do.call(pmax, lapply(seq_len(ncol(eta)), function(j) eta[, j]))
  E  <- exp(eta - mx)
  E / rowSums(E)
}
row_lse <- function(eta) {
  mx <- do.call(pmax, lapply(seq_len(ncol(eta)), function(j) eta[, j]))
  mx + log(rowSums(exp(eta - mx)))
}

fit_membership <- function(Z, H, G0, ridge) {
  C <- ncol(H); p <- ncol(Z)
  if (C == 1) return(matrix(0, p, 0))
  fn <- function(g) {
    G   <- matrix(g, p, C - 1)
    eta <- cbind(0, Z %*% G)
    -(sum(H * (eta - row_lse(eta))) - 0.5 * ridge * sum(g^2))
  }
  gr <- function(g) {
    G   <- matrix(g, p, C - 1)
    eta <- cbind(0, Z %*% G)
    Pi  <- row_softmax(eta)
    -as.vector(crossprod(Z, (H - Pi)[, -1, drop = FALSE]) - ridge * G)
  }
  o <- optim(as.vector(G0), fn, gr, method = "BFGS", control = list(maxit = 300))
  matrix(o$par, p, C - 1)
}

membership_prob <- function(Z, G) {
  if (!ncol(G)) return(matrix(1, nrow(Z), 1))
  row_softmax(cbind(0, Z %*% G))
}

# =============================================================================
# 4. EM
# =============================================================================
# resp : length-N integer, which row of Z (and of H) each task belongs to
# Returns betas (list), G, H, and the training mixture log-likelihood.

em_fit <- function(X, yrow, resp, Z, C, H0, n_iter, betas0 = NULL, G0 = NULL,
                   verbose = TRUE, tag = "", Qb = LAMBDA_B) {
  R <- nrow(Z); P <- ncol(X)
  betas <- if (is.null(betas0)) replicate(C, rep(0, P), simplify = FALSE) else betas0
  G     <- if (is.null(G0)) matrix(0, ncol(Z), max(C - 1, 0)) else G0
  H     <- H0
  path  <- numeric(0)
  prev  <- -Inf

  for (it in seq_len(n_iter)) {
    ## ---- M-step: class-specific tastes ------------------------------------
    # Generalized EM: the M-step need not be solved exactly, only improved. Each
    # Newton step costs a P x P Hessian (the dominant cost of the whole fit), so
    # take a few early while beta is far from the optimum and one thereafter --
    # warm-started from the previous iteration, one step is close to exact.
    nst <- if (it <= 2) 3L else if (it <= 5) 2L else 1L
    for (cc in seq_len(C)) {
      w <- H[resp, cc]
      betas[[cc]] <- cl_newton(X, yrow, w, betas[[cc]], Qb, nsteps = nst)$beta
    }
    ## ---- M-step: membership ------------------------------------------------
    G <- fit_membership(Z, H, G, LAMBDA_G)

    ## ---- E-step ------------------------------------------------------------
    # respondent-level log-likelihood under each class
    LLr <- matrix(0, R, C)
    for (cc in seq_len(C)) {
      llt <- cl_ll_task(X, betas[[cc]], yrow)
      s   <- rowsum(llt, resp, reorder = TRUE)   # rows come back in order 1..R
      stopifnot(nrow(s) == R, identical(rownames(s), as.character(seq_len(R))))
      LLr[, cc] <- as.vector(s)
    }
    logpi <- log(pmax(membership_prob(Z, G), 1e-12))
    num   <- logpi + LLr
    ll    <- sum(row_lse(num))                    # training mixture log-likelihood
    H     <- row_softmax(num)
    path  <- c(path, ll)
    if (verbose) cat(sprintf("  %s iter %2d  LL = %.3f  shares = %s\n", tag, it, ll,
                             paste(sprintf("%.3f", colMeans(H)), collapse = " ")))
    if (is.finite(prev) && abs(ll - prev) / (abs(prev) + 1) < TOL) { prev <- ll; break }
    prev <- ll
  }
  list(betas = betas, G = G, H = H, ll = prev, path = path)
}

# resp must be 1..R in increasing blocks for the rowsum trick above; build it that way
make_resp <- function(case_of_task) {
  u <- sort(unique(case_of_task))
  match(case_of_task, u)
}

init_H <- function(C, R, none_rate, kind, seed = 1) {
  if (C == 1) return(matrix(1, R, 1))
  if (kind == "det") {
    # deterministic split on how often the respondent takes the none-option --
    # a real and stable axis of heterogeneity (30.2% of all choices)
    grp <- cut(rank(none_rate, ties.method = "first"),
               breaks = C, labels = FALSE)
    H <- matrix(0.15 / (C - 1), R, C)
    H[cbind(seq_len(R), grp)] <- 0.85
  } else {
    set.seed(seed)
    H <- matrix(rgamma(R * C, 1), R, C)
    H <- H / rowSums(H)
  }
  H
}

# =============================================================================
# 5. Driver
# =============================================================================
# The "finalize" mode inherited from iteration 11 (pick the better of C = 2/3)
# is deliberately dropped: C is fixed at the incumbent's 3 and there is nothing
# to select between.
cat("=== latent-class conditional logit, C =", C,
    "| task-position mode:", TASKMODE, "===\n")

bd   <- build_data()
long <- bd$long; keep <- bd$keep; Z <- bd$Z
DSM  <- build_smoother(long, keep)
cat("lambda_s grid:", paste(LAMBDA_S_GRID, collapse = ", "),
    "| inner budget", N_INNER, "EM iters, 2 respondent-grouped inner folds\n")

tasks <- unique(long[, .(No, Case, y, is_test)]); setorder(tasks, No)
folds <- readRDS("model/artifacts/folds.rds")
tasks <- merge(tasks, folds[, .(No, fold)], by = "No", all.x = TRUE)
setorder(tasks, No)
# task i in `tasks` == long rows 4i-3 .. 4i.  Everything downstream assumes this.
stopifnot(identical(tasks$No, long$No[seq(1L, nrow(long), by = 4L)]))
stopifnot(nrow(tasks) == nrow(long) / 4)
stopifnot(sum(tasks$is_test == FALSE) == 21565, sum(tasks$is_test == TRUE) == 4997)
stopifnot(!anyNA(tasks$fold[tasks$is_test == FALSE]))

Xall <- as.matrix(long[, ..keep])
tr_t <- which(tasks$is_test == FALSE)
te_t <- which(tasks$is_test == TRUE)
rows_of <- function(ti) as.vector(rbind(4L * ti - 3L, 4L * ti - 2L, 4L * ti - 1L, 4L * ti))

# respondent-level none-rate, used only to seed the EM (training choices, fitting set)
none_rate_all <- tasks[is_test == FALSE, .(nr = mean(y == 4)), by = Case]

fit_one <- function(train_tasks, lambda_s = 0, n_iter = NULL, quiet = FALSE) {
  ti   <- train_tasks
  X    <- Xall[rows_of(ti), , drop = FALSE]
  yrow <- tasks$y[ti]
  cases <- tasks$Case[ti]
  ucase <- sort(unique(cases))
  resp <- match(cases, ucase)
  Zf   <- Z[as.character(ucase), , drop = FALSE]
  nr   <- none_rate_all[match(ucase, Case), nr]

  if (C == 1) {
    b <- cl_newton(X, yrow, rep(1, length(yrow)), rep(0, ncol(X)), LAMBDA_B, nsteps = 60)$beta
    return(list(betas = list(b), G = matrix(0, ncol(Z), 0), H = matrix(1, length(ucase), 1),
                ll = NA_real_, path = numeric(0), ucase = ucase))
  }

  # THE PENALTY. lambda_s = 0 gives exactly LAMBDA_B * I, i.e. the incumbent.
  Qb <- if (lambda_s > 0) LAMBDA_B * diag(ncol(X)) + lambda_s * crossprod(DSM)
        else LAMBDA_B

  # Inner CV fits run one deterministic start at a reduced budget; the outer fits
  # keep the incumbent's two-start screen and full budget, unchanged.
  if (!is.null(n_iter)) {
    fin <- em_fit(X, yrow, resp, Zf, C, init_H(C, length(ucase), nr, "det", 0),
                  n_iter, verbose = FALSE, tag = "[inner]", Qb = Qb)
    fin$ucase <- ucase
    return(fin)
  }

  starts <- list(list(kind = "det", seed = 0),
                 list(kind = "rand", seed = 4242))
  scr <- lapply(seq_along(starts), function(i) {
    s <- starts[[i]]
    em_fit(X, yrow, resp, Zf, C, init_H(C, length(ucase), nr, s$kind, s$seed),
           N_SCREEN, verbose = !quiet, tag = sprintf("[screen %s]", s$kind), Qb = Qb)
  })
  lls <- sapply(scr, function(z) tail(z$path, 1))
  cat("  start screening LL:", paste(sprintf("%.2f", lls), collapse = " / "),
      "-> keeping", starts[[which.max(lls)]]$kind, "\n")
  w <- scr[[which.max(lls)]]
  fin <- em_fit(X, yrow, resp, Zf, C, w$H, N_MAX, betas0 = w$betas, G0 = w$G,
                verbose = !quiet, tag = "[final]", Qb = Qb)
  fin$path  <- c(w$path, fin$path)
  fin$ucase <- ucase
  fin
}

# =============================================================================
# 4b. Nested inner CV that chooses LAMBDA_S  (new in this iteration)
# =============================================================================
# Given the tasks of ONE outer fitting set, split its RESPONDENTS in two, fit
# each candidate LAMBDA_S on one inner half and score it on the other, and return
# the grid minimiser. The outer fold being predicted is never touched, so the
# decision number never sees this tuning.
#
# Splitting by respondent, not by row, for the same reason the outer folds do:
# a respondent contributes 19 correlated tasks, and splitting rows would let a
# fit learn the very people it is about to be scored on.
choose_lambda_s <- function(train_tasks, tag) {
  cs   <- tasks$Case[train_tasks]
  ifold <- make_case_folds(cs, k = 2, seed = INNER_SEED)
  tot <- numeric(length(LAMBDA_S_GRID))
  for (j in 1:2) {
    itr <- train_tasks[ifold != j]
    iva <- train_tasks[ifold == j]
    stopifnot(length(intersect(tasks$Case[itr], tasks$Case[iva])) == 0)
    for (m in seq_along(LAMBDA_S_GRID)) {
      f <- fit_one(itr, lambda_s = LAMBDA_S_GRID[m], n_iter = N_INNER, quiet = TRUE)
      tot[m] <- tot[m] + logloss(tasks$y[iva], predict_lc(f, iva)) / 2
    }
  }
  best <- LAMBDA_S_GRID[which.min(tot)]
  cat(sprintf("  [%s] inner CV over lambda_s: %s -> chose %g\n", tag,
              paste(sprintf("%g=%.5f", LAMBDA_S_GRID, tot), collapse = "  "), best))
  list(best = best, curve = tot)
}

# The ONLY inputs are the fitted (betas, G), the task design, and the target
# respondents' DEMOGRAPHICS. Their choices are structurally unreachable from here.
# `fixed_pi` is an ablation switch used for diagnostics: it replaces the
# demographic-driven mixing weights with one constant vector for everybody, which
# isolates how much of the gain comes from the demographic channel as opposed to
# the mixture form itself.
predict_lc <- function(fit, target_tasks, fixed_pi = NULL) {
  ti <- target_tasks
  X  <- Xall[rows_of(ti), , drop = FALSE]
  cs <- tasks$Case[ti]
  if (is.null(fixed_pi)) {
    Zp <- Z[as.character(cs), , drop = FALSE]   # one row per TASK, demographics only
    Pi <- membership_prob(Zp, fit$G)
  } else {
    Pi <- matrix(fixed_pi, length(ti), length(fixed_pi), byrow = TRUE)
  }
  out <- matrix(0, length(ti), 4)
  for (cc in seq_along(fit$betas)) {
    out <- out + Pi[, cc] * cl_prob(X, fit$betas[[cc]])$P
  }
  out
}

# ---- 5-fold OOF -------------------------------------------------------------
oof <- matrix(NA_real_, length(tr_t), 4)
oof_fix <- matrix(NA_real_, length(tr_t), 4)   # ablation, diagnostic only
t0  <- proc.time()
lam_chosen <- numeric(5)
for (k in 1:5) {
  cat("--- fold", k, "---\n")
  itr <- tr_t[tasks$fold[tr_t] != k]
  iva <- tr_t[tasks$fold[tr_t] == k]
  stopifnot(length(intersect(tasks$Case[itr], tasks$Case[iva])) == 0)
  # lambda_s is chosen using ONLY the respondents in itr. Fold k is untouched.
  sel <- choose_lambda_s(itr, sprintf("fold %d", k))
  lam_chosen[k] <- sel$best
  fit <- fit_one(itr, lambda_s = sel$best)
  oof[match(iva, tr_t), ] <- predict_lc(fit, iva)
  oof_fix[match(iva, tr_t), ] <- predict_lc(fit, iva, fixed_pi = colMeans(fit$H))
  cat("  fold", k, "logloss:",
      round(logloss(tasks$y[iva], oof[match(iva, tr_t), ]), 5),
      " elapsed", round((proc.time() - t0)[3] / 60, 1), "min\n")
}
ll_oof <- logloss(tasks$y[tr_t], oof)
cat(sprintf("\n>>> %s honest OOF logloss: %.5f   (control lcmnl3_both 1.13863)\n",
            NAME, ll_oof))
cat("    lambda_s chosen per fold:", paste(lam_chosen, collapse = "  "), "\n")
if (all(lam_chosen == 0))
  cat("    => the inner CV rejected smoothing in EVERY fold. This run reproduces\n",
      "      its own control, and that IS the result: at LAMBDA_B = 2 the\n",
      "      part-worths are already shrunk enough that ordinal pooling adds\n",
      "      nothing. Report it as a negative, do not adopt.\n")
if (C > 1) {
  ll_fix <- logloss(tasks$y[tr_t], oof_fix)
  cat(sprintf("    ablation, constant mixing weights for everyone: %.5f\n", ll_fix))
  cat(sprintf("    => demographic membership channel is worth %+.5f of the total %+.5f\n",
              ll_fix - ll_oof, 1.15686 - ll_oof))
}

oof <- clip_norm(oof)
saveRDS(data.table(No = tasks$No[tr_t], p1 = oof[, 1], p2 = oof[, 2],
                   p3 = oof[, 3], p4 = oof[, 4]),
        sprintf("model/artifacts/oof_%s.rds", NAME))

# ---- refit on ALL training respondents, predict test -------------------------
# The refit re-runs the inner CV on the full training set rather than reusing a
# fold's lambda_s: the selection procedure, not one of its outputs, is what this
# iteration is testing, and the test predictions must come from the same
# procedure the OOF predictions did.
cat("--- full refit ---\n")
sel_full <- choose_lambda_s(tr_t, "full")
full <- fit_one(tr_t, lambda_s = sel_full$best)
te <- clip_norm(predict_lc(full, te_t))
saveRDS(data.table(No = tasks$No[te_t], p1 = te[, 1], p2 = te[, 2],
                   p3 = te[, 3], p4 = te[, 4]),
        sprintf("model/artifacts/test_%s.rds", NAME))

# =============================================================================
# 6. Report material
# =============================================================================
# Label switching: the mixture likelihood is invariant to permuting the classes,
# so the raw indices mean nothing across runs. Relabel by descending posterior
# share before printing anything. Predictions are unaffected.
if (C > 1) {
  ord <- order(colMeans(full$H), decreasing = TRUE)
  betas <- full$betas[ord]
  Hs    <- full$H[, ord, drop = FALSE]
  cat("\n--- SEGMENTS (full-data fit, relabelled by descending share) ---\n")
  cat("posterior share (avg over training respondents):",
      paste(sprintf("%.3f", colMeans(Hs)), collapse = "  "), "\n")
  cat("hard assignment counts:", paste(table(factor(max.col(Hs), levels = 1:C)),
                                       collapse = "  "), "\n")

  # how often each segment takes the none-option, and how price-sensitive it is
  nr_f <- none_rate_all[match(full$ucase, Case), nr]
  cat("weighted mean none-rate by segment:",
      paste(sprintf("%.3f", colSums(Hs * nr_f) / colSums(Hs)), collapse = "  "), "\n")

  B <- do.call(cbind, betas); rownames(B) <- keep
  cat("\nnone-option constant (asc4) by segment:\n")
  if ("asc4" %in% keep) print(round(B["asc4", ], 3))
  pl <- grep("^Price_L", keep, value = TRUE)
  if (length(pl)) {
    cat("\nprice part-worths by segment (utility relative to Price level 1):\n")
    print(round(B[pl, , drop = FALSE], 3))
  }
  cat("\nmost DISTINGUISHING tastes (largest spread across segments):\n")
  sp <- apply(B, 1, function(r) max(r) - min(r))
  top <- names(sort(sp, decreasing = TRUE))[1:15]
  print(round(B[top, , drop = FALSE], 3))

  cat("\nmembership model: strongest demographic drivers (vs class 1 baseline)\n")
  Gs <- full$G
  if (ncol(Gs)) {
    rownames(Gs) <- colnames(Z)
    for (j in seq_len(ncol(Gs))) {
      v <- sort(Gs[, j], decreasing = TRUE)
      cat(sprintf(" class %d vs 1: %s\n", j + 1,
                  paste(sprintf("%s %+.2f", names(c(head(v, 4), tail(v, 4))),
                                c(head(v, 4), tail(v, 4))), collapse = " | ")))
    }
    # how much does the demographic channel actually move the mixture?
    Pi <- membership_prob(Z[as.character(full$ucase), , drop = FALSE], full$G)
    cat(" predicted-share spread across respondents (sd of pi_rc):",
        paste(sprintf("%.3f", apply(Pi, 2, sd)), collapse = "  "), "\n")
  }
  cat("\nLL path (training mixture log-likelihood):\n")
  print(round(full$path, 2))
  # What the penalty actually did: curvature of the fitted part-worth profiles.
  # If lambda_s > 0 was selected, ||D beta|| must be visibly smaller than the
  # incumbent's. If it is not, the penalty is not biting and the result is a
  # wash for a mechanical reason rather than a substantive one.
  cat("\n--- curvature of the fitted part-worth profiles, ||D beta|| by segment ---\n")
  cat(paste(sprintf("%.3f", apply(B, 2, function(b) sqrt(sum((DSM %*% b)^2)))),
            collapse = "  "), "\n")
  cat("lambda_s selected for this full-data fit:", sel_full$best, "\n")

  saveRDS(list(betas = betas, G = full$G, keep = keep, zcols = colnames(Z),
               shares = colMeans(Hs), path = full$path, oof = ll_oof,
               lambda_s = lam_chosen, lambda_s_full = sel_full$best,
               inner_curve = sel_full$curve, D = DSM),
          file.path(OUTDIR, sprintf("fit_C%d.rds", C)))
}
cat("OK\n")
