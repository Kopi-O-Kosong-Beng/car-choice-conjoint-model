# =============================================================================
# ITERATION 14 — Extending the latent-class conditional logit (iteration 11).
#
# CONTEXT
# -------
# lcmnl3 (3 classes, demographic membership) scored honest OOF 1.14396, beat the
# fixed part-worth logit mnl_pw (1.15686) by z = 3.77, and took the largest weight
# in the production blend (0.447), displacing mnl_pw entirely. It is the confirmed
# winner of the round, so it is the thing worth pushing on.
#
# This script is a faithful refactor of experiments/iter11_latent_class/run.R with
# three orthogonal extension axes bolted on, each selectable by a single CLI mode
# so that exactly ONE thing changes per experiment.
#
#   mode  artifact    change relative to lcmnl3
#   ----  ----------  ----------------------------------------------------------
#   3x    lcmnl3x     NOTHING. Harness fidelity check: must reproduce 1.14396.
#   C4    lcmnl4      C = 4 classes.
#   C5    lcmnl5      C = 5 classes.
#   3b    lcmnl3b     richer class-membership model (income x age, income x
#                     segment, quadratics in log-income and age).
#   3p    lcmnl3p     roughness (2nd-difference) penalty on each class's price
#                     part-worth curve. See "AXIS C" below.
#
# HYPOTHESES
# ----------
# (a) MORE CLASSES. The C = 3 solution splits the population into a price-elastic
#     buyer, a near-price-insensitive buyer, and a heavy none-taker (posterior
#     none-rates 0.228 / 0.049 / 0.648). Adding classes buys resolution if there
#     is more discrete structure, but each class costs 85 taste parameters fitted
#     on an ever-thinner slice of 1,135 respondents. Expectation: C = 4 is roughly
#     a coin flip; C = 5 more likely to degenerate (a class collapsing to a
#     near-empty share, or a class share driven to a corner). The log-likelihood
#     path is printed for every fold so under-convergence and collapse are visible
#     rather than assumed away.
#
# (b) RICHER MEMBERSHIP. The membership multinomial currently enters demographics
#     linearly. Note the base build ALREADY log-transforms the heavy-tailed money
#     and mileage variables (log1p on incomea/milesa/nighta) — verified and
#     re-stated at runtime — so "use log(income) not raw income" is already true
#     and cannot be the source of a gain. What is genuinely absent is INTERACTION:
#     a $150k 30-year-old and a $150k 60-year-old are assumed to sit at the same
#     point on the income axis. Adds log-income x age, log-income x segment (5),
#     log-income^2, age^2. 29 -> 37 membership covariates.
#
# (c) CLASS-SPECIFIC PRICE SHAPE. The brief asked whether each class gets its own
#     full price part-worth vector or shares a shape. ANSWER, verified in code and
#     re-asserted at runtime below: the shape is ALREADY FULLY FREE. `keep`
#     contains Price_L2..Price_L12 and every class carries its own independent
#     copy, so there is nothing to "free" — the C = 3 fit already estimates three
#     completely unrelated price curves, and they are not scale multiples of each
#     other (one is monotone-decreasing, one is essentially flat-to-increasing).
#     So the interesting question runs the OTHER way. Look at the fitted class-1
#     curve: ... -0.654, -0.972, -2.002, -1.654. It falls off a cliff at level 11
#     and then goes back UP at level 12. That non-monotonicity is not an economic
#     story, it is 11 free parameters fitted on a third of the respondents. Mode
#     3p therefore keeps the curve class-specific but adds a second-difference
#     roughness penalty, which shrinks each class's curve toward a straight line
#     WITHOUT forcing the classes to share anything and WITHOUT flattening genuine
#     concavity (concavity is a constant second difference; only WIGGLE is
#     penalised). If 3p wins, the free curves were partly noise; if it loses, the
#     kinks are real and that is a finding in itself. LAMBDA_S = 10 is fixed a
#     priori from the observed curvature scale (prior sd 1/sqrt(10) = 0.32 per
#     second difference: loose against the 0.05-0.2 wiggle in the smooth region,
#     tight against the 0.7-1.4 spike at the top). It is NOT tuned on OOF.
#
# THE LEAK THAT WOULD FAKE A LARGE GAIN
# -------------------------------------
# At prediction time class membership MUST come from demographics only:
#     pi_rc = softmax_c(z_r' gamma_c),  z_r = demographics of respondent r.
# Running an E-step on a held-out respondent — i.e. using their own 19 choices to
# work out which segment they are in — is leakage and would look like a huge win.
# Three defences, all executed at runtime, all printed:
#   LEAK CHECK A (structural, decisive): the held-out respondents' choices are
#     randomly permuted and the fold is re-predicted. Predictions must be
#     BIT-IDENTICAL. They are, because predict_lc() never reads y. stopifnot.
#   LEAK CHECK B (quantitative): the leaky variant is computed on fold 1 and
#     reported, so the size of the trap is on the record. It is never saved.
#   LEAK CHECK C (input hygiene): asserts the membership design Z is built purely
#     from demographic columns, and that no validation Case appears in the fitting
#     set of its own fold.
# Standardisation of Z uses all 1,398 respondents but only FEATURES, never
# outcomes, so it is not leakage. The EM seed uses the none-rate of FITTING
# respondents only — asserted.
#
# GUARDS CARRIED OVER FROM ITERATION 11
#   * folds read from model/artifacts/folds.rds, grouped by Case, never regenerated
#   * label switching handled by relabelling classes by descending share before
#     any coefficient is printed; predictions are invariant to labelling
#   * multiple EM starts screened on TRAINING log-likelihood only
#   * ridge penalties fixed a priori (lambda_b = 2, lambda_g = 2)
#   * identical convergence budget (3 screening + <=25 EM iters, rel tol 1e-5) for
#     every mode, so the C = 3 / 4 / 5 comparison is like-for-like. Under-
#     converging can only HURT, so reported OOF is conservative, never flattered.
#
# USAGE
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter14_lc_extend/run.R 3x
#   ... run.R C4 | C5 | 3b | 3p
# =============================================================================

suppressMessages({ library(data.table) })
source("model/99_utils.R")

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: run.R <3x|C4|C5|3b|3p>")
MODE <- args[1]

CFG <- list(
  "3x" = list(C = 3L, memb = "base", lsm = 0,  name = "lcmnl3x"),
  "C4" = list(C = 4L, memb = "base", lsm = 0,  name = "lcmnl4"),
  "C5" = list(C = 5L, memb = "base", lsm = 0,  name = "lcmnl5"),
  "3b" = list(C = 3L, memb = "rich", lsm = 0,  name = "lcmnl3b"),
  "3p" = list(C = 3L, memb = "base", lsm = 10, name = "lcmnl3p")
)
if (!MODE %in% names(CFG)) stop("mode must be one of: ", paste(names(CFG), collapse = ", "))
cfg      <- CFG[[MODE]]
C        <- cfg$C
MEMB     <- cfg$memb
LAMBDA_S <- cfg$lsm
NAME     <- cfg$name

LAMBDA_B <- 2.0     # ridge on class-specific taste parameters   (as iteration 11)
LAMBDA_G <- 2.0     # ridge on membership-model coefficients     (as iteration 11)
# Budget identical to iteration 11 so C = 3 / 4 / 5 is a like-for-like comparison.
# The env vars exist ONLY so the code path can be smoke tested cheaply; every
# reported run uses the defaults below.
N_SCREEN <- as.integer(Sys.getenv("LC_SCREEN", "3"))   # EM iters to screen starts
N_MAX    <- as.integer(Sys.getenv("LC_MAX", "25"))     # further EM iters for winner
TOL      <- as.numeric(Sys.getenv("LC_TOL", "1e-5"))   # relative tol on mixture LL
SMOKE    <- nzchar(Sys.getenv("LC_SMOKE", ""))         # skip artifact writes

OUTDIR <- "experiments/iter14_lc_extend"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# never silently clobber someone else's artifact
if (!SMOKE) for (f in sprintf("model/artifacts/%s_%s.rds", c("oof", "test"), NAME)) {
  if (file.exists(f)) stop("refusing to overwrite existing artifact: ", f)
}

cat("=== iteration 14 | mode", MODE, "| artifact", NAME, "===\n")
cat("    C =", C, " membership =", MEMB, " price roughness lambda =", LAMBDA_S, "\n\n")

# =============================================================================
# 1. Design matrices
# =============================================================================
build_data <- function(memb = "base") {
  long <- readRDS("model/artifacts/long.rds")
  setorder(long, No, alt)

  stopifnot(nrow(long) %% 4 == 0)
  stopifnot(all(long$alt == rep(1:4, nrow(long) / 4)))
  stopifnot(all(long$No == rep(long$No[seq(1, nrow(long), 4)], each = 4)))

  # ---- part-worth coding -----------------------------------------------------
  # Reference level = lowest level occurring on a REAL bundle (alt 1-3). Price is
  # 0 only on the all-zero none-option, so 0 as reference would make the price
  # dummies collinear with the none-constant (singular Hessian).
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
  cand <- c("asc2", "asc3", "asc4", pw, px)

  # ---- identification: rank-revealing QR on the TASK-DEMEANED design ---------
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
  # LEAK CHECK C (input hygiene): the membership design may only see demographics.
  OUTCOME_COLS <- c("y", "chosen", ATTRS, "alt", "Task", "No", "fold")
  stopifnot(length(intersect(c(DEMO_CAT, DEMO_NUM), OUTCOME_COLS)) == 0)

  dem <- unique(long[, c("Case", "is_test", DEMO_CAT, DEMO_NUM), with = FALSE])
  setorder(dem, Case)
  stopifnot(nrow(dem) == uniqueN(long$Case))

  Zl <- list(Intercept = rep(1, nrow(dem)))
  for (v in DEMO_CAT) {                      # dummy code, first level = reference
    lv <- sort(unique(dem[[v]]))
    for (l in lv[-1]) Zl[[sprintf("%s_%s", v, l)]] <- as.numeric(dem[[v]] == l)
  }
  # Heavy-tailed money/mileage counts get a log (incomea runs 9e3 .. 5e6, a 555x
  # range with a 1% tail above $500k); then standardise so the ridge penalty means
  # the same thing for every covariate. Uses FEATURES only, so computing it over
  # all 1398 respondents is not leakage.
  tf <- function(v, x) if (v %in% c("incomea", "milesa", "nighta")) log1p(x) else x
  for (v in DEMO_NUM) {
    x <- tf(v, as.numeric(dem[[v]]))
    Zl[[v]] <- (x - mean(x)) / sd(x)
  }
  cat("log-transformed membership covariates:",
      paste(intersect(DEMO_NUM, c("incomea", "milesa", "nighta")), collapse = ", "),
      "  (so 'use log income' is ALREADY the base spec)\n")

  if (memb == "rich") {
    # --- AXIS (b): the one change. Interactions absent from the base spec. ------
    std  <- function(x) (x - mean(x)) / sd(x)
    linc <- Zl[["incomea"]]                 # standardised log1p(income)
    ag   <- Zl[["agea"]]                    # standardised age
    Zl[["linc_x_age"]] <- std(linc * ag)
    Zl[["linc_sq"]]    <- std(linc^2)
    Zl[["age_sq"]]     <- std(ag^2)
    for (l in sort(unique(dem$segmentind))[-1])
      Zl[[sprintf("linc_x_seg%s", l)]] <- std(linc * as.numeric(dem$segmentind == l))
    cat("RICH membership: added linc_x_age, linc_sq, age_sq, linc_x_seg*\n")
  }

  Z <- do.call(cbind, Zl)
  rownames(Z) <- as.character(dem$Case)
  stopifnot(!anyNA(Z), all(is.finite(Z)))
  cat("membership covariates:", ncol(Z), "\n")

  list(long = long, keep = keep, Z = Z, dem = dem)
}

# =============================================================================
# 2. Weighted conditional logit  (Newton-Raphson, analytic Hessian)
# =============================================================================
# Pen2 is NULL for every mode except 3p, so the arithmetic for the other modes is
# byte-for-byte the iteration 11 path (mode 3x exists to prove that).

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

cl_obj <- function(X, beta, yrow, wtask, ridge, Pen2 = NULL) {
  v <- sum(wtask * cl_ll_task(X, beta, yrow)) - 0.5 * ridge * sum(beta^2)
  if (!is.null(Pen2)) v <- v - 0.5 * as.numeric(crossprod(beta, Pen2 %*% beta))
  v
}

cl_newton <- function(X, yrow, wtask, beta, ridge, nsteps = 2, tol = 1e-8, Pen2 = NULL) {
  N   <- length(yrow)
  i1  <- seq(1L, 4L * N, by = 4L)
  wr  <- rep(wtask, each = 4L)
  Xch <- X[i1 - 1L + yrow, , drop = FALSE]
  f   <- cl_obj(X, beta, yrow, wtask, ridge, Pen2)
  for (s in seq_len(nsteps)) {
    cp <- cl_prob(X, beta)
    pv <- as.vector(t(cp$P))
    Xp <- X * pv
    Xb <- Xp[i1, , drop = FALSE] + Xp[i1 + 1L, , drop = FALSE] +
          Xp[i1 + 2L, , drop = FALSE] + Xp[i1 + 3L, , drop = FALSE]
    g  <- colSums(wtask * (Xch - Xb)) - ridge * beta
    A  <- crossprod(X, X * (wr * pv))
    B  <- crossprod(Xb, Xb * wtask)
    H  <- A - B                                   # = -(Hessian of the loglik)
    diag(H) <- diag(H) + ridge
    if (!is.null(Pen2)) { g <- g - as.vector(Pen2 %*% beta); H <- H + Pen2 }
    step <- tryCatch(solve(H, g), error = function(e) {
      diag(H) <- diag(H) + 1e-4 * max(diag(H)); solve(H, g)
    })
    ok <- FALSE
    for (h in 0:12) {                             # step halving
      bnew <- beta + step / 2^h
      fnew <- cl_obj(X, bnew, yrow, wtask, ridge, Pen2)
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
  Cn <- ncol(H); p <- ncol(Z)
  if (Cn == 1) return(matrix(0, p, 0))
  fn <- function(g) {
    G   <- matrix(g, p, Cn - 1)
    eta <- cbind(0, Z %*% G)
    -(sum(H * (eta - row_lse(eta))) - 0.5 * ridge * sum(g^2))
  }
  gr <- function(g) {
    G   <- matrix(g, p, Cn - 1)
    eta <- cbind(0, Z %*% G)
    Pi  <- row_softmax(eta)
    -as.vector(crossprod(Z, (H - Pi)[, -1, drop = FALSE]) - ridge * G)
  }
  o <- optim(as.vector(G0), fn, gr, method = "BFGS", control = list(maxit = 300))
  matrix(o$par, p, Cn - 1)
}

membership_prob <- function(Z, G) {
  if (!ncol(G)) return(matrix(1, nrow(Z), 1))
  row_softmax(cbind(0, Z %*% G))
}

# =============================================================================
# 4. EM
# =============================================================================
em_fit <- function(X, yrow, resp, Z, Cn, H0, n_iter, betas0 = NULL, G0 = NULL,
                   verbose = TRUE, tag = "") {
  R <- nrow(Z); P <- ncol(X)
  betas <- if (is.null(betas0)) replicate(Cn, rep(0, P), simplify = FALSE) else betas0
  G     <- if (is.null(G0)) matrix(0, ncol(Z), max(Cn - 1, 0)) else G0
  H     <- H0
  path  <- numeric(0)
  prev  <- -Inf

  for (it in seq_len(n_iter)) {
    ## ---- M-step: class-specific tastes ------------------------------------
    # Generalized EM: the M-step need only improve, not solve. Each Newton step
    # costs a P x P Hessian (the dominant cost), so take a few early and one
    # thereafter, warm-started from the previous iteration.
    nst <- if (it <= 2) 3L else if (it <= 5) 2L else 1L
    for (cc in seq_len(Cn)) {
      w <- H[resp, cc]
      betas[[cc]] <- cl_newton(X, yrow, w, betas[[cc]], LAMBDA_B,
                               nsteps = nst, Pen2 = PEN2)$beta
    }
    ## ---- M-step: membership ------------------------------------------------
    G <- fit_membership(Z, H, G, LAMBDA_G)

    ## ---- E-step ------------------------------------------------------------
    LLr <- matrix(0, R, Cn)
    for (cc in seq_len(Cn)) {
      llt <- cl_ll_task(X, betas[[cc]], yrow)
      s   <- rowsum(llt, resp, reorder = TRUE)
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

init_H <- function(Cn, R, none_rate, kind, seed = 1) {
  if (Cn == 1) return(matrix(1, R, 1))
  if (kind == "det") {
    # deterministic split on how often the respondent takes the none-option --
    # a real and stable axis of heterogeneity (30.2% of all choices)
    grp <- cut(rank(none_rate, ties.method = "first"), breaks = Cn, labels = FALSE)
    H <- matrix(0.15 / (Cn - 1), R, Cn)
    H[cbind(seq_len(R), grp)] <- 0.85
  } else {
    set.seed(seed)
    H <- matrix(rgamma(R * Cn, 1), R, Cn)
    H <- H / rowSums(H)
  }
  H
}

# =============================================================================
# 5. Setup
# =============================================================================
bd   <- build_data(MEMB)
long <- bd$long; keep <- bd$keep; Z <- bd$Z; dem <- bd$dem

# ---- AXIS (c): is the price curve already class-specific? --------------------
PL     <- grep("^Price_L", keep, value = TRUE)
PL_lev <- as.integer(sub("^Price_L", "", PL))
PL     <- PL[order(PL_lev)]; PL_lev <- sort(PL_lev)
cat("\nAXIS (c) CHECK -- price part-worths in the taste vector:", length(PL),
    "free parameters per class\n")
cat("   levels:", paste(PL_lev, collapse = " "),
    "\n   every class carries its OWN independent copy of these, so the price SHAPE",
    "\n   is already unconstrained across classes. Nothing to free.",
    "\n   Mode 3p instead tests whether the free shape is partly noise.\n\n")
stopifnot(length(PL) >= 10)

# ---- price roughness penalty (mode 3p only) ---------------------------------
PEN2 <- NULL
if (LAMBDA_S > 0) {
  # curve over levels 1..12 is f = (0, beta_L2, ..., beta_L12); f_1 = 0 by the
  # part-worth reference. Penalise squared second differences of f. Shrinks toward
  # a straight line, i.e. penalises WIGGLE, not concavity (constant 2nd diff).
  L    <- max(PL_lev)
  D    <- matrix(0, L - 2, L)
  for (j in seq_len(L - 2)) { D[j, j] <- 1; D[j, j + 1] <- -2; D[j, j + 2] <- 1 }
  D    <- D[, -1, drop = FALSE]                # drop the f_1 = 0 column
  stopifnot(ncol(D) == length(PL))
  PEN2 <- matrix(0, length(keep), length(keep))
  idx  <- match(PL, keep)
  PEN2[idx, idx] <- LAMBDA_S * crossprod(D)
  cat("price roughness penalty active: lambda_s =", LAMBDA_S,
      " (", nrow(D), "second differences penalised per class )\n\n")
}

tasks <- unique(long[, .(No, Case, y, is_test)]); setorder(tasks, No)
folds <- readRDS("model/artifacts/folds.rds")
tasks <- merge(tasks, folds[, .(No, fold)], by = "No", all.x = TRUE)
setorder(tasks, No)
# task i in `tasks` == long rows 4i-3 .. 4i. Everything downstream assumes this.
stopifnot(identical(tasks$No, long$No[seq(1L, nrow(long), by = 4L)]))
stopifnot(nrow(tasks) == nrow(long) / 4)
stopifnot(sum(tasks$is_test == FALSE) == 21565, sum(tasks$is_test == TRUE) == 4997)
stopifnot(!anyNA(tasks$fold[tasks$is_test == FALSE]))

Xall <- as.matrix(long[, ..keep])
tr_t <- which(tasks$is_test == FALSE)
te_t <- which(tasks$is_test == TRUE)
rows_of <- function(ti) as.vector(rbind(4L * ti - 3L, 4L * ti - 2L, 4L * ti - 1L, 4L * ti))

# respondent-level none-rate, used ONLY to seed the EM, and only for respondents
# inside the fitting set (asserted in fit_one)
none_rate_all <- tasks[is_test == FALSE, .(nr = mean(y == 4)), by = Case]

fit_one <- function(train_tasks) {
  ti    <- train_tasks
  X     <- Xall[rows_of(ti), , drop = FALSE]
  yrow  <- tasks$y[ti]
  cases <- tasks$Case[ti]
  ucase <- sort(unique(cases))
  resp  <- match(cases, ucase)
  Zf    <- Z[as.character(ucase), , drop = FALSE]
  nr    <- none_rate_all[match(ucase, Case), nr]

  starts <- list(list(kind = "det", seed = 0),
                 list(kind = "rand", seed = 4242))
  scr <- lapply(seq_along(starts), function(i) {
    s <- starts[[i]]
    em_fit(X, yrow, resp, Zf, C, init_H(C, length(ucase), nr, s$kind, s$seed),
           N_SCREEN, verbose = TRUE, tag = sprintf("[screen %s]", s$kind))
  })
  lls <- sapply(scr, function(z) tail(z$path, 1))
  cat("  start screening LL:", paste(sprintf("%.2f", lls), collapse = " / "),
      "-> keeping", starts[[which.max(lls)]]$kind, "\n")
  w <- scr[[which.max(lls)]]
  fin <- em_fit(X, yrow, resp, Zf, C, w$H, N_MAX, betas0 = w$betas, G0 = w$G,
                verbose = TRUE, tag = "[final]")
  fin$path  <- c(w$path, fin$path)
  fin$ucase <- ucase
  fin
}

# THE PREDICTION PATH. Inputs: fitted (betas, G), the task DESIGN, and the target
# respondents' DEMOGRAPHICS via Z. `tasks$y` is structurally unreachable from this
# function -- LEAK CHECK A proves it empirically.
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
  for (cc in seq_along(fit$betas)) out <- out + Pi[, cc] * cl_prob(X, fit$betas[[cc]])$P
  out
}

# LEAK CHECK B: what the leak WOULD have bought. Runs an E-step on the held-out
# respondents using their OWN choices. Diagnostic only; never saved.
predict_lc_LEAKY <- function(fit, target_tasks) {
  ti   <- target_tasks
  X    <- Xall[rows_of(ti), , drop = FALSE]
  yrow <- tasks$y[ti]
  cs   <- tasks$Case[ti]; uc <- sort(unique(cs)); rp <- match(cs, uc)
  Zv   <- Z[as.character(uc), , drop = FALSE]
  Cn   <- length(fit$betas)
  LLr  <- matrix(0, length(uc), Cn)
  for (cc in seq_len(Cn))
    LLr[, cc] <- as.vector(rowsum(cl_ll_task(X, fit$betas[[cc]], yrow), rp, reorder = TRUE))
  H   <- row_softmax(log(pmax(membership_prob(Zv, fit$G), 1e-12)) + LLr)
  out <- matrix(0, length(ti), 4)
  for (cc in seq_len(Cn)) out <- out + H[rp, cc] * cl_prob(X, fit$betas[[cc]])$P
  out
}

# =============================================================================
# 6. 5-fold OOF
# =============================================================================
oof     <- matrix(NA_real_, length(tr_t), 4)
oof_fix <- matrix(NA_real_, length(tr_t), 4)   # ablation: constant mixing weights
fold_ll <- numeric(5)
min_share <- numeric(5)
t0 <- proc.time()
for (k in 1:5) {
  cat("--- fold", k, "---\n")
  itr <- tr_t[tasks$fold[tr_t] != k]
  iva <- tr_t[tasks$fold[tr_t] == k]
  # LEAK CHECK C: no validation respondent may appear in the fitting set
  stopifnot(length(intersect(tasks$Case[itr], tasks$Case[iva])) == 0)
  fit <- fit_one(itr)
  # the EM seed must not have seen validation respondents either
  stopifnot(length(intersect(fit$ucase, tasks$Case[iva])) == 0)

  oof[match(iva, tr_t), ]     <- predict_lc(fit, iva)
  oof_fix[match(iva, tr_t), ] <- predict_lc(fit, iva, fixed_pi = colMeans(fit$H))
  fold_ll[k]   <- logloss(tasks$y[iva], oof[match(iva, tr_t), ])
  min_share[k] <- min(colMeans(fit$H))

  if (k == 1) {
    ## ---- LEAK CHECK A (structural, decisive) --------------------------------
    y_true <- copy(tasks$y)
    set.seed(999)
    yperm  <- y_true; yperm[iva] <- sample(yperm[iva])
    stopifnot(!identical(yperm, y_true))       # the permutation actually did something
    tasks[, y := yperm]
    p_scr <- predict_lc(fit, iva)
    tasks[, y := y_true]
    same <- identical(p_scr, oof[match(iva, tr_t), ])
    cat("  LEAK CHECK A: held-out choices randomly permuted, fold re-predicted.\n")
    cat("                predictions bit-identical:", same, "  <- must be TRUE\n")
    stopifnot(same, identical(tasks$y, y_true))

    ## ---- LEAK CHECK B (quantitative) ----------------------------------------
    ll_leak <- logloss(tasks$y[iva], predict_lc_LEAKY(fit, iva))
    cat(sprintf("  LEAK CHECK B: if we HAD used held-out respondents' own choices to\n"))
    cat(sprintf("                infer their class, fold 1 would read %.5f vs the honest\n", ll_leak))
    cat(sprintf("                %.5f (a fake gain of %.5f). Not saved, not used.\n",
                fold_ll[1], fold_ll[1] - ll_leak))
  }
  cat("  fold", k, "logloss:", round(fold_ll[k], 5),
      " min class share", round(min_share[k], 3),
      " elapsed", round((proc.time() - t0)[3] / 60, 1), "min\n")
}
ll_oof <- logloss(tasks$y[tr_t], oof)
cat(sprintf("\n>>> %s honest OOF logloss: %.5f   (lcmnl3 1.14396, mnl_pw 1.15686)\n",
            NAME, ll_oof))
cat("    per-fold:", paste(sprintf("%.5f", fold_ll), collapse = " "), "\n")
ll_fix <- logloss(tasks$y[tr_t], oof_fix)
cat(sprintf("    ablation, constant mixing weights for everyone: %.5f\n", ll_fix))
cat(sprintf("    => demographic membership channel is worth %+.5f\n", ll_fix - ll_oof))
cat(sprintf("    smallest class share across folds: %.3f %s\n", min(min_share),
            if (min(min_share) < 0.05) "  <-- DEGENERATE / near-empty class" else ""))

oof <- clip_norm(oof)
stopifnot(nrow(oof) == 21565, max(abs(rowSums(oof) - 1)) < 1e-10)
if (!SMOKE) saveRDS(data.table(No = tasks$No[tr_t], p1 = oof[, 1], p2 = oof[, 2],
                               p3 = oof[, 3], p4 = oof[, 4]),
                    sprintf("model/artifacts/oof_%s.rds", NAME))

# =============================================================================
# 7. Refit on ALL training respondents, predict test
# =============================================================================
cat("--- full refit ---\n")
full <- fit_one(tr_t)
te <- clip_norm(predict_lc(full, te_t))
stopifnot(nrow(te) == 4997, max(abs(rowSums(te) - 1)) < 1e-10)
if (!SMOKE) saveRDS(data.table(No = tasks$No[te_t], p1 = te[, 1], p2 = te[, 2],
                               p3 = te[, 3], p4 = te[, 4]),
                    sprintf("model/artifacts/test_%s.rds", NAME))

# =============================================================================
# 8. Report material
# =============================================================================
# Label switching: the mixture likelihood is invariant to permuting classes, so
# raw indices mean nothing across runs. Relabel by descending share before
# printing. Predictions are unaffected.
ord   <- order(colMeans(full$H), decreasing = TRUE)
betas <- full$betas[ord]
Hs    <- full$H[, ord, drop = FALSE]
Gs    <- full$G
cat("\n--- SEGMENTS (full-data fit, relabelled by descending share) ---\n")
cat("posterior share:", paste(sprintf("%.3f", colMeans(Hs)), collapse = "  "), "\n")
cat("hard assignment counts:",
    paste(table(factor(max.col(Hs), levels = 1:C)), collapse = "  "), "\n")

nr_f <- none_rate_all[match(full$ucase, Case), nr]
cat("posterior-weighted mean none-rate by segment:",
    paste(sprintf("%.3f", colSums(Hs * nr_f) / colSums(Hs)), collapse = "  "), "\n")

# posterior-weighted demographic profile: the human description of each segment
dm <- dem[match(full$ucase, Case)]
cat("\nposterior-weighted demographic profile by segment:\n")
prof <- rbind(
  age        = colSums(Hs * dm$agea)    / colSums(Hs),
  income     = colSums(Hs * dm$incomea) / colSums(Hs),
  log_income = colSums(Hs * log(dm$incomea)) / colSums(Hs),
  miles      = colSums(Hs * dm$milesa)  / colSums(Hs),
  night_pct  = colSums(Hs * dm$nighta)  / colSums(Hs),
  pct_urban  = 100 * colSums(Hs * (dm$Urbind == max(dm$Urbind))) / colSums(Hs),
  pct_female = 100 * colSums(Hs * (dm$genderind == 2)) / colSums(Hs)
)
print(round(prof, 1))

B <- do.call(cbind, betas); rownames(B) <- keep
if ("asc4" %in% keep) {
  cat("\nnone-option constant (asc4) by segment:\n"); print(round(B["asc4", ], 3))
}
cat("\nprice part-worths by segment (utility relative to Price level 1):\n")
print(round(B[PL, , drop = FALSE], 3))
cat("\nprice-curve diagnostics per segment:\n")
pc <- rbind(
  total_drop_L1_L12 = B[PL[length(PL)], ],
  monotone_violations = apply(B[PL, , drop = FALSE], 2,
                              function(v) sum(diff(c(0, v)) > 0)),
  mean_abs_2nd_diff = apply(B[PL, , drop = FALSE], 2,
                            function(v) mean(abs(diff(diff(c(0, v))))))
)
print(round(pc, 3))

cat("\nmost DISTINGUISHING tastes (largest spread across segments):\n")
sp  <- apply(B, 1, function(r) max(r) - min(r))
top <- names(sort(sp, decreasing = TRUE))[1:20]
print(round(B[top, , drop = FALSE], 3))

cat("\nmost distinguishing NON-price tastes:\n")
spn <- sp[!grepl("^Price", names(sp))]
print(round(B[names(sort(spn, decreasing = TRUE))[1:12], , drop = FALSE], 3))

cat("\nmembership model: strongest demographic drivers (vs class 1 baseline)\n")
if (ncol(Gs)) {
  rownames(Gs) <- colnames(Z)
  for (j in seq_len(ncol(Gs))) {
    v <- sort(Gs[, j], decreasing = TRUE)
    cat(sprintf(" class %d vs 1: %s\n", j + 1,
                paste(sprintf("%s %+.2f", names(c(head(v, 5), tail(v, 5))),
                              c(head(v, 5), tail(v, 5))), collapse = " | ")))
  }
  Pi <- membership_prob(Z[as.character(full$ucase), , drop = FALSE], full$G)
  cat(" predicted-share spread across respondents (sd of pi_rc):",
      paste(sprintf("%.3f", apply(Pi, 2, sd)), collapse = "  "), "\n")
  cat(" predicted-share range (min-max of pi_rc):\n")
  print(round(apply(Pi, 2, range), 3))
}
cat("\nLL path (training mixture log-likelihood, full refit):\n")
print(round(full$path, 2))
cat("converged flag: last relative LL change =",
    signif(abs(diff(tail(full$path, 2))) / (abs(tail(full$path, 1)) + 1), 3),
    " (tol", TOL, ")\n")

if (!SMOKE) saveRDS(list(mode = MODE, name = NAME, C = C, memb = MEMB, lambda_s = LAMBDA_S,
             betas = betas, G = full$G, keep = keep, zcols = colnames(Z),
             shares = colMeans(Hs), profile = prof, price = B[PL, , drop = FALSE],
             path = full$path, oof = ll_oof, fold_ll = fold_ll, oof_fixed_pi = ll_fix),
        file.path(OUTDIR, sprintf("fit_%s.rds", NAME)))
cat("\nOK\n")
