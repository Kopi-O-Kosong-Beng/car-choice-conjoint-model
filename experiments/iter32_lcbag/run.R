# =============================================================================
# ITERATION 32 — Bagging the EM starts of the latent-class model.
#
# WHAT THIS FILE IS. A byte-for-byte copy of experiments/iter25_taskpos/run.R
# with ONE change, described below. Every design matrix, penalty, convergence
# budget, fold and seed is identical to the incumbent `lcmnl3_both`. The copy is
# deliberate: it makes the change a diff rather than a rewrite, so attribution
# cannot slip.
#
# THE CHANGE. The incumbent screens two EM starts for 3 iterations, keeps the one
# with the higher TRAINING mixture log-likelihood, and runs only that one to
# budget. This version runs each start to the full budget and averages the
# resulting PREDICTIONS across starts. One invocation = one start.
#
# CORRECTION TO THE RECORD, ESTABLISHED BEFORE THIS RUN
# -----------------------------------------------------
# The idea was originally motivated as "the incumbent selects the best start on
# the fitting data, so it is optimistically biased." Reading the code and the log
# shows that is wrong on both counts, and the honest hypothesis is narrower:
#
#   1. The selection is `which.max` on the TRAINING mixture log-likelihood after
#      3 EM iterations (run.R:485-487 of iteration 25), not a selection on the
#      scored fold. Selection on training LL is not selection on the metric, so
#      it carries no optimism about held-out logloss.
#   2. The deterministic start won the screen in ALL SIX fits — five folds plus
#      the full refit (log_both.txt lines 15, 34, 57, 79, 100, 127). The random
#      start never won once. So the selection step is, empirically, a no-op, and
#      `lcmnl3_both` is exactly "det start, run to budget".
#
# So there is no bias to remove. What remains is a real and separate question:
# EM on a 3-class mixture has local optima, and a start that reaches a LOWER
# training log-likelihood does not necessarily generalise worse. Averaging over
# starts is then a variance reduction, not a debiasing.
#
# WHY IT IS WORTH THE COMPUTE ANYWAY. The same argument iteration 26 made for
# seed-bagging the trees: bagging consults no fold, tunes no hyperparameter and
# selects no feature, so unlike almost every other gain in this log it cannot
# overfit the seed-42 fold structure. That matters here because the local->public
# transfer rate has decayed from ~58% to ~1/3 across eighteen experiments against
# one fixed partition. A variance reduction should transfer at close to 100%.
#
# THE SECOND OUTPUT MATTERS AS MUCH AS THE FIRST. The spread of OOF logloss across
# EM starts is a number this repository has never measured, on the member carrying
# the LARGEST blend weight (0.504). Iteration 26 measured the equivalent number
# for the trees (seed sd 0.0026-0.0032) and it retroactively refuted the monotone
# constraint's accepted margin of 0.0017. If the latent-class start spread is of
# the same order, then `lcmnl3_both`'s 0.00534 win over `lcmnl3` — and the 0.00018
# that separates `_both` from `_tilt` — need rereading. If the spread is tiny, the
# accepted results stand. Either answer is worth having and goes in the report.
#
# BOTH OUTCOMES ARE PRE-REGISTERED
#   * Starts converge to one optimum  -> the bag equals the incumbent, the gain is
#     ~0, and the finding is that EM start-sensitivity here is nil. Publishable,
#     NOT adoptable.
#   * Starts land in distinct optima  -> averaging is a genuine variance reduction
#     and the gain is real. Adoptable if it clears the gates below.
#
# EXPECTATION. 0 to -0.003 at member level, and LESS at blend level: three
# separate demonstrations on 26 Jul (bagging, gating and carriers) all produced
# ~0 once blended, because the ensemble absorbs member-level repairs.
#
# PRE-REGISTERED ADOPTION GATES (all three must pass; declared before running)
#   1. MEMBER   Rscript model/compare.R lcmnl3_both lcmnl3_bag  ->  z >= 2,
#      paired with respondent-clustered SE.
#   2. BLEND    swap the name into model/members.txt, Rscript model/06_blend.R,
#      nested must improve on 1.12867. This is a non-regression check, not a
#      paired test — a paired blend-level test needs the nested loop's per-fold
#      held-out blend predictions, as EXPERIMENTS.md already records.
#   3. SHIFT    Rscript model/shift_audit.R, retention >= ~80% (the level at which
#      lcmnl3 was flagged as carrying real extrapolation risk).
#   Fail any one -> revert members.txt and log it as a failure.
#
# AVERAGING SPACE, PRE-DECLARED. The GEOMETRIC mean of the per-start probability
# vectors is primary; the arithmetic mean is reported as secondary and is NOT
# eligible for adoption. Reason: logloss is a function of log p, so the geometric
# mean is its natural average. Iteration 26 computed both and kept whichever
# scored better on the OOF it was about to be judged on; that is a small selection
# on the decision data and this run does not repeat it.
#
# LABEL SWITCHING. The mixture likelihood is invariant to permuting classes, so
# class 2 of one start has no relationship to class 2 of another. Parameters
# therefore CANNOT be averaged. Only the 4-column predicted probabilities are.
#
# LEAKAGE: none possible, and none newly introduced. Every start's fold-k
# prediction comes from a fit that never saw fold k, exactly as before; class
# membership for a held-out respondent still comes from demographics alone.
#
# RESUMABILITY (the pattern from iteration 26, which exists because iteration 17
# finished and looked like a failure when its caller died). Each start writes its
# own artifact as its last act. Re-running skips starts already on disk, and
# `combine` averages whatever is present. A crash costs one start, not the run.
#
#   Rscript experiments/iter32_lcbag/run.R det          # the incumbent's own start
#   Rscript experiments/iter32_lcbag/run.R rand4242     # the incumbent's 2nd start
#   Rscript experiments/iter32_lcbag/run.R rand101      # ... rand202, rand303
#   Rscript experiments/iter32_lcbag/run.R combine      # -> oof/test_lcmnl3_bag
#
# The start set is a strict SUPERSET of the incumbent's two, so the bag contains
# the incumbent's own fit. `det` alone reproduces `lcmnl3_both` exactly, which is
# also this repository's R-version gate: this session runs R 4.3.3 against the
# 4.6.0 the artifacts were built on, and `det` must land on 1.13863.
# =============================================================================

suppressMessages({ library(data.table) })
source("model/99_utils.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: run.R <det|rand4242|rand101|rand202|rand303|combine>")
JOB <- args[1]

# The configuration is FIXED at the incumbent's: 3 classes, both task-position
# terms. Nothing here is a free choice -- the whole point of this iteration is
# that exactly one thing differs from `lcmnl3_both`.
C        <- 3L
TASKMODE <- "both"

# The start set. `det` is the incumbent's deterministic start (init_H seeds off
# the respondent none-rate and ignores the seed argument); `rand4242` is the
# incumbent's second start. The other three are new random draws.
STARTS <- list(det      = list(kind = "det",  seed = 0L),
               rand4242 = list(kind = "rand", seed = 4242L),
               rand101  = list(kind = "rand", seed = 101L),
               rand202  = list(kind = "rand", seed = 202L),
               rand303  = list(kind = "rand", seed = 303L))

LAMBDA_B <- 2.0     # ridge on class-specific taste parameters
LAMBDA_G <- 2.0     # ridge on membership-model coefficients
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
N_SCREEN <- as.integer(Sys.getenv("LC_SCREEN", "3"))   # EM iters the incumbent spent screening
N_MAX    <- as.integer(Sys.getenv("LC_MAX", "25"))     # further EM iters for the winner
TOL      <- as.numeric(Sys.getenv("LC_TOL", "1e-5"))   # relative tol on the mixture LL
# Each start here runs N_SCREEN + N_MAX iterations in one uninterrupted EM, which
# is the same total budget the incumbent's WINNING start received (3 screening +
# 25 final, warm-started from the screen). Per-start cost therefore matches the
# incumbent run, and the bag costs K times it.
N_TOTAL  <- N_SCREEN + N_MAX

OUTDIR <- "experiments/iter32_lcbag"

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

cl_obj <- function(X, beta, yrow, wtask, ridge) {
  sum(wtask * cl_ll_task(X, beta, yrow)) - 0.5 * ridge * sum(beta^2)
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
    g  <- colSums(wtask * (Xch - Xb)) - ridge * beta
    A  <- crossprod(X, X * (wr * pv))
    B  <- crossprod(Xb, Xb * wtask)
    H  <- A - B                                   # = -(Hessian of the loglik)
    diag(H) <- diag(H) + ridge
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
                   verbose = TRUE, tag = "") {
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
      betas[[cc]] <- cl_newton(X, yrow, w, betas[[cc]], LAMBDA_B, nsteps = nst)$beta
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
if (JOB == "combine") {
  long <- readRDS("model/artifacts/long.rds")
  ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
  ytr  <- ymap$y

  files <- list.files(OUTDIR, pattern = "^start_.*\\.rds$", full.names = TRUE)
  if (!length(files)) stop("no start_*.rds in ", OUTDIR, " -- run the starts first")
  ps <- lapply(files, readRDS)
  names(ps) <- sapply(ps, `[[`, "tag")
  cat("combining", length(ps), "starts:", paste(names(ps), collapse = ", "), "\n\n")

  # ---- the second output: how much do the starts actually differ? ------------
  lls <- sapply(ps, function(p) logloss(ytr, p$oof))
  cat("--- PER-START OOF (the number this repository has never measured) ---\n")
  for (nm in names(lls)) cat(sprintf("  %-10s %.5f   (%.1f min)\n", nm, lls[[nm]],
                                     ps[[nm]]$mins))
  cat(sprintf("  mean %.5f   sd %.5f   range %.5f\n", mean(lls), sd(lls),
              max(lls) - min(lls)))
  cat(sprintf("  for scale: lcmnl3_both beat lcmnl3 by 0.00534, and _both beats\n"))
  cat(sprintf("             _tilt by 0.00018. xgb seed sd was 0.0026-0.0032.\n\n"))

  # Training mixture LL at the end of each start: do they find the same optimum?
  mll <- sapply(ps, function(p) p$ll_full)
  cat("--- final TRAINING mixture LL of the full refit, by start ---\n")
  for (nm in names(mll)) cat(sprintf("  %-10s %.3f\n", nm, mll[[nm]]))
  cat(sprintf("  spread %.3f  (near zero => every start found the same optimum,\n",
              max(mll) - min(mll)))
  cat( "             and averaging cannot help)\n\n")

  # ---- average, in both spaces ----------------------------------------------
  gmean <- function(key) {
    L <- Reduce(`+`, lapply(ps, function(p) log(pmax(p[[key]], 1e-15)))) / length(ps)
    P <- exp(L); P / rowSums(P)
  }
  amean <- function(key) Reduce(`+`, lapply(ps, `[[`, key)) / length(ps)

  G <- gmean("oof"); A <- amean("oof")
  cat(sprintf("--- BAGGED OOF ---\n  geometric (PRIMARY, pre-declared) %.5f\n", logloss(ytr, G)))
  cat(sprintf("  arithmetic (secondary, not adoptable) %.5f\n", logloss(ytr, A)))
  cat(sprintf("  incumbent lcmnl3_both                 %.5f\n", 1.13863))
  cat(sprintf("  best single start                     %.5f\n", min(lls)))
  cat(sprintf("  bag vs mean single start: %+.5f\n", mean(lls) - logloss(ytr, G)))

  # PRE-DECLARED: geometric is what ships. No selection on the OOF it is judged on.
  best  <- G
  tbest <- gmean("test")

  No_tr <- ymap$No
  No_te <- sort(unique(long[is_test == TRUE, No]))
  stopifnot(length(No_tr) == 21565, length(No_te) == 4997)
  best <- clip_norm(best); tbest <- clip_norm(tbest)
  stopifnot(!anyNA(best), !anyNA(tbest),
            all(abs(rowSums(best) - 1) < 1e-9), all(abs(rowSums(tbest) - 1) < 1e-9))
  saveRDS(data.table(No = No_tr, p1 = best[,1], p2 = best[,2], p3 = best[,3], p4 = best[,4]),
          "model/artifacts/oof_lcmnl3_bag.rds")
  saveRDS(data.table(No = No_te, p1 = tbest[,1], p2 = tbest[,2], p3 = tbest[,3], p4 = tbest[,4]),
          "model/artifacts/test_lcmnl3_bag.rds")
  cat(sprintf("\nwrote lcmnl3_bag  (predicted test none-rate %.4f; lcmnl3_both 0.2237,\n",
              mean(tbest[, 4])))
  cat("                   trees ~0.275, train 0.302)\n")
  cat("next: Rscript model/compare.R lcmnl3_both lcmnl3_bag\n")
  cat("OK\n"); quit(save = "no")
}

if (!JOB %in% names(STARTS)) stop("unknown start: ", JOB)
START <- STARTS[[JOB]]
NAME  <- sprintf("lcmnl%d_%s", C, TASKMODE)
cat("=== latent-class conditional logit, C =", C,
    "| task-position mode:", TASKMODE, "===\n")

bd   <- build_data()
long <- bd$long; keep <- bd$keep; Z <- bd$Z

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

fit_one <- function(train_tasks) {
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

  # ===========================================================================
  # THE ONE CHANGE vs experiments/iter25_taskpos/run.R.
  #
  # The incumbent ran BOTH starts for N_SCREEN iterations, compared their
  # training mixture log-likelihoods, and continued only the higher one. This
  # runs the ONE start this invocation is responsible for, straight through for
  # N_SCREEN + N_MAX iterations -- the same total budget the incumbent's winner
  # received -- and never compares anything. The averaging happens later, across
  # invocations, in `combine`.
  #
  # Note that `det` is deterministic: init_H ignores the seed for kind = "det",
  # so this branch reproduces the incumbent's winning path exactly.
  # ===========================================================================
  fin <- em_fit(X, yrow, resp, Zf, C,
                init_H(C, length(ucase), nr, START$kind, START$seed),
                N_TOTAL, verbose = TRUE, tag = sprintf("[%s]", JOB))
  fin$ucase <- ucase
  fin
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

# ---- resumability: one start per invocation, skip if already on disk ---------
outf <- file.path(OUTDIR, sprintf("start_%s.rds", JOB))
if (file.exists(outf)) {
  cat("start", JOB, "already done ->", outf, "-- skipping\n"); quit(save = "no")
}

# ---- 5-fold OOF -------------------------------------------------------------
oof <- matrix(NA_real_, length(tr_t), 4)
oof_fix <- matrix(NA_real_, length(tr_t), 4)   # ablation, diagnostic only
t0  <- proc.time()
for (k in 1:5) {
  cat("--- fold", k, "---\n")
  itr <- tr_t[tasks$fold[tr_t] != k]
  iva <- tr_t[tasks$fold[tr_t] == k]
  stopifnot(length(intersect(tasks$Case[itr], tasks$Case[iva])) == 0)
  fit <- fit_one(itr)
  oof[match(iva, tr_t), ] <- predict_lc(fit, iva)
  oof_fix[match(iva, tr_t), ] <- predict_lc(fit, iva, fixed_pi = colMeans(fit$H))
  cat("  fold", k, "logloss:",
      round(logloss(tasks$y[iva], oof[match(iva, tr_t), ]), 5),
      " elapsed", round((proc.time() - t0)[3] / 60, 1), "min\n")
}
ll_oof <- logloss(tasks$y[tr_t], oof)
cat(sprintf("\n>>> %s honest OOF logloss: %.5f   (mnl_pw 1.15686, xgb_lw2 1.14152)\n",
            NAME, ll_oof))
if (C > 1) {
  ll_fix <- logloss(tasks$y[tr_t], oof_fix)
  cat(sprintf("    ablation, constant mixing weights for everyone: %.5f\n", ll_fix))
  cat(sprintf("    => demographic membership channel is worth %+.5f of the total %+.5f\n",
              ll_fix - ll_oof, 1.15686 - ll_oof))
}

oof <- clip_norm(oof)

# ---- refit on ALL training respondents, predict test -------------------------
cat("--- full refit ---\n")
full <- fit_one(tr_t)
te <- clip_norm(predict_lc(full, te_t))

# ---- this start's artifact, written as the last act -------------------------
# NOTE: no model/artifacts/oof_* or test_* is written here. A single start is not
# a model this repository ships; only `combine` emits lcmnl3_bag. That also keeps
# the rule "one artifact name, one producing script" intact -- iteration 25 owns
# lcmnl3_both and nothing in this file can overwrite it.
stopifnot(nrow(oof) == 21565, nrow(te) == 4997)
saveRDS(list(tag = JOB, kind = START$kind, seed = START$seed,
             oof = oof, test = te, ll_oof = ll_oof, ll_full = tail(full$path, 1),
             mins = (proc.time() - t0)[3] / 60), outf)
cat(sprintf("start %s done in %.1f min -> %s\n", JOB,
            (proc.time() - t0)[3] / 60, outf))
cat(sprintf("  test none-rate %.4f  (lcmnl3_both 0.2237)\n", mean(te[, 4])))
if (JOB == "det")
  cat(sprintf("  R-VERSION GATE: expected 1.13863, got %.5f -> %s\n", ll_oof,
              if (abs(ll_oof - 1.13863) < 1e-4) "REPRODUCES" else "*** DISAGREES ***"))

# =============================================================================
# The per-segment report material from iteration 25 is deliberately NOT carried
# over. It describes ONE fit, and this script produces K of them -- five copies
# would collide on a single fit_C3.rds and none of them would describe the bag.
# The segment interpretation in the report comes from the incumbent run, which
# is unchanged by this iteration.
# =============================================================================
cat("OK\n")
