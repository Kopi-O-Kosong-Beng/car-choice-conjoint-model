# =============================================================================
# ITERATION 11 — Latent-class (finite mixture) conditional logit, hand-written EM.
#
# HYPOTHESIS
# ----------
# Iteration 05 tried a mixed logit with CONTINUOUS random coefficients and lost
# (1.17281 vs 1.15686 for the fixed part-worth logit). The diagnosis recorded in
# EXPERIMENTS.md: every test respondent is unseen, so the only thing we can emit is
# a population-averaged probability, and integrating over a smooth taste
# distribution blurs the prediction without buying anything back.
#
# A FINITE mixture may behave differently. Instead of a continuum, posit a small
# number of DISCRETE segments (C = 2 or 3), each with its own part-worth vector,
# and make class membership a function of DEMOGRAPHICS. For a new respondent:
#
#     P(choice) = sum_c  P(class c | demographics_r) * P(choice | class c)
#
# The mixing weights are then informative rather than blind: a 25-year-old renter
# in a city gets a different mixture than a 60-year-old suburban homeowner. If
# demographics carry any signal about taste, this is sharper than mixed logit while
# still capturing heterogeneity. If they do not, this collapses toward a fixed
# mixture and should land close to mnl_pw (and the experiment tells us that the
# demographic channel is empty, which is itself worth knowing).
#
# The specification is a strict GENERALISATION of mnl_pw: identical x-variables
# (part-worths + ASCs + the 12 price x demographic interactions), so C = 1 must
# reproduce mnl_pw. That is run as a sanity check.
#
# METHOD (EM, written directly in R)
# ----------------------------------
#   E-step  h_rc  proportional to  pi_rc(gamma) * prod_t P_c(choice_rt)
#           -- posterior class membership per RESPONDENT, using their 19 choices.
#   M-step  beta_c : weighted conditional logit, weight = h_rc on every task of r.
#                    Newton-Raphson with analytic gradient and Hessian, ridge lambda_b.
#           gamma  : multinomial logit of the soft targets h on demographics,
#                    BFGS with analytic gradient, ridge lambda_g.
#
# PREDICTION (the part that must not leak)
# ----------------------------------------
# For a held-out respondent the class probabilities come from DEMOGRAPHICS ONLY:
#     pi_rc = softmax_c(z_r' gamma_c)
# Their own choices are never touched. Running an E-step on a validation
# respondent would be leakage and would invalidate the whole comparison, so the
# posterior h is computed only for respondents inside the fitting set.
#
# GUARDS
#   * folds come from model/artifacts/folds.rds, grouped by Case. Never regenerated.
#   * label switching: classes are relabelled canonically (descending share) before
#     any coefficient is printed. Predictions are invariant to labelling.
#   * multiple starts (deterministic + random), screened on TRAINING log-likelihood
#     only, never on the fold being scored.
#   * ridge penalties fixed a priori (lambda_b = 2, lambda_g = 2), not tuned on OOF.
#
# USAGE
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter11_latent_class/run.R 1
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter11_latent_class/run.R 2
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter11_latent_class/run.R 3
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter11_latent_class/run.R finalize
#
#   C = 1 is the sanity check (writes lcmnl1, expected ~1.157).
#   "finalize" copies the better of C=2 / C=3 to oof_lcmnl.rds / test_lcmnl.rds.
# =============================================================================

suppressMessages({ library(data.table) })
source("model/99_utils.R")

### ITERATION 21 PATCH ##########################################################
# Verbatim copy of experiments/iter11_latent_class/run.R. The ONLY changes are:
#   1. the fold file is a command-line argument (folds_b.rds / folds_c.rds),
#   2. artifacts get a _<split> suffix,
#   3. the "finalize" mode and the report-material block are removed (not needed).
# Everything statistical -- LAMBDA_B, LAMBDA_G, N_SCREEN, N_MAX, TOL, the starts,
# the EM schedule -- is identical to production.
#   Rscript experiments/iter21_foldrobust/lcmnl_split.R <C> <b|c>
#################################################################################
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: lcmnl_split.R <1|2|3> <b|c>")
MODE  <- args[1]
SPLIT <- args[2]
stopifnot(SPLIT %in% c("b", "c"))

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
N_SCREEN <- as.integer(Sys.getenv("LC_SCREEN", "3"))   # EM iters used to screen starts
N_MAX    <- as.integer(Sys.getenv("LC_MAX", "25"))     # further EM iters for the winner
TOL      <- as.numeric(Sys.getenv("LC_TOL", "1e-5"))   # relative tol on the mixture LL

OUTDIR <- "experiments/iter21_foldrobust"

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
  cand <- c("asc2", "asc3", "asc4", pw, px)

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

C <- as.integer(MODE)
if (!C %in% 1:4) stop("C must be 1..4")
NAME <- sprintf("lcmnl%d_%s", C, SPLIT)
cat("=== latent-class conditional logit, C =", C, "===\n")

bd   <- build_data()
long <- bd$long; keep <- bd$keep; Z <- bd$Z

tasks <- unique(long[, .(No, Case, y, is_test)]); setorder(tasks, No)
folds <- readRDS(sprintf("model/artifacts/folds_%s.rds", SPLIT))   # NOT folds.rds
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
saveRDS(data.table(No = tasks$No[tr_t], p1 = oof[, 1], p2 = oof[, 2],
                   p3 = oof[, 3], p4 = oof[, 4]),
        sprintf("model/artifacts/oof_%s.rds", NAME))

# ---- refit on ALL training respondents, predict test -------------------------
cat("--- full refit ---\n")
full <- fit_one(tr_t)
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
  saveRDS(list(betas = betas, G = full$G, keep = keep, zcols = colnames(Z),
               shares = colMeans(Hs), path = full$path, oof = ll_oof),
          file.path(OUTDIR, sprintf("fit_C%d_%s.rds", C, SPLIT)))
}
cat("OK\n")
