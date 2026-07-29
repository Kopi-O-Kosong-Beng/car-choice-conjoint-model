# =============================================================================
# ITERATION 58 -- CAN THE LATENT-CLASS MODEL BE MADE TO EXTRAPOLATE LESS?
#
# This file is experiments/iter25_taskpos/run.R with ONE thing changed: the
# regularisation and weighting of the MEMBERSHIP model (gamma). The utility
# model (beta, LAMBDA_B = 2), the EM schedule, the starts, the folds, the design
# and the task-position terms are byte-for-byte the production code path, so a
# difference in the reported number is attributable to the membership channel
# and to nothing else.
#
# HYPOTHESIS AND DECISION RULE, WRITTEN BEFORE ANY OF THESE RUNS
# --------------------------------------------------------------
# diag.R and diag2.R (read-only, already run) established four things:
#
#   H0  The membership softmax is NOT saturated: mean max(pi) is 0.4775 on test
#       respondents and ZERO respondents exceed 0.9. => "cap the demographic
#       contribution" has nothing to cap and is NOT run here.
#   H1  The softmax does not extrapolate on the luxury axis: it assigns TEST
#       luxury respondents class shares 0.441/0.384/0.176 against 0.450/0.374/
#       0.176 for TRAIN luxury respondents.
#   H2  Across 10 archived models the correlation between "OOF luxury none-rate
#       miss" and "global TEST none-rate miss" is +0.988. The models whose
#       shipped none-rate is closest to the measured r* = 0.266521 are exactly
#       the models MOST wrong about luxury respondents in-population.
#   H3  A model perfectly calibrated to the training population ships
#       0.68821*0.15986 + 0.31179*0.31712 = 0.20890. That is 0.0576 BELOW r*.
#
# PREDICTION (pre-registered):
#   P1  Raising LAMBDA_G monotonically RAISES the shipped test none-rate toward
#       r*, and monotonically WORSENS the nested OOF. It buys the moment by
#       re-introducing the tree family's documented luxury defect.
#   P2  Lowering LAMBDA_G below the a-priori 2 does not improve the nested OOF
#       (the a-priori choice was not obviously wrong).
#   P3  Reweighting the membership fit toward the test segment mix moves the
#       shipped none-rate DOWN, not up -- the wrong way -- because the OOF
#       luxury prediction (0.18252) is already ABOVE the observed rate
#       (0.15986), so fitting luxury respondents harder pushes it lower.
#
# DECISION RULE. A variant is worth reporting as an IMPROVEMENT only if it beats
# lcmnl3_both's 1.13863 plain OOF by more than the model-level seed sd of
# 0.00283 AND does not worsen the 2-member nested blend. Any variant that only
# moves the shipped none-rate closer to r* is recorded as TUNED TO THE
# MEASUREMENT and is worth exactly what a 1-parameter none-margin shift is
# worth -- which is available already, without damaging the model.
#
# USAGE:  Rscript experiments/iter58_lcextrap/run.R <TAG> <LAMBDA_G> <LUXW>
#   TAG    short name; artifacts go to oof_lcxg_<TAG>.rds / test_lcxg_<TAG>.rds
#   LUXW   relative weight on luxury respondents in the membership M-step only
# The script REFUSES to overwrite an existing artifact.
# =============================================================================

suppressMessages({ library(data.table) })
source("model/99_utils.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run.R <TAG> <LAMBDA_G> <LUXW>")
TAG      <- args[1]
LAMBDA_G <- as.numeric(args[2])
LUXW     <- as.numeric(args[3])
stopifnot(grepl("^[a-z0-9]+$", TAG), is.finite(LAMBDA_G), is.finite(LUXW))

# ---- artifact names. Prefix is a string LITERAL and is unique in the repo. ----
OOF_OUT  <- paste0("model/artifacts/oof_lcxg_",  TAG, ".rds")
TEST_OUT <- paste0("model/artifacts/test_lcxg_", TAG, ".rds")
if (file.exists(OOF_OUT) || file.exists(TEST_OUT))
  stop("refusing to overwrite an existing artifact: ", OOF_OUT)
# belt and braces: never write over anything that predates this iteration
prot <- setdiff(list.files("model/artifacts", pattern = "\\.rds$"),
                c(basename(OOF_OUT), basename(TEST_OUT)))
stopifnot(!basename(OOF_OUT) %in% prot, !basename(TEST_OUT) %in% prot)

C        <- 3L
LAMBDA_B <- 2.0                                   # UNCHANGED from production
N_SCREEN <- as.integer(Sys.getenv("LC_SCREEN", "3"))
N_MAX    <- as.integer(Sys.getenv("LC_MAX", "25"))
TOL      <- as.numeric(Sys.getenv("LC_TOL", "1e-5"))
FOLDFILE <- "model/artifacts/folds.rds"           # READ ONLY, never written

cat("=== iter58: latent class C=3, task mode 'both' ===\n")
cat(sprintf("    TAG %s | LAMBDA_G %.4g (production 2) | luxury membership weight %.4g\n",
            TAG, LAMBDA_G, LUXW))

# =============================================================================
# 1. Design matrices  -- identical to experiments/iter25_taskpos/run.R
# =============================================================================
build_data <- function() {
  long <- readRDS("model/artifacts/long.rds")
  setorder(long, No, alt)
  stopifnot(nrow(long) %% 4 == 0)
  stopifnot(all(long$alt == rep(1:4, nrow(long) / 4)))
  stopifnot(all(long$No == rep(long$No[seq(1, nrow(long), 4)], each = 4)))

  pw <- character(0)
  for (a in ATTRS) {
    lv  <- sort(unique(long[alt != 4][[a]])); ref <- lv[1]
    for (l in setdiff(lv, ref)) {
      nm <- sprintf("%s_L%s", a, l); long[, (nm) := as.numeric(get(a) == l)]; pw <- c(pw, nm)
    }
  }
  px <- c("Price_x_age", "Price_x_ppark", "Price_x_inc",
          grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))

  long[, Task_c := (as.numeric(Task) - 10) / 9]
  long[, none_x_Task := asc4 * Task_c]
  long[, Price_x_Task := Price * Task_c]
  tp <- c("none_x_Task", "Price_x_Task")

  cand <- c("asc2", "asc3", "asc4", pw, px, tp)
  Xall  <- as.matrix(long[, ..cand])
  i1    <- seq(1L, nrow(long), by = 4L)
  tr_ti <- which(long$is_test[i1] == FALSE)
  ridx  <- as.vector(rbind(4L*tr_ti-3L, 4L*tr_ti-2L, 4L*tr_ti-1L, 4L*tr_ti))
  Xt    <- Xall[ridx, , drop = FALSE]
  j1    <- seq(1L, nrow(Xt), by = 4L)
  M     <- (Xt[j1, ] + Xt[j1+1L, ] + Xt[j1+2L, ] + Xt[j1+3L, ]) / 4
  Xc    <- Xt - M[rep(seq_len(nrow(M)), each = 4L), ]
  rm(Xt, M, Xall); gc(verbose = FALSE)
  q    <- qr(Xc, tol = 1e-7)
  keep <- cand[sort(q$pivot[seq_len(q$rank)])]
  rm(Xc); gc(verbose = FALSE)
  cat("taste parameters per class:", length(keep), "\n")

  DEMO_CAT <- c("segmentind", "pparkind", "genderind", "educind", "regionind", "Urbind")
  DEMO_NUM <- c("agea", "incomea", "milesa", "nighta", "yearind", "milesind", "nightind")
  dem <- unique(long[, c("Case", "is_test", DEMO_CAT, DEMO_NUM), with = FALSE])
  setorder(dem, Case)
  Zl <- list(Intercept = rep(1, nrow(dem)))
  for (v in DEMO_CAT) {
    lv <- sort(unique(dem[[v]]))
    for (l in lv[-1]) Zl[[sprintf("%s_%s", v, l)]] <- as.numeric(dem[[v]] == l)
  }
  tf <- function(v, x) if (v %in% c("incomea", "milesa", "nighta")) log1p(x) else x
  for (v in DEMO_NUM) { x <- tf(v, as.numeric(dem[[v]])); Zl[[v]] <- (x - mean(x)) / sd(x) }
  Z <- do.call(cbind, Zl); rownames(Z) <- as.character(dem$Case)

  # luxury flag per respondent -- the ONLY new object this file introduces
  luxr <- setNames(as.integer(dem$segmentind %in% c(3, 5)), as.character(dem$Case))
  list(long = long, keep = keep, Z = Z, luxr = luxr)
}

# =============================================================================
# 2-3. Weighted conditional logit and membership model -- unchanged except that
#      fit_membership now takes a per-respondent weight vector.
# =============================================================================
cl_prob <- function(X, beta) {
  Vm <- matrix(as.vector(X %*% beta), ncol = 4, byrow = TRUE)
  mx <- pmax(Vm[,1], Vm[,2], Vm[,3], Vm[,4]); E <- exp(Vm - mx); S <- rowSums(E)
  list(P = E / S, lse = mx + log(S), Vm = Vm)
}
cl_ll_task <- function(X, beta, yrow) {
  cp <- cl_prob(X, beta); cp$Vm[cbind(seq_along(yrow), yrow)] - cp$lse
}
cl_obj <- function(X, beta, yrow, wtask, ridge)
  sum(wtask * cl_ll_task(X, beta, yrow)) - 0.5 * ridge * sum(beta^2)

cl_newton <- function(X, yrow, wtask, beta, ridge, nsteps = 2, tol = 1e-8) {
  N <- length(yrow); i1 <- seq(1L, 4L*N, by = 4L); wr <- rep(wtask, each = 4L)
  Xch <- X[i1 - 1L + yrow, , drop = FALSE]
  f <- cl_obj(X, beta, yrow, wtask, ridge)
  for (s in seq_len(nsteps)) {
    cp <- cl_prob(X, beta); pv <- as.vector(t(cp$P)); Xp <- X * pv
    Xb <- Xp[i1,,drop=FALSE] + Xp[i1+1L,,drop=FALSE] + Xp[i1+2L,,drop=FALSE] + Xp[i1+3L,,drop=FALSE]
    g <- colSums(wtask * (Xch - Xb)) - ridge * beta
    H <- crossprod(X, X * (wr * pv)) - crossprod(Xb, Xb * wtask)
    diag(H) <- diag(H) + ridge
    step <- tryCatch(solve(H, g), error = function(e) {
      diag(H) <- diag(H) + 1e-4 * max(diag(H)); solve(H, g) })
    ok <- FALSE
    for (h in 0:12) {
      bnew <- beta + step / 2^h; fnew <- cl_obj(X, bnew, yrow, wtask, ridge)
      if (is.finite(fnew) && fnew >= f) { ok <- TRUE; break }
    }
    if (!ok) break
    rel <- abs(fnew - f) / (abs(f) + 1); beta <- bnew; f <- fnew
    if (rel < tol) break
  }
  list(beta = beta, obj = f)
}

row_softmax <- function(eta) {
  mx <- do.call(pmax, lapply(seq_len(ncol(eta)), function(j) eta[, j]))
  E <- exp(eta - mx); E / rowSums(E)
}
row_lse <- function(eta) {
  mx <- do.call(pmax, lapply(seq_len(ncol(eta)), function(j) eta[, j]))
  mx + log(rowSums(exp(eta - mx)))
}

# THE ONE CHANGE: `wresp` (a per-respondent weight) and `ridge` (= LAMBDA_G).
fit_membership <- function(Z, H, G0, ridge, wresp) {
  Cc <- ncol(H); p <- ncol(Z)
  if (Cc == 1) return(matrix(0, p, 0))
  fn <- function(g) {
    G <- matrix(g, p, Cc - 1); eta <- cbind(0, Z %*% G)
    -(sum(wresp * H * (eta - row_lse(eta))) - 0.5 * ridge * sum(g^2))
  }
  gr <- function(g) {
    G <- matrix(g, p, Cc - 1); eta <- cbind(0, Z %*% G); Pi <- row_softmax(eta)
    -as.vector(crossprod(Z, (wresp * (H - Pi))[, -1, drop = FALSE]) - ridge * G)
  }
  o <- optim(as.vector(G0), fn, gr, method = "BFGS", control = list(maxit = 300))
  matrix(o$par, p, Cc - 1)
}
membership_prob <- function(Z, G) {
  if (!ncol(G)) return(matrix(1, nrow(Z), 1))
  row_softmax(cbind(0, Z %*% G))
}

# =============================================================================
# 4. EM -- unchanged apart from threading wresp into the membership M-step
# =============================================================================
em_fit <- function(X, yrow, resp, Z, wresp, H0, n_iter, betas0 = NULL, G0 = NULL,
                   verbose = TRUE, tag = "") {
  R <- nrow(Z); P <- ncol(X)
  betas <- if (is.null(betas0)) replicate(C, rep(0, P), simplify = FALSE) else betas0
  G     <- if (is.null(G0)) matrix(0, ncol(Z), C - 1) else G0
  H     <- H0; path <- numeric(0); prev <- -Inf
  for (it in seq_len(n_iter)) {
    nst <- if (it <= 2) 3L else if (it <= 5) 2L else 1L
    for (cc in seq_len(C)) {
      w <- H[resp, cc]
      betas[[cc]] <- cl_newton(X, yrow, w, betas[[cc]], LAMBDA_B, nsteps = nst)$beta
    }
    G <- fit_membership(Z, H, G, LAMBDA_G, wresp)
    LLr <- matrix(0, R, C)
    for (cc in seq_len(C)) {
      llt <- cl_ll_task(X, betas[[cc]], yrow)
      s <- rowsum(llt, resp, reorder = TRUE)
      stopifnot(nrow(s) == R, identical(rownames(s), as.character(seq_len(R))))
      LLr[, cc] <- as.vector(s)
    }
    logpi <- log(pmax(membership_prob(Z, G), 1e-12))
    num <- logpi + LLr
    ll  <- sum(row_lse(num))
    H   <- row_softmax(num)
    path <- c(path, ll)
    if (verbose) cat(sprintf("  %s iter %2d  LL = %.3f  shares = %s\n", tag, it, ll,
                             paste(sprintf("%.3f", colMeans(H)), collapse = " ")))
    if (is.finite(prev) && abs(ll - prev) / (abs(prev) + 1) < TOL) { prev <- ll; break }
    prev <- ll
  }
  list(betas = betas, G = G, H = H, ll = prev, path = path)
}

init_H <- function(R, none_rate, kind, seed = 1) {
  if (kind == "det") {
    grp <- cut(rank(none_rate, ties.method = "first"), breaks = C, labels = FALSE)
    H <- matrix(0.15 / (C - 1), R, C); H[cbind(seq_len(R), grp)] <- 0.85
  } else {
    set.seed(seed); H <- matrix(rgamma(R * C, 1), R, C); H <- H / rowSums(H)
  }
  H
}

# =============================================================================
# 5. Driver
# =============================================================================
bd <- build_data()
long <- bd$long; keep <- bd$keep; Z <- bd$Z; luxr <- bd$luxr

tasks <- unique(long[, .(No, Case, y, is_test)]); setorder(tasks, No)
folds <- readRDS(FOLDFILE)
tasks <- merge(tasks, folds[, .(No, fold)], by = "No", all.x = TRUE); setorder(tasks, No)
stopifnot(identical(tasks$No, long$No[seq(1L, nrow(long), by = 4L)]))
stopifnot(nrow(tasks) == nrow(long) / 4)
stopifnot(sum(tasks$is_test == FALSE) == 21565, sum(tasks$is_test == TRUE) == 4997)
stopifnot(!anyNA(tasks$fold[tasks$is_test == FALSE]))

Xall <- as.matrix(long[, ..keep])
tr_t <- which(tasks$is_test == FALSE); te_t <- which(tasks$is_test == TRUE)
rows_of <- function(ti) as.vector(rbind(4L*ti-3L, 4L*ti-2L, 4L*ti-1L, 4L*ti))
none_rate_all <- tasks[is_test == FALSE, .(nr = mean(y == 4)), by = Case]

fit_one <- function(train_tasks) {
  ti <- train_tasks
  X <- Xall[rows_of(ti), , drop = FALSE]
  yrow <- tasks$y[ti]; cases <- tasks$Case[ti]
  ucase <- sort(unique(cases)); resp <- match(cases, ucase)
  Zf <- Z[as.character(ucase), , drop = FALSE]
  nr <- none_rate_all[match(ucase, Case), nr]
  # membership weights: luxury respondents get LUXW, others 1, then renormalised
  # so the total weight equals the respondent count (keeps LAMBDA_G comparable).
  lx <- luxr[as.character(ucase)]
  wr <- ifelse(lx == 1, LUXW, 1); wr <- wr * length(wr) / sum(wr)

  starts <- list(list(kind = "det", seed = 0), list(kind = "rand", seed = 4242))
  scr <- lapply(starts, function(s)
    em_fit(X, yrow, resp, Zf, wr, init_H(length(ucase), nr, s$kind, s$seed),
           N_SCREEN, verbose = TRUE, tag = sprintf("[screen %s]", s$kind)))
  lls <- sapply(scr, function(z) tail(z$path, 1))
  cat("  start screening LL:", paste(sprintf("%.2f", lls), collapse = " / "),
      "-> keeping", starts[[which.max(lls)]]$kind, "\n")
  w <- scr[[which.max(lls)]]
  fin <- em_fit(X, yrow, resp, Zf, wr, w$H, N_MAX, betas0 = w$betas, G0 = w$G,
                verbose = TRUE, tag = "[final]")
  fin$path <- c(w$path, fin$path); fin$ucase <- ucase
  fin
}

predict_lc <- function(fit, target_tasks) {
  ti <- target_tasks
  X <- Xall[rows_of(ti), , drop = FALSE]
  Zp <- Z[as.character(tasks$Case[ti]), , drop = FALSE]
  Pi <- membership_prob(Zp, fit$G)
  out <- matrix(0, length(ti), 4)
  for (cc in seq_along(fit$betas)) out <- out + Pi[, cc] * cl_prob(X, fit$betas[[cc]])$P
  out
}

oof <- matrix(NA_real_, length(tr_t), 4)
t0 <- proc.time()
for (k in 1:5) {
  cat("--- fold", k, "---\n")
  itr <- tr_t[tasks$fold[tr_t] != k]; iva <- tr_t[tasks$fold[tr_t] == k]
  stopifnot(length(intersect(tasks$Case[itr], tasks$Case[iva])) == 0)
  fit <- fit_one(itr)
  oof[match(iva, tr_t), ] <- predict_lc(fit, iva)
  cat("  fold", k, "logloss:", round(logloss(tasks$y[iva], oof[match(iva, tr_t), ]), 5),
      " elapsed", round((proc.time() - t0)[3] / 60, 1), "min\n")
}
ll_oof <- logloss(tasks$y[tr_t], oof)
cat(sprintf("\n>>> lcxg_%s honest OOF logloss: %.5f   (lcmnl3_both 1.13863)\n", TAG, ll_oof))

cat("--- full refit ---\n")
full <- fit_one(tr_t)
te <- predict_lc(full, te_t)
cat(sprintf(">>> shipped TEST none-rate: %.5f   (r* = 0.266521, lcmnl3_both 0.22367)\n",
            mean(te[, 4])))
segt <- as.integer(unique(long[, .(Case, segmentind)])[match(tasks$Case[te_t], Case)]$segmentind)
lx_te <- as.integer(segt %in% c(3, 5))
cat(sprintf(">>> shipped TEST none-rate  lux %.5f | non %.5f\n",
            mean(te[lx_te == 1, 4]), mean(te[lx_te == 0, 4])))
sego <- as.integer(unique(long[, .(Case, segmentind)])[match(tasks$Case[tr_t], Case)]$segmentind)
lx_tr <- as.integer(sego %in% c(3, 5))
cat(sprintf(">>> OOF none-rate  all %.5f | lux %.5f (observed 0.15986) | non %.5f (observed 0.31712)\n",
            mean(oof[, 4]), mean(oof[lx_tr == 1, 4]), mean(oof[lx_tr == 0, 4])))

# ---- artifacts written as the LAST act -------------------------------------
oof <- clip_norm(oof)          # production does this; keep the artifacts comparable
saveRDS(data.table(No = tasks$No[tr_t], p1 = oof[,1], p2 = oof[,2], p3 = oof[,3], p4 = oof[,4]),
        OOF_OUT)
te <- clip_norm(te)
saveRDS(data.table(No = tasks$No[te_t], p1 = te[,1], p2 = te[,2], p3 = te[,3], p4 = te[,4]),
        TEST_OUT)
cat("wrote", OOF_OUT, "and", TEST_OUT, "\n")
cat("OK\n")
