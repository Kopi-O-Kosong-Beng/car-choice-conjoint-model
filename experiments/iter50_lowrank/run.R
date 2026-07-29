# =============================================================================
# ITERATION 50 -- REDUCED-RANK DEMOGRAPHIC x ATTRIBUTE INTERACTION
#                 (matrix factorization as the MODEL, not as features)
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# WHY, AND WHY THIS IS NOT ITERATION 49
# -----------------------------------------------------------------------------
# Iteration 49 tested factorization as a FEATURE TRANSFORM (PCA/SVD blocks fed to
# xgboost). This tests factorization as the MODEL: a low-rank constraint on the
# matrix of demographic-by-attribute interaction coefficients.
#
# Write the respondent-varying conditional logit as
#
#     u_rj = x_j' beta  +  z_r' C x_j
#
# where x_j are the ~80 part-worth columns and z_r the 15 demographics. C is the
# 15 x 80 matrix of "how does this person's taste for this attribute level differ
# from the pooled average". FULL RANK C is 1,200 free parameters on 21,565 choice
# sets -- unestimable, and nobody in this repo has ever tried it. Production
# instead carries TWELVE hand-picked cells of C (the Price_x_age, Price_x_inc,
# Price_x_seg*, Price_x_reg* terms in 02_mnl_partworth.R:56). Every other cell is
# pinned to zero by fiat, never by evidence.
#
# A RANK-F factorization C = Gamma V' costs F*(15+80) parameters instead of 1,200
# -- 190 at F = 2. It lets EVERY attribute have demographically-varying
# part-worths while regularising the whole thing through the rank. That is
# exactly SVD's "best rank-k approximation" (Topic 8) applied to the object that
# actually has structure here, rather than to a design matrix that (iteration 49,
# Section 3) has a measurably FLAT spectrum and therefore nothing to compress.
#
# Equivalently: u_rj = x_j'beta + sum_f (z_r' gamma_f)(x_j' v_f). Each factor f is
# a latent taste direction v_f in attribute space, and a demographic score
# z_r'gamma_f saying how much of it respondent r carries. This is a latent-factor
# recommender with content-based cold start (Topic 7) -- the ONLY factorization
# form that survives this competition's structure, because test respondents are
# 263 entirely new people with zero observed choices, so a free per-respondent
# factor is unidentified and must be reached through demographics or not at all.
#
# -----------------------------------------------------------------------------
# WHAT WOULD MAKE THIS FAIL, STATED FIRST
# -----------------------------------------------------------------------------
# Iteration 18 measured demographics at R^2 0.0968 against the TRUE none-propensity
# heterogeneity -- roughly 11% reach, ~89% of the relevant taste variation is
# unobservable in this dataset. z_r' C x_j can only ever recover the demographic
# share. So the honest prior is a SMALL gain or none, and the informative outcome
# is the size of the ceiling, which is report material either way.
#
# Iteration 49 also just demonstrated the specific failure mode to avoid: turning
# coarse ordinal demographics into dense real-valued columns let the learner
# fingerprint individual respondents, costing 0.0123. This model cannot do that
# -- z_r enters ONLY through F linear scores that multiply attribute contrasts,
# never as a free per-person effect, and a conditional logit cannot use a
# respondent-constant term at all (it cancels in the within-task softmax).
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# A rank-F demographic x attribute interaction beats the part-worth conditional
# logit (mnl_pw, OOF 1.15686) that it nests at F = 0, AND -- the gate that
# actually matters -- earns weight in the production blend, because it is a THIRD
# error family. Iteration 19 found 93% of blend error variance on a single
# tree-vs-logit axis (eigenvalues 3.726/0.202/0.056/0.016); a member that errs
# off that axis can pay even at a worse solo score.
#
# DIRECTIONAL PRE-REGISTRATION: if the low-rank interaction is real, F = 1 or 2
# should capture most of the gain and F = 4 should not be materially better than
# F = 2. A gain that only appears at the largest F is the model buying capacity,
# not structure, and is to be read as noise.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
#   1. F = 0 is refitted here by the SAME optimiser as F > 0. Every comparison is
#      against that internal control, never against mlogit's 1.15686, so that an
#      optimiser difference can never be read as a modelling gain.
#   2. Beat the F = 0 control by more than the model-level seed sd 0.00283. (This
#      model is deterministic given the fold, so that floor is conservative.)
#   3. BLEND GATE, and this is the decision number: added to xgb_lw2bag +
#      lcmnl3_both it must improve the nested blend by more than the blend-level
#      seed sd 0.00048. Iteration 39 gained +0.00252 at member level and +0.00020
#      at blend level -- member gains do not automatically reach the blend.
#   4. SHIFT AUDIT >= ~100% (model/shift_audit.R). 77% killed the design encoding
#      as population-specific; 64% killed a fatigue term.
#   5. REPLICATE under folds_b before any production change.
# ADOPT only if all five hold. A null closes reduced-rank heterogeneity and is
# reported as such -- iteration 18 is the template for a useful negative result.
#
# ARTIFACTS: oof_mnl_lr.rds / test_mnl_lr.rds -- NEW names, STRING LITERALS only.
# Iteration 39 overwrote a live blend member by inheriting an artifact name
# through a variable and had to recover it from git. Verified absent before write.
# =============================================================================
suppressMessages({ library(data.table) })
source("model/99_utils.R")

DIR <- "experiments/iter50_lowrank"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
FS   <- as.integer(strsplit(Sys.getenv("ITER50_F", "0,1,2,4"), ",")[[1]])
LAMB <- as.numeric(Sys.getenv("ITER50_LAM", "5"))   # ridge on gamma and v
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

# ============================================================== SECTION 1 =====
rule("SECTION 1 -- DESIGN, built exactly as model/02_mnl_partworth.R builds it")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
setorder(long, No, alt)

pw_cols <- character(0)
for (a in ATTRS) {
  lv_real <- sort(unique(long[alt != 4][[a]]))
  ref <- lv_real[1]
  for (l in setdiff(lv_real, ref)) {
    nm <- sprintf("%s_L%s", a, l)
    long[, (nm) := as.numeric(get(a) == l)]
    pw_cols <- c(pw_cols, nm)
  }
}
# Identification: a conditional logit identifies only columns that VARY WITHIN a
# choice set. Task-demean, then rank-revealing QR. This is the check that caught
# HU_L2, which hand-reasoning missed.
cand <- c("asc2", "asc3", "asc4", pw_cols)
X0 <- as.matrix(long[, ..cand])
Xc <- X0 - as.matrix(long[, lapply(.SD, mean), by = No, .SDcols = cand][
  match(long$No, unique(long$No)), ..cand])
qrx <- qr(Xc, tol = 1e-7)
keep <- cand[sort(qrx$pivot[seq_len(qrx$rank)])]
if (length(setdiff(cand, keep)))
  cat("  dropped as unidentified:", paste(setdiff(cand, keep), collapse = ", "), "\n")

# The FACTORISED block excludes the ASCs: an alternative-specific constant is not
# an attribute taste, and letting demographics load on asc4 would silently
# duplicate the none-rate machinery that lcmnl3_both already owns.
xv <- keep
fv <- setdiff(keep, c("asc2", "asc3", "asc4"))
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")

X  <- as.matrix(long[, ..xv])            # utility design
XF <- as.matrix(long[, ..fv])            # factorised design (no ASCs)
NO <- long$No; ALT <- long$alt
nt <- length(unique(NO))
cat(sprintf("  rows %d  tasks %d  utility cols %d  factorised cols %d  demographics %d\n",
            nrow(X), nt, ncol(X), ncol(XF), length(demo)))

# respondent-level demographics, standardised (scale fitted per fold in SECTION 3)
rsp <- unique(long[, c("No", "Case", demo), with = FALSE])
setorder(rsp, No)
stopifnot(nrow(rsp) == nt)
Zraw <- as.matrix(rsp[, ..demo])
caseOf <- rsp$Case
y_task <- unique(long[, .(No, y)])[order(No), y]
is_te  <- unique(long[, .(No, is_test)])[order(No), is_test]
fold_of <- folds[match(rsp$No[!is_te], folds$No), fold]
cat(sprintf("  train tasks %d   test tasks %d\n", sum(!is_te), sum(is_te)))

# ============================================================== SECTION 2 =====
# Likelihood, analytic gradient. Utility u = X b + sum_f (Z g_f)[task] * (XF v_f).
rule("SECTION 2 -- OPTIMISER (analytic gradient, verified against finite differences)")

sm <- function(u) { M <- matrix(u, ncol = 4L, byrow = TRUE); M <- M - apply(M, 1, max)
                    E <- exp(M); E / rowSums(E) }

make_nll <- function(Xs, XFs, Zs, ys, F) {
  n <- length(ys); p <- ncol(Xs); q <- ncol(XFs); m <- ncol(Zs)
  Y <- matrix(0, n, 4L); Y[cbind(seq_len(n), ys)] <- 1
  yv <- as.vector(t(Y))
  function(th, grad = FALSE) {
    b <- th[seq_len(p)]
    u <- as.vector(Xs %*% b)
    if (F > 0) {
      G <- matrix(th[p + seq_len(m * F)], m, F)          # demographics -> loadings
      V <- matrix(th[p + m * F + seq_len(q * F)], q, F)  # latent taste directions
      A <- Zs %*% G                                       # n x F loadings
      S <- XFs %*% V                                      # (4n) x F attribute scores
      Arep <- A[rep(seq_len(n), each = 4L), , drop = FALSE]
      u <- u + rowSums(Arep * S)
    }
    P <- sm(u); pv <- as.vector(t(P))
    nll <- -sum(log(pmax(pv[yv == 1], 1e-300))) / n
    if (F > 0) nll <- nll + LAMB * (sum(G^2) + sum(V^2)) / (2 * n)
    if (!grad) return(nll)
    r <- yv - pv                                          # residual, (4n)
    gb <- -as.vector(crossprod(Xs, r)) / n
    if (F == 0) return(c(nll = nll, g = list(gb)))
    rS <- r * S                                           # (4n) x F
    rSt <- rowsum(rS, rep(seq_len(n), each = 4L), reorder = FALSE)   # n x F
    gG <- -crossprod(Zs, rSt) / n + LAMB * G / n
    rA <- r * Arep
    gV <- -crossprod(XFs, rA) / n + LAMB * V / n
    list(nll = nll, g = c(gb, as.vector(gG), as.vector(gV)))
  }
}

fit_one <- function(idx_tasks, F, seed = 1L) {
  rows <- which(NO %in% rsp$No[idx_tasks])
  Xs <- X[rows, , drop = FALSE]; XFs <- XF[rows, , drop = FALSE]
  Zc <- Zraw[idx_tasks, , drop = FALSE]
  ctr <- colMeans(Zc); scl <- apply(Zc, 2, sd); scl[scl == 0] <- 1
  Zs <- scale(Zc, center = ctr, scale = scl)
  ys <- y_task[idx_tasks]
  f <- make_nll(Xs, XFs, Zs, ys, F)
  p <- ncol(Xs); q <- ncol(XFs); m <- ncol(Zs)
  set.seed(seed)
  th0 <- c(rep(0, p), if (F > 0) rnorm(m * F, sd = 0.01) else numeric(0),
           if (F > 0) rnorm(q * F, sd = 0.01) else numeric(0))
  o <- optim(th0, fn = function(t) f(t), gr = function(t) f(t, grad = TRUE)$g,
             method = "BFGS", control = list(maxit = 600, reltol = 1e-11))
  list(par = o$par, ctr = ctr, scl = scl, F = F, p = p, q = q, m = m, conv = o$convergence)
}

predict_one <- function(fit, idx_tasks) {
  rows <- which(NO %in% rsp$No[idx_tasks])
  Xs <- X[rows, , drop = FALSE]; XFs <- XF[rows, , drop = FALSE]
  Zs <- scale(Zraw[idx_tasks, , drop = FALSE], center = fit$ctr, scale = fit$scl)
  b <- fit$par[seq_len(fit$p)]
  u <- as.vector(Xs %*% b)
  if (fit$F > 0) {
    G <- matrix(fit$par[fit$p + seq_len(fit$m * fit$F)], fit$m, fit$F)
    V <- matrix(fit$par[fit$p + fit$m * fit$F + seq_len(fit$q * fit$F)], fit$q, fit$F)
    A <- Zs %*% G; S <- XFs %*% V
    u <- u + rowSums(A[rep(seq_len(length(idx_tasks)), each = 4L), , drop = FALSE] * S)
  }
  sm(u)
}

# gradient check -- a wrong analytic gradient is the single most likely defect here
set.seed(7); chk <- sample(which(!is_te), 400)
{
  rows <- which(NO %in% rsp$No[chk])
  Zs <- scale(Zraw[chk, , drop = FALSE])
  f <- make_nll(X[rows, ], XF[rows, ], Zs, y_task[chk], 2L)
  th <- rnorm(ncol(X) + ncol(Zs) * 2 + ncol(XF) * 2, sd = 0.05)
  ga <- f(th, grad = TRUE)$g
  ii <- c(1, 5, ncol(X) + 3, ncol(X) + ncol(Zs) * 2 + 7)
  gn <- vapply(ii, function(i) { e <- rep(0, length(th)); e[i] <- 1e-6
                                 (f(th + e) - f(th - e)) / 2e-6 }, 0)
  cat(sprintf("  gradient check (analytic vs finite difference), max abs error %.3e\n",
              max(abs(ga[ii] - gn))))
  stopifnot(max(abs(ga[ii] - gn)) < 1e-6)
}

# ============================================================== SECTION 3 =====
rule(sprintf("SECTION 3 -- NESTED OOF over F = %s   (ridge lambda %.1f)", paste(FS, collapse = ", "), LAMB))

tr_idx <- which(!is_te); te_idx <- which(is_te)
res <- list(); OOF <- list()
for (F in FS) {
  t0 <- Sys.time()
  P <- matrix(NA_real_, length(tr_idx), 4L)
  for (k in 1:5) {
    ins <- tr_idx[fold_of != k]; out <- tr_idx[fold_of == k]
    fit <- fit_one(ins, F)
    P[which(fold_of == k), ] <- predict_one(fit, out)
  }
  ll <- logloss(y_task[tr_idx], P)
  OOF[[as.character(F)]] <- P
  res[[length(res) + 1]] <- data.table(F = F, npar = ncol(X) + F * (length(demo) + ncol(XF)),
                                       oof = ll, mins = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  cat(sprintf("  F = %d   npar %4d   nested OOF %.5f   (%.1f min)\n",
              F, ncol(X) + F * (length(demo) + ncol(XF)), ll, res[[length(res)]]$mins))
  saveRDS(rbindlist(res), file.path(DIR, "rank_curve.rds"))
  saveRDS(OOF, file.path(DIR, "oof_by_F.rds"))
}
R <- rbindlist(res)
rule("RANK CURVE")
print(R)
fwrite(R, file.path(DIR, "rank_curve.csv"))

ctrl <- R$oof[R$F == 0]
best <- R[which.min(oof)]
cat(sprintf("\n  internal control F=0        : %.5f\n", ctrl))
cat(sprintf("  best rank F=%d              : %.5f   (delta %+.5f)\n", best$F, best$oof, best$oof - ctrl))
cat(sprintf("  model-level seed sd 0.00283 -> delta is %.2f sd\n", abs(best$oof - ctrl) / 0.00283))
if (best$F == max(FS) && length(FS) > 2)
  cat("  WARNING: optimum at the LARGEST F -- pre-registered as capacity, not structure.\n")

# ============================================================== SECTION 4 =====
# Emit artifacts for the winning F so the blend gate can be run. Written as the
# LAST act: two experiments in this repo looked like failures for hours because
# the caller died before assembling them.
rule("SECTION 4 -- ARTIFACTS")
Fb <- best$F
if (Fb == 0) {
  cat("  best rank is F = 0; the factorisation adds nothing. No member emitted.\n")
} else {
  oof_dt <- data.table(No = rsp$No[tr_idx])
  Pb <- OOF[[as.character(Fb)]]
  oof_dt[, `:=`(p1 = Pb[, 1], p2 = Pb[, 2], p3 = Pb[, 3], p4 = Pb[, 4])]
  fit_all <- fit_one(tr_idx, Fb)
  Pt <- predict_one(fit_all, te_idx)
  te_dt <- data.table(No = rsp$No[te_idx], p1 = Pt[, 1], p2 = Pt[, 2], p3 = Pt[, 3], p4 = Pt[, 4])
  stopifnot(nrow(oof_dt) == 21565L, nrow(te_dt) == 4997L,
            max(abs(rowSums(oof_dt[, .(p1,p2,p3,p4)]) - 1)) < 1e-9,
            max(abs(rowSums(te_dt[,  .(p1,p2,p3,p4)]) - 1)) < 1e-9)
  stopifnot(!file.exists("model/artifacts/oof_mnl_lr.rds"))
  saveRDS(oof_dt, "model/artifacts/oof_mnl_lr.rds")
  saveRDS(te_dt,  "model/artifacts/test_mnl_lr.rds")
  cat(sprintf("  wrote oof_mnl_lr.rds / test_mnl_lr.rds at F = %d (OOF %.5f)\n", Fb, best$oof))
  cat("  NEXT: compare.R mnl_pw mnl_lr, then the BLEND GATE, then shift_audit.R, then folds_b.\n")
}
cat("\ndone\n")
