# =============================================================================
# ITERATION 17 — Hierarchical Bayes conditional logit (bayesm::rhierMnlRwMixture)
#
# HYPOTHESIS (stated before running)
# ---------------------------------
# Conjoint data with a panel structure is what HB was built for: every respondent
# gets their own part-worth vector beta_i, drawn from a population distribution
# estimated jointly with the individual draws. Iteration 05 already showed that a
# mixed logit with a SINGLE normal population distribution loses here (1.17281,
# zero blend weight), and the diagnosis was structural: every test respondent is
# unseen, so the only object we can emit is the POPULATION-AVERAGED probability
#
#        P(y | X) = INT softmax(X beta) dF(beta | z)
#
# and integrating a smooth unimodal F blurs the prediction without buying anything.
#
# Two things could make HB behave differently from that failure:
#   (a) F is a MIXTURE of normals, not one normal. Iteration 11 found three very
#       distinct segments (none-rates 0.049 / 0.228 / 0.648), so the true F is
#       strongly multimodal and a single normal cannot represent it.
#   (b) F is conditioned on demographics through beta_i = Delta' z_i + u_i, which
#       is the continuous analogue of the demographic membership model that carries
#       93% of lcmnl3's gain.
# On top of that, Bayesian shrinkage is better behaved than simulated-ML with only
# 19 observations per respondent.
#
# EXPECTATION: modest at best. lcmnl3 (1.14396) already captures a discrete version
# of the same structure, and HB adds continuous within-segment spread, which can
# only BLUR the population-averaged prediction relative to a point-mass segment.
# A clean negative result is a perfectly good outcome and is reported as such.
#
# THE LEAK THAT WOULD FAKE A HUGE GAIN — and how it is closed
# ----------------------------------------------------------
# HB's headline output, betadraw, is the per-respondent posterior. Using a held-out
# respondent's own beta_i (or running an E-step / posterior update on their choices)
# would be reading their labels straight back. Even using TRAINING respondents'
# fitted betas as an empirical predictive distribution is wrong here, because the
# fitted betas are shrunken toward the fold's own likelihood.
# Prediction below therefore touches exactly three things:
#     the posterior draws of (Delta, pvec, {mu_c, Sigma_c})  -- fit on folds != k,
#     the held-out task DESIGN matrix,
#     the held-out respondents' DEMOGRAPHICS.
# betadraw is never indexed for prediction. Assertions enforce that the Cases in
# the fitting set and the scored fold are disjoint.
#
# MODEL
#   y_it ~ MNL(X_it, beta_i),  beta_i = Delta' z_i + u_i,  u_i ~ sum_c pvec_c N(mu_c, Sigma_c)
#   X    : part-worth dummies for all 20 attributes + asc2/asc3/asc4, reference level
#          = lowest level occurring on a REAL bundle, unidentified columns removed by
#          a rank-revealing QR on the TASK-DEMEANED design (same recipe as mnl_pw).
#          The explicit Price x demographic interactions used by mnl_pw/lcmnl3 are
#          DROPPED: individual-level price part-worths plus Delta'z_i subsume them.
#   Z    : demographics, no intercept, centred and scaled on the FITTING respondents
#          only (the mixture mu absorbs the intercept).
#
# PRIOR (fixed a priori, never tuned on OOF)
#   bayesm's default IW prior is nu = nvar+3, V = nu*I, i.e. E[Sigma] = (nu/2)*I.
#   With nvar ~ 73 that is E[Sigma_jj] ~ 38 -- a part-worth standard deviation of 6
#   utiles, which is absurd for 0/1 dummies whose fitted part-worths are O(1) (see
#   the lcmnl3 segment table). Left at the default it inflates Sigma by roughly 50%
#   and blurs exactly the quantity we predict with. We instead set
#        nu = nvar + 5,  V = (nu - nvar - 1) * I = 4I   =>   E[Sigma] = I,
#   i.e. an a-priori part-worth sd of 1 utile. This is a scale judgement made from
#   the design coding, not from any OOF number. Config "dflt" re-runs the identical
#   model under bayesm's default prior so the choice can be audited.
#
# USAGE
#   Rscript experiments/iter17_hb/run.R <config> <1|2|3|4|5|full|combine>
#   configs: main (ncomp=3), c1 (ncomp=1), c5 (ncomp=5), dflt (ncomp=3, bayesm prior)
#   env HB_R (MCMC draws), HB_KEEP (thinning), HB_S (mixture sims per kept draw)
# =============================================================================

suppressMessages({ library(data.table); library(bayesm) })
source("model/99_utils.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: run.R <config> <1..5|full|combine>")
CFG <- args[1]; JOB <- args[2]

CONFIGS <- list(
  main = list(ncomp = 3, prior = "tight", name = "hbmnl"),
  c1   = list(ncomp = 1, prior = "tight", name = "hbmnl_c1"),
  c5   = list(ncomp = 5, prior = "tight", name = "hbmnl_c5"),
  dflt = list(ncomp = 3, prior = "bayesm", name = "hbmnl_dflt")
)
if (!CFG %in% names(CONFIGS)) stop("unknown config: ", CFG)
CO   <- CONFIGS[[CFG]]
NAME <- CO$name

R_MCMC  <- as.integer(Sys.getenv("HB_R",    "40000"))
KEEP    <- as.integer(Sys.getenv("HB_KEEP", "50"))
S_SIM   <- as.integer(Sys.getenv("HB_S",    "30"))   # mixture draws per kept MCMC draw
BURNFRAC <- 0.5                                       # fraction of KEPT draws discarded
OUTDIR  <- "experiments/iter17_hb"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# 1. Design
# =============================================================================
build <- function() {
  long <- readRDS("model/artifacts/long.rds")
  setorder(long, No, alt)
  stopifnot(nrow(long) %% 4 == 0)
  stopifnot(all(long$alt == rep(1:4, nrow(long) / 4)))
  stopifnot(all(long$No == rep(long$No[seq(1, nrow(long), 4)], each = 4)))

  # ---- part-worth coding ----------------------------------------------------
  pw <- character(0)
  for (a in ATTRS) {
    lv <- sort(unique(long[alt != 4][[a]])); ref <- lv[1]
    for (l in setdiff(lv, ref)) {
      nm <- sprintf("%s_L%s", a, l)
      long[, (nm) := as.numeric(get(a) == l)]
      pw <- c(pw, nm)
    }
  }
  cand <- c("asc2", "asc3", "asc4", pw)

  # ---- identification: rank-revealing QR on the TASK-DEMEANED training design
  Xall  <- as.matrix(long[, ..cand])
  i1    <- seq(1L, nrow(long), by = 4L)
  tr_ti <- which(long$is_test[i1] == FALSE)
  ridx  <- as.vector(rbind(4L*tr_ti - 3L, 4L*tr_ti - 2L, 4L*tr_ti - 1L, 4L*tr_ti))
  Xt    <- Xall[ridx, , drop = FALSE]
  j1    <- seq(1L, nrow(Xt), by = 4L)
  M     <- (Xt[j1, ] + Xt[j1+1L, ] + Xt[j1+2L, ] + Xt[j1+3L, ]) / 4
  Xc    <- Xt - M[rep(seq_len(nrow(M)), each = 4L), ]
  rm(Xt, M); gc(verbose = FALSE)
  q    <- qr(Xc, tol = 1e-7)
  keep <- cand[sort(q$pivot[seq_len(q$rank)])]
  rm(Xc); gc(verbose = FALSE)
  drp  <- setdiff(cand, keep)
  if (length(drp)) cat("dropped as unidentified:", paste(drp, collapse = ", "), "\n")
  cat("nvar (part-worths per respondent):", length(keep), "\n")

  # ---- demographics, raw (centring/scaling happens per fold) -----------------
  DEMO_CAT <- c("segmentind", "pparkind", "genderind", "educind", "regionind", "Urbind")
  DEMO_NUM <- c("agea", "incomea", "milesa", "nighta", "yearind", "milesind", "nightind")
  dem <- unique(long[, c("Case", DEMO_CAT, DEMO_NUM), with = FALSE])
  setorder(dem, Case)
  stopifnot(nrow(dem) == uniqueN(long$Case))
  Zl <- list()
  for (v in DEMO_CAT) {
    lv <- sort(unique(dem[[v]]))
    for (l in lv[-1]) Zl[[sprintf("%s_%s", v, l)]] <- as.numeric(dem[[v]] == l)
  }
  for (v in DEMO_NUM) {
    x <- as.numeric(dem[[v]])
    Zl[[v]] <- if (v %in% c("incomea", "milesa", "nighta")) log1p(x) else x
  }
  Zraw <- do.call(cbind, Zl); rownames(Zraw) <- as.character(dem$Case)
  cat("nz (demographic covariates, no intercept):", ncol(Zraw), "\n")

  tasks <- unique(long[, .(No, Case, y, is_test)]); setorder(tasks, No)
  folds <- readRDS("model/artifacts/folds.rds")
  tasks <- merge(tasks, folds[, .(No, fold)], by = "No", all.x = TRUE)
  setorder(tasks, No)
  stopifnot(identical(tasks$No, long$No[seq(1L, nrow(long), by = 4L)]))
  stopifnot(sum(tasks$is_test == FALSE) == 21565, sum(tasks$is_test == TRUE) == 4997)
  stopifnot(!anyNA(tasks$fold[tasks$is_test == FALSE]))

  list(X = as.matrix(long[, ..keep]), keep = keep, Zraw = Zraw, tasks = tasks)
}

rows_of <- function(ti) as.vector(rbind(4L*ti - 3L, 4L*ti - 2L, 4L*ti - 1L, 4L*ti))

# =============================================================================
# 2. Fit
# =============================================================================
# train_tasks: integer task indices (rows of `tasks`) used for fitting.
# Returns ONLY the population-level posterior draws needed for prediction, plus
# diagnostics. betadraw is deliberately summarised and discarded.
fit_hb <- function(bd, train_tasks, tag) {
  X <- bd$X; tasks <- bd$tasks
  ti <- train_tasks
  cs <- tasks$Case[ti]
  ucase <- sort(unique(cs))
  nvar <- ncol(X); nlgt <- length(ucase)

  # per-fold centring/scaling of Z, fitting respondents only
  Ztr_raw <- bd$Zraw[as.character(ucase), , drop = FALSE]
  zmu <- colMeans(Ztr_raw)
  zsd <- apply(Ztr_raw, 2, sd); zsd[zsd < 1e-8] <- 1
  Z <- sweep(sweep(Ztr_raw, 2, zmu, "-"), 2, zsd, "/")
  stopifnot(max(abs(colMeans(Z))) < 1e-8)     # bayesm requires centred Z

  lgtdata <- vector("list", nlgt)
  for (i in seq_len(nlgt)) {
    ii <- ti[cs == ucase[i]]
    lgtdata[[i]] <- list(y = tasks$y[ii], X = X[rows_of(ii), , drop = FALSE])
  }
  ni <- vapply(lgtdata, function(z) length(z$y), 1L)
  cat(sprintf("[%s] nlgt=%d  nvar=%d  nz=%d  tasks/resp=%s  ncomp=%d\n",
              tag, nlgt, nvar, ncol(Z), paste(range(ni), collapse = "-"), CO$ncomp))

  Prior <- list(ncomp = CO$ncomp)
  if (CO$prior == "tight") {
    nu <- nvar + 5
    Prior$nu <- nu
    Prior$V  <- (nu - nvar - 1) * diag(nvar)   # E[Sigma] = I
  }
  Mcmc <- list(R = R_MCMC, keep = KEEP, nprint = max(R_MCMC %/% 10, 1))

  t0 <- Sys.time()
  out <- rhierMnlRwMixture(Data = list(p = 4L, lgtdata = lgtdata, Z = Z),
                           Prior = Prior, Mcmc = Mcmc)
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  cat(sprintf("[%s] MCMC done in %.1f min\n", tag, mins))

  D  <- length(out$nmix$compdraw)
  pd <- out$nmix$probdraw; dim(pd) <- c(D, CO$ncomp)
  b0 <- as.integer(ceiling(BURNFRAC * D))
  sel <- (b0 + 1L):D
  cat(sprintf("[%s] kept draws %d (thin %d of R=%d); burn-in = first %d kept; using %d\n",
              tag, D, KEEP, R_MCMC, b0, length(sel)))

  # in-sample individual-level fit -- DIAGNOSTIC ONLY, never used for prediction
  bmean <- apply(out$betadraw[, , sel, drop = FALSE], c(1, 2), mean)

  res <- list(compdraw = out$nmix$compdraw[sel],
              probdraw = pd[sel, , drop = FALSE],
              Deltadraw = out$Deltadraw[sel, , drop = FALSE],
              loglike = as.numeric(out$loglike), burn = b0, D = D,
              zmu = zmu, zsd = zsd, ucase = ucase, nvar = nvar, nz = ncol(Z),
              mins = mins, beta_mean = bmean)
  rm(out); gc(verbose = FALSE)
  res
}

# =============================================================================
# 3. Population-averaged prediction  (the no-leak path)
# =============================================================================
# Inputs: posterior draws (fit), held-out DESIGN, held-out DEMOGRAPHICS. Nothing else.
# For each retained MCMC draw d we draw S vectors u ~ mixture_d, form
#   beta = Delta_d' z_r + u,  and average softmax(X beta) over all D*S samples.
# use_delta = FALSE sets z_r := 0 for everyone, i.e. ablates the demographic
# channel while keeping the mixture -- the direct analogue of lcmnl3's ablation.
predict_pop <- function(fit, Xho, resp_idx, Zho, S = S_SIM, seed = 20260726,
                        use_delta = TRUE) {
  set.seed(seed)
  n <- nrow(Xho); nvar <- ncol(Xho); ntask <- n %/% 4L
  stopifnot(n %% 4L == 0L, nvar == fit$nvar, ncol(Zho) == fit$nz,
            length(resp_idx) == n, max(resp_idx) <= nrow(Zho))
  D <- length(fit$compdraw)
  acc <- matrix(0, ntask, 4)
  Zrow <- Zho[resp_idx, , drop = FALSE]
  for (d in seq_len(D)) {
    base <- if (use_delta) {
      Dm <- matrix(fit$Deltadraw[d, ], nrow = fit$nz, ncol = nvar)   # nz x nvar
      rowSums((Xho %*% t(Dm)) * Zrow)
    } else numeric(n)
    cmp <- fit$compdraw[[d]]; pv <- fit$probdraw[d, ]
    # Sigma_c = t(root) %*% root with root = backsolve(rooti, I)  [bayesm::momMix]
    # => L = t(root) satisfies L L' = Sigma_c
    Ls <- lapply(cmp, function(cc) t(backsolve(cc$rooti, diag(nvar))))
    ci <- if (length(pv) == 1L) rep(1L, S) else
            sample.int(length(pv), S, replace = TRUE, prob = pv)
    Zr <- matrix(rnorm(nvar * S), nvar, S)
    U  <- matrix(0, nvar, S)
    for (s in seq_len(S)) U[, s] <- cmp[[ci[s]]]$mu + Ls[[ci[s]]] %*% Zr[, s]
    XU <- Xho %*% U
    for (s in seq_len(S)) {
      Vm <- matrix(base + XU[, s], ncol = 4, byrow = TRUE)
      m  <- pmax(Vm[, 1], Vm[, 2], Vm[, 3], Vm[, 4])
      E  <- exp(Vm - m)
      acc <- acc + E / rowSums(E)
    }
  }
  acc / (D * S)
}

# probabilities from a fixed beta per respondent (used ONLY for the in-sample
# individual-level diagnostic on the training fit)
probs_fixed <- function(Xho, resp_idx, B) {
  V  <- rowSums(Xho * B[resp_idx, , drop = FALSE])
  Vm <- matrix(V, ncol = 4, byrow = TRUE)
  m  <- pmax(Vm[, 1], Vm[, 2], Vm[, 3], Vm[, 4])
  E  <- exp(Vm - m); E / rowSums(E)
}

# =============================================================================
# 4. Driver
# =============================================================================
if (JOB == "combine") {
  bd <- build(); tasks <- bd$tasks
  tr_t <- which(tasks$is_test == FALSE); te_t <- which(tasks$is_test == TRUE)
  oof <- matrix(NA_real_, length(tr_t), 4)
  oof_nod <- matrix(NA_real_, length(tr_t), 4)
  oof_s2  <- matrix(NA_real_, length(tr_t), 4)
  for (k in 1:5) {
    f <- file.path(OUTDIR, sprintf("pred_%s_f%d.rds", CFG, k))
    if (!file.exists(f)) stop("missing ", f)
    p <- readRDS(f)
    idx <- match(p$task_idx, tr_t); stopifnot(!anyNA(idx))
    oof[idx, ] <- p$P; oof_nod[idx, ] <- p$P_nodelta; oof_s2[idx, ] <- p$P_seed2
  }
  stopifnot(!anyNA(oof))
  y <- tasks$y[tr_t]
  ll  <- logloss(y, oof)
  cat(sprintf("\n>>> %s (%s) honest OOF logloss: %.5f\n", NAME, CFG, ll))
  cat(sprintf("    second RNG seed for the mixture simulation: %.5f  (MC noise = %.5f)\n",
              logloss(y, oof_s2), abs(ll - logloss(y, oof_s2))))
  cat(sprintf("    ablation, demographic channel removed (z := 0): %.5f\n",
              logloss(y, oof_nod)))
  cat("    reference: lcmnl3 1.14396 | mnl_pw 1.15686 | xgb_lw2 1.14152 | mixl 1.17281\n")
  for (k in 1:5) {
    p <- readRDS(file.path(OUTDIR, sprintf("pred_%s_f%d.rds", CFG, k)))
    cat(sprintf("    fold %d: %.5f  (%.1f min)\n", k,
                logloss(tasks$y[p$task_idx], p$P), p$mins))
  }
  saveRDS(data.table(No = tasks$No[tr_t], p1 = oof[,1], p2 = oof[,2],
                     p3 = oof[,3], p4 = oof[,4]),
          sprintf("model/artifacts/oof_%s.rds", NAME))

  ftest <- file.path(OUTDIR, sprintf("pred_%s_full.rds", CFG))
  if (file.exists(ftest)) {
    p <- readRDS(ftest); stopifnot(identical(p$task_idx, te_t))
    te <- clip_norm(p$P)
    saveRDS(data.table(No = tasks$No[te_t], p1 = te[,1], p2 = te[,2],
                       p3 = te[,3], p4 = te[,4]),
            sprintf("model/artifacts/test_%s.rds", NAME))
    cat(sprintf("    wrote test_%s.rds  (predicted none-rate %.3f vs train %.3f)\n",
                NAME, mean(te[, 4]), mean(tasks$y[tr_t] == 4)))
  } else cat("    NOTE: no full refit yet, test artifact not written\n")
  cat("OK\n")
  quit(save = "no")
}

bd <- build(); tasks <- bd$tasks
tr_t <- which(tasks$is_test == FALSE); te_t <- which(tasks$is_test == TRUE)

if (JOB == "full") { itr <- tr_t; iho <- te_t; tag <- sprintf("%s-full", CFG)
} else {
  k <- as.integer(JOB); stopifnot(k %in% 1:5)
  itr <- tr_t[tasks$fold[tr_t] != k]; iho <- tr_t[tasks$fold[tr_t] == k]
  tag <- sprintf("%s-f%d", CFG, k)
}
# --- the guard that matters: fitting respondents and scored respondents disjoint
stopifnot(length(intersect(tasks$Case[itr], tasks$Case[iho])) == 0)
cat(sprintf("[%s] fit on %d respondents / %d tasks; score %d respondents / %d tasks\n",
            tag, uniqueN(tasks$Case[itr]), length(itr),
            uniqueN(tasks$Case[iho]), length(iho)))

t0 <- Sys.time()
fit <- fit_hb(bd, itr, tag)

# held-out design + demographics only
ho_case <- tasks$Case[iho]
ucase_ho <- sort(unique(ho_case))
resp_idx <- rep(match(ho_case, ucase_ho), each = 4L)
Zho <- sweep(sweep(bd$Zraw[as.character(ucase_ho), , drop = FALSE], 2, fit$zmu, "-"),
             2, fit$zsd, "/")
Xho <- bd$X[rows_of(iho), , drop = FALSE]

P       <- predict_pop(fit, Xho, resp_idx, Zho, seed = 20260726)
P_seed2 <- predict_pop(fit, Xho, resp_idx, Zho, seed = 815491)
P_nod   <- predict_pop(fit, Xho, resp_idx, Zho, seed = 20260726, use_delta = FALSE)

if (JOB != "full") {
  cat(sprintf("[%s] held-out logloss: %.5f  (seed2 %.5f, no-demog %.5f)\n", tag,
              logloss(tasks$y[iho], P), logloss(tasks$y[iho], P_seed2),
              logloss(tasks$y[iho], P_nod)))
}

mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
saveRDS(list(task_idx = iho, P = clip_norm(P), P_seed2 = clip_norm(P_seed2),
             P_nodelta = clip_norm(P_nod), mins = mins, loglike = fit$loglike,
             burn = fit$burn, D = fit$D, R = R_MCMC, keep = KEEP, S = S_SIM,
             probmean = colMeans(fit$probdraw), cfg = CFG),
        file.path(OUTDIR, sprintf("pred_%s_%s.rds", CFG, JOB)))

# =============================================================================
# 5. Report material (full fit only)
# =============================================================================
if (JOB == "full") {
  keepn <- bd$keep; nvar <- fit$nvar
  cat("\n--- MCMC diagnostics ---\n")
  llk <- fit$loglike
  cat("loglike trace (kept draws, every 10th):\n")
  print(round(llk[seq(1, length(llk), by = max(length(llk) %/% 12, 1))], 1))
  post <- llk[(fit$burn + 1):length(llk)]
  cat(sprintf("post-burn-in loglike: mean %.1f  sd %.1f  first-half %.1f  second-half %.1f\n",
              mean(post), sd(post), mean(head(post, length(post) %/% 2)),
              mean(tail(post, length(post) %/% 2))))

  cat("\n--- POPULATION DISTRIBUTION ---\n")
  pm <- colMeans(fit$probdraw)
  cat("posterior mean mixture weights:", paste(sprintf("%.3f", pm), collapse = "  "), "\n")
  cat("effective # components (weight > 0.05):", sum(pm > 0.05), "of", CO$ncomp, "\n")

  mm <- momMix(fit$probdraw, fit$compdraw)
  names(mm$mu) <- keepn; names(mm$sd) <- keepn
  cat("\npopulation mean part-worth (top 10 positive):\n")
  print(round(sort(mm$mu, decreasing = TRUE)[1:10], 3))
  cat("\npopulation heterogeneity sd (top 10):\n")
  print(round(sort(mm$sd, decreasing = TRUE)[1:10], 3))
  cat(sprintf("\nasc4 (none-option): population mean %.3f, sd %.3f\n",
              mm$mu["asc4"], mm$sd["asc4"]))
  pl <- grep("^Price_L", keepn, value = TRUE)
  cat("\nPrice part-worths: population mean (sd) by level\n")
  print(round(cbind(mean = mm$mu[pl], sd = mm$sd[pl]), 3))

  if (CO$ncomp > 1) {
    MU <- sapply(seq_len(CO$ncomp), function(c)
      rowMeans(sapply(fit$compdraw, function(dd) dd[[c]]$mu)))
    rownames(MU) <- keepn
    ordc <- order(pm, decreasing = TRUE)
    MU <- MU[, ordc, drop = FALSE]
    cat("\ncomponent means, relabelled by descending weight",
        sprintf("(%s)\n", paste(sprintf("%.3f", pm[ordc]), collapse = " ")))
    cat("  asc4 by component:", paste(sprintf("%.3f", MU["asc4", ]), collapse = "  "), "\n")
    cat("  Price part-worths by component:\n")
    print(round(MU[pl, , drop = FALSE], 3))
    sp <- apply(MU, 1, function(r) max(r) - min(r))
    cat("  most distinguishing coefficients across components:\n")
    print(round(MU[names(sort(sp, decreasing = TRUE))[1:12], , drop = FALSE], 3))
  }

  cat("\n--- DEMOGRAPHIC CHANNEL (Delta) ---\n")
  Dm <- matrix(colMeans(fit$Deltadraw), nrow = fit$nz, ncol = nvar)
  rownames(Dm) <- colnames(bd$Zraw); colnames(Dm) <- keepn
  cat("|Delta| row norms -- which demographics move tastes most:\n")
  print(round(sort(sqrt(rowSums(Dm^2)), decreasing = TRUE)[1:8], 3))
  cat("effect on asc4 (propensity to take 'none'):\n")
  print(round(sort(Dm[, "asc4"])[c(1:4, (fit$nz-3):fit$nz)], 3))

  cat("\n--- INDIVIDUAL vs POPULATION (why HB cannot help a cold-start test set) ---\n")
  Xin <- bd$X[rows_of(tr_t), , drop = FALSE]
  ci  <- rep(match(tasks$Case[tr_t], fit$ucase), each = 4L)
  Pin <- probs_fixed(Xin, ci, fit$beta_mean)
  Zin <- sweep(sweep(bd$Zraw[as.character(fit$ucase), , drop = FALSE], 2, fit$zmu, "-"),
               2, fit$zsd, "/")
  Ppop <- predict_pop(fit, Xin, ci, Zin, seed = 20260726)
  cat(sprintf("in-sample logloss with each respondent's OWN posterior mean beta: %.5f\n",
              logloss(tasks$y[tr_t], Pin)))
  cat(sprintf("same rows, POPULATION-averaged (what a new respondent gets):        %.5f\n",
              logloss(tasks$y[tr_t], Ppop)))
  cat("the gap is the part of HB's fit that is structurally unavailable at test time.\n")

  saveRDS(list(cfg = CFG, name = NAME, ncomp = CO$ncomp, prior = CO$prior,
               probmean = pm, mu = mm$mu, sd = mm$sd, Delta = Dm, keep = keepn,
               loglike = llk, burn = fit$burn, R = R_MCMC, KEEP = KEEP,
               ll_indiv = logloss(tasks$y[tr_t], Pin),
               ll_pop = logloss(tasks$y[tr_t], Ppop)),
          file.path(OUTDIR, sprintf("report_%s.rds", CFG)))
}
cat(sprintf("[%s] total %.1f min\nOK\n", tag, mins))
