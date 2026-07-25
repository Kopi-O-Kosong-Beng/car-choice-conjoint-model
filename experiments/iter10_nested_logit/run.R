# =============================================================================
# ITERATION 10 — nested logit with the outside option in its own nest
# Artifacts: model/artifacts/{oof,test}_nlogit.rds
#
# HYPOTHESIS (stated before running).
# The MNL we use in production (mnl_pw, OOF 1.15686) imposes IIA: the relative
# odds of any two alternatives are unaffected by what else is on offer. That
# assumption is least credible at the boundary between the three real safety
# bundles and the "none of these" option. Bundles 1-3 plausibly share unobserved
# "product-ness" -- a respondent who is in the market at all finds all three more
# attractive than opting out -- which "none" by construction does not share.
# A two-level nested logit with nests {1,2,3} and {4} relaxes IIA exactly there
# and nowhere else, and buys a single extra parameter (lambda, mlogit's "iv").
#
# PREDICTION. If bundles substitute for each other more readily than for the
# outside option, lambda < 1, and the model should beat mnl_pw on logloss.
#   lambda = 1  -> nested logit collapses to MNL, IIA holds, no gain
#   lambda < 1  -> positive within-nest correlation (1 - lambda^2); the three
#                  bundles are closer substitutes for each other than for "none"
#   lambda > 1  -> negative implied correlation; NOT globally consistent with
#                  random utility maximisation, and evidence AGAINST this nest
#                  structure being the right relaxation of IIA
#
# The nest {4} is degenerate (one alternative), so its own lambda is not
# identified -- with a degenerate nest the log-sum collapses:
# lambda*log(exp(V4/lambda)) = V4 for any lambda. We therefore use
# un.nest.el = TRUE (one shared lambda), which is in effect estimating the
# elasticity of the {1,2,3} nest only. Using un.nest.el = FALSE would ask the
# optimiser to identify a parameter that has no effect on the likelihood.
#
# Utility spec is deliberately identical to model/02_mnl_partworth.R so that the
# ONLY difference from the incumbent is the nest structure (one change per
# experiment): part-worth dummies (reference = lowest level occurring on a REAL
# bundle), ASCs via "| 1", and the Price interactions.
#
# NOTE (cost me one wasted run): the incumbent mnl_pw does NOT use only
# Price_x_age/ppark/inc -- it also carries Price_x_seg2..6 and Price_x_reg2..5.
# A first version of this script omitted those nine terms, which made the OOF
# gap versus mnl_pw a measure of "nest structure PLUS nine dropped covariates"
# rather than of the nest structure. xvars below is now taken verbatim from
# model/02_mnl_partworth.R. As an internal control the script also refits the
# plain MNL on the same folds with the same spec and reports its OOF, which must
# reproduce mnl_pw's 1.15686; if it does not, the difference is not the nest.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter10_nested_logit/run.R
# Runtime: ~4 min (6 fits at ~15-25 s each, plus 6 MNL reference fits).
# =============================================================================
suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")
t_start <- Sys.time()

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")   # FIXED folds, grouped by Case. Never regenerate.

# ---- part-worth dummies (verbatim logic from model/02_mnl_partworth.R) -------
# Reference level = lowest level that actually occurs on a real bundle (alt 1-3).
# For Price that is 1, not 0: Price is 0 only on the all-zero "none" option, so a
# 0 reference makes the price dummies collinear with the none-constant.
pw_cols <- character(0)
for (a in ATTRS) {
  lv_real <- sort(unique(long[alt != 4][[a]]))
  ref <- lv_real[1]
  for (l in setdiff(lv_real, ref)) {
    nm <- sprintf("%s_L%s", a, l)
    long[, (nm) := as.numeric(get(a) == l)]
    pw_cols <- c(pw_cols, nm)
  }
  if (ref != 0) cat(sprintf("  note: %s reference level = %s (0 never occurs on a real bundle)\n", a, ref))
}
cat("part-worth parameters before rank check:", length(pw_cols), "\n")

# ---- identification: rank-revealing QR on the TASK-DEMEANED design -----------
cand <- c("asc2", "asc3", "asc4", pw_cols)
X  <- as.matrix(long[, ..cand])
Xc <- X - as.matrix(long[, lapply(.SD, mean), by = No, .SDcols = cand][
  match(long$No, unique(long$No)), ..cand])
qrx <- qr(Xc, tol = 1e-7)
keep <- cand[sort(qrx$pivot[seq_len(qrx$rank)])]
dropped <- setdiff(cand, keep)
if (length(dropped)) cat("dropped as unidentified:", paste(dropped, collapse = ", "), "\n")
pw_cols <- setdiff(keep, c("asc2", "asc3", "asc4"))
cat("part-worth parameters:", length(pw_cols), "\n")

xvars <- c(pw_cols, "Price_x_age", "Price_x_ppark", "Price_x_inc",
           grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))
fml   <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 1"))
cat("utility terms:", length(xvars), "(matches model/02_mnl_partworth.R)\n")
NESTS <- list(buy = c("1", "2", "3"), none = c("4"))

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
setorder(trl, No, alt)
tel <- long[is_test == TRUE]; setorder(tel, No, alt)

# ---- guard: recompute the nested-logit probabilities by hand -----------------
# mlogit >= 2.0 returns a matrix from predict(); we must also be sure the nest
# structure is actually APPLIED to newdata rather than silently ignored (which
# would make this experiment a relabelled copy of mnl_pw). Recompute
#   P(j in nest l) = e^{V_j/L} (sum_{k in l} e^{V_k/L})^{L-1} / sum_m (...)^{L}
# from the raw coefficients and assert agreement.
# d must be sorted by (No, alt): each task is 4 consecutive rows.
lin_util <- function(co, d) {
  b <- co[setdiff(names(co), "iv")]
  icept <- grep("^\\(Intercept\\)", names(b), value = TRUE)
  bx <- b[setdiff(names(b), icept)]
  V  <- as.vector(as.matrix(as.data.frame(d)[, names(bx)]) %*% bx)
  for (nm in icept) {
    a <- as.integer(sub(".*:", "", nm))
    V <- V + as.numeric(d$alt == a) * b[nm]
  }
  matrix(V, ncol = 4, byrow = TRUE)
}
# Nested-logit probabilities. Shifting every V in a task by a constant leaves
# these unchanged (the exp(c/lambda)^lambda in the nest term cancels the exp(c)
# in the none term), so subtracting the row max is a safe stabilisation.
nl_prob <- function(Vm, lam) {
  Vm   <- Vm - apply(Vm, 1, max)
  Ebuy <- exp(Vm[, 1:3] / lam)
  nb   <- Ebuy * (rowSums(Ebuy)^(lam - 1))
  nn   <- exp(Vm[, 4])                        # degenerate nest {4}
  den  <- rowSums(nb) + nn
  cbind(nb / den, nn / den)
}
mnl_prob <- function(Vm) {
  E <- exp(Vm - apply(Vm, 1, max)); E / rowSums(E)
}

# We evaluate probabilities from the closed form above rather than through
# predict.mlogit, because predict.mlogit RE-INVOKES mlogit() on the newdata.
# Price_x_seg6 is identically zero on the test set (segment 6 does not occur
# among the 263 test respondents), which makes that internal re-fit exactly
# singular: "Lapack dgesv: U[81,81] = 0", and 81 = 3 intercepts + xvar #78 =
# Price_x_seg6. The closed form has no such problem. On every fold, where
# predict.mlogit DOES work, we assert the two agree to 1e-8.
fit_predict <- function(dtr, dva, check_predict = TRUE) {
  dtr_x <- dfidx(as.data.frame(dtr), idx = c("No", "alt"), choice = "chosen")
  mnl <- mlogit(fml, data = dtr_x)                       # reference fit, same spec
  # Seed the nested fit at the MNL solution with lambda = 1 -- at lambda = 1 the
  # nested logit IS the MNL, so this is the canonical warm start. A multi-start
  # check below confirms the optimum is not an artefact of starting there.
  fit <- mlogit(fml, data = dtr_x, nests = NESTS, un.nest.el = TRUE,
                start = c(coef(mnl), 1), method = "bfgs")
  dva <- copy(dva); setorder(dva, No, alt)
  stopifnot(all(dva[, .N, by = No]$N == 4))
  lam <- as.numeric(coef(fit)["iv"])
  Vm  <- lin_util(coef(fit), dva)
  P   <- nl_prob(Vm, lam)
  P0  <- mnl_prob(lin_util(coef(mnl), dva))              # matched-spec MNL control
  nos <- sort(unique(dva$No))
  stopifnot(nrow(P) == length(nos), max(abs(rowSums(P) - 1)) < 1e-10)
  # the nest must actually BEND the probabilities, else this is a relabelled MNL
  gap <- max(abs(P - mnl_prob(Vm)))
  if (gap < 1e-6) stop("nest structure had no effect -- P equals plain MNL")
  if (check_predict) {
    dva2 <- copy(dva); dva2[, chosen := FALSE]; dva2[alt == 1, chosen := TRUE]
    dva_x <- dfidx(as.data.frame(dva2), idx = c("No", "alt"), choice = "chosen")
    Pp <- predict(fit, newdata = dva_x)
    if (!is.matrix(Pp) || ncol(Pp) != 4) stop("predict() did not return an n x 4 matrix")
    stopifnot(identical(as.integer(rownames(Pp)), as.integer(nos)))
    stopifnot(max(abs(P - Pp)) < 1e-8)                   # closed form == mlogit
    stopifnot(max(abs(P0 - predict(mnl, newdata = dva_x))) < 1e-8)
  }
  list(P = P, P0 = P0, nos = nos, fit = fit, gap_vs_mnl = gap,
       lam = lam, lam_se = summary(fit)$CoefTable["iv", "Std. Error"],
       ll = as.numeric(logLik(fit)), ll_mnl = as.numeric(logLik(mnl)))
}

# ---- out-of-fold ------------------------------------------------------------
oof <- data.table(No = sort(unique(trl$No)), p1 = NA_real_, p2 = NA_real_,
                  p3 = NA_real_, p4 = NA_real_)
ctl <- copy(oof)                                     # matched-spec MNL control
lam_tab <- data.table(fold = 1:5, lambda = NA_real_, se = NA_real_,
                      ll_nl = NA_real_, ll_mnl = NA_real_)
for (k in 1:5) {
  r <- fit_predict(trl[fold != k], trl[fold == k])   # trained ONLY on folds != k
  oof[match(r$nos, No), `:=`(p1 = r$P[, 1], p2 = r$P[, 2], p3 = r$P[, 3], p4 = r$P[, 4])]
  ctl[match(r$nos, No), `:=`(p1 = r$P0[, 1], p2 = r$P0[, 2], p3 = r$P0[, 3], p4 = r$P0[, 4])]
  lam_tab[fold == k, `:=`(lambda = r$lam, se = r$lam_se, ll_nl = r$ll, ll_mnl = r$ll_mnl)]
  cat(sprintf("fold %d done  lambda = %.4f (se %.4f)  LR vs MNL = %.2f\n",
              k, r$lam, r$lam_se, 2 * (r$ll - r$ll_mnl)))
}
stopifnot(!anyNA(oof), !anyNA(ctl))

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
oof_ll <- logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)]))
ctl_ll <- logloss(ytr, as.matrix(ctl[, .(p1, p2, p3, p4)]))
cat(sprintf("\n>>> matched-spec MNL control OOF: %.5f   (must reproduce mnl_pw = 1.15686)\n", ctl_ll))
cat(sprintf(">>> nested logit OOF logloss:     %.5f   (blend: 1.13556)\n", oof_ll))
cat(sprintf(">>> attributable to the NEST alone: %+.5f\n", oof_ll - ctl_ll))
saveRDS(ctl, "experiments/iter10_nested_logit/oof_mnl_control.rds")
stopifnot(nrow(oof) == 21565)
saveRDS(oof, "model/artifacts/oof_nlogit.rds")

# ---- refit on ALL training data, predict test -------------------------------
rf <- fit_predict(trl, tel, check_predict = FALSE)   # see note above: seg6 is 0 on test
test <- data.table(No = rf$nos, p1 = rf$P[, 1], p2 = rf$P[, 2],
                   p3 = rf$P[, 3], p4 = rf$P[, 4])
stopifnot(nrow(test) == 4997)
saveRDS(test, "model/artifacts/test_nlogit.rds")

# ---- report material --------------------------------------------------------
lam <- rf$lam; se <- rf$lam_se
LR  <- 2 * (rf$ll - rf$ll_mnl)
cat("\n================ NEST ELASTICITY (full-data refit) ================\n")
cat(sprintf("lambda (mlogit 'iv') = %.4f   se = %.4f\n", lam, se))
cat(sprintf("Wald test of H0: lambda = 1  ->  z = %.3f,  p = %.4f\n",
            (lam - 1) / se, 2 * pnorm(-abs((lam - 1) / se))))
cat(sprintf("LR test vs MNL (1 df): LR = %.3f, p = %.4f\n", LR, pchisq(LR, 1, lower.tail = FALSE)))
cat(sprintf("implied within-nest error correlation 1 - lambda^2 = %.4f\n", 1 - lam^2))
cat(sprintf("RUM-consistent for all parameter values (needs 0 < lambda <= 1): %s\n",
            if (lam <= 1) "YES" else "NO"))
cat("per-fold lambda:\n"); print(lam_tab)

# multi-start robustness: is lambda an artefact of warm-starting at lambda = 1?
trl_x <- dfidx(as.data.frame(trl), idx = c("No", "alt"), choice = "chosen")
mnl_full <- mlogit(fml, data = trl_x)
cat("\nmulti-start check on the full-data refit:\n")
for (s in c(0.5, 0.8, 1.3)) {
  z <- try(mlogit(fml, data = trl_x, nests = NESTS, un.nest.el = TRUE,
                  start = c(coef(mnl_full), s), method = "bfgs"), silent = TRUE)
  cat(sprintf("  start lambda = %.1f -> %s\n", s,
      if (inherits(z, "try-error")) "FAILED"
      else sprintf("lambda = %.4f, ll = %.3f", coef(z)["iv"], as.numeric(logLik(z)))))
}
saveRDS(list(lambda = lam, se = se, LR = LR, per_fold = lam_tab,
             coefs = coef(rf$fit), oof_logloss = oof_ll, control_oof_logloss = ctl_ll),
        "experiments/iter10_nested_logit/nlogit_summary.rds")
cat("\nelapsed:", round(difftime(Sys.time(), t_start, units = "mins"), 1), "min\nOK\n")
