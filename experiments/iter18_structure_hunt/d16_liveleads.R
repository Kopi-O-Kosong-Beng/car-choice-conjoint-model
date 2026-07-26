# =============================================================================
# Diagnostic 16 (follow-up, written after re-reading d1..d15).
#
# WHY: four quantities were PRINTED by d3/d4/d12/d14/d15 but never converted into
# an honest out-of-fold logloss. Without that number you cannot say whether the
# lead is alive. This script supplies the missing number for each. It writes NO
# artifact and builds NO model -- every correction below is a 1-3 parameter
# reweighting of an EXISTING oof_*.rds, with the parameters fitted on folds != k
# and scored on fold k, on the fixed seed-42 respondent-grouped folds.
#
# HYPOTHESES (stated before running):
#   A  d3-D printed per-quartile temperatures 0.987/0.973/0.938/0.935 by mean
#      price. If task-level scale really varies with design dispersion, a
#      dispersion-conditional temperature should beat a single global one.
#   B  d15 found the incumbent's residual Price_c x Task coefficient is
#      -0.000200, clustered z = -2.50 -- statistically real. The run.R header
#      asserts it is "worth ~3e-5 of logloss" but NO script ever computed that.
#      Compute it.
#   C  d12 found lag_minp z = -2.74 / lag_bestq z = -2.58 against a partly-dirty
#      placebo (lead_minp z = -1.52). Convert to honest logloss, and run the same
#      pipeline on the placebo as a control.
#   D  d4 found R2(demographics -> OBSERVED none rate) = 0.0968 but
#      R2(demographics -> PREDICTED none rate) = 0.5301. The incumbent spreads
#      respondents apart along the demographic axis ~5.5x more (in R2) than
#      reality warrants. Demographics ARE available for test respondents, so
#      unlike d6's respondent shrinkage this correction is actually applicable.
#      Shrink the demographic component and measure.
#
# LEAK STATEMENT (this repo has been burned twice): every fitted quantity below
# comes from respondents in folds != k only. For a held-out respondent the
# correction uses (i) their own OOF predictions -- predictions, not labels -- and
# (ii) their demographics, which are observed for test respondents too. No label
# belonging to a fold-k respondent enters any coefficient applied to fold k.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
setorder(long, No, alt)
tr <- long[is_test == FALSE]; setorder(tr, No, alt)
tk <- unique(tr[, .(No, Case, Task, y)]); setorder(tk, No)
tk <- merge(tk, folds[, .(No, fold)], by = "No"); setorder(tk, No)
y <- tk$y; fmap <- tk$fold; N <- nrow(tk)
stopifnot(N == 21565L, !anyNA(fmap))

getP <- function(nm) { d <- readRDS(sprintf("model/artifacts/oof_%s.rds", nm)); setorder(d, No)
  stopifnot(identical(d$No, tk$No)); as.matrix(d[, .(p1,p2,p3,p4)]) }
blendP <- function() {
  b <- readRDS("model/artifacts/blend.rds"); memb <- b$members; M <- length(memb)
  OOF <- lapply(memb, getP)
  th <- b$par; w <- exp(th[1:M]); w <- w/sum(w); Tt <- exp(th[M+1]); eA <- plogis(th[M+2])*0.10
  L <- Reduce(`+`, Map(function(wi,P) wi*log(pmax(P,1e-12)), w, OOF))/Tt
  P <- exp(L - apply(L,1,max)); P <- P/rowSums(P); (1-eA)*P + eA*0.25
}
MODELS <- list(xgb_lw2 = getP("xgb_lw2"), BLEND = blendP())

# generic honest evaluator: fit par on folds != k by minimising logloss there,
# apply to fold k. adj(P, par) must return a full 21565x4 probability matrix.
honest <- function(P, adj, npar, lo, hi, start) {
  out <- numeric(5); pars <- matrix(NA_real_, 5, npar)
  for (k in 1:5) {
    ii <- which(fmap != k); jj <- which(fmap == k)
    f <- function(th) logloss(y[ii], adj(P, th)[ii, , drop = FALSE])
    o <- if (npar == 1) optimize(function(t) f(t), c(lo, hi))
         else optim(start, f, method = "Nelder-Mead", control = list(maxit = 2000))
    th <- if (npar == 1) o$minimum else o$par
    pars[k, ] <- th
    out[k] <- logloss(y[jj], adj(P, th)[jj, , drop = FALSE])
  }
  list(ll = mean(out), pars = pars)
}
base_by_fold <- function(P) mean(sapply(1:5, function(k) logloss(y[fmap==k], P[fmap==k,,drop=FALSE])))

# =====================================================================
cat("############ A. dispersion-conditional temperature (from d3-D) ############\n")
real <- tr[alt <= 3]
dd <- real[, .(spread_price = max(Price)-min(Price), mean_price = mean(Price),
               spread_lvl = max(lvlsum)-min(lvlsum)), by = No]
setorder(dd, No); stopifnot(identical(dd$No, tk$No))

temp_adj <- function(P, lam_vec, grp) {
  L <- log(pmax(P, 1e-12)) * lam_vec[grp]
  Q <- exp(L - apply(L, 1, max)); Q/rowSums(Q)
}
for (v in c("mean_price", "spread_price", "spread_lvl")) {
  x <- dd[[v]]
  for (nm in names(MODELS)) {
    P <- MODELS[[nm]]; b <- base_by_fold(P)
    # global temperature, fitted out-of-fold
    g <- honest(P, function(P, th) temp_adj(P, rep(th, 1), rep(1L, N)), 1, 0.5, 1.5, NULL)
    # 4 quartile temperatures; cut points from the TRAINING folds of each k
    q4 <- numeric(5)
    for (k in 1:5) {
      ii <- which(fmap != k); jj <- which(fmap == k)
      qs <- quantile(x[ii], c(.25,.5,.75)); grp <- as.integer(cut(x, c(-Inf, qs, Inf)))
      f <- function(th) logloss(y[ii], temp_adj(P, th, grp)[ii,,drop=FALSE])
      o <- optim(rep(1,4), f, method="Nelder-Mead", control=list(maxit=2000))
      q4[k] <- logloss(y[jj], temp_adj(P, o$par, grp)[jj,,drop=FALSE])
    }
    cat(sprintf("  %-12s by %-13s base %.5f | global T %.5f (%+.5f) | 4 quartile T %.5f (%+.5f)  EXTRA from conditioning %+.5f\n",
                nm, v, b, g$ll, b-g$ll, mean(q4), b-mean(q4), g$ll-mean(q4)))
  }
}

# =====================================================================
cat("\n############ B. residual Price_c x Task drift (from d14/d15) ############\n")
PC <- matrix(tr$Price_c, ncol = 4, byrow = TRUE)
TC <- tk$Task - 10
b_adj <- function(P, w) { L <- log(pmax(P,1e-12)) + w * PC * TC
  Q <- exp(L - apply(L,1,max)); Q/rowSums(Q) }
for (nm in names(MODELS)) {
  P <- MODELS[[nm]]; b <- base_by_fold(P)
  h <- honest(P, b_adj, 1, -0.05, 0.05, NULL)
  cat(sprintf("  %-12s base %.5f -> %.5f   delta %+.6f   w per fold: %s\n",
              nm, b, h$ll, b - h$ll, paste(sprintf("%+.4f", h$pars[,1]), collapse=" ")))
}

# =====================================================================
cat("\n############ C. sequential anchoring, with placebo (from d12) ############\n")
NP <- setdiff(ATTRS, "Price"); MAXL <- sapply(ATTRS, function(a) max(long[[a]]))
Zn <- sapply(NP, function(a) ifelse(real[[a]] > 0, (real[[a]]-1)/pmax(MAXL[a]-1,1), 0))
r2 <- copy(real); r2[, nsum := rowSums(Zn)]
q <- r2[, .(minp = min(Price), meanp = mean(Price), bestq = max(nsum)), by = .(Case, Task, No)]
setorder(q, Case, Task)
q[, `:=`(lag_minp = shift(minp), lag_meanp = shift(meanp), lag_bestq = shift(bestq),
         lead_minp = shift(minp,-1), lead_meanp = shift(meanp,-1), lead_bestq = shift(bestq,-1)), by = Case]
setorder(q, No); stopifnot(identical(q$No, tk$No))
mk <- function(cols) { Z <- as.matrix(q[, ..cols]); Z[is.na(Z)] <- 0
  sweep(Z, 2, colMeans(Z)) }
ZL <- mk(c("lag_minp","lag_meanp","lag_bestq"))
ZF <- mk(c("lead_minp","lead_meanp","lead_bestq"))
anch_adj <- function(Z) function(P, th) {
  L <- log(pmax(P,1e-12)); L[,4] <- L[,4] + as.numeric(Z %*% th)
  Q <- exp(L - apply(L,1,max)); Q/rowSums(Q) }
for (nm in names(MODELS)) {
  P <- MODELS[[nm]]; b <- base_by_fold(P)
  hL <- honest(P, anch_adj(ZL), 3, NA, NA, rep(0,3))
  hF <- honest(P, anch_adj(ZF), 3, NA, NA, rep(0,3))
  cat(sprintf("  %-12s base %.5f | PREVIOUS task %.5f (%+.6f) | PLACEBO next task %.5f (%+.6f)\n",
              nm, b, hL$ll, b-hL$ll, hF$ll, b-hF$ll))
}

# =====================================================================
cat("\n############ D. demographic over-dispersion of the none rate (from d4) ############\n")
dem <- unique(tr[, .(Case, incomea, agea, genderind, educind, milesa, segmentind,
                     regionind, Urbind, nighta, pparkind, yearind)])
dem[, loginc := log(pmax(incomea, 1))]
for (v in c("genderind","segmentind","regionind","Urbind","pparkind")) dem[[v]] <- factor(dem[[v]])
FML <- ~ loginc + agea + genderind + educind + milesa + segmentind +
  regionind + Urbind + nighta + pparkind + yearind
MM <- model.matrix(FML, dem)                     # design only, no labels
cmap <- match(tk$Case, dem$Case)
cat(sprintf("  respondents %d, demographic design matrix %d cols\n", nrow(dem), ncol(MM)))

for (nm in names(MODELS)) {
  P <- MODELS[[nm]]; b <- base_by_fold(P)
  nobs_c <- tapply(as.numeric(y == 4), tk$Case, mean)[as.character(dem$Case)]
  nhat_c <- tapply(P[,4],               tk$Case, mean)[as.character(dem$Case)]
  cat(sprintf("\n  -- %s --\n", nm))
  cat(sprintf("     sd(observed none rate) %.4f  sd(predicted) %.4f\n", sd(nobs_c), sd(nhat_c)))
  r_o <- summary(lm(nobs_c ~ MM - 1))$r.squared; r_p <- summary(lm(nhat_c ~ MM - 1))$r.squared
  cat(sprintf("     R2 demog->observed %.4f   R2 demog->predicted %.4f\n", r_o, r_p))
  cat(sprintf("     sd of demog-fitted observed %.4f vs demog-fitted predicted %.4f  (ratio %.2f)\n",
              sd(fitted(lm(nobs_c ~ MM - 1))), sd(fitted(lm(nhat_c ~ MM - 1))),
              sd(fitted(lm(nhat_c ~ MM - 1)))/sd(fitted(lm(nobs_c ~ MM - 1)))))
  # honest 2-parameter correction, coefficients from folds != k only
  out <- numeric(5); pp <- matrix(NA_real_, 5, 2)
  for (k in 1:5) {
    ii <- which(fmap != k); jj <- which(fmap == k)
    trc <- unique(tk$Case[ii]); idx <- which(dem$Case %in% trc)
    bo <- qr.coef(qr(MM[idx,,drop=FALSE]), nobs_c[idx]); bo[is.na(bo)] <- 0
    bp <- qr.coef(qr(MM[idx,,drop=FALSE]), nhat_c[idx]); bp[is.na(bp)] <- 0
    oh <- as.numeric(MM %*% bo); ph <- as.numeric(MM %*% bp)
    oh <- oh - mean(oh[idx]); ph <- ph - mean(ph[idx])
    Zo <- cbind(oh[cmap], ph[cmap])
    adj <- function(P, th) { L <- log(pmax(P,1e-12)); L[,4] <- L[,4] + as.numeric(Zo %*% th)
      Q <- exp(L - apply(L,1,max)); Q/rowSums(Q) }
    f <- function(th) logloss(y[ii], adj(P, th)[ii,,drop=FALSE])
    o <- optim(c(0,0), f, method="Nelder-Mead", control=list(maxit=2000))
    pp[k,] <- o$par; out[k] <- logloss(y[jj], adj(P, o$par)[jj,,drop=FALSE])
  }
  cat(sprintf("     HONEST demographic re-dispersion: %.5f vs base %.5f   delta %+.6f\n",
              mean(out), b, b - mean(out)))
  cat(sprintf("     coefficients (a on demog-observed, b on demog-predicted) per fold:\n"))
  for (k in 1:5) cat(sprintf("       fold %d  a %+.3f  b %+.3f\n", k, pp[k,1], pp[k,2]))
}
cat("\nNOTE: no artifact written; nothing in model/ touched.\n")
