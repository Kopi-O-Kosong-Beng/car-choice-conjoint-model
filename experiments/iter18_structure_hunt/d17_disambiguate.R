# =============================================================================
# Diagnostic 17: pin down the ONE lead that d16 showed alive, and repair the ONE
# measurement d16 got wrong. Still diagnostic-only: no artifact, no model.
#
# 1. d16-B gave the BLEND +0.00130 from a single term w * Price_c * (Task-10).
#    But Price_c on the none row is a large negative task constant (mean -4.84),
#    so that single term is CONFOUNDED: it simultaneously (i) tilts price
#    sensitivity by task position and (ii) puts a linear task trend on the
#    none-utility. d3 measured a real none-rate drift 0.237 -> 0.337. Which of
#    the two is paying? Decompose:
#       B1  none-drift only     : L[,4] += w1 * (Task-10)
#       B2  price tilt only     : L      += w2 * Price_c * (Task-10), with the
#                                 alt-4 column of Price_c zeroed out
#       B3  both, fitted jointly
#    Report a respondent-clustered paired z for each (same method as compare.R).
#
# 2. d16-D fitted the weight (a,b) on folds where the demographic regressions oh
#    and ph were themselves IN-SAMPLE. That inflates their apparent value, the
#    optimiser cranks a to ~+6, and it collapses out-of-fold (-0.003). That is a
#    measurement artifact, not evidence. Redo with a NESTED construction: inside
#    folds != k, oh/ph for an inner group come from a regression fitted on the
#    other inner groups, so oh/ph are out-of-sample everywhere the weight is fit.
#    Also fix the R2 print (d16 used lm(y ~ MM - 1) with an intercept column
#    inside MM, which reports the UNCENTRED R2 -- that is why d16 printed 0.5775
#    where d4 correctly printed 0.0968).
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
setorder(long, No, alt)
tr <- long[is_test == FALSE]; setorder(tr, No, alt)
tk <- unique(tr[, .(No, Case, Task, y)]); setorder(tk, No)
tk <- merge(tk, folds[, .(No, fold)], by = "No"); setorder(tk, No)
y <- tk$y; fmap <- tk$fold; N <- nrow(tk); stopifnot(N == 21565L)

getP <- function(nm) { d <- readRDS(sprintf("model/artifacts/oof_%s.rds", nm)); setorder(d, No)
  stopifnot(identical(d$No, tk$No)); as.matrix(d[, .(p1,p2,p3,p4)]) }
blendP <- function() {
  b <- readRDS("model/artifacts/blend.rds"); memb <- b$members; M <- length(memb)
  OOF <- lapply(memb, getP); th <- b$par
  w <- exp(th[1:M]); w <- w/sum(w); Tt <- exp(th[M+1]); eA <- plogis(th[M+2])*0.10
  L <- Reduce(`+`, Map(function(wi,P) wi*log(pmax(P,1e-12)), w, OOF))/Tt
  P <- exp(L - apply(L,1,max)); P <- P/rowSums(P); (1-eA)*P + eA*0.25
}
MODELS <- list(mnl_pw = getP("mnl_pw"), lcmnl3 = getP("lcmnl3"),
               xgb_lw2 = getP("xgb_lw2"), xgb_mono = getP("xgb_mono"), BLEND = blendP())

rowloss <- function(P) -log(pmax(P[cbind(1:N, y)], 1e-15))
clz <- function(la, lb) {                       # positive => corrected is better
  cl <- data.table(Case = tk$Case, d = la - lb)[, .(dm = mean(d)), by = Case]
  c(est = mean(cl$dm), se = sd(cl$dm)/sqrt(nrow(cl)), z = mean(cl$dm)/(sd(cl$dm)/sqrt(nrow(cl))))
}

cat("############ 0. does each model track the none-rate drift over Task? ############\n")
obs <- tapply(as.numeric(y == 4), tk$Task, mean)
cat(sprintf("  %-9s task1 %.3f  task19 %.3f  slope/task %+.5f\n", "OBSERVED",
            obs[1], obs[19], coef(lm(obs ~ I(1:19)))[2]))
for (nm in names(MODELS)) {
  m <- tapply(MODELS[[nm]][,4], tk$Task, mean)
  cat(sprintf("  %-9s task1 %.3f  task19 %.3f  slope/task %+.5f\n", nm, m[1], m[19],
              coef(lm(m ~ I(1:19)))[2]))
}

# =====================================================================
cat("\n############ 1. decomposing d16-B ############\n")
PC  <- matrix(tr$Price_c, ncol = 4, byrow = TRUE)
PC0 <- PC; PC0[, 4] <- 0                        # price tilt on REAL bundles only
TC  <- tk$Task - 10
E4  <- cbind(0, 0, 0, 1)

adjB1 <- function(P, th) { L <- log(pmax(P,1e-12)) + th[1] * (TC %o% c(0,0,0,1))
  Q <- exp(L - apply(L,1,max)); Q/rowSums(Q) }
adjB2 <- function(P, th) { L <- log(pmax(P,1e-12)) + th[1] * PC0 * TC
  Q <- exp(L - apply(L,1,max)); Q/rowSums(Q) }
adjB3 <- function(P, th) { L <- log(pmax(P,1e-12)) + th[1]*(TC %o% c(0,0,0,1)) + th[2]*PC0*TC
  Q <- exp(L - apply(L,1,max)); Q/rowSums(Q) }

run_honest <- function(P, adj, npar) {          # fit on folds != k, score fold k
  Pc <- matrix(NA_real_, N, 4); pars <- matrix(NA_real_, 5, npar)
  for (k in 1:5) {
    ii <- which(fmap != k); jj <- which(fmap == k)
    f <- function(th) logloss(y[ii], adj(P, th)[ii,,drop=FALSE])
    o <- optim(rep(0, max(npar,2)), function(th) f(th[1:npar]), method="Nelder-Mead",
               control=list(maxit=3000))
    th <- o$par[1:npar]; pars[k,] <- th
    Pc[jj,] <- adj(P, th)[jj,,drop=FALSE]
  }
  list(P = Pc, pars = pars)
}
for (nm in names(MODELS)) {
  P <- MODELS[[nm]]; lb <- rowloss(P)
  cat(sprintf("\n  -- %s (base %.5f) --\n", nm, mean(lb)))
  for (lab in c("B1 none-drift only", "B2 price-tilt only (alt4 zeroed)", "B3 both")) {
    ad <- switch(lab, "B1 none-drift only"=adjB1, "B2 price-tilt only (alt4 zeroed)"=adjB2, adjB3)
    np <- if (lab == "B3 both") 2 else 1
    r  <- run_honest(P, ad, np)
    la <- rowloss(r$P); s <- clz(lb, la)
    cat(sprintf("     %-33s %.5f  delta %+.5f  clustered SE %.5f  z %+.2f   par %s\n",
                lab, mean(la), s["est"], s["se"], s["z"],
                paste(sprintf("%+.4f", colMeans(r$pars)), collapse=" ")))
  }
}

# =====================================================================
cat("\n############ 2. d16-D redone NESTED (and with the R2 print fixed) ############\n")
dem <- unique(tr[, .(Case, incomea, agea, genderind, educind, milesa, segmentind,
                     regionind, Urbind, nighta, pparkind, yearind)])
dem[, loginc := log(pmax(incomea, 1))]
for (v in c("genderind","segmentind","regionind","Urbind","pparkind")) dem[[v]] <- factor(dem[[v]])
MM <- model.matrix(~ loginc + agea + genderind + educind + milesa + segmentind +
                     regionind + Urbind + nighta + pparkind + yearind, dem)
cmap <- match(tk$Case, dem$Case)
nobs_c <- as.numeric(tapply(as.numeric(y == 4), tk$Case, mean)[as.character(dem$Case)])
set.seed(42)
inner <- sample(rep_len(1:4, nrow(dem)))        # inner groups over RESPONDENTS

fitpred <- function(idx, v) { b <- qr.coef(qr(MM[idx,,drop=FALSE]), v[idx]); b[is.na(b)] <- 0
  as.numeric(MM %*% b) }

for (nm in c("xgb_lw2", "BLEND")) {
  P <- MODELS[[nm]]; lb <- rowloss(P)
  nhat_c <- as.numeric(tapply(P[,4], tk$Case, mean)[as.character(dem$Case)])
  cat(sprintf("\n  -- %s --\n", nm))
  cat(sprintf("     CENTRED R2 demog->observed none rate  %.4f\n", summary(lm(nobs_c ~ MM[,-1]))$r.squared))
  cat(sprintf("     CENTRED R2 demog->predicted none rate %.4f   (d4's 0.0968 / 0.5301 reproduced)\n",
              summary(lm(nhat_c ~ MM[,-1]))$r.squared))
  so <- sd(fitted(lm(nobs_c ~ MM[,-1]))); sp <- sd(fitted(lm(nhat_c ~ MM[,-1])))
  vres <- var(nobs_c)*(1 - summary(lm(nobs_c ~ MM[,-1]))$r.squared)
  so_adj <- sqrt(max(so^2 - (ncol(MM)-1)/nrow(dem)*vres, 0))
  cat(sprintf("     sd of demographic COMPONENT: observed %.4f (overfit-adjusted %.4f) vs model %.4f\n",
              so, so_adj, sp))
  cat(sprintf("     -> model uses %.0f%% of the demographic spread that is really there\n", 100*sp/so_adj))
  Pc <- matrix(NA_real_, N, 4); pars <- matrix(NA_real_, 5, 2)
  for (k in 1:5) {
    trc <- which(dem$Case %in% unique(tk$Case[fmap != k]))
    oh <- numeric(nrow(dem)); ph <- numeric(nrow(dem))
    for (m in 1:4) {                            # NESTED: oh/ph out-of-sample on train folds
      fit_idx <- trc[inner[trc] != m]; tgt <- trc[inner[trc] == m]
      oh[tgt] <- fitpred(fit_idx, nobs_c)[tgt]; ph[tgt] <- fitpred(fit_idx, nhat_c)[tgt]
    }
    tst <- setdiff(seq_len(nrow(dem)), trc)     # fold-k respondents: fit on all of folds != k
    oh[tst] <- fitpred(trc, nobs_c)[tst]; ph[tst] <- fitpred(trc, nhat_c)[tst]
    oh <- oh - mean(oh[trc]); ph <- ph - mean(ph[trc])
    Z <- cbind(oh[cmap], ph[cmap])
    adj <- function(th) { L <- log(pmax(P,1e-12)); L[,4] <- L[,4] + as.numeric(Z %*% th)
      Q <- exp(L - apply(L,1,max)); Q/rowSums(Q) }
    ii <- which(fmap != k); jj <- which(fmap == k)
    o <- optim(c(0,0), function(th) logloss(y[ii], adj(th)[ii,,drop=FALSE]),
               method="Nelder-Mead", control=list(maxit=3000))
    pars[k,] <- o$par; Pc[jj,] <- adj(o$par)[jj,,drop=FALSE]
  }
  la <- rowloss(Pc); s <- clz(lb, la)
  cat(sprintf("     NESTED honest: %.5f vs base %.5f  delta %+.5f  clustered SE %.5f  z %+.2f\n",
              mean(la), mean(lb), s["est"], s["se"], s["z"]))
  cat(sprintf("     mean coefficients  a(on demog-observed) %+.3f   b(on demog-predicted) %+.3f\n",
              mean(pars[,1]), mean(pars[,2])))
}
cat("\nNOTE: no artifact written; nothing in model/ touched.\n")
