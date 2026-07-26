# Diagnostic 6: respondent-level over-dispersion. The model spreads respondents'
# predicted alternative-rates further apart than reality warrants (slope 0.74-0.94).
# Test: shrink each respondent's deviation from the population mean, in log space.
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); folds <- readRDS("model/artifacts/folds.rds")
ymap <- unique(long[is_test==FALSE, .(No, y, Case)]); setorder(ymap, No)
y <- ymap$y; cs <- ymap$Case; fmap <- folds[order(No), fold]

blend_oof <- function() {
  b <- readRDS("model/artifacts/blend.rds"); memb <- b$members; M <- length(memb)
  OOF <- lapply(memb, function(m){d<-readRDS(sprintf("model/artifacts/oof_%s.rds",m));setorder(d,No);as.matrix(d[,.(p1,p2,p3,p4)])})
  th <- b$par; w <- exp(th[1:M]); w <- w/sum(w); Tt <- exp(th[M+1]); eA <- plogis(th[M+2])*0.10
  L <- Reduce(`+`, Map(function(wi,P) wi*log(pmax(P,1e-12)), w, OOF)); L <- L/Tt
  P <- exp(L - apply(L,1,max)); P <- P/rowSums(P); (1-eA)*P + eA*0.25
}

adjust <- function(P, cs, cvec, Tt) {
  L <- log(pmax(P, 1e-12))
  mb <- rowsum(L, cs) / as.vector(table(cs))            # per-respondent mean log-prob per alt
  gb <- colMeans(mb)
  dev <- sweep(mb, 2, gb)                                # respondent deviation
  A <- dev[match(cs, as.integer(rownames(mb))), , drop=FALSE]
  L2 <- (L - sweep(A, 2, cvec, `*`)) / Tt                # shrink deviation by (1-c)
  Q <- exp(L2 - apply(L2,1,max)); Q/rowSums(Q)
}

fit_adj <- function(P, rows, per_alt = TRUE) {
  f <- function(th) {
    cv <- if (per_alt) plogis(th[1:4]) else rep(plogis(th[1]),4)
    Tt <- exp(th[length(th)])
    Q <- adjust(P, cs, cv, Tt)
    logloss(y[rows], Q[rows,,drop=FALSE])
  }
  n <- if (per_alt) 5 else 2
  optim(c(rep(-3,n-1),0), f, method="Nelder-Mead", control=list(maxit=2000))
}
score <- function(P, th, rows, per_alt=TRUE) {
  cv <- if (per_alt) plogis(th[1:4]) else rep(plogis(th[1]),4)
  logloss(y[rows], adjust(P, cs, cv, exp(th[length(th)]))[rows,,drop=FALSE])
}

for (nm in c("xgb_lw2","xgb_resenc2","BLEND")) {
  P <- if (nm=="BLEND") blend_oof() else { d<-readRDS(sprintf("model/artifacts/oof_%s.rds",nm)); setorder(d,No); as.matrix(d[,.(p1,p2,p3,p4)]) }
  base <- logloss(y, P)
  o <- fit_adj(P, rep(TRUE,length(y)))
  cat(sprintf("\n%-12s base %.5f | in-sample ceiling with shrinkage+T %.5f  (c = %s, T = %.3f)\n",
              nm, base, o$value, paste(sprintf("%.3f", plogis(o$par[1:4])), collapse=" "), exp(o$par[5])))
  # T alone, for separation of axes
  oT <- optim(0, function(t) logloss(y, adjust(P, cs, rep(0,4), exp(t))), method="Brent", lower=-1, upper=1)
  cat(sprintf("%-12s   temperature ALONE: %.5f (T = %.3f)  -> extra from respondent shrinkage: %+.5f\n",
              "", oT$value, exp(oT$par), oT$value - o$value))
  # honest: fit (c,T) on folds != k, apply to fold k
  hon <- numeric(5)
  for (k in 1:5) { ok <- fit_adj(P, fmap!=k); hon[k] <- score(P, ok$par, fmap==k) }
  bk <- sapply(1:5, function(k) logloss(y[fmap==k], P[fmap==k,,drop=FALSE]))
  cat(sprintf("%-12s   HONEST (params fit out-of-fold): %.5f vs base %.5f  delta %+.5f\n",
              "", mean(hon), mean(bk), mean(bk)-mean(hon)))
}
