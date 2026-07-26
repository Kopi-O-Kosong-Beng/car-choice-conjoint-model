# Diagnostic 5 (pilot): (a) pairwise attribute interactions in a penalised listwise
# conditional logit; (b) is the model's per-respondent prediction over-dispersed?
suppressMessages({library(data.table); library(Matrix)})
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); folds <- readRDS("model/artifacts/folds.rds")
setorder(long, No, alt)
tr <- merge(long[is_test==FALSE], folds[, .(No, fold)], by="No"); setorder(tr, No, alt)

MAXL <- sapply(ATTRS, function(a) max(long[[a]]))
build_X <- function(d, with_int) {
  n <- nrow(d); cols <- list(); nm <- character()
  for (j in 2:4) { cols[[length(cols)+1]] <- Matrix(as.numeric(d$alt==j), ncol=1, sparse=TRUE); nm <- c(nm, paste0("asc",j)) }
  for (a in ATTRS) for (l in 2:MAXL[a]) {
    cols[[length(cols)+1]] <- Matrix(as.numeric(d[[a]]==l), ncol=1, sparse=TRUE); nm <- c(nm, paste0(a,"_L",l))
  }
  if (with_int) {
    Z <- sapply(ATTRS, function(a) d[[a]]/MAXL[a])
    for (i in 1:(length(ATTRS)-1)) for (j in (i+1):length(ATTRS)) {
      cols[[length(cols)+1]] <- Matrix(Z[,i]*Z[,j], ncol=1, sparse=TRUE); nm <- c(nm, paste0(ATTRS[i],"x",ATTRS[j]))
    }
  }
  X <- do.call(cbind, cols); colnames(X) <- nm; X
}
# listwise softmax nll with ridge on a subset of columns
fit_cl <- function(X, y, pen_idx, lam, b0=NULL) {
  n4 <- nrow(X); ntask <- n4/4
  pen <- numeric(ncol(X)); pen[pen_idx] <- lam
  f <- function(b) {
    s <- as.numeric(X %*% b); M <- matrix(s, ncol=4, byrow=TRUE)
    M <- M - apply(M,1,max); E <- exp(M); S <- rowSums(E); P <- E/S
    -sum(log(pmax(as.vector(t(P))[y==1],1e-15)))/ntask + sum(pen*b^2)/2
  }
  g <- function(b) {
    s <- as.numeric(X %*% b); M <- matrix(s, ncol=4, byrow=TRUE)
    M <- M - apply(M,1,max); E <- exp(M); P <- E/rowSums(E)
    r <- as.vector(t(P)) - y
    as.numeric(crossprod(X, r))/ntask + pen*b
  }
  if (is.null(b0)) b0 <- numeric(ncol(X))
  o <- optim(b0, f, g, method="L-BFGS-B", control=list(maxit=1500, factr=1e8))
  o$par
}
pred_ll <- function(X, b, y) {
  s <- as.numeric(X %*% b); M <- matrix(s, ncol=4, byrow=TRUE)
  M <- M - apply(M,1,max); E <- exp(M); P <- E/rowSums(E)
  -mean(log(pmax(as.vector(t(P))[y==1],1e-15)))
}

cat("== (a) PILOT: interactions, train folds 2-5 -> eval fold 1 ==\n")
dtr <- tr[fold!=1]; dte <- tr[fold==1]
ytr <- as.numeric(dtr$chosen); yte <- as.numeric(dte$chosen)
X0tr <- build_X(dtr, FALSE); X0te <- build_X(dte, FALSE)
b0 <- fit_cl(X0tr, ytr, integer(0), 0)
cat(sprintf("main effects only (%d params): fold-1 logloss %.5f\n", ncol(X0tr), pred_ll(X0te, b0, yte)))
X1tr <- build_X(dtr, TRUE); X1te <- build_X(dte, TRUE)
pen <- (ncol(X0tr)+1):ncol(X1tr)
for (lam in c(1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1)) {
  b1 <- fit_cl(X1tr, ytr, pen, lam)
  cat(sprintf("main + 190 interactions, lambda %.4g: fold-1 logloss %.5f  (train %.5f)\n",
              lam, pred_ll(X1te, b1, yte), pred_ll(X1tr, b1, ytr)))
  if (abs(lam - 1e-2) < 1e-12) {
    g <- b1[pen]; names(g) <- colnames(X1tr)[pen]
    cat("  top 15 |interaction| at lambda=0.01:\n")
    print(round(head(sort(g[order(-abs(g))]),0),3)); print(round(g[order(-abs(g))][1:15],3))
  }
}
saveRDS(list(MAXL=MAXL), "experiments/iter18_structure_hunt/pilot_meta.rds")

cat("\n== (b) per-respondent over-dispersion of the incumbent's predictions ==\n")
tk <- unique(tr[, .(No, Case, y)]); setorder(tk, No)
oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(oof, No)
P <- as.matrix(oof[, .(p1,p2,p3,p4)])
for (a in 1:4) {
  tk[, obs := as.numeric(y==a)]; tk[, ph := P[,a]]
  rl <- tk[, .(o=mean(obs), p=mean(ph)), by=Case]
  cf <- coef(lm(o ~ p, rl))
  cat(sprintf("alt %d: slope of observed on predicted per-respondent rate = %.3f (1.0 = calibrated), var(pred)=%.5f var(obs)=%.5f\n",
              a, cf[2], var(rl$p), var(rl$o)))
}
