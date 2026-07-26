# H4, confirmed properly: full 5-fold honest OOF on the fixed folds, main effects vs
# main + all 190 pairwise attribute interactions. Ridge lambda for the interaction block
# is chosen INSIDE each training fold by a Case-grouped inner split, so the reported
# number is not self-tuned.
suppressMessages({library(data.table); library(Matrix)})
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); folds <- readRDS("model/artifacts/folds.rds")
setorder(long, No, alt)
tr <- merge(long[is_test==FALSE], folds[, .(No, fold)], by="No"); setorder(tr, No, alt)
ymap <- unique(tr[, .(No, Case, y, fold)]); setorder(ymap, No)
MAXL <- sapply(ATTRS, function(a) max(long[[a]]))

build_X <- function(d, with_int) {
  cols <- list(); nm <- character()
  for (j in 2:4) { cols[[length(cols)+1]] <- Matrix(as.numeric(d$alt==j), ncol=1, sparse=TRUE); nm <- c(nm, paste0("asc",j)) }
  for (a in ATTRS) for (l in 2:MAXL[a]) { cols[[length(cols)+1]] <- Matrix(as.numeric(d[[a]]==l), ncol=1, sparse=TRUE); nm <- c(nm, paste0(a,"_L",l)) }
  if (with_int) { Z <- sapply(ATTRS, function(a) d[[a]]/MAXL[a])
    for (i in 1:19) for (j in (i+1):20) { cols[[length(cols)+1]] <- Matrix(Z[,i]*Z[,j], ncol=1, sparse=TRUE); nm <- c(nm, paste0(ATTRS[i],"x",ATTRS[j])) } }
  X <- do.call(cbind, cols); colnames(X) <- nm; X
}
fit_cl <- function(X, y, pen_idx, lam) {
  ntask <- nrow(X)/4; pen <- numeric(ncol(X)); pen[pen_idx] <- lam
  f <- function(b){s<-as.numeric(X%*%b);M<-matrix(s,ncol=4,byrow=TRUE);M<-M-apply(M,1,max);E<-exp(M);P<-E/rowSums(E)
    -sum(log(pmax(as.vector(t(P))[y==1],1e-15)))/ntask + sum(pen*b^2)/2}
  g <- function(b){s<-as.numeric(X%*%b);M<-matrix(s,ncol=4,byrow=TRUE);M<-M-apply(M,1,max);E<-exp(M);P<-E/rowSums(E)
    as.numeric(crossprod(X, as.vector(t(P))-y))/ntask + pen*b}
  optim(numeric(ncol(X)), f, g, method="L-BFGS-B", control=list(maxit=1500, factr=1e8))$par
}
pll <- function(X,b,y){s<-as.numeric(X%*%b);M<-matrix(s,ncol=4,byrow=TRUE);M<-M-apply(M,1,max);E<-exp(M);P<-E/rowSums(E)
  -mean(log(pmax(as.vector(t(P))[y==1],1e-15)))}

LAMS <- c(3e-3, 1e-2, 3e-2, 1e-1)
ll_main <- numeric(5); ll_int <- numeric(5); picked <- numeric(5)
set.seed(42)
for (k in 1:5) {
  dtr <- tr[fold!=k]; dte <- tr[fold==k]
  ytr <- as.numeric(dtr$chosen); yte <- as.numeric(dte$chosen)
  X0tr <- build_X(dtr,FALSE); X0te <- build_X(dte,FALSE)
  ll_main[k] <- pll(X0te, fit_cl(X0tr,ytr,integer(0),0), yte)
  # inner Case-grouped split to pick lambda without touching fold k
  ic <- sample(unique(dtr$Case), round(0.25*uniqueN(dtr$Case)))
  itr <- dtr[!Case %in% ic]; ite <- dtr[Case %in% ic]
  X1i <- build_X(itr,TRUE); X1v <- build_X(ite,TRUE); npen <- (ncol(X0tr)+1):ncol(X1i)
  inner <- sapply(LAMS, function(l) pll(X1v, fit_cl(X1i, as.numeric(itr$chosen), npen, l), as.numeric(ite$chosen)))
  lam <- LAMS[which.min(inner)]; picked[k] <- lam
  X1tr <- build_X(dtr,TRUE); X1te <- build_X(dte,TRUE)
  ll_int[k] <- pll(X1te, fit_cl(X1tr,ytr,npen,lam), yte)
  cat(sprintf("fold %d  main %.5f   main+190int %.5f  (inner-picked lambda %.3g)\n", k, ll_main[k], ll_int[k], lam))
}
cat(sprintf("\n>>> 5-fold OOF, main effects only        : %.5f\n", mean(ll_main)))
cat(sprintf(">>> 5-fold OOF, + 190 pairwise interactions: %.5f   (delta %+.5f)\n",
            mean(ll_int), mean(ll_main)-mean(ll_int)))
cat("lambda picked per fold:", picked, "\n")
