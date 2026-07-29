# Segment-reweighted nested OOF for the production blend vs the rotation blend.
# Read-only. The reweighted metric is used to JUDGE, never to fit (iteration 07).
suppressMessages(library(data.table)); source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); wide <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No); y <- ymap$y
fmap <- folds[order(No), fold]; eps0 <- 1e-12

mk_w <- function(v) {
  resp <- unique(wide[, c("Case","is_test",v), with = FALSE])
  a <- resp[is_test == FALSE, .(ptr = .N/sum(!resp$is_test)), by = v]
  b <- resp[is_test == TRUE,  .(pte = .N/sum(resp$is_test)),  by = v]
  t2 <- merge(a,b,by=v,all.x=TRUE); t2[is.na(pte), pte := 0]
  t2[, wt := pmin(pmax(pte/ptr, 0.2), 5)]
  merge(unique(wide[is_test == FALSE, c("No",v), with = FALSE]),
        t2[, c(v,"wt"), with = FALSE], by = v)[order(No), wt]
}
w_seg <- mk_w("segmentind")
rwll <- function(P,w){ l <- -log(pmax(P[cbind(seq_along(y),y)],1e-15)); sum(w*l)/sum(w) }

nested_oof <- function(memb) {
  M <- length(memb)
  OOF <- lapply(memb, function(m){ d <- readRDS(sprintf("model/artifacts/oof_%s.rds",m))
                                   setorder(d,No); as.matrix(d[,.(p1,p2,p3,p4)]) })
  blend <- function(th, Ps, rows) {
    w <- exp(th[1:M]); w <- w/sum(w); Tt <- exp(th[M+1]); eA <- plogis(th[M+2])*0.10
    L <- Reduce(`+`, Map(function(wi,P) wi*log(pmax(P[rows,,drop=FALSE],eps0)), w, Ps))/Tt
    P <- exp(L - apply(L,1,max)); P <- P/rowSums(P); (1-eA)*P + eA*0.25
  }
  obj <- function(th, rows) logloss(y[rows], blend(th, OOF, rows))
  P <- matrix(NA_real_, 21565L, 4L)
  for (k in 1:5) {
    o <- optim(c(rep(0,M),0,-3), obj, rows = fmap != k, method="Nelder-Mead",
               control=list(maxit=3000))
    P[fmap==k,] <- blend(o$par, OOF, fmap==k)
  }
  P
}
cat(sprintf("%-34s %9s %9s %11s\n","blend","plain","seg-rwt","pred public"))
for (nm in list(c("xgb_lw2bag","lcmnl3_both"), c("xgb_rot","lcmnl3_both"))) {
  P <- nested_oof(nm); s <- rwll(P, w_seg)
  cat(sprintf("%-34s %9.5f %9.5f %11.3f\n", paste(nm, collapse=" + "),
              logloss(y,P), s, s + 0.001))
}
cat("\n  calibration anchor: production blend seg-reweighted 1.19610 -> live public 1.197\n")
cat("  (the +0.001 column applies that single observed offset; it is ONE data point)\n")
