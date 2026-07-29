# Predict the new blend's public score from DIFFERENCES against the production
# blend, whose public score is known (1.197). Differencing cancels each
# weighting scheme's own offset, so the prediction does not depend on choosing
# the "right" scheme -- if the schemes agree, the answer is robust; if they
# disagree, that disagreement is the honest error bar.
suppressMessages(library(data.table)); source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); wide <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No); y <- ymap$y
fmap <- folds[order(No), fold]; eps0 <- 1e-12
LIVE_PUBLIC <- 1.197

mk_w <- function(v, clip) {
  resp <- unique(wide[, c("Case","is_test",v), with = FALSE])
  a <- resp[is_test == FALSE, .(ptr = .N/sum(!resp$is_test)), by = v]
  b <- resp[is_test == TRUE,  .(pte = .N/sum(resp$is_test)),  by = v]
  t2 <- merge(a,b,by=v,all.x=TRUE); t2[is.na(pte), pte := 0]
  t2[, wt := if (is.null(clip)) pte/ptr else pmin(pmax(pte/ptr, clip[1]), clip[2])]
  merge(unique(wide[is_test == FALSE, c("No",v), with = FALSE]),
        t2[, c(v,"wt"), with = FALSE], by = v)[order(No), wt]
}
# luxury binary: segmentind in {3,5} is 68.8% of TEST rows vs 9.4% of train
lux <- unique(wide[, .(No, is_test, lux = as.integer(segmentind %in% c(3,5)))])
ptr <- lux[is_test == FALSE, mean(lux)]; pte <- lux[is_test == TRUE, mean(lux)]
wl <- lux[is_test == FALSE][order(No), ifelse(lux == 1, pte/ptr, (1-pte)/(1-ptr))]

SCHEMES <- list(
  `segment  clipped [0.2,5]` = mk_w("segmentind", c(0.2,5)),
  `segment  UNCLIPPED`       = mk_w("segmentind", NULL),
  `luxury binary UNCLIPPED`  = wl,
  `income   clipped [0.2,5]` = mk_w("incomeind", c(0.2,5)),
  `plain (no reweighting)`   = rep(1, length(y))
)
rwll <- function(P,w){ l <- -log(pmax(P[cbind(seq_along(y),y)],1e-15)); sum(w*l)/sum(w) }
ess  <- function(w) sum(w)^2/sum(w^2)

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
Pprod <- nested_oof(c("xgb_lw2bag","lcmnl3_both"))
Pnew  <- nested_oof(c("xgb_rot","lcmnl3_both"))

cat(sprintf("\n  production blend plain %.5f   |   rotation blend plain %.5f\n\n",
            logloss(y,Pprod), logloss(y,Pnew)))
cat(sprintf("  %-26s %7s %9s %9s %9s %11s\n","weighting","ESS","prod","new","delta","pred public"))
for (nm in names(SCHEMES)) {
  w <- SCHEMES[[nm]]; a <- rwll(Pprod,w); b <- rwll(Pnew,w)
  cat(sprintf("  %-26s %7.0f %9.5f %9.5f %+9.5f %11.3f\n",
              nm, ess(w), a, b, b-a, LIVE_PUBLIC + (b-a)))
}
cat(sprintf("\n  Anchor: the production blend scored %.3f public. Each row predicts the\n", LIVE_PUBLIC))
cat("  rotation blend by applying that scheme's OWN measured delta to the anchor,\n")
cat("  so a scheme's absolute offset cancels. Spread across schemes = the error bar.\n")
cat("  Leader today is 1.186; we are 12th at 1.197.\n")
