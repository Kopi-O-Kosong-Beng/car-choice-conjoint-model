# Diagnostic 10: partial profile implies lvlsum sums raw levels across attributes with
# 2-6 levels, and WHICH attributes are in the sum changes every task. Test normalised /
# rank-based bundle summaries against the incumbent's per-row residual.
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); folds <- readRDS("model/artifacts/folds.rds")
NP <- setdiff(ATTRS,"Price"); MAXL <- sapply(ATTRS, function(a) max(long[[a]]))
tr <- long[is_test==FALSE]; setorder(tr, No, alt)
oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(oof, No)
P <- as.matrix(oof[,.(p1,p2,p3,p4)])
tr[, phat := as.vector(t(P))]
tr[, r := as.numeric(chosen) - phat]

Zn <- sapply(NP, function(a) ifelse(tr[[a]]>0, (tr[[a]]-1)/pmax(MAXL[a]-1,1), 0))
tr[, nsum := rowSums(Zn)]
tr[, pnorm_ := ifelse(Price>0, (Price-1)/11, 0)]
# within-task comparisons among the 3 real bundles only
rl <- tr[alt<=3]
Zr <- Zn[tr$alt<=3, ]
rl[, `:=`(nsum_c = nsum - mean(nsum), price_c2 = Price - mean(Price)), by=No]
# how many of the 9 shown attributes is this bundle strictly best / worst on
cnt <- rl[, {
  M <- Zr[.I, , drop=FALSE]
  mx <- apply(M,2,max); mn <- apply(M,2,min); act <- mx>mn
  .(alt=alt, nbest=rowSums(sweep(M,2,mx,`==`)[,act,drop=FALSE]),
    nworst=rowSums(sweep(M,2,mn,`==`)[,act,drop=FALSE]), nvary=sum(act))
}, by=No]
rl <- cbind(rl, cnt[, .(nbest, nworst, nvary)])
rl[, `:=`(fbest = nbest/pmax(nvary,1), fworst = nworst/pmax(nvary,1))]

cat("== do normalised / rank summaries explain the incumbent's residual on real bundles? ==\n")
for (fs in list(c("nsum_c"), c("fbest","fworst"), c("nsum_c","fbest","fworst","nvary","price_c2"))) {
  s <- summary(lm(reformulate(fs, "r"), data=rl))
  cat(sprintf("  %-40s R2 %.6f  F %.2f  p %.3g\n", paste(fs, collapse="+"), s$r.squared,
      s$fstatistic[1], pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3], lower.tail=FALSE)))
}
cat("\n  sanity: same features on the OUTCOME itself\n")
s <- summary(lm(reformulate(c("nsum_c","fbest","fworst","nvary","price_c2"), "chosen"), data=rl))
cat(sprintf("  R2 %.5f (so the features are informative; the model already has the signal)\n", s$r.squared))
cat("\n  correlation of lvlsum with the better-scaled nsum:",
    round(cor(rl$lvlsum, rl$nsum),4), "\n")
