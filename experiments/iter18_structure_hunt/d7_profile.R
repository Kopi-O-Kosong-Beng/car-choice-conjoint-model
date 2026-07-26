# Diagnostic 7: PARTIAL PROFILE. Exactly 9 of 19 non-price attributes are shown per task.
# The "none" row is all zeros, so it carries NO information about which attributes were on
# offer or how good the bundles were. Does the model's none-residual still depend on the
# design's attribute set / bundle quality? If yes, there is unexploited design structure.
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); folds <- readRDS("model/artifacts/folds.rds")
NP <- setdiff(ATTRS, "Price")
MAXL <- sapply(ATTRS, function(a) max(long[[a]]))
tr <- long[is_test==FALSE]; setorder(tr, No, alt)
real <- tr[alt<=3]

# normalised quality of each real bundle: sum over the 9 shown attrs of (lvl-1)/(max-1)
Zn <- as.data.table(sapply(NP, function(a) ifelse(real[[a]]>0, (real[[a]]-1)/pmax(MAXL[a]-1,1), 0)))
real2 <- cbind(real[, .(No, alt, Price)], nsum = rowSums(Zn))
tskq <- real2[, .(best_nsum=max(nsum), mean_nsum=mean(nsum), range_nsum=max(nsum)-min(nsum),
                  min_price=min(Price), mean_price=mean(Price), range_price=max(Price)-min(Price)), by=No]
# which attributes are shown (design-only, no choices involved)
shown <- real[, lapply(.SD, function(x) as.integer(any(x>0))), by=No, .SDcols=NP]
setnames(shown, NP, paste0("sh_", NP))
# level headroom of the shown attribute set
shown[, headroom := as.numeric(as.matrix(.SD) %*% (MAXL[NP]-1)), .SDcols=paste0("sh_",NP)]
D <- merge(tskq, shown, by="No")

tk <- unique(tr[, .(No, Case, Task, y)]); setorder(tk, No)
oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(oof, No)
P <- as.matrix(oof[, .(p1,p2,p3,p4)])
tk[, `:=`(obs4 = as.numeric(y==4), p4 = P[,4])]
tk <- merge(tk, D, by="No"); setorder(tk, No)
tk[, r4 := obs4 - p4]

shv <- paste0("sh_", NP)
qv  <- c("best_nsum","mean_nsum","range_nsum","min_price","mean_price","range_price","headroom")
cat("== does the none-residual depend on design-set features the none row cannot see? ==\n")
for (set in list(shown=shv, quality=qv, both=c(shv,qv))) {
  f <- lm(reformulate(set, "r4"), data=tk)
  s <- summary(f)
  cat(sprintf("%-8s: R2 %.5f  F %.2f on (%d,%d) df  p %.3g\n",
      paste(deparse(substitute(set))), s$r.squared, s$fstatistic[1], s$fstatistic[2], s$fstatistic[3],
      pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3], lower.tail=FALSE)))
}
f <- lm(reformulate(c(shv,qv), "r4"), data=tk); s <- summary(f)
cat("\nnames:", paste(names(list(a=1)),collapse=""), "\n")
print(round(head(s$coefficients[order(-abs(s$coefficients[,3])),], 12), 4))
cat("\nsd of fitted residual-model:", round(sd(fitted(f)),5), " (sd of p4 itself:", round(sd(tk$p4),5), ")\n")

cat("\n== sanity: same test on the OBSERVED none rate (should be strongly significant) ==\n")
s2 <- summary(lm(reformulate(c(shv,qv), "obs4"), data=tk))
cat(sprintf("R2 %.5f  F %.2f  p %.3g\n", s2$r.squared, s2$fstatistic[1],
    pf(s2$fstatistic[1], s2$fstatistic[2], s2$fstatistic[3], lower.tail=FALSE)))
s3 <- summary(lm(reformulate(c(shv,qv), "p4"), data=tk))
cat(sprintf("and on the PREDICTED none rate: R2 %.5f\n", s3$r.squared))

cat("\n== honest check: does adding these features improve a direct logistic fit? ==\n")
# out-of-fold logistic correction of the none probability, features = design set only
tk <- merge(tk, folds[, .(No, fold)], by="No"); setorder(tk, No)
tk[, off := log(pmax(p4,1e-9)/pmax(1-p4,1e-9))]
ll_base <- numeric(5); ll_new <- numeric(5)
for (k in 1:5) {
  g <- glm(reformulate(c(shv,qv,"offset(off)"), "obs4"), data=tk[fold!=k], family=binomial)
  pk <- predict(g, newdata=tk[fold==k], type="response")
  o <- tk[fold==k]
  ll_base[k] <- -mean(log(pmax(ifelse(o$obs4==1, o$p4, 1-o$p4),1e-15)))
  ll_new[k]  <- -mean(log(pmax(ifelse(o$obs4==1, pk, 1-pk),1e-15)))
}
cat(sprintf("binary none-vs-rest logloss: base %.5f -> with design-set features %.5f (delta %+.5f)\n",
            mean(ll_base), mean(ll_new), mean(ll_base)-mean(ll_new)))
saveRDS(D, "experiments/iter18_structure_hunt/design_feats.rds")
