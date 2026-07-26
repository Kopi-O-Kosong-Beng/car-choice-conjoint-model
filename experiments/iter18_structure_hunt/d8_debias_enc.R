# Diagnostic 8: the design encoding averages ~3.7 reference respondents. Because the
# questionnaire is a fixed 299-VERSION block, those same ~3.7 people supply all 19 designs
# of a version -- so their personal taste (e.g. a none-lover) biases all 19 shares the same
# way. Test a version-mate-debiased share: subtract each reference respondent's own
# leave-one-task-out rate for that alternative. No model predictions are involved, so the
# iteration-12 double-OOF leak cannot occur.
suppressMessages(library(data.table))
source("model/99_utils.R"); source("model/encode_design.R")
long <- readRDS("model/artifacts/long.rds"); wide <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")
long <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
tr <- merge(long[is_test==FALSE], folds[, .(No, fold)], by="No"); setorder(tr, No, alt)
ymap <- unique(tr[, .(No, Case, y, fold)]); setorder(ymap, No)
y <- ymap$y

# leave-one-task-out own rate for each (Case, alt)
tr[, ntot := sum(chosen), by=.(Case, alt)]
tr[, q_loo := (ntot - as.numeric(chosen)) / 18]

enc_both <- function(ref, tgt, alpha = 5) {
  prior <- ref[, .(pr = sum(chosen)/.N), by=alt]
  cnt <- ref[, .(n_ch = sum(chosen), n = .N, adj = sum(as.numeric(chosen) - q_loo)), by=.(dkey, alt)]
  t2 <- copy(tgt)[, .rid := .I][, .(.rid, dkey, alt)]
  t2 <- merge(t2, cnt, by=c("dkey","alt"), all.x=TRUE)
  t2 <- merge(t2, prior, by="alt", all.x=TRUE)
  t2[is.na(n), `:=`(n=0, n_ch=0, adj=0)]
  t2[, s_raw := (n_ch + alpha*pr)/(n + alpha)]
  t2[, s_adj := pr + adj/(n + alpha)]          # same shrinkage, respondent-effect removed
  setorder(t2, .rid); t2[, .(s_raw, s_adj, n)]
}

E <- rbindlist(lapply(1:5, function(k) {
  e <- enc_both(tr[fold!=k], tr[fold==k])
  cbind(tr[fold==k, .(No, alt, chosen)], e)
}))
setorder(E, No, alt)

cat("== leak detector (a): correlation of encoding with the row's OWN held-out outcome ==\n")
cat(sprintf("  raw share vs chosen: %+.4f   debiased vs chosen: %+.4f  (positive = genuine signal)\n",
            cor(E$s_raw, as.numeric(E$chosen)), cor(E$s_adj, as.numeric(E$chosen))))

cat("\n== leak detector (b): direct signal test on a design-free baseline (mnl_pw) ==\n")
b <- readRDS("model/artifacts/oof_mnl_pw.rds"); setorder(b, No)
stopifnot(identical(b$No, ymap$No))
Pb <- as.matrix(b[, .(p1,p2,p3,p4)])
Sr <- matrix(E$s_raw, ncol=4, byrow=TRUE); Sa <- matrix(E$s_adj, ncol=4, byrow=TRUE)
pop <- matrix(rep(colMeans(matrix(as.numeric(E$chosen), ncol=4, byrow=TRUE)), each=nrow(Sr)), ncol=4)
apply_w <- function(P, S, w) { L <- log(pmax(P,1e-12)) + w*(S - pop); Q <- exp(L - apply(L,1,max)); Q/rowSums(Q) }
cat(sprintf("  baseline mnl_pw: %.5f\n", logloss(y, Pb)))
for (w in c(0.25, 0.5, 1, 1.5, 2, 3)) {
  cat(sprintf("  w = %-4g  raw %.5f   debiased %.5f\n", w,
              logloss(y, apply_w(Pb, Sr, w)), logloss(y, apply_w(Pb, Sa, w))))
}
o1 <- optimize(function(w) logloss(y, apply_w(Pb, Sr, w)), c(0,6))
o2 <- optimize(function(w) logloss(y, apply_w(Pb, Sa, w)), c(0,6))
cat(sprintf("  best raw      w=%.2f -> %.5f  (gain %+.5f)\n", o1$minimum, o1$objective, logloss(y,Pb)-o1$objective))
cat(sprintf("  best debiased w=%.2f -> %.5f  (gain %+.5f)\n", o2$minimum, o2$objective, logloss(y,Pb)-o2$objective))

cat("\n== how much of the share's variance is reference-respondent noise? ==\n")
cat(sprintf("  sd(raw share) %.4f  sd(debiased) %.4f  cor %.3f\n", sd(E$s_raw), sd(E$s_adj), cor(E$s_raw,E$s_adj)))
cat(sprintf("  mean support n per (design,alt): %.2f\n", mean(E$n)))
saveRDS(E, "experiments/iter18_structure_hunt/enc_compare.rds")
