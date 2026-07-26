# Diagnostic 12: sequential ANCHORING / CONTRAST. Tasks arrive in a fixed order within a
# version, so we know every test respondent's PREVIOUS task design too. If a task looks
# cheap/good relative to the one before it, choice may shift. Placebo: the NEXT task's
# design must have no effect (the respondent has not seen it yet).
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
NP <- setdiff(ATTRS,"Price"); MAXL <- sapply(ATTRS, function(a) max(long[[a]]))
tr <- long[is_test==FALSE]; setorder(tr, No, alt)
oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(oof, No)
P <- as.matrix(oof[,.(p1,p2,p3,p4)])

real <- tr[alt<=3]
Zn <- sapply(NP, function(a) ifelse(real[[a]]>0,(real[[a]]-1)/pmax(MAXL[a]-1,1),0))
real[, nsum := rowSums(Zn)]
q <- real[, .(minp=min(Price), meanp=mean(Price), bestq=max(nsum), meanq=mean(nsum)), by=.(Case,Task,No)]
setorder(q, Case, Task)
q[, `:=`(lag_minp = shift(minp), lag_meanp = shift(meanp), lag_bestq = shift(bestq),
         lead_minp = shift(minp,-1), lead_meanp = shift(meanp,-1), lead_bestq = shift(bestq,-1)), by=Case]
tk <- unique(tr[, .(No, Case, Task, y)]); setorder(tk, No)
tk[, `:=`(obs4=as.numeric(y==4), p4=P[,4])]; tk[, r4 := obs4-p4]
tk <- merge(tk, q[, !c("Case","Task")], by="No")
tk[, `:=`(dp = minp - lag_minp, dq = bestq - lag_bestq,
          dp_f = lead_minp - minp, dq_f = lead_bestq - bestq)]

cat("== effect of the PREVIOUS task's design on the current none-residual ==\n")
m1 <- lm(r4 ~ lag_minp + lag_meanp + lag_bestq, data=tk[!is.na(lag_minp)])
print(round(summary(m1)$coefficients,5))
cat(sprintf("R2 %.6f  F-p %.4g\n", summary(m1)$r.squared,
    pf(summary(m1)$fstatistic[1],summary(m1)$fstatistic[2],summary(m1)$fstatistic[3],lower.tail=FALSE)))

cat("\n== contrast form: current minus previous ==\n")
m2 <- lm(r4 ~ dp + dq, data=tk[!is.na(dp)]); print(round(summary(m2)$coefficients,5))

cat("\n== PLACEBO: the NEXT task's design (respondent has not seen it) ==\n")
m3 <- lm(r4 ~ lead_minp + lead_meanp + lead_bestq, data=tk[!is.na(lead_minp)])
print(round(summary(m3)$coefficients,5))
cat(sprintf("R2 %.6f  F-p %.4g\n", summary(m3)$r.squared,
    pf(summary(m3)$fstatistic[1],summary(m3)$fstatistic[2],summary(m3)$fstatistic[3],lower.tail=FALSE)))

cat("\n== respondent-clustered check on the strongest lag term ==\n")
for (v in c("lag_minp","lag_bestq","lead_minp","lead_bestq")) {
  d <- tk[!is.na(get(v))]
  b <- coef(lm(reformulate(v,"r4"), d))[2]
  cl <- d[, .(cv = cov(r4, get(v)), vv = var(get(v))), by=Case]
  se <- sd(cl$cv, na.rm=TRUE)/sqrt(nrow(cl))/mean(cl$vv, na.rm=TRUE)
  cat(sprintf("  %-11s beta %+.5f  clustered SE %.5f  z %+.2f\n", v, b, se, b/se))
}

cat("\n== state dependence (NOT usable at test time, reported for completeness) ==\n")
tk2 <- merge(unique(tr[,.(No,Case,Task,y)]), q[,.(No)], by="No"); setorder(tk2, Case, Task)
tk2[, prev_y := shift(y), by=Case]
print(round(prop.table(table(prev = tk2$prev_y, cur = tk2$y), 1), 4))
