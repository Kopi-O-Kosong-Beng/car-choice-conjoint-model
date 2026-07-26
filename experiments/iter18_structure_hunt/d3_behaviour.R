# Diagnostic 3: task-order / fatigue, dominance, per-task scale, respondent predictability.
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); folds <- readRDS("model/artifacts/folds.rds")
tr <- long[is_test == FALSE]
setorder(tr, No, alt)
tk <- unique(tr[, .(No, Case, Task, y)])
setorder(tk, No)
oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(oof, No)
stopifnot(identical(tk$No, oof$No))
P <- as.matrix(oof[, .(p1,p2,p3,p4)])
tk[, loss := -log(pmax(P[cbind(.I, y)], 1e-15))]
tk[, phat_none := P[,4]]

# ---------------- A. task-order effects -------------------------------------
cat("== A. choice shares and model loss by Task position ==\n")
a <- tk[, .(n=.N, none_rate = mean(y==4), p_none_hat = mean(phat_none),
            a1=mean(y==1), a2=mean(y==2), a3=mean(y==3), loss = mean(loss)), by=Task][order(Task)]
print(a)
cat("\nlinear trend, none-rate ~ Task:\n")
print(summary(glm(I(y==4) ~ Task, data = tk, family = binomial))$coef)
cat("\nlinear trend, loss ~ Task (respondent-clustered SE via case means):\n")
cm <- tk[, .(l = mean(loss)), by = .(Case, Task)]
print(summary(lm(l ~ Task, data = cm))$coef)
cat("\nHOW MUCH is Task worth? compare a Task-only reweighting of the OOF:\n")
# fit per-task-position temperature on OOF (in-sample upper bound)
tmp <- function(P, y, tvec) {
  ll <- 0
  for (t in sort(unique(tvec))) {
    idx <- which(tvec == t)
    f <- function(lam) { Q <- P[idx,,drop=FALSE]^lam; Q <- Q/rowSums(Q); -sum(log(pmax(Q[cbind(seq_along(idx), y[idx])],1e-15))) }
    o <- optimize(f, c(0.3, 3)); ll <- ll + o$objective
  }
  ll / length(y)
}
cat("  base OOF loss          :", round(mean(tk$loss),5), "\n")
cat("  per-Task temp (IN-samp):", round(tmp(P, tk$y, tk$Task),5), "\n")

# ---------------- B. is fatigue respondent-specific? -------------------------
cat("\n== B. per-respondent slope of loss on Task ==\n")
sl <- tk[, .(b = coef(lm(loss ~ Task))[2]), by = Case]
cat("mean slope", round(mean(sl$b),5), " sd", round(sd(sl$b),5),
    " sd expected under noise ~", round(sd(tk$loss)/sqrt(sum((1:19-10)^2)),5), "\n")

# ---------------- C. dominance ------------------------------------------------
cat("\n== C. dominance among the 3 real bundles ==\n")
real <- tr[alt <= 3]
dom <- real[, {
  M <- as.matrix(.SD[, ATTRS[ATTRS!="Price"], with=FALSE]); pr <- Price
  nd <- integer(3); ndby <- integer(3)
  for (i in 1:3) for (j in 1:3) if (i!=j) {
    if (all(M[i,] >= M[j,]) && pr[i] <= pr[j] && (any(M[i,]>M[j,]) || pr[i]<pr[j])) { nd[i]<-nd[i]+1L; ndby[j]<-ndby[j]+1L }
  }
  .(alt=alt, chosen=chosen, dominates=nd, dominated_by=ndby)
}, by = No, .SDcols = c(ATTRS,"Price","alt","chosen")]
cat("share of real alternatives that are dominated by a rival:", round(mean(dom$dominated_by>0),4), "\n")
cat("choice rate | dominated  :", round(mean(dom[dominated_by>0]$chosen),4), "\n")
cat("choice rate | not dominated:", round(mean(dom[dominated_by==0]$chosen),4), "\n")
cat("tasks containing >=1 dominated alt:", round(mean(dom[, any(dominated_by>0), by=No]$V1),4), "\n")

# ---------------- D. per-task scale driven by design dispersion --------------
cat("\n== D. does the optimal temperature vary with design dispersion? ==\n")
dd <- real[, .(spread_lvl = max(lvlsum)-min(lvlsum), spread_price = max(Price)-min(Price),
               mean_price = mean(Price), mean_lvl = mean(lvlsum)), by = No]
tk2 <- merge(tk, dd, by="No")
for (v in c("spread_lvl","spread_price","mean_price","mean_lvl")) {
  q <- cut(tk2[[v]], quantile(tk2[[v]], 0:4/4), include.lowest=TRUE)
  out <- sapply(levels(q), function(L) {
    idx <- which(q==L)
    f <- function(lam){Q<-P[idx,,drop=FALSE]^lam; Q<-Q/rowSums(Q); -mean(log(pmax(Q[cbind(seq_along(idx),tk2$y[idx])],1e-15)))}
    o <- optimize(f, c(0.3,3)); c(lam=o$minimum, loss=o$objective, base=mean(tk2$loss[idx]))
  })
  cat(sprintf("%-13s  lam by quartile: %s   | base loss: %s\n", v,
      paste(sprintf("%.3f", out["lam",]), collapse=" "), paste(sprintf("%.3f", out["base",]), collapse=" ")))
}

# ---------------- E. respondent predictability vs demographics --------------
cat("\n== E. is per-respondent mean loss predictable from demographics? ==\n")
dem <- unique(tr[, .(Case, incomea, agea, genderind, educind, milesa, segmentind,
                     regionind, Urbind, nighta, pparkind, yearind, nightind, milesind)])
rl <- tk[, .(mloss = mean(loss), nrate = mean(y==4)), by=Case]
rl <- merge(rl, dem, by="Case"); rl[, loginc := log(pmax(incomea,1))]
f <- lm(mloss ~ loginc + agea + factor(genderind) + educind + milesa + factor(segmentind) +
          factor(regionind) + factor(Urbind) + nighta + factor(pparkind) + yearind, data=rl)
cat("R2 of demographics on per-respondent mean loss:", round(summary(f)$r.squared,4),
    " adjR2", round(summary(f)$adj.r.squared,4), " F-p", signif(pf(summary(f)$fstatistic[1],
    summary(f)$fstatistic[2], summary(f)$fstatistic[3], lower.tail=FALSE),3), "\n")
cat("sd of per-respondent mean loss:", round(sd(rl$mloss),4),
    "; sd of fitted:", round(sd(fitted(f)),4), "\n")
g <- lm(nrate ~ loginc + agea + factor(genderind) + educind + milesa + factor(segmentind) +
          factor(regionind) + factor(Urbind) + nighta + factor(pparkind) + yearind, data=rl)
cat("R2 of demographics on per-respondent NONE rate:", round(summary(g)$r.squared,4), "\n")
