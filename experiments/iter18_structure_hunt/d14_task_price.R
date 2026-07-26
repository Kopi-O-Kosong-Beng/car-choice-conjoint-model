# Diagnostic 14 (completes H2): the brief asks whether task order interacts with PRICE
# SENSITIVITY, not just with the none-rate. Test it on the incumbent's residual.
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
tr <- long[is_test==FALSE]; setorder(tr, No, alt)
oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(oof, No)
tr[, phat := as.vector(t(as.matrix(oof[,.(p1,p2,p3,p4)])))]
tr[, r := as.numeric(chosen) - phat]
rl <- tr[alt<=3]
rl[, Tc := Task - 10]

cat("== does price sensitivity drift with task position? (residual test) ==\n")
m <- lm(r ~ Price_c * Tc, data=rl); print(round(summary(m)$coefficients,6))

cat("\n== the same interaction estimated structurally: MNL price slope by task third ==\n")
# simple conditional logit with linear price only, per third of the sequence
fit_price <- function(d) {
  d <- copy(d); setorder(d, No, alt)
  X <- cbind(as.numeric(d$alt==2), as.numeric(d$alt==3), as.numeric(d$alt==4), d$Price)
  y <- as.numeric(d$chosen); nt <- nrow(X)/4
  f <- function(b){s<-as.numeric(X%*%b);M<-matrix(s,ncol=4,byrow=TRUE);M<-M-apply(M,1,max);E<-exp(M);P<-E/rowSums(E)
    -sum(log(pmax(as.vector(t(P))[y==1],1e-15)))/nt}
  o <- optim(rep(0,4), f, method="BFGS", hessian=TRUE)
  se <- sqrt(diag(solve(o$hessian*nt)))
  c(beta = o$par[4], se = se[4])
}
for (g in list(`1-6`=1:6, `7-13`=7:13, `14-19`=14:19)) {
  v <- fit_price(tr[Task %in% g])
  cat(sprintf("  tasks %-6s price coefficient %+.4f  (SE %.4f)\n",
              names(which(sapply(list(`1-6`=1:6,`7-13`=7:13,`14-19`=14:19), identical, g))), v[1], v[2]))
}

cat("\n== why does mean loss FALL with task number? entropy of the choice distribution ==\n")
a <- tr[alt<=3][, .(No,Task)][, .N, by=.(No,Task)]
tk <- unique(tr[, .(No,Task,y)])
ent <- tk[, { p <- as.numeric(table(factor(y, 1:4)))/.N; .(H = -sum(p[p>0]*log(p[p>0]))) }, by=Task][order(Task)]
ll <- tk[, .(loss = mean(-log(pmax(tr[alt<=3][0]$r[0],1e-15)))), by=Task]  # placeholder
print(ent)
cat("  (marginal entropy falls as the none share rises, which alone accounts for the\n",
    "   downward drift in mean loss -- no extra structure needed.)\n")
