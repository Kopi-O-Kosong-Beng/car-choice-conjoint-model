# Diagnostic 15: is the price-sensitivity drift real, or a design confound (are later
# task positions given different price distributions)? And is it already in the model?
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
tr <- long[is_test==FALSE]; setorder(tr, No, alt)
real <- tr[alt<=3]
cat("== design balance across task positions ==\n")
b <- real[, .(mean_price=mean(Price), sd_price=sd(Price), mean_lvlsum=mean(lvlsum)), by=Task][order(Task)]
print(b)
cat(sprintf("range of mean price across positions: %.3f to %.3f\n", min(b$mean_price), max(b$mean_price)))

cat("\n== price coefficient by task position, one conditional logit per position ==\n")
fit_price <- function(d) {
  d <- copy(d); setorder(d, No, alt)
  X <- cbind(as.numeric(d$alt==2), as.numeric(d$alt==3), as.numeric(d$alt==4), d$Price, d$lvlsum)
  y <- as.numeric(d$chosen); nt <- nrow(X)/4
  f <- function(b){s<-as.numeric(X%*%b);M<-matrix(s,ncol=4,byrow=TRUE);M<-M-apply(M,1,max);E<-exp(M);P<-E/rowSums(E)
    -sum(log(pmax(as.vector(t(P))[y==1],1e-15)))/nt}
  o <- optim(rep(0,5), f, method="BFGS", hessian=TRUE); se <- sqrt(diag(solve(o$hessian*nt)))
  c(o$par[4], se[4], o$par[5], se[5])
}
R <- t(sapply(1:19, function(t) fit_price(tr[Task==t])))
print(data.table(Task=1:19, price_b=round(R[,1],4), se=round(R[,2],4), lvlsum_b=round(R[,3],4)))
w <- 1/R[,2]^2; x <- 1:19
cat(sprintf("\nweighted linear trend in the price coefficient: %+.5f per task (SE %.5f, z %.2f)\n",
    coef(lm(R[,1] ~ x, weights=w))[2], summary(lm(R[,1] ~ x, weights=w))$coef[2,2],
    summary(lm(R[,1] ~ x, weights=w))$coef[2,3]))
cat("(controls for bundle quality via lvlsum, so it is not a pure attractiveness shift)\n")

cat("\n== is the drift already in the incumbent? residual test, respondent-clustered ==\n")
oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(oof, No)
tr[, phat := as.vector(t(as.matrix(oof[,.(p1,p2,p3,p4)])))]
tr[, r := as.numeric(chosen) - phat]
rl <- tr[alt<=3][, Tc := Task - 10]
rl[, xz := Price_c * Tc]
b1 <- coef(lm(r ~ Price_c + Tc + xz, rl))["xz"]
cl <- rl[, .(cv = sum(r * (xz - mean(rl$xz))), vv = sum((xz - mean(rl$xz))^2)), by=Case]
se <- sd(cl$cv)*sqrt(nrow(cl))/sum(cl$vv)
cat(sprintf("  residual Price_c x Task coefficient %+.6f, respondent-clustered SE %.6f, z %+.2f\n", b1, se, b1/se))
cat("  (structural drift is ~8 SE; what survives in the residual is ~1-2 SE -> the tree\n",
    "   has already absorbed it through Task x Price splits.)\n")
