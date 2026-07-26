# Diagnostic 9: where does the remaining loss live? Variance decomposition of the choice
# outcome into respondent / design components, and an ORACLE bound on how much a
# population-averaged model can ever achieve when every test respondent is a stranger.
suppressMessages(library(data.table))
source("model/99_utils.R"); source("model/encode_design.R")
long <- readRDS("model/artifacts/long.rds"); wide <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")
long <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
tr <- merge(long[is_test==FALSE], folds[, .(No,fold)], by="No"); setorder(tr, No, alt)
ymap <- unique(tr[, .(No, Case, y, fold)]); setorder(ymap, No); y <- ymap$y

cat("== 0. sanity on production features ==\n")
for (v in c("richness","rich_task_mean","price_rank")) {
  cat(sprintf("  %-15s real bundles: %s | none rows: %s\n", v,
      paste(sort(unique(tr[alt<=3][[v]])), collapse=","), paste(sort(unique(tr[alt==4][[v]])), collapse=",")))
}

cat("\n== 1. variance decomposition of the choice indicator, per alternative ==\n")
# version id = design key at Task 1
vmap <- unique(tr[Task==1, .(Case, ver = dkey)])
tr2 <- merge(tr, vmap, by="Case")
for (j in 1:4) {
  d <- tr2[alt==j, .(No, Case, dkey, ver, c = as.numeric(chosen))]
  p <- mean(d$c); tot <- p*(1-p)
  # respondent component
  rm_ <- d[, .(m=mean(c), n=.N), by=Case]
  v_resp <- var(rm_$m) - tot/mean(rm_$n)
  # design component: designs have ~4.68 respondents; observed spread minus sampling noise
  dm <- d[, .(m=mean(c), n=.N), by=dkey]
  v_des <- var(dm$m) - tot/mean(dm$n)
  vm <- d[, .(m=mean(c), n=.N), by=ver]
  v_ver <- var(vm$m) - tot/mean(vm$n)
  cat(sprintf("  alt %d  p=%.3f  total var %.4f | between-RESPONDENT %.4f (%.0f%%) | between-DESIGN %.4f (%.0f%%) | between-VERSION %.4f\n",
      j, p, tot, v_resp, 100*v_resp/tot, v_des, 100*v_des/tot, v_ver))
}
cat("  (between-DESIGN is inflated: a design's ~4.7 respondents are the SAME people for\n",
    "  all 19 designs of a version, so respondent taste leaks into the design estimate.)\n")

cat("\n== 2. ORACLE: what would knowing the respondent be worth? ==\n")
cat("   (leave-one-task-out own rates -- IMPOSSIBLE for test respondents, this is a bound)\n")
tr[, ntot := sum(chosen), by=.(Case, alt)]
tr[, q_loo := (ntot - as.numeric(chosen))/18]
Q <- matrix(tr$q_loo, ncol=4, byrow=TRUE)
popq <- matrix(rep(colMeans(Q), each=nrow(Q)), ncol=4)
bl <- readRDS("model/artifacts/blend.rds"); memb <- bl$members; M <- length(memb)
OOF <- lapply(memb, function(m){d<-readRDS(sprintf("model/artifacts/oof_%s.rds",m));setorder(d,No);as.matrix(d[,.(p1,p2,p3,p4)])})
th <- bl$par; w <- exp(th[1:M]); w<-w/sum(w); Tt<-exp(th[M+1]); eA<-plogis(th[M+2])*0.10
L <- Reduce(`+`, Map(function(wi,P) wi*log(pmax(P,1e-12)), w, OOF))/Tt
Pb <- exp(L-apply(L,1,max)); Pb <- Pb/rowSums(Pb); Pb <- (1-eA)*Pb + eA*0.25
for (nm in c("mnl_pw","xgb_lw2","BLEND")) {
  P <- if (nm=="BLEND") Pb else {d<-readRDS(sprintf("model/artifacts/oof_%s.rds",nm));setorder(d,No);as.matrix(d[,.(p1,p2,p3,p4)])}
  f <- function(a){ LL <- log(pmax(P,1e-12)) + a*(Q-popq); QQ<-exp(LL-apply(LL,1,max)); logloss(y, QQ/rowSums(QQ)) }
  o <- optimize(f, c(0,15))
  cat(sprintf("  %-8s %.5f -> %.5f with oracle respondent rates (a=%.2f), gap %+.4f\n",
              nm, logloss(y,P), o$objective, o$minimum, logloss(y,P)-o$objective))
}

cat("\n== 3. best possible DESIGN-only predictor (population choice probability) ==\n")
cat("   full-information limit: encode every design from ALL other respondents, no shrinkage\n")
for (a in c(0.5, 2, 5, 20)) {
  P <- matrix(NA_real_, nrow(ymap), 4)
  for (k in 1:5) {
    e <- encode_aligned(tr[fold!=k], tr[fold==k]); ii <- which(ymap$fold==k)
    S <- matrix(e[[paste0("share_a", if (a %in% c(1,5,20)) a else 5)]], ncol=4, byrow=TRUE)
    if (!(a %in% c(1,5,20))) S <- S
    P[ii,] <- S/rowSums(S)
  }
  if (a %in% c(1,5,20)) cat(sprintf("  design shares alone, alpha=%2g: %.5f\n", a, logloss(y,P)))
}
