# Emit the iteration-61 winner at 10 seeds, then answer the only question left:
# does an HONEST tree add anything to lcmnl3_both, given it is individually the
# worst member we have on the graded population?
#
# Individually xgb_syn is worse than every existing member (SEG 1.23138 vs
# lcmnl3_both 1.20493). That does NOT settle it: iteration 19 found 93% of blend
# error variance on a single tree-vs-logit axis, so a weak tree can still pay if
# it errs orthogonally. The blend is the decision, not the member.
#
# GATE, fixed before running: the 2-member blend (xgb_syn + lcmnl3_both) must
# beat the production blend's SEG of 1.19610. Anything above that and this
# whole line of work is closed and we say so.
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
DIR <- "experiments/iter61_synthesis"
SEEDS <- 10L; DEPTH <- 6L; NR <- 250L

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
NOS <- sort(unique(trl$No)); TNOS <- sort(unique(tel$No))
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
allA <- rbind(trl[, ..ATTRS], tel[, ..ATTRS])
lev <- lapply(ATTRS, function(a) sort(unique(allA[[a]]))); names(lev) <- ATTRS
onehot <- function(D) do.call(cbind, lapply(ATTRS, function(a) {
  M <- outer(D[[a]], lev[[a]], "==") * 1.0
  colnames(M) <- paste0("oh_", a, "_L", lev[[a]]); M }))
OH_tr <- onehot(trl[, ..ATTRS]); OH_te <- onehot(tel[, ..ATTRS])
oh_cols <- colnames(OH_tr)
for (j in seq_along(oh_cols)) { set(trl, NULL, oh_cols[j], OH_tr[, j])
                                set(tel, NULL, oh_cols[j], OH_te[, j]) }
MF_RANK <- 3L; mf_cols <- c("mf_fit", "mf_outside")
person_signal <- function(D) {
  d3 <- D[alt %in% 1:3]
  mm <- d3[, lapply(.SD, mean), by = No, .SDcols = ATTRS]
  setnames(mm, ATTRS, paste0(ATTRS, "__m"))
  ch <- D[chosen == TRUE & alt %in% 1:3, c("No", ATTRS), with = FALSE]
  del <- merge(ch, mm, by = "No")
  for (a in ATTRS) set(del, NULL, a, del[[a]] - del[[paste0(a, "__m")]])
  key <- unique(D[, .(No, Case)])
  del <- merge(del[, c("No", ATTRS), with = FALSE], key, by = "No")
  ps <- del[, lapply(.SD, mean), by = Case, .SDcols = ATTRS]
  y4 <- unique(D[, .(No, Case, out = as.integer(y == 4L))])
  orate <- y4[, .(out = mean(out)), by = Case]
  ps <- merge(data.table(Case = sort(unique(D$Case))), ps, by = "Case", all.x = TRUE)
  for (a in ATTRS) set(ps, which(!is.finite(ps[[a]])), a, 0)
  merge(ps, orate, by = "Case")
}
fit_mf <- function(tr_rows) {
  P <- person_signal(trl[tr_rows]); R <- as.matrix(P[, ..ATTRS]); ctr <- colMeans(R)
  Rc <- sweep(R, 2, ctr, "-"); sv <- svd(Rc, nu = MF_RANK, nv = MF_RANK); Z <- Rc %*% sv$v
  dtab <- unique(trl[, c("Case", demo), with = FALSE]); dz <- dtab[match(P$Case, dtab$Case)]
  X <- cbind(1, as.matrix(dz[, ..demo]))
  cf <- qr.solve(crossprod(X) + diag(1e-3, ncol(X)), crossprod(X, cbind(Z, P$out)))
  list(ctr = ctr, V = sv$v, cf = cf)
}
apply_mf <- function(m, D) {
  dz <- unique(D[, c("Case", demo), with = FALSE]); X <- cbind(1, as.matrix(dz[, ..demo]))
  pred <- X %*% m$cf
  taste <- sweep(pred[, seq_len(MF_RANK), drop = FALSE] %*% t(m$V), 2, m$ctr, "+")
  idx <- match(D$Case, dz$Case)
  fitv <- rowSums(as.matrix(D[, ..ATTRS]) * taste[idx, , drop = FALSE]); fitv[D$alt == 4L] <- 0
  data.table(mf_fit = fitv, mf_outside = pred[idx, MF_RANK + 1L])
}
sbt <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
                     E <- exp(M); E / rowSums(E) }
obj_lw <- function(preds, dtrain) {
  yy <- getinfo(dtrain, "label"); p <- as.vector(t(sbt(preds)))
  list(grad = p - yy, hess = pmax(2*p*(1-p), 1e-6)) }
FEAT <- c(ATTRS, paste0(ATTRS,"_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt",1:4), demo, oh_cols, mf_cols)
tr1 <- function(d, v, seed) {
  fit <- xgb.train(params = list(eta = 0.05, max_depth = DEPTH, min_child_weight = 5,
                                 subsample = 0.8, colsample_bytree = 0.8,
                                 base_score = 0, nthread = 4, seed = seed),
                   data = xgb.DMatrix(as.matrix(d[, ..FEAT]), label = as.numeric(d$chosen)),
                   nrounds = NR, verbose = 0, obj = obj_lw, maximize = FALSE)
  sbt(as.vector(predict(fit, xgb.DMatrix(as.matrix(v[, ..FEAT])), outputmargin = TRUE)))
}
accO <- matrix(0, 21565L, 4L); accT <- matrix(0, 4997L, 4L)
for (s in 1:SEEDS) {
  P <- matrix(NA_real_, 21565L, 4L)
  for (k in 1:5) {
    ins <- which(trl$fold != k); out <- which(trl$fold == k)
    d <- trl[ins]; v <- trl[out]; m <- fit_mf(ins)
    fd <- apply_mf(m, d); fv <- apply_mf(m, v)
    for (cc in mf_cols) { set(d, NULL, cc, fd[[cc]]); set(v, NULL, cc, fv[[cc]]) }
    P[match(unique(trl$No[out]), NOS), ] <- tr1(d, v, s)
  }
  accO <- accO + log(pmax(P, 1e-15))
  m <- fit_mf(seq_len(nrow(trl))); d <- copy(trl); v <- copy(tel)
  fd <- apply_mf(m, d); fv <- apply_mf(m, v)
  for (cc in mf_cols) { set(d, NULL, cc, fd[[cc]]); set(v, NULL, cc, fv[[cc]]) }
  accT <- accT + log(pmax(tr1(d, v, s), 1e-15))
  cat(sprintf("  seed %2d done (OOF %.5f)\n", s, logloss(ytr, P)))
}
nrm <- function(A) { E <- exp(A/SEEDS - apply(A/SEEDS, 1, max)); E / rowSums(E) }
PO <- nrm(accO); PT <- nrm(accT)
stopifnot(!file.exists("model/artifacts/oof_xgb_syn.rds"))
saveRDS(data.table(No = NOS,  p1=PO[,1], p2=PO[,2], p3=PO[,3], p4=PO[,4]), "model/artifacts/oof_xgb_syn.rds")
saveRDS(data.table(No = TNOS, p1=PT[,1], p2=PT[,2], p3=PT[,3], p4=PT[,4]), "model/artifacts/test_xgb_syn.rds")
cat(sprintf("\n  xgb_syn %d-seed OOF %.5f   shipped mean p4 %.5f\n",
            SEEDS, logloss(ytr, PO), mean(PT[,4])))
cat("  wrote oof_xgb_syn.rds / test_xgb_syn.rds\n")
cat("\n  NEXT: BLEND_MEMBERS='xgb_syn lcmnl3_both' -> compare SEG against 1.19610\n")
