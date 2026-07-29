# =============================================================================
# ITERATION 29 — give the TREES the alternative x demographic carrier they
# structurally lack. HYPOTHESIS AND FALSIFICATION TESTS WRITTEN BEFORE RUNNING.
#
# THE DEFECT, derived. The listwise objective is a softmax WITHIN each task, so
# the gradient depends only on within-task contrasts of the score. Every
# demographic is CONSTANT across a task's four rows, hence has identically zero
# within-task contrast on its own. A tree can therefore express "luxury
# respondents decline the outside option less often" ONLY as a path interaction:
# a split on `alt4` composed with a split on `segmentind`. That interaction earns
# gradient on just the 9.4% of tasks belonging to luxury respondents, so boosting
# stops short of the true level.
#
# THE EVIDENCE. Respondent-clustered alt-4 bias on the 107 luxury respondents:
#     xgb_lw2      +0.099  (z = +4.74)      <- broken
#     xgb_mono     +0.098  (z = +4.68)      <- broken
#     mnl_pw       +0.020  (z =  0.96)      <- fine (has explicit Price_x_seg)
#     lcmnl3_both  +0.023  (z =  1.08)      <- fine (segment routes membership)
# Only the tree family is miscalibrated, and only on the demographic axis. A
# single-fold probe adding the carriers moved predicted luxury p4 from 0.2781 to
# 0.1618 -- the observed training rate -- confirming the mechanism.
#
# THE CHANGE (one change, six binary columns):
#     a4_seg2..a4_seg6 = alt4 * (segmentind == s),  s = 2..6   (segment 1 = ref)
#     a4_lux           = alt4 * (segmentind in {3,5})
# These are deterministic products of the alternative structure with an observed
# demographic -- the exact analogue of `Price_x_seg*`, which already works in the
# logit family. NOT the rejected "segment importance weights" (that reweighted the
# LOSS and wrecked trees); this changes the FEATURE BASIS. NOT the rejected
# within-respondent context features (those z-scored attributes against a
# respondent's random design draw, i.e. noise).
#
# EXPLICITLY EXCLUDED: continuous products (Price_x_age, Price_x_inc, ...). A probe
# measured them at -0.020 (z = -3.34); they seize importance slots and halve the
# early-stopping horizon (476 -> 188 rounds).
#
# HYPOTHESIS: plain nested OOF roughly FLAT (the fix touches 9.4% of training
# rows, so the decision number structurally cannot price it), luxury alt-4 bias
# collapses from +0.099 to under 0.03, and the segment-reweighted metric improves.
# Adoption is MECHANISM-gated, not z-gated, for that reason -- stated up front so
# it cannot be rationalised afterwards.
#
# FALSIFICATION, pre-registered:
#   F1  plain OOF must not be a real loss (compare.R z > -2 vs xgb_lw2)
#   F2  luxury alt-4 |bias| must fall below 0.03
#   F3  the change must NOT be concentrated in a single fold (leak signature).
#       Fold 1 is adversarial: its 16 luxury respondents have observed none-rate
#       0.306 vs 0.116-0.152 in folds 2-5, so expect fold 1 flat/negative and the
#       gain in folds 2-5. Anything else is suspicious.
#   F4  segment-reweighted OOF must improve.
# K = 3 bags (distinct seeds AND early-stopping carves): single-fit tree
# comparisons are inadmissible, carve SD is 0.0026-0.0032.
# Emits oof_xgb_lw3.rds / test_xgb_lw3.rds.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R"); source("model/encode_design.R")
long <- readRDS("model/artifacts/long.rds"); folds <- readRDS("model/artifacts/folds.rds")
wide <- readRDS("model/artifacts/wide.rds")
long <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
trl <- merge(long[is_test==FALSE], folds[, .(No, fold)], by="No")
tel <- long[is_test==TRUE]; setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)

# ---- THE ONE CHANGE: alternative x demographic carriers ----------------------
CARRY <- c(paste0("a4_seg", 2:6), "a4_lux")
for (d in list(trl, tel)) {
  for (s in 2:6) d[, paste0("a4_seg", s) := as.numeric(alt == 4L & segmentind == s)]
  d[, a4_lux := as.numeric(alt == 4L & segmentind %in% c(3L, 5L))]
}
cat("carriers added:", paste(CARRY, collapse=", "), "\n")
cat("nonzero rate in train:", paste(sprintf("%s %.3f", CARRY,
    sapply(CARRY, function(c) mean(trl[[c]]))), collapse=" | "), "\n\n")

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS,"_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt",1:4), demo, ENC_COLS, CARRY)

sbt <- function(s){M<-matrix(s,ncol=4,byrow=TRUE); M<-M-apply(M,1,max); E<-exp(M); E/rowSums(E)}
obj <- function(p,d){yv<-getinfo(d,"label"); q<-as.vector(t(sbt(p)))
  list(grad=q-yv, hess=pmax(2*q*(1-q),1e-6))}
ev <- function(p,d){yv<-getinfo(d,"label"); P<-sbt(p); Y<-matrix(yv,ncol=4,byrow=TRUE)
  list(metric="tl", value=-mean(log(pmax(P[Y==1],1e-15))))}
na <- packageVersion("xgboost") >= "2.1.0"
gbi <- function(f){b<-tryCatch(xgb.attr(f,"best_iteration"),error=function(e)NULL)
  if(is.null(b)) b<-tryCatch(f$best_iteration,error=function(e)NULL)
  if(is.null(b)||is.na(suppressWarnings(as.numeric(b)))) NULL else as.integer(b)}
rp <- function(f,d){dm<-xgb.DMatrix(as.matrix(d[,..feat])); b<-gbi(f)
  as.vector(if(!is.null(b)) tryCatch(predict(f,dm,iterationrange=c(1,b+1),outputmargin=TRUE),
    error=function(e)predict(f,dm,outputmargin=TRUE)) else predict(f,dm,outputmargin=TRUE))}
PAR <- list(eta=0.03, max_depth=8, min_child_weight=20, subsample=0.8,
            colsample_bytree=0.8, base_score=0, nthread=0)
tr1 <- function(dtr,des){a<-list(params=PAR, data=xgb.DMatrix(as.matrix(dtr[,..feat]),
  label=as.numeric(dtr$chosen)), nrounds=5000, early_stopping_rounds=150, verbose=0,
  obj=obj, maximize=FALSE)
  m<-xgb.DMatrix(as.matrix(des[,..feat]), label=as.numeric(des$chosen))
  if(na){a$evals<-list(es=m); a$custom_metric<-ev} else {a$watchlist<-list(es=m); a$feval<-ev}
  do.call(xgb.train,a)}

K <- 3L; SEEDS <- 1000L + seq_len(K)
ymap <- unique(long[is_test==FALSE,.(No,y,segmentind)])[order(No)]
ytr <- ymap$y; nos_tr <- ymap$No; lux <- ymap$segmentind %in% c(3L,5L)
nos_te <- sort(unique(tel$No))
ps <- array(NA_real_, dim=c(length(nos_tr),4,K))
for (k in 1:5) {
  d <- trl[fold==k]; rk <- match(unique(d$No), nos_tr)
  da <- trl[fold!=k]; cse <- unique(da$Case); it <- integer(K)
  for (b in seq_len(K)) {
    set.seed(SEEDS[b]+17L*k); es <- sample(cse, round(0.1*length(cse)))
    f <- tr1(da[!Case %in% es], da[Case %in% es])
    ps[rk,,b] <- sbt(rp(f,d)); bb <- gbi(f); it[b] <- if(is.null(bb)) NA_integer_ else bb
  }
  # F3: per-fold luxury diagnostics
  P <- apply(ps[rk,,,drop=FALSE], c(1,2), mean); P <- P/rowSums(P)
  lk <- lux[rk]
  cat(sprintf("  fold %d iters %s | luxury n=%d obs_p4 %.3f pred_p4 %.3f bias %+.3f\n",
      k, paste(it,collapse="/"), sum(lk), mean(ytr[rk][lk]==4), mean(P[lk,4]),
      mean(P[lk,4])-mean(ytr[rk][lk]==4))); flush.console()
}
lls <- apply(ps,3,function(P) logloss(ytr,P))
Pa <- apply(ps,c(1,2),mean); Pa <- Pa/rowSums(Pa)
cat(sprintf("\nper-seed %s | mean %.5f SD %.5f | BAGGED %.5f\n",
    paste(sprintf("%.5f",lls),collapse=" "), mean(lls), sd(lls), logloss(ytr,Pa)))
cat(sprintf("F2  luxury alt-4: obs %.4f pred %.4f BIAS %+.4f   (xgb_lw2 was +0.099)\n",
    mean(ytr[lux]==4), mean(Pa[lux,4]), mean(Pa[lux,4])-mean(ytr[lux]==4)))
cat(sprintf("    non-luxury  : obs %.4f pred %.4f bias %+.4f\n",
    mean(ytr[!lux]==4), mean(Pa[!lux,4]), mean(Pa[!lux,4])-mean(ytr[!lux]==4)))
saveRDS(data.table(No=nos_tr,p1=Pa[,1],p2=Pa[,2],p3=Pa[,3],p4=Pa[,4]),
        "model/artifacts/oof_xgb_lw3.rds")
cse <- unique(trl$Case); Pt <- array(NA_real_, dim=c(length(nos_te),4,K))
for (b in seq_len(K)) { set.seed(SEEDS[b]+977L); es <- sample(cse, round(0.1*length(cse)))
  f <- tr1(trl[!Case %in% es], trl[Case %in% es]); Pt[,,b] <- sbt(rp(f,tel)) }
Pte <- apply(Pt,c(1,2),mean); Pte <- Pte/rowSums(Pte)
stopifnot(nrow(Pte)==4997, all(abs(rowSums(Pte)-1)<1e-9))
saveRDS(data.table(No=nos_te,p1=Pte[,1],p2=Pte[,2],p3=Pte[,3],p4=Pte[,4]),
        "model/artifacts/test_xgb_lw3.rds")
telseg <- unique(tel[,.(No,segmentind)])[order(No)]
cat(sprintf("\nTEST luxury pred p4: %.4f  (blend was 0.223; target [0.17,0.20])\n",
    mean(Pte[telseg$segmentind %in% c(3L,5L),4])))
cat("\nNOT promoted. Next: model/compare.R xgb_lw2 xgb_lw3, then blend swap.\nOK\n")
