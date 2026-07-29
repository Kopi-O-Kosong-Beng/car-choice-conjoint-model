# =============================================================================
# ITERATION 31 — the SAME carriers, ported to the second tree (xgb_mono).
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
#   F1  plain OOF must not be a real loss: compare.R z > -2 vs **xgb_mono**
#       (NOT xgb_lw2 — the control must match the challenger's config)
#   F2  luxury alt-4 |bias| must fall below 0.03
#   F3  the change must NOT be concentrated in a single fold (leak signature).
#       Fold 1 is adversarial: its 16 luxury respondents have observed none-rate
#       0.306 vs 0.116-0.152 in folds 2-5, so expect fold 1 flat/negative and the
#       gain in folds 2-5. Anything else is suspicious.
#   F4  segment-reweighted OOF must improve.
# CAVEAT ON SEEDS, recorded before results: this reuses iteration 29's exact seeds
# (1000+1:3, +17L*k, +977), so xgb_lw3 and xgb_mono2 share early-stopping carves.
# That correlates their bagging noise and will INFLATE any measured correlation
# between the two corrections. So a high r_twin is expected and only an extreme
# value (>= ~0.9) is evidence the two trees encode one and the same correction.
#
# TWIN DETECTOR, declared now: over the 107 luxury respondents compute
#   r_twin = cor( loss(mono2) - loss(mono), loss(lw3) - loss(lw2) )
# r_twin >= 0.8 means the trees carry ONE correction -- the blend can get all of it
# from lw3 alone by weight, so mono2 adds concentration rather than information and
# no OOF "gain" may be claimed for it. r_twin < 0.5 means partially independent.
#
# K = 3 bags (distinct seeds AND early-stopping carves): single-fit tree
# comparisons are inadmissible, carve SD is 0.0026-0.0032.
# Emits oof_xgb_mono2.rds / test_xgb_mono2.rds.
#
# WHY THIS EXISTS. Iteration 29 fixed the same defect in `xgb_lw2` and it worked:
# pooled luxury alt-4 bias +0.099 -> +0.023, segment-reweighted member OOF
# 1.21976 -> 1.21153, blend test luxury p4 0.2227 -> 0.2005, all four
# pre-registered falsification tests passed with the mechanism fingerprint intact
# (luxury deltas negative only in the adversarial fold, positive in the other
# four; non-luxury deltas centred on zero; gains on carrier segments 3/4/5 only).
#
# But `xgb_mono` STILL CARRIES THE IDENTICAL DEFECT -- measured bias +0.098
# (z = 4.68), test luxury p4 0.2609 -- and it retains 0.306 of the blend weight
# after the lw3 swap. That residual is precisely what holds the blend at 0.2005
# rather than the ~0.185-0.19 the two unbiased members independently support
# (mnl_pw 0.1844, lcmnl3_both 0.1895).
#
# HYPOTHESIS: fixing the second tree closes the blend's remaining outside-option
# over-prediction on the two segments that are 68.8% of the graded population.
# Expect the same shape as iteration 29 -- roughly flat on plain OOF, clear
# improvement on the segment-reweighted metric, blend test luxury p4 falling
# into 0.185-0.195.
# FALSIFIED IF: the F3 fingerprint is absent (gains not concentrated on luxury
# respondents in folds 2-5), or the blend's segment-reweighted metric does not
# improve, or test luxury p4 overshoots below 0.17.
#
# ONE CHANGE vs iteration 29: the base config gains the monotone price constraint,
# i.e. this is `xgb_mono` + the six carriers. Same K=3 bagging, same seeds, same
# falsification battery.
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
mono_vec <- rep(0, length(feat)); mono_vec[match(c("Price","Price_c"), feat)] <- -1
PAR <- list(eta=0.03, max_depth=8, min_child_weight=20, subsample=0.8,
            colsample_bytree=0.8, base_score=0, nthread=0,
            monotone_constraints = mono_vec)
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
        "model/artifacts/oof_xgb_mono2.rds")
cse <- unique(trl$Case); Pt <- array(NA_real_, dim=c(length(nos_te),4,K))
for (b in seq_len(K)) { set.seed(SEEDS[b]+977L); es <- sample(cse, round(0.1*length(cse)))
  f <- tr1(trl[!Case %in% es], trl[Case %in% es]); Pt[,,b] <- sbt(rp(f,tel)) }
Pte <- apply(Pt,c(1,2),mean); Pte <- Pte/rowSums(Pte)
stopifnot(nrow(Pte)==4997, all(abs(rowSums(Pte)-1)<1e-9))
saveRDS(data.table(No=nos_te,p1=Pte[,1],p2=Pte[,2],p3=Pte[,3],p4=Pte[,4]),
        "model/artifacts/test_xgb_mono2.rds")
telseg <- unique(tel[,.(No,segmentind)])[order(No)]
cat(sprintf("\nTEST luxury pred p4: %.4f  (blend was 0.223; target [0.17,0.20])\n",
    mean(Pte[telseg$segmentind %in% c(3L,5L),4])))
cat("\nNOT promoted. Next: model/compare.R xgb_mono xgb_mono2, then blend swap.\nOK\n")
