# =============================================================================
# ITERATION 27 — focal weighting inside the listwise objective.
# HYPOTHESIS WRITTEN BEFORE RUNNING, INCLUDING THE REASON IT MAY FAIL.
#
# THE OBSERVATION. Loss is concentrated: the worst 10% of tasks carry 21.9% of
# total loss and the worst 25% carry 44.7%, with mean predicted probability of
# the realised choice only 0.090 in the worst decile. 20.5% of respondents score
# worse than ln(4).
#
# THE MECHANISM (focal loss, Lin et al. 2017, adapted to a within-task softmax).
# Replace  L = -log p_y  with  L = -(1-p_y)^gamma * log p_y.  Easy tasks
# (p_y -> 1) get down-weighted by (1-p_y)^gamma, so tree capacity is spent on
# the hard tail instead of on cases already solved. Differentiating,
#     dL/ds_j = w * (p_j - 1{j=y}),
#     w = (1-p_y)^gamma - gamma * p_y * (1-p_y)^(gamma-1) * log p_y
# so it is EXACTLY the existing listwise gradient scaled by a per-task weight w
# that depends only on p_y. At gamma = 0, w = 1 and this reduces identically to
# the production objective -- the control is nested exactly, not approximately.
#
# WHY I EXPECT IT TO FAIL AT ITS STATED PURPOSE, stated up front.
# Focal loss is NOT a proper scoring rule. Our metric IS logloss, so deliberately
# down-weighting confident-correct tasks biases calibration in the direction the
# metric punishes. Worse, a diagnostic run today shows respondent difficulty
# correlates with the none-rate at r = -0.342 and with luxury-segment membership
# at only r = +0.098: the hard cases are hard because discriminating between
# THREE SIMILAR BUNDLES is intrinsically harder than predicting the attribute-free
# outside option -- not because of a correctable bias. Emphasising those rows
# cannot manufacture signal that is not in the features.
#
# SO THE REAL HYPOTHESIS IS DIVERSITY, NOT ACCURACY. A focally-trained tree makes
# systematically DIFFERENT errors from a logloss-trained one (it trades
# calibration on easy rows for sharpness on hard ones). The blend has a fitted
# temperature and eps that can undo the calibration damage, and the pool has
# repeatedly rewarded members that are worse alone but decorrelated -- xgb_mono
# is exactly that (1.13980 alone, 0.311 weight). Judged on the BLEND, not alone.
#
# PRE-REGISTERED: gamma in {0, 1, 2}. gamma = 0 is the exact control. K = 3 bags
# per fold (distinct seeds AND distinct early-stopping carves) because single-fit
# tree comparisons are inadmissible here -- the carve alone has SD 0.0026-0.0032,
# which exceeds the effect under test. Params are xgb_lw2's production values.
#
# ADOPTION GATE: not the single-model OOF. A gamma variant is adopted only if
# adding it to the production blend improves the NESTED blend number, and does
# not regress the segment-reweighted metric by more than 1 bootstrap SE.
#
# Emits oof_xgb_focal{1,2}.rds / test_xgb_focal{1,2}.rds and a bagged control.
# Run: Rscript experiments/iter27_focal/run.R     Runtime: ~45 fits, 1.5-2 h.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")

long <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, ENC_COLS)

softmax_by_task <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE)
  M <- M - apply(M, 1, max); E <- exp(M); E / rowSums(E) }

# ---- focal objective: standard listwise gradient scaled by w(p_y) -------------
make_obj <- function(gamma) function(preds, dtrain) {
  yv <- getinfo(dtrain, "label")
  P  <- softmax_by_task(preds)                     # ntask x 4
  Y  <- matrix(yv, ncol = 4, byrow = TRUE)
  py <- P[Y == 1]                                  # prob of the realised choice
  w  <- if (gamma == 0) rep(1, length(py)) else
        (1 - py)^gamma - gamma * py * (1 - py)^(gamma - 1) * log(pmax(py, 1e-12))
  w  <- pmax(w, 1e-6)
  W  <- rep(w, each = 4)                           # broadcast to the 4 rows
  p  <- as.vector(t(P))
  list(grad = W * (p - yv), hess = pmax(W * 2 * p * (1 - p), 1e-6))
}
# evaluation metric stays PLAIN logloss -- we are scored on that, not on focal
eval_tl <- function(preds, dtrain) { yv <- getinfo(dtrain, "label")
  P <- softmax_by_task(preds); Y <- matrix(yv, ncol = 4, byrow = TRUE)
  list(metric = "tasklogloss", value = -mean(log(pmax(P[Y == 1], 1e-15)))) }

new_api <- packageVersion("xgboost") >= "2.1.0"
gbi <- function(f) { b <- tryCatch(xgb.attr(f, "best_iteration"), error = function(e) NULL)
  if (is.null(b)) b <- tryCatch(f$best_iteration, error = function(e) NULL)
  if (is.null(b) || is.na(suppressWarnings(as.numeric(b)))) NULL else as.integer(b) }
rp <- function(f, d) { dm <- xgb.DMatrix(as.matrix(d[, ..feat])); b <- gbi(f)
  as.vector(if (!is.null(b)) tryCatch(predict(f, dm, iterationrange = c(1, b+1), outputmargin = TRUE),
    error = function(e) predict(f, dm, outputmargin = TRUE)) else predict(f, dm, outputmargin = TRUE)) }
PAR <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,     # xgb_lw2 production
            subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = 0)
tr1 <- function(dtr, des, gamma) {
  a <- list(params = PAR, data = xgb.DMatrix(as.matrix(dtr[, ..feat]),
            label = as.numeric(dtr$chosen)), nrounds = 5000,
            early_stopping_rounds = 150, verbose = 0,
            obj = make_obj(gamma), maximize = FALSE)
  m <- xgb.DMatrix(as.matrix(des[, ..feat]), label = as.numeric(des$chosen))
  if (new_api) { a$evals <- list(es = m); a$custom_metric <- eval_tl }
  else         { a$watchlist <- list(es = m); a$feval <- eval_tl }
  do.call(xgb.train, a)
}

GAMMAS <- c(0, 1, 2); K <- 3L; SEEDS <- 2000L + seq_len(K)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
nos_tr <- unique(long[is_test == FALSE, .(No)])[order(No), No]
nos_te <- sort(unique(tel$No))
res <- list()

for (g in GAMMAS) {
  nm <- if (g == 0) "focal0" else sprintf("focal%d", g)
  cat("\n===== gamma =", g, "(", nm, ") =====\n"); flush.console()
  ps <- array(NA_real_, dim = c(length(nos_tr), 4, K))
  for (k in 1:5) {
    d <- trl[fold == k]; rk <- match(unique(d$No), nos_tr)
    da <- trl[fold != k]; cse <- unique(da$Case); it <- integer(K)
    for (b in seq_len(K)) {
      set.seed(SEEDS[b] + 17L * k)
      es <- sample(cse, round(0.1 * length(cse)))
      f <- tr1(da[!Case %in% es], da[Case %in% es], g)
      ps[rk, , b] <- softmax_by_task(rp(f, d))
      bb <- gbi(f); it[b] <- if (is.null(bb)) NA_integer_ else bb
    }
    cat(sprintf("  fold %d  iters %s\n", k, paste(it, collapse = "/"))); flush.console()
  }
  lls <- apply(ps, 3, function(P) logloss(ytr, P))
  Pa <- apply(ps, c(1, 2), mean); Pa <- Pa / rowSums(Pa)
  cat(sprintf("  per-seed %s | seed mean %.5f SD %.5f | BAGGED %.5f\n",
              paste(sprintf("%.5f", lls), collapse = " "), mean(lls), sd(lls),
              logloss(ytr, Pa)))
  saveRDS(data.table(No = nos_tr, p1=Pa[,1], p2=Pa[,2], p3=Pa[,3], p4=Pa[,4]),
          sprintf("model/artifacts/oof_xgb_%s.rds", nm))
  cse <- unique(trl$Case); Pt <- array(NA_real_, dim = c(length(nos_te), 4, K))
  for (b in seq_len(K)) {
    set.seed(SEEDS[b] + 977L)
    es <- sample(cse, round(0.1 * length(cse)))
    f <- tr1(trl[!Case %in% es], trl[Case %in% es], g)
    Pt[, , b] <- softmax_by_task(rp(f, tel))
  }
  Pte <- apply(Pt, c(1,2), mean); Pte <- Pte / rowSums(Pte)
  stopifnot(nrow(Pte) == 4997, all(abs(rowSums(Pte) - 1) < 1e-9))
  saveRDS(data.table(No = nos_te, p1=Pte[,1], p2=Pte[,2], p3=Pte[,3], p4=Pte[,4]),
          sprintf("model/artifacts/test_xgb_%s.rds", nm))
  res[[nm]] <- list(gamma = g, seed_ll = lls, bagged = logloss(ytr, Pa))
  saveRDS(res, "experiments/iter27_focal/results.rds")
}

cat("\n================ SUMMARY ================\n")
for (nm in names(res)) cat(sprintf("%-8s gamma %g   seed mean %.5f (SD %.5f)   BAGGED %.5f\n",
  nm, res[[nm]]$gamma, mean(res[[nm]]$seed_ll), sd(res[[nm]]$seed_ll), res[[nm]]$bagged))
cat("\nControl gamma=0 must be close to xgb_lw2 (1.14152) up to bagging gain.\n")
cat("NOT promoted. Judge on the BLEND: add each to members and rerun 06_blend.R.\nOK\n")
