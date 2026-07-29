# =============================================================================
# ITERATION 30 - the parallel track's tuned config as a DECORRELATION MEMBER.
# HYPOTHESIS BEFORE RESULTS.
#
# WHY. Six iterations established that this blend ABSORBS member-level repairs:
# iter29 fixed a real structural defect in the trees (luxury alt-4 bias
# +0.099 -> +0.023) and the blend gained +0.0003, because the pool had already
# routed around the defect by down-weighting the trees. Improving a member is
# therefore near-worthless. What a log-opinion pool rewards is a member whose
# ERRORS ARE DECORRELATED from the incumbents'.
#
# The blend currently has a redundancy problem: xgb_lw2 and xgb_mono have loss
# correlation 0.984 and carry 0.185 + 0.311 = 0.496 of the weight between them,
# and iter12 showed the monotone constraint that supposedly distinguishes them is
# worth +0.0001 once carve noise is averaged out. Two near-duplicates, half the
# weight.
#
# THE CHANGE. A tree from a genuinely different region of hyperparameter space:
# the parallel track's searched config - max_depth 5 (vs 8), min_child_weight 80
# (vs 20), colsample_bytree 0.30 (vs 0.80), alpha 1, lambda 10 (vs 0). Shallow,
# heavily column-subsampled and strongly regularised. It sees a different column
# mix on every split, so it cannot learn the same function as the incumbents even
# on identical data. Confirmed on 5 fresh partitions in its own codebase
# (p = 0.0001, survives a Hansen SPA correction) - but on a DIFFERENT feature set,
# and this repo's iteration 06 found heavy regularisation LOSES on these features
# (its `reg` config was worst of five). So the standalone score is likely mediocre.
#
# THAT IS FINE AND IS THE POINT. xgb_mono scores 1.13980 alone - worse than
# lcmnl3_both - yet earns 0.311 weight because it is different. The hypothesis is
# not that this config is better; it is that it is DIFFERENT.
#
# GATE, pre-registered, in this order:
#   G1  loss correlation with xgb_lw2 < 0.97. At >= 0.98 it is a third twin and is
#       dropped regardless of its own score - no further testing.
#   G2  nested blend improves under the respondent-clustered PAIRED test, not a
#       headline delta (the repo's own method note; violated earlier tonight).
#   G3  segment-reweighted metric not worse by more than 1 bootstrap SE.
# Standalone OOF is explicitly NOT a gate.
#
# K = 5 bags (distinct seeds AND early-stopping carves); single-fit tree
# comparisons are inadmissible here.
# Emits oof_xgb_pt.rds / test_xgb_pt.rds.
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
# WITHDRAWN 26 Jul — THIS SCRIPT NEVER RAN THE EXPERIMENT ITS HEADER DESCRIBES.
# It was generated by sed from experiments/iter27_focal/run.R; the pattern anchored
# on "min_child_weight = 20,$" but that line carries a trailing comment, so the
# substitution silently failed. What executed was xgb_lw2's config with only
# subsample 0.8->0.85 and colsample 0.8->0.30 — a TWO-change experiment, in the
# incumbent's hyperparameter region, mislabelled as the parallel track's config
# (depth 5 / mcw 80 / eta 0.04 / alpha 1 / lambda 10). Killed at fold 3.
# Any `xgb_pt` artifact from this script is misnamed and must not be cited.
PAR <- list(eta = 0.04, max_depth = 5, min_child_weight = 80, alpha = 1, lambda = 10,
            subsample = 0.85, colsample_bytree = 0.30, base_score = 0, nthread = 0)
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

GAMMAS <- c(0); K <- 5L; SEEDS <- 3000L + seq_len(K)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
nos_tr <- unique(long[is_test == FALSE, .(No)])[order(No), No]
nos_te <- sort(unique(tel$No))
res <- list()

for (g in GAMMAS) {
  nm <- "pt"
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
  saveRDS(res, "experiments/iter30_decorr/results.rds")
}

cat("\n================ SUMMARY ================\n")
for (nm in names(res)) cat(sprintf("%-8s gamma %g   seed mean %.5f (SD %.5f)   BAGGED %.5f\n",
  nm, res[[nm]]$gamma, mean(res[[nm]]$seed_ll), sd(res[[nm]]$seed_ll), res[[nm]]$bagged))
cat("\nControl gamma=0 must be close to xgb_lw2 (1.14152) up to bagging gain.\n")
cat("NOT promoted. Judge on the BLEND: add each to members and rerun 06_blend.R.\nOK\n")
