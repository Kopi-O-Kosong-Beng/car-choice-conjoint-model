# =============================================================================
# ITERATION 12 — bagged head-to-head of the two candidate listwise configs.
# HYPOTHESIS AND PROTOCOL WRITTEN BEFORE RUNNING. Nothing is selected on the folds.
#
# WHY THIS EXISTS. Iteration 06 promoted `slow_deep` (eta .03/depth 8/mcw 20) over
# `base` (eta .05/depth 6/mcw 10) on a single fit per fold: 1.14152 vs 1.14462,
# paired z = 2.48. An independent rebuild on this machine reproduced `base` to
# 5e-5 but got 1.14655 for `slow_deep` -- the sign of the effect REVERSES. Two
# repeat runs here are BIT-IDENTICAL (max|A-B| = 0), so this is not run-to-run
# noise; a single fit per fold is simply a point estimate of a random quantity
# whose randomness lives in the seed, and z = 2.48 sits at the expected max-|t|
# (~2.07) after ~30 comparisons on one partition.
#
# DERIVED MOTIVATION (why variance, not bias, is the problem). The design is
# partial-profile: exactly 9 of 19 non-price attributes are shown per bundle and
# the shown-set is identical across the three real bundles in a task. So a given
# PAIR of attributes is co-shown in only ~(9/19)(8/18) = 21% of tasks, and the
# two segments carrying 69% of the graded rows are 9.4% of training. Depth-8
# trees therefore spend capacity precisely where the design gives least support.
# The efficient response to a variance problem is averaging, not more capacity.
#
# HYPOTHESIS: with K = 5 bags per fold (distinct seed AND distinct early-stopping
# carve), (a) both configs improve materially over their single-fit selves, and
# (b) the gap between the two configs shrinks below 0.002, i.e. iteration 06's
# choice was largely noise. REGISTERED PREDICTION, per the chief: |bagA - bagB| < 0.002.
#
# COMBINATION RULE, pre-registered: ARITHMETIC mean of probabilities. For log
# loss this dominates the geometric (margin) mean by Jensen -- log(mean p) >=
# mean(log p). The geometric mean is computed too, but ONLY as a reported
# diagnostic; the adopted artifact is the arithmetic one regardless of which
# scores better, so this is not a selection.
#
# NO SELECTION ANYWHERE: both configs are pre-existing, both are kept, and both
# bagged artifacts are emitted. Which one (or both) enters the blend is decided
# afterwards by model/compare.R and the convex blend, not by this script.
#
# TEST REFIT also fixes a real defect: model/03_xgb_listwise.R carves 10% of
# respondents out for early stopping and never trains on them, so the SUBMITTED
# model sees only ~90% of the data. Averaging K fits with DISTINCT carves puts
# every respondent in ~(K-1)/K of the fits. This is invisible in OOF by
# construction -- it must never be claimed as a local gain.
#
# Emits: oof_xgb_bagA/.rds, test_xgb_bagA.rds  (base config)
#        oof_xgb_bagB.rds, test_xgb_bagB.rds  (slow_deep config)
# Does NOT touch oof_xgb_lw2.rds or members.txt.
#
# Run: Rscript experiments/iter12_bagged_configs/run.R
# Runtime: 2 configs x (5 folds + 1 test) x 5 bags ~ 60 fits, 2-4 h. Background it.
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

softmax_by_task <- function(scores) {
  M <- matrix(scores, ncol = 4, byrow = TRUE)
  M <- M - apply(M, 1, max); E <- exp(M); E / rowSums(E)
}
obj_listwise <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  p <- as.vector(t(softmax_by_task(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
eval_tasklogloss <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  P <- softmax_by_task(preds); Y <- matrix(y, ncol = 4, byrow = TRUE)
  list(metric = "tasklogloss", value = -mean(log(pmax(P[Y == 1], 1e-15))))
}
new_api <- packageVersion("xgboost") >= "2.1.0"
get_best_iter <- function(fit) {
  bi <- tryCatch(xgb.attr(fit, "best_iteration"), error = function(e) NULL)
  if (is.null(bi)) bi <- tryCatch(fit$best_iteration, error = function(e) NULL)
  if (is.null(bi) || is.na(suppressWarnings(as.numeric(bi)))) NULL else as.integer(bi)
}
raw_pred <- function(fit, d) {
  dm <- xgb.DMatrix(as.matrix(d[, ..feat])); bi <- get_best_iter(fit)
  as.vector(if (!is.null(bi))
    tryCatch(predict(fit, dm, iterationrange = c(1, bi + 1), outputmargin = TRUE),
             error = function(e) predict(fit, dm, outputmargin = TRUE))
    else predict(fit, dm, outputmargin = TRUE))
}
train_one <- function(dtr, des, p) {
  a <- list(params = c(p, list(base_score = 0, nthread = 0)),
            data = xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen)),
            nrounds = 5000, early_stopping_rounds = 150, verbose = 0,
            obj = obj_listwise, maximize = FALSE)
  m <- xgb.DMatrix(as.matrix(des[, ..feat]), label = as.numeric(des$chosen))
  if (new_api) { a$evals <- list(es = m); a$custom_metric <- eval_tasklogloss }
  else         { a$watchlist <- list(es = m); a$feval <- eval_tasklogloss }
  do.call(xgb.train, a)
}

# monotone constraint vector: utility must not increase in price (as iteration 06)
mono_vec <- rep(0, length(feat))
mono_vec[match(c("Price", "Price_c"), feat)] <- -1

CONFIGS <- list(
  bagA = list(eta = 0.05, max_depth = 6, min_child_weight = 10,
              subsample = 0.8, colsample_bytree = 0.8),   # 'base' == xgb_lw params
  bagB = list(eta = 0.03, max_depth = 8, min_child_weight = 20,
              subsample = 0.8, colsample_bytree = 0.8),   # 'slow_deep' == xgb_lw2 params
  bagC = list(eta = 0.05, max_depth = 6, min_child_weight = 10,
              subsample = 0.8, colsample_bytree = 0.8,
              monotone_constraints = mono_vec)            # 'mono' == base + price prior
)
K <- 5L                       # bags per fold; seeds 1001..1005
SEEDS <- 1000L + seq_len(K)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
nos_tr <- unique(long[is_test == FALSE, .(No)])[order(No), No]
nos_te <- sort(unique(tel$No))

run_bag <- function(nm, p) {
  cat("\n===== config", nm, "=====\n"); flush.console()
  # per-seed OOF probability matrices, so seed-to-seed SD is measurable
  perseed <- array(NA_real_, dim = c(length(nos_tr), 4, K))
  iters <- matrix(NA_integer_, 5, K)
  for (k in 1:5) {
    d  <- trl[fold == k]
    rows_k <- match(unique(d$No), nos_tr)
    dtr_all <- trl[fold != k]
    cases <- unique(dtr_all$Case)
    for (b in seq_len(K)) {
      set.seed(SEEDS[b] + 17L * k)                 # distinct carve per (fold, bag)
      es_cases <- sample(cases, round(0.1 * length(cases)))
      fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases], p)
      perseed[rows_k, , b] <- softmax_by_task(raw_pred(fit, d))
      bi <- get_best_iter(fit); iters[k, b] <- if (is.null(bi)) NA_integer_ else bi
    }
    cat(sprintf("  fold %d  iters %s\n", k, paste(iters[k, ], collapse = "/"))); flush.console()
  }
  ll_seed <- apply(perseed, 3, function(P) logloss(ytr, P))
  P_arith <- apply(perseed, c(1, 2), mean)
  P_arith <- P_arith / rowSums(P_arith)
  Lg <- apply(log(pmax(perseed, 1e-12)), c(1, 2), mean)
  P_geo <- exp(Lg - apply(Lg, 1, max)); P_geo <- P_geo / rowSums(P_geo)
  cat(sprintf("  per-seed OOF: %s\n", paste(sprintf("%.5f", ll_seed), collapse = " ")))
  cat(sprintf("  seed mean %.5f  SD %.5f   <-- seed-to-seed variance\n",
              mean(ll_seed), sd(ll_seed)))
  cat(sprintf("  BAGGED arithmetic %.5f   (geometric %.5f, diagnostic only)\n",
              logloss(ytr, P_arith), logloss(ytr, P_geo)))
  saveRDS(data.table(No = nos_tr, p1 = P_arith[,1], p2 = P_arith[,2],
                     p3 = P_arith[,3], p4 = P_arith[,4]),
          sprintf("model/artifacts/oof_xgb_%s.rds", nm))

  # ---- test refit: K fits, DISTINCT carves, trained on all training respondents
  cases <- unique(trl$Case); Pt <- array(NA_real_, dim = c(length(nos_te), 4, K))
  for (b in seq_len(K)) {
    set.seed(SEEDS[b] + 977L)
    es_cases <- sample(cases, round(0.1 * length(cases)))
    fit <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases], p)
    Pt[, , b] <- softmax_by_task(raw_pred(fit, tel))
  }
  Pte <- apply(Pt, c(1, 2), mean); Pte <- Pte / rowSums(Pte)
  stopifnot(nrow(Pte) == 4997, all(abs(rowSums(Pte) - 1) < 1e-9))
  saveRDS(data.table(No = nos_te, p1 = Pte[,1], p2 = Pte[,2], p3 = Pte[,3], p4 = Pte[,4]),
          sprintf("model/artifacts/test_xgb_%s.rds", nm))
  cat("  test refit done (", K, "carves averaged; every respondent trained on in",
      K - 1, "of", K, "fits)\n")
  list(ll_seed = ll_seed, bagged = logloss(ytr, P_arith),
       bagged_geo = logloss(ytr, P_geo), iters = iters)
}

res <- list()
for (nm in names(CONFIGS)) {
  res[[nm]] <- run_bag(nm, CONFIGS[[nm]])
  saveRDS(res, "experiments/iter12_bagged_configs/results.rds")
}

cat("\n================ SUMMARY ================\n")
for (nm in names(res)) {
  r <- res[[nm]]
  cat(sprintf("%-6s single-fit mean %.5f (SD %.5f)   BAGGED %.5f   gain %+.5f\n",
              nm, mean(r$ll_seed), sd(r$ll_seed), r$bagged, r$bagged - mean(r$ll_seed)))
}
cat(sprintf("\nconfig gap |bagA - bagB| = %.5f   (registered prediction: < 0.002)\n",
            abs(res$bagA$bagged - res$bagB$bagged)))
cat(sprintf("mono effect  bagC - bagA = %+.5f   (iteration 06 single-carve claim: -0.0018)\n",
            res$bagC$bagged - res$bagA$bagged))
cat("\nNOT promoted. Run model/compare.R, then the convex blend, before adopting.\nOK\n")
