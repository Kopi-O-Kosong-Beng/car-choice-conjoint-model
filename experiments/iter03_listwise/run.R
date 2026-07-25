# =============================================================================
# ITERATION 03 — Listwise softmax objective for xgboost.
#
# HYPOTHESIS: our xgboost is optimizing the wrong loss. It treats every
# alternative as an independent binary "was this chosen?" problem, then we
# normalize the four probabilities afterwards. But the metric is a softmax over
# the four alternatives in a choice set: only RELATIVE utility matters, and a
# model that inflates all four equally is penalized by the binary objective for
# no reason. Supplying a custom gradient makes the training loss identical to
# the competition metric.
#
# The gradient is clean: with p = softmax(scores within task),
#     dL/ds_j = p_j - y_j      and      d2L/ds_j2 = p_j(1 - p_j).
# This is the same algebra as multiclass softmax, but grouped DOWN rows
# (4 consecutive rows = 1 task) instead of across columns.
#
# Held constant vs iteration 01 (features, folds, hyperparameters) so the only
# change under test is the objective. Baseline to beat: xgb_de = 1.15055.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter03_listwise/run.R
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

# ---- the custom objective ----------------------------------------------------
# CRITICAL: rows must be sorted by (No, alt) so each task is 4 consecutive rows.
softmax_by_task <- function(scores) {
  M <- matrix(scores, ncol = 4, byrow = TRUE)
  M <- M - apply(M, 1, max)              # stabilize before exponentiating
  E <- exp(M)
  E / rowSums(E)
}
obj_listwise <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  p <- as.vector(t(softmax_by_task(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
eval_tasklogloss <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  P <- softmax_by_task(preds)
  Y <- matrix(y, ncol = 4, byrow = TRUE)
  list(metric = "tasklogloss", value = -mean(log(pmax(P[Y == 1], 1e-15))))
}

params <- list(eta = 0.05, max_depth = 6, min_child_weight = 10,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = 0)
new_api <- packageVersion("xgboost") >= "2.1.0"
get_best_iter <- function(fit) {
  bi <- tryCatch(xgb.attr(fit, "best_iteration"), error = function(e) NULL)
  if (is.null(bi)) bi <- tryCatch(fit$best_iteration, error = function(e) NULL)
  if (is.null(bi) || is.na(suppressWarnings(as.numeric(bi)))) NULL else as.integer(bi)
}
raw_pred <- function(fit, d) {
  dm <- xgb.DMatrix(as.matrix(d[, ..feat]))
  bi <- get_best_iter(fit)
  p <- if (!is.null(bi)) tryCatch(predict(fit, dm, iterationrange = c(1, bi + 1), outputmargin = TRUE),
                                  error = function(e) predict(fit, dm, outputmargin = TRUE))
       else predict(fit, dm, outputmargin = TRUE)
  as.vector(p)
}
train_one <- function(dtr, des) {
  a <- list(params = params,
            data = xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen)),
            nrounds = 3000, early_stopping_rounds = 100, verbose = 0,
            obj = obj_listwise, maximize = FALSE)
  m_es <- xgb.DMatrix(as.matrix(des[, ..feat]), label = as.numeric(des$chosen))
  if (new_api) { a$evals <- list(es = m_es); a$custom_metric <- eval_tasklogloss }
  else         { a$watchlist <- list(es = m_es); a$feval <- eval_tasklogloss }
  do.call(xgb.train, a)
}

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
set.seed(123)
oof_list <- list()
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  # keep tasks intact on both sides of the split -- the objective needs whole tasks
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  d <- trl[fold == k]
  P <- softmax_by_task(raw_pred(fit, d))
  oof_list[[k]] <- data.table(No = unique(d$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
  cat("fold", k, "best_iter", get_best_iter(fit), "\n")
}
oof <- rbindlist(oof_list); setorder(oof, No)
cat(">>> XGB listwise OOF logloss:",
    round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5),
    "  (pointwise + same features: 1.15055)\n")
saveRDS(oof, "model/artifacts/oof_xgb_lw.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_lw.rds")
print(head(as.data.frame(xgb.importance(model = fit_full))[, c("Feature", "Gain")], 10))
cat("OK\n")
