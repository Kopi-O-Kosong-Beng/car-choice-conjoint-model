# =============================================================================
# ITERATION 08b — Monotone constraint on Price_c ONLY (raw Price left free).
#
# HYPOTHESIS: the run.R header names this as the fix if the two-constraint version
# fails. The all-zero "none of these" alternative has Price = 0, so it is ALWAYS the
# cheapest option in its choice set. A monotone constraint on raw Price therefore
# forces "score is non-increasing in raw Price" across the none-vs-bundle boundary,
# where the utility difference is the none-constant, not a price effect. The model
# then cannot give the none-option a lower score than a cheap bundle purely on the
# Price split, and has to route the none-constant through alt4 / richness / lvlsum
# instead. Price_c (the within-task centered price contrast) has no such problem:
# within a task it is a genuine relative-expensiveness signal, and demanding utility
# be non-increasing in it is exactly the intended economics.
#
# ONE CHANGE UNDER TEST vs experiments/iter08_mono_tuned/run.R: the Price entry of
# `mono` is 0 instead of -1. Everything else -- features, folds, seeds, objective,
# early stopping, hyperparameters -- is identical.
#
# HOW TO RUN
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter08_mono_tuned/run_pricec_only.R
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/compare.R xgb_lw2 xgb_monoc
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
  M <- M - apply(M, 1, max)
  E <- exp(M); E / rowSums(E)
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

# ONLY the within-task price contrast is constrained; raw Price is free so the tree
# can still use it to express the none-constant.
mono <- rep(0, length(feat))
mono[match("Price_c", feat)] <- -1
cat("constrained features:", paste(feat[mono != 0], collapse = ", "), "\n")

params <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = 4,
               monotone_constraints = mono)

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
train_one <- function(dtr, des) {
  a <- list(params = params,
            data = xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen)),
            nrounds = 5000, early_stopping_rounds = 150, verbose = 0,
            obj = obj_listwise, maximize = FALSE)
  m_es <- xgb.DMatrix(as.matrix(des[, ..feat]), label = as.numeric(des$chosen))
  if (new_api) { a$evals <- list(es = m_es); a$custom_metric <- eval_tasklogloss }
  else         { a$watchlist <- list(es = m_es); a$feval <- eval_tasklogloss }
  do.call(xgb.train, a)
}

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
set.seed(123)                      # same seed as production, so folds/ES splits match
oof_list <- list()
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  d <- trl[fold == k]
  P <- softmax_by_task(raw_pred(fit, d))
  oof_list[[k]] <- data.table(No = unique(d$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
  cat("fold", k, "best_iter", get_best_iter(fit), "\n")
}
oof <- rbindlist(oof_list); setorder(oof, No)
ll <- logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)]))
cat(sprintf("\n>>> XGB listwise + monotone Price_c ONLY OOF: %.5f\n", ll))
cat("    production (tuned, unconstrained): 1.14152\n")
stopifnot(nrow(oof) == 21565, all(abs(rowSums(as.matrix(oof[, -1])) - 1) < 1e-9))
saveRDS(oof, "model/artifacts/oof_xgb_monoc.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_monoc.rds")
cat("OK -- wrote oof_xgb_monoc.rds and test_xgb_monoc.rds\n")
