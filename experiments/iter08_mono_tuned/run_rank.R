# =============================================================================
# ITERATION 08c — add a monotone constraint on price_rank ON TOP of xgb_mono.
#
# CONTEXT: run.R (constrain Price and Price_c, -1 each) scored 1.13980 OOF, beating
# production xgb_lw2 = 1.14152. run_pricec_only.R (constrain Price_c alone) scored
# 1.14123, i.e. no real gain -- so the header's predicted failure mode (the raw-Price
# constraint fighting the all-zero none-option) did NOT materialise. The stronger,
# more complete constraint is the better one. That suggests the prior is doing real
# work rather than binding awkwardly, and invites one more of the same kind.
#
# HYPOTHESIS: price_rank is rank(Price) within the choice set, 1 = cheapest
# (model/00_load.R:38). A higher rank means the alternative is more expensive
# RELATIVE TO ITS RIVALS, which should not raise its utility -- so the economically
# correct sign is -1, exactly as for Price and Price_c. Adding it makes the model's
# price behaviour monotone in all three of its price representations (absolute,
# task-centered, ordinal-relative) instead of two, closing the remaining route by
# which a tree can fit a non-monotone price artefact in a thin region.
#
# ONE CHANGE UNDER TEST vs experiments/iter08_mono_tuned/run.R: the price_rank entry
# of `mono` is -1 instead of 0. Everything else -- features, folds, seeds, objective,
# early stopping, hyperparameters -- is identical.
#
# EXPECTED: small. Price_rank is a coarser, partly redundant encoding of information
# already carried by Price_c, so most of the constraint's content is already imposed.
# Anything from 1.138 to 1.141 is plausible; a paired z of ~0 against xgb_mono would
# mean the constraint is redundant, which is itself a clean finding.
#
# HOW TO RUN
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter08_mono_tuned/run_rank.R
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/compare.R xgb_mono xgb_monor
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

mono <- rep(0, length(feat))
mono[match("Price",      feat)] <- -1
mono[match("Price_c",    feat)] <- -1
mono[match("price_rank", feat)] <- -1
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
cat(sprintf("\n>>> XGB listwise + monotone Price/Price_c/price_rank OOF: %.5f\n", ll))
cat("    xgb_mono  (Price + Price_c):        1.13980\n")
cat("    xgb_monoc (Price_c only):           1.14123\n")
cat("    xgb_lw2   (production, no constr.): 1.14152\n")
stopifnot(nrow(oof) == 21565, all(abs(rowSums(as.matrix(oof[, -1])) - 1) < 1e-9))
saveRDS(oof, "model/artifacts/oof_xgb_monor.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_monor.rds")
cat("OK -- wrote oof_xgb_monor.rds and test_xgb_monor.rds\n")
