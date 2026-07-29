suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds_b.rds")

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo)

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
params <- list(objective = "binary:logistic", eta = 0.05, max_depth = 6,
               min_child_weight = 10, subsample = 0.8, colsample_bytree = 0.8,
               eval_metric = "logloss", nthread = 0)
use_evals <- packageVersion("xgboost") >= "2.1.0"

get_best_iter <- function(fit) {
  bi <- tryCatch(xgb.attr(fit, "best_iteration"), error = function(e) NULL)
  if (is.null(bi)) bi <- tryCatch(fit$best_iteration, error = function(e) NULL)
  if (is.null(bi) || is.na(suppressWarnings(as.numeric(bi)))) NULL else as.integer(bi)
}
pred_best <- function(fit, dm) {
  bi <- get_best_iter(fit)
  if (!is.null(bi)) tryCatch(predict(fit, dm, iterationrange = c(1, bi + 1)),
                             error = function(e) predict(fit, dm))
  else predict(fit, dm)
}
train_one <- function(dtr, dva_es, nrounds = 3000) {
  m_tr <- xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen))
  m_es <- xgb.DMatrix(as.matrix(dva_es[, ..feat]), label = as.numeric(dva_es$chosen))
  a <- list(params = params, data = m_tr, nrounds = nrounds,
            early_stopping_rounds = 100, verbose = 0)
  if (use_evals) a$evals <- list(es = m_es) else a$watchlist <- list(es = m_es)
  do.call(xgb.train, a)
}
predict_norm <- function(fit, d) {
  p <- pred_best(fit, xgb.DMatrix(as.matrix(d[, ..feat])))
  data.table(No = d$No, alt = d$alt, p = norm_by_group(p, d$No))
}

set.seed(123)
oof_list <- list(); best_iters <- integer(5)
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  bi <- get_best_iter(fit); best_iters[k] <- if (is.null(bi)) NA_integer_ else bi
  oof_list[[k]] <- predict_norm(fit, trl[fold == k])
  cat("fold", k, "best_iter", best_iters[k], "\n")
}
oof_l <- rbindlist(oof_list)
oof <- dcast(oof_l, No ~ alt, value.var = "p")
setnames(oof, c("No", "p1", "p2", "p3", "p4")); setorder(oof, No)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
cat(">>> XGB-long OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5), "\n")
saveRDS(oof, "/private/tmp/claude-501/-Users-sheil-Desktop-SUTD-Y2-T5-40-016-The-Analytics-Edge-TAE-R-izzlers/87738d43-6ae7-4a89-b6d6-d01a77b3f47c/scratchpad/oof_xgb_long_b.rds")

cat("DONE xgb_long_b
")
