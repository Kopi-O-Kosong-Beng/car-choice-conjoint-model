suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
wide  <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")

acols <- as.vector(outer(setdiff(ATTRS, "Price"), 1:3, paste0))
pcols <- paste0("Price", 1:3)
wide[, `:=`(pd12 = Price1 - Price2, pd13 = Price1 - Price3, pd23 = Price2 - Price3,
            pmin123 = pmin(Price1, Price2, Price3), pmax123 = pmax(Price1, Price2, Price3))]
for (j in 1:3) wide[, paste0("rich", j) := rowSums(.SD != 0), .SDcols = paste0(setdiff(ATTRS, "Price"), j)]
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(acols, pcols, "pd12","pd13","pd23","pmin123","pmax123", paste0("rich", 1:3), "Task", demo)

trw <- merge(wide[is_test == FALSE], folds[, .(No, fold)], by = "No")
tew <- wide[is_test == TRUE]
params <- list(objective = "multi:softprob", num_class = 4, eta = 0.05, max_depth = 6,
               min_child_weight = 10, subsample = 0.8, colsample_bytree = 0.8,
               eval_metric = "mlogloss", nthread = 0)
use_evals <- packageVersion("xgboost") >= "2.1.0"

get_best_iter <- function(fit) {
  bi <- tryCatch(xgb.attr(fit, "best_iteration"), error = function(e) NULL)
  if (is.null(bi)) bi <- tryCatch(fit$best_iteration, error = function(e) NULL)
  if (is.null(bi) || is.na(suppressWarnings(as.numeric(bi)))) NULL else as.integer(bi)
}
train_one <- function(dtr, des) {
  m_tr <- xgb.DMatrix(as.matrix(dtr[, ..feat]), label = dtr$y - 1)
  m_es <- xgb.DMatrix(as.matrix(des[, ..feat]), label = des$y - 1)
  a <- list(params = params, data = m_tr, nrounds = 3000,
            early_stopping_rounds = 100, verbose = 0)
  if (use_evals) a$evals <- list(es = m_es) else a$watchlist <- list(es = m_es)
  do.call(xgb.train, a)
}
pred_mat <- function(fit, d) {
  dm <- xgb.DMatrix(as.matrix(d[, ..feat]))
  bi <- get_best_iter(fit)
  p <- if (!is.null(bi)) tryCatch(predict(fit, dm, iterationrange = c(1, bi + 1)),
                                  error = function(e) predict(fit, dm)) else predict(fit, dm)
  if (is.matrix(p)) p else matrix(p, ncol = 4, byrow = TRUE)  # xgboost >=3 returns a matrix
}

set.seed(123)
oof <- data.table(No = trw$No, p1 = NA_real_, p2 = NA_real_, p3 = NA_real_, p4 = NA_real_)
for (k in 1:5) {
  dtr_all <- trw[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  P <- pred_mat(fit, trw[fold == k])
  oof[match(trw[fold == k, No], No), `:=`(p1 = P[, 1], p2 = P[, 2], p3 = P[, 3], p4 = P[, 4])]
  cat("fold", k, "best_iter", get_best_iter(fit), "\n")
}
setorder(oof, No)
cat(">>> XGB-wide OOF logloss:", round(logloss(trw[order(No), y], as.matrix(oof[, .(p1, p2, p3, p4)])), 5), "\n")
saveRDS(oof, "model/artifacts/oof_xgb_wide.rds")

es_cases <- sample(unique(trw$Case), round(0.1 * uniqueN(trw$Case)))
fit_full <- train_one(trw[!Case %in% es_cases], trw[Case %in% es_cases])
P <- pred_mat(fit_full, tew)
tp <- data.table(No = tew$No, p1 = P[, 1], p2 = P[, 2], p3 = P[, 3], p4 = P[, 4])
setorder(tp, No)
saveRDS(tp, "model/artifacts/test_xgb_wide.rds")
cat("OK\n")
