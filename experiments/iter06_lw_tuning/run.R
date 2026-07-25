# =============================================================================
# ITERATION 06 — Re-tune the listwise model, and test a monotone price constraint.
#
# WHY: iteration 03 changed the objective but kept hyperparameters chosen for the
# OLD pointwise objective. Best-iteration counts jumped from ~250 to 400-600, which
# says the optimum moved -- probably toward a lower learning rate and/or different
# depth. Separately, economics says utility must DECREASE in price; imposing that as
# a monotone constraint is a free prior that should help most where training support
# is thinnest, which is exactly the richer test population.
#
# Each config gets a full honest 5-fold OOF on the fixed folds. All configs are
# reported, not just the winner, so the selection is visible and auditable.
# NOTE: picking the best of N configs on the same folds is itself a mild selection
# bias. Treat the winner's OOF as slightly optimistic and confirm with compare.R
# against xgb_lw before promoting it.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter06_lw_tuning/run.R
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

# monotone constraint vector: utility must not increase in price
mono <- rep(0, length(feat))
mono[match(c("Price", "Price_c"), feat)] <- -1

CONFIGS <- list(
  base      = list(eta=0.05, max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8),
  slow_deep = list(eta=0.03, max_depth=8, min_child_weight=20, subsample=0.8, colsample_bytree=0.8),
  slow_shal = list(eta=0.03, max_depth=4, min_child_weight=5,  subsample=0.9, colsample_bytree=0.7),
  reg       = list(eta=0.04, max_depth=6, min_child_weight=30, subsample=0.7, colsample_bytree=0.6,
                   lambda=3, alpha=0.5),
  mono      = list(eta=0.05, max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                   monotone_constraints = mono)
)

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
run_config <- function(cfg) {
  p <- c(cfg, list(base_score = 0, nthread = 0))
  train_one <- function(dtr, des) {
    a <- list(params = p,
              data = xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen)),
              nrounds = 5000, early_stopping_rounds = 150, verbose = 0,
              obj = obj_listwise, maximize = FALSE)
    m_es <- xgb.DMatrix(as.matrix(des[, ..feat]), label = as.numeric(des$chosen))
    if (new_api) { a$evals <- list(es = m_es); a$custom_metric <- eval_tasklogloss }
    else         { a$watchlist <- list(es = m_es); a$feval <- eval_tasklogloss }
    do.call(xgb.train, a)
  }
  set.seed(123)
  oof_list <- list(); iters <- integer(5)
  for (k in 1:5) {
    dtr_all <- trl[fold != k]
    es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
    fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
    bi <- get_best_iter(fit); iters[k] <- if (is.null(bi)) NA_integer_ else bi
    d <- trl[fold == k]; P <- softmax_by_task(raw_pred(fit, d))
    oof_list[[k]] <- data.table(No = unique(d$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
  }
  oof <- rbindlist(oof_list); setorder(oof, No)
  list(oof = oof, ll = logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)])),
       iters = iters, train = train_one)
}

results <- list()
for (nm in names(CONFIGS)) {
  t0 <- Sys.time()
  r <- run_config(CONFIGS[[nm]])
  results[[nm]] <- r
  cat(sprintf("%-10s OOF %.5f   iters %s   [%.1f min]\n", nm, r$ll,
              paste(r$iters, collapse = "/"),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  flush.console()
}
cat("\n(reference: xgb_lw with inherited hyperparameters = 1.14477)\n")

best <- names(which.min(sapply(results, function(x) x$ll)))
cat(">>> best config:", best, "at", round(results[[best]]$ll, 5), "\n")
saveRDS(results[[best]]$oof, "model/artifacts/oof_xgb_lw2.rds")
saveRDS(lapply(results, function(x) list(ll = x$ll, iters = x$iters)),
        "experiments/iter06_lw_tuning/all_results.rds")

# refit winner on all training data for the test predictions
set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- results[[best]]$train(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_lw2.rds")
cat("OK -- promoted as xgb_lw2; confirm with: model/compare.R xgb_lw xgb_lw2\n")
