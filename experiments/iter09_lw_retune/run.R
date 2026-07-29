# =============================================================================
# ITERATION 09 — retune the listwise xgboost. HYPOTHESIS WRITTEN BEFORE RUNNING.
#
# HYPOTHESIS: `xgb_lw`'s hyperparameters (eta 0.05, depth 6, min_child_weight 10,
# colsample_bytree 0.8, lambda 1 -- model/03_xgb_listwise.R:64-65) were inherited
# from the POINTWISE objective and were never re-searched after the objective
# changed to listwise softmax. Round counts already jumped 250 -> 600 when the
# objective changed, which is direct evidence the optimum moved. An independent
# 40-config search in the parallel track (different feature set, different folds)
# consistently preferred MUCH heavier regularisation -- min_child_weight 80,
# colsample_bytree 0.30, lambda 10 -- worth -0.00214 there, confirmed on 5 fresh
# fold partitions (p = 0.0001) and surviving a Hansen SPA correction (p = 0.006).
# Crucially it helped under the EXACT softmax objective too, so the result is
# objective-robust rather than an artefact of their approximation.
#
# I expect the DIRECTION (heavier regularisation) to transfer, not the literal
# config, because this feature set differs: it adds ENC_COLS (design-encoding
# shares) and `price_min_rival`, so colsample 0.30 samples a different column mix.
#
# PRE-REGISTERED GRID (fixed before any result is seen, so the winner cannot be
# cherry-picked): a 2x2x2x2 factorial bracketing incumbent against donor winner,
#   max_depth {5,6} x min_child_weight {30,80} x colsample_bytree {0.30,0.50}
#   x lambda {3,10}
# plus two controls: the incumbent verbatim, and the donor winner verbatim.
# eta 0.04, subsample 0.85, alpha 1 held fixed at the donor's low-sensitivity values.
#
# EARLY STOPPING: kept, NOT replaced by fixed rounds -- switching would be a second
# simultaneous change and would break comparability with the incumbent's fitting
# protocol. Patience raised 100 -> 150 for ALL configs including the controls, so
# patience is not confounded with the params under test: eta 0.04 learns more
# slowly than 0.05, giving flatter validation curves, and a premature stop would
# bias systematically against the heavy-regularisation region being tested.
#
# ONE CHANGE UNDER TEST: the parameter vector. Features, folds, objective, seeds,
# ES protocol and loop structure are byte-identical to model/03_xgb_listwise.R.
#
# Emits per-config OOF matrices to results.rds for PAIRED comparison. It does NOT
# write oof_xgb_lw2.rds -- promoting a winner is a separate, deliberate step after
# model/compare.R and fresh-partition confirmation.
#
# Run: Rscript experiments/iter09_lw_retune/run.R
# Runtime: ~18 configs x ~10-14 min = 3-4 h. Run in the background.
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
setorder(trl, No, alt)
tel <- long[is_test == TRUE]; setorder(tel, No, alt)
apply_design_encoding(trl, tel)

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, ENC_COLS)

# ---- objective (identical to model/03_xgb_listwise.R:46-62) ------------------
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
  p <- if (!is.null(bi))
    tryCatch(predict(fit, dm, iterationrange = c(1, bi + 1), outputmargin = TRUE),
             error = function(e) predict(fit, dm, outputmargin = TRUE))
  else predict(fit, dm, outputmargin = TRUE)
  as.vector(p)
}
train_one <- function(dtr, des, params, patience) {
  a <- list(params = params,
            data = xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen)),
            nrounds = 4000, early_stopping_rounds = patience, verbose = 0,
            obj = obj_listwise, maximize = FALSE)
  m_es <- xgb.DMatrix(as.matrix(des[, ..feat]), label = as.numeric(des$chosen))
  if (new_api) { a$evals <- list(es = m_es); a$custom_metric <- eval_tasklogloss }
  else         { a$watchlist <- list(es = m_es); a$feval <- eval_tasklogloss }
  do.call(xgb.train, a)
}

# ---- pre-registered grid ----------------------------------------------------
g <- CJ(max_depth = c(5L, 6L), min_child_weight = c(30, 80),
        colsample_bytree = c(0.30, 0.50), lambda = c(3, 10))
g[, `:=`(eta = 0.04, subsample = 0.85, alpha = 1, tag = sprintf(
  "d%d_mcw%d_cs%.2f_l%d", max_depth, min_child_weight, colsample_bytree, lambda))]
ctrl <- rbindlist(list(
  data.table(max_depth=6L, min_child_weight=10, colsample_bytree=0.80, lambda=1,
             eta=0.05, subsample=0.80, alpha=0, tag="CTRL_incumbent"),
  data.table(max_depth=5L, min_child_weight=80, colsample_bytree=0.30, lambda=10,
             eta=0.04, subsample=0.85, alpha=1, tag="CTRL_donor")))
grid <- rbind(ctrl, g)
cat("configs to run:", nrow(grid), "\n")

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
PATIENCE <- 150L
res <- list()

for (i in seq_len(nrow(grid))) {
  gi <- grid[i]
  params <- list(eta = gi$eta, max_depth = gi$max_depth,
                 min_child_weight = gi$min_child_weight, subsample = gi$subsample,
                 colsample_bytree = gi$colsample_bytree, alpha = gi$alpha,
                 lambda = gi$lambda, base_score = 0, nthread = 0)
  t0 <- Sys.time()
  set.seed(123)                      # identical ES splits across configs => paired
  oof_list <- list(); iters <- integer(5)
  for (k in 1:5) {
    dtr_all <- trl[fold != k]
    es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
    fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases],
                     params, PATIENCE)
    d <- trl[fold == k]
    P <- softmax_by_task(raw_pred(fit, d))
    oof_list[[k]] <- data.table(No = unique(d$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
    bi <- get_best_iter(fit); iters[k] <- if (is.null(bi)) NA_integer_ else bi
  }
  oof <- rbindlist(oof_list); setorder(oof, No)
  ll <- logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)]))
  mins <- round(as.numeric(difftime(Sys.time(), t0, units="mins")), 1)
  res[[gi$tag]] <- list(tag = gi$tag, params = params, oof = oof, logloss = ll,
                        best_iters = iters)
  cat(sprintf("[%2d/%2d] %-22s OOF %.5f  iters %s  (%.1f min)\n",
              i, nrow(grid), gi$tag, ll, paste(iters, collapse=","), mins))
  flush.console()
  saveRDS(res, "experiments/iter09_lw_retune/results.rds")   # checkpoint each config
}

cat("\n=== ranked ===\n")
tb <- data.table(tag = names(res), OOF = sapply(res, `[[`, "logloss"))[order(OOF)]
print(tb)
cat("\nincumbent (CTRL_incumbent):", round(res$CTRL_incumbent$logloss, 5), "\n")
cat("best:", tb$tag[1], round(tb$OOF[1], 5),
    "  delta:", round(tb$OOF[1] - res$CTRL_incumbent$logloss, 5), "\n")
cat("\nNOT promoted. Run model/compare.R + fresh-partition check before adopting.\nOK\n")
