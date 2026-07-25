# =============================================================================
# ITERATION 08 — Monotone price constraint COMBINED with the tuned hyperparameters.
#
# HYPOTHESIS: iteration 06 tested these separately and both beat the old settings:
#     base (eta .05, depth 6)                    1.14477
#     mono (monotone price, base settings)       1.14298
#     slow_deep (eta .03, depth 8, mcw 20)       1.14152   <- now in production
# They were never combined. They address different things -- the constraint injects a
# structural prior (utility cannot increase with price), the tuning gives the trees more
# capacity for the interaction-shaped contrasts the listwise objective rewards -- so
# there is no obvious reason they should conflict.
#
# The constraint should matter most where training support is thinnest, which is the
# rich end of the distribution, and the test population is twice as wealthy as training.
# So a gain here should transfer BETTER than average to the leaderboard. Check that with
# model/shift_audit.R afterwards, not just the headline number.
#
# ONE CHANGE UNDER TEST: adding monotone_constraints to the production configuration.
# Everything else -- features, folds, seeds, objective, early stopping -- is identical to
# model/03_xgb_listwise.R.
#
# HOW TO RUN
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter08_mono_tuned/run.R
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/compare.R xgb_lw2 xgb_mono
# If the paired test says CONFIRMED, add `xgb_mono` to model/members.txt and run:
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/run_all.R blend submit
# Then record the result and a reflection in EXPERIMENTS.md -- including if it fails.
#
# IF IT SCORES WORSE: the likely cause is that the all-zero "none" alternative has
# Price = 0, so it is always the cheapest option in its choice set, and forcing utility
# to be non-increasing in raw Price fights the none-constant. The fix to try next is
# constraining ONLY Price_c (the within-task contrast) and leaving raw Price free --
# set the Price entry of `mono` below to 0 and rerun.
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

# utility must not increase with price; 0 = unconstrained for every other feature
mono <- rep(0, length(feat))
mono[match("Price",   feat)] <- -1
mono[match("Price_c", feat)] <- -1
cat("constrained features:", paste(feat[mono != 0], collapse = ", "), "\n")

# production hyperparameters (iteration 06 "slow_deep") + the constraint
params <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = 0,
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
cat(sprintf("\n>>> XGB listwise + monotone price OOF: %.5f\n", ll))
cat("    production (tuned, unconstrained): 1.14152\n")
cat("    monotone with OLD hyperparameters: 1.14298\n")
cat(if (ll < 1.14152) "    -> candidate; confirm with model/compare.R xgb_lw2 xgb_mono\n"
    else "    -> no gain. See the header for the Price_c-only variant to try next.\n")
saveRDS(oof, "model/artifacts/oof_xgb_mono.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_mono.rds")
cat("OK -- wrote oof_xgb_mono.rds and test_xgb_mono.rds\n")
