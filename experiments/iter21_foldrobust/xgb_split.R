# =============================================================================
# ITERATION 21 — listwise xgboost refit under an ALTERNATIVE respondent-grouped split.
#
# This is a verbatim copy of model/03_xgb_listwise.R (variant "lw2") and
# experiments/iter08_mono_tuned/run.R (variant "mono"); the ONLY differences are
#   1. the fold file is a command-line argument (folds_b.rds / folds_c.rds),
#   2. the artifact names get a _<split> suffix,
#   3. nthread = 3 (shared machine), which cannot change the fitted model.
# The two variants differ from each other only by monotone_constraints, exactly as
# in production.
#
#   Rscript experiments/iter21_foldrobust/xgb_split.R <lw2|mono> <b|c>
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: xgb_split.R <lw2|mono> <b|c>")
VARIANT <- args[1]; SPLIT <- args[2]
stopifnot(VARIANT %in% c("lw2", "mono"), SPLIT %in% c("b", "c"))
FOLDFILE <- sprintf("model/artifacts/folds_%s.rds", SPLIT)
NAME <- sprintf("xgb_%s_%s", VARIANT, SPLIT)
cat("=== variant", VARIANT, " split", SPLIT, " -> oof_", NAME, ".rds ===\n", sep = "")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS(FOLDFILE)          # NOT model/artifacts/folds.rds
wide  <- readRDS("model/artifacts/wide.rds")
stopifnot(nrow(folds) == 21565, all(folds[, uniqueN(fold), by = Case]$V1 == 1))

long <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)     # fold-aware: re-derived under the NEW split

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

params <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = 3)
if (VARIANT == "mono") {
  mono <- rep(0, length(feat))
  mono[match("Price",   feat)] <- -1
  mono[match("Price_c", feat)] <- -1
  params$monotone_constraints <- mono
  cat("constrained features:", paste(feat[mono != 0], collapse = ", "), "\n")
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
set.seed(123)                       # same seed as production
oof_list <- list()
t0 <- proc.time()
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  d <- trl[fold == k]
  P <- softmax_by_task(raw_pred(fit, d))
  oof_list[[k]] <- data.table(No = unique(d$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
  cat("fold", k, "best_iter", get_best_iter(fit),
      " fold logloss", round(logloss(unique(d[, .(No, y)])[order(No), y],
                                     as.matrix(oof_list[[k]][, .(p1,p2,p3,p4)])), 5),
      " elapsed", round((proc.time() - t0)[3] / 60, 1), "min\n")
}
oof <- rbindlist(oof_list); setorder(oof, No)
stopifnot(nrow(oof) == 21565, identical(oof$No, sort(unique(trl$No))), !anyNA(oof))
ll <- logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)]))
cat(sprintf("\n>>> %s OOF logloss: %.5f\n", NAME, ll))
saveRDS(oof, sprintf("model/artifacts/oof_%s.rds", NAME))

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, !anyNA(tp), all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, sprintf("model/artifacts/test_%s.rds", NAME))
cat("OK -- wrote oof_", NAME, ".rds and test_", NAME, ".rds\n", sep = "")
