# =============================================================================
# ITERATION 15, STEP 3 — test-set refit only.
#
# run.R was killed by the harness after it had written oof_xgb_resenc3.rds (the
# decision number) but before the full-data refit that produces the test
# predictions. This script reproduces ONLY that final block: identical feature
# construction, identical hyperparameters, identical seed (7). It writes
# model/artifacts/test_xgb_resenc3.rds and nothing else.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter15_nested_resenc/test_refit.R
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

nb <- readRDS("experiments/iter15_nested_resenc/nested_base.rds")
stopifnot(identical(as.numeric(nb$No), as.numeric(trl$No)),
          identical(as.numeric(nb$alt), as.numeric(trl$alt)),
          identical(as.numeric(nb$fold), as.numeric(trl$fold)))
NB <- nb$NB

base_oof <- readRDS("model/artifacts/oof_mnl_pw.rds"); setorder(base_oof, No)
pm <- melt(base_oof, id.vars = "No", measure.vars = c("p1","p2","p3","p4"),
           variable.name = "altf", value.name = "p_model")
pm[, alt := as.integer(sub("p", "", as.character(altf)))][, altf := NULL]
trl <- merge(trl, pm, by = c("No","alt"), all.x = TRUE)
setorder(trl, No, alt)
trl[, resid_oof := as.numeric(chosen) - p_model]

RES_ALPHAS <- c(1, 5, 20)
RES_COLS   <- c(paste0("resid_a", RES_ALPHAS), "resid_n")
encode_resid <- function(ref, target) {
  agg <- ref[, .(rs = sum(resid), n = .N), by = .(dkey, alt)]
  t2 <- copy(target)[, .rid := .I][, .(.rid, dkey, alt)]
  t2 <- merge(t2, agg, by = c("dkey", "alt"), all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, rs = 0)]
  for (a in RES_ALPHAS) t2[, paste0("resid_a", a) := rs / (n + a)]
  t2[, resid_n := n]
  setorder(t2, .rid)
  t2[, .SD, .SDcols = RES_COLS]
}
for (cc in RES_COLS) { trl[, (cc) := NA_real_]; tel[, (cc) := NA_real_] }
for (k in 1:5) {
  ii  <- which(trl$fold != k)
  ref <- copy(trl[ii]); ref[, resid := as.numeric(chosen) - NB[ii, k]]
  e <- encode_resid(ref, trl[fold == k])
  for (cc in RES_COLS) set(trl, which(trl$fold == k), cc, e[[cc]])
}
ref_all <- copy(trl)[, resid := resid_oof]
e <- encode_resid(ref_all, tel)
for (cc in RES_COLS) set(tel, seq_len(nrow(tel)), cc, e[[cc]])
stopifnot(!anyNA(trl[, ..RES_COLS]), !anyNA(tel[, ..RES_COLS]))

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, ENC_COLS,
          RES_COLS)
softmax_by_task <- function(scores) {
  M <- matrix(scores, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
  E <- exp(M); E / rowSums(E)
}
obj_listwise <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(softmax_by_task(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
eval_tasklogloss <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); P <- softmax_by_task(preds)
  Y <- matrix(y, ncol = 4, byrow = TRUE)
  list(metric = "tasklogloss", value = -mean(log(pmax(P[Y == 1], 1e-15))))
}
params <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = 4)
new_api <- packageVersion("xgboost") >= "2.1.0"
get_best_iter <- function(fit) {
  bi <- tryCatch(xgb.attr(fit, "best_iteration"), error = function(e) NULL)
  if (is.null(bi)) bi <- tryCatch(fit$best_iteration, error = function(e) NULL)
  if (is.null(bi) || is.na(suppressWarnings(as.numeric(bi)))) NULL else as.integer(bi)
}
raw_pred <- function(fit, d) {
  dm <- xgb.DMatrix(as.matrix(d[, ..feat])); bi <- get_best_iter(fit)
  p <- if (!is.null(bi)) tryCatch(predict(fit, dm, iterationrange = c(1, bi + 1), outputmargin = TRUE),
                                  error = function(e) predict(fit, dm, outputmargin = TRUE))
       else predict(fit, dm, outputmargin = TRUE)
  as.vector(p)
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

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_resenc3.rds")
cat("full-fit best_iter", get_best_iter(fit_full), "\n")
imp <- as.data.frame(xgb.importance(model = fit_full))
print(head(imp[, c("Feature", "Gain")], 12))
cat("\nresidual-feature gains:\n")
print(imp[imp$Feature %in% RES_COLS, c("Feature", "Gain")])
cat("rank of each residual feature:",
    paste(RES_COLS, match(RES_COLS, imp$Feature), collapse = "  "), "\n")
cat("TEST_REFIT_DONE\n")
