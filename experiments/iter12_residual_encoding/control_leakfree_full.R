# =============================================================================
# ITERATION 12 — LEAKAGE CONTROL for the residual encoding.
#
# The main run scored OOF 1.09962 against a 1.14152 incumbent (z = 9.52). That is
# nine times the entire project's accumulated gain, from one derived feature, and
# it contradicts the direct diagnostic in run.R, where adding the same encoding to
# p_model as a correction made every configuration WORSE. Something is leaking.
#
# THE LEAK. The encoding for a target row in fold k is built from reference
# respondents in folds != k -- that rule is honoured. But each reference
# respondent i (in fold j) contributes chosen_i - p_model_i, and p_model_i is
# xgb_lw2's OOF prediction, made by a model trained on all folds EXCEPT j. That
# training set INCLUDES fold k. A depth-8 xgboost with the design-share encoding
# can partially memorise an individual design, so p_model_i on design d already
# embeds what fold-k respondents chose on design d -- including the target's own
# choice. The residual then carries that memorised label with a flipped sign,
# which is exactly why the encoding is NEGATIVELY correlated (-0.07) with the
# target's own held-out residual, and why the least-shrunk feature resid_a1 (the
# one that retains the most memorisation) is the model's #2 feature by gain.
#
# THE CONTROL. Rebuild the identical encoding from mnl_pw instead. mnl_pw is a
# part-worth conditional logit: ~150 GLOBAL coefficients, no design-level
# features, structurally incapable of memorising an individual choice set. The
# leak path is therefore closed while the design-level signal, if any exists, is
# preserved. Everything else -- features, hyperparameters, seeds, objective, fold
# handling -- is byte-identical to run.R.
#
#   If the +0.042 is real design signal  -> this run also gains a lot.
#   If the +0.042 is memorisation leaking -> this run gains ~nothing.
#
# Writes NO artifacts on purpose; it only prints its OOF.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter12_residual_encoding/control_leakfree.R
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

BASE <- "mnl_pw"                       # <- the ONLY change versus run.R
base_oof <- readRDS(sprintf("model/artifacts/oof_%s.rds", BASE)); setorder(base_oof, No)
pm <- melt(base_oof, id.vars = "No", measure.vars = c("p1","p2","p3","p4"),
           variable.name = "altf", value.name = "p_model")
pm[, alt := as.integer(sub("p", "", as.character(altf)))][, altf := NULL]
trl <- merge(trl, pm, by = c("No","alt"), all.x = TRUE)
setorder(trl, No, alt)
trl[, resid := as.numeric(chosen) - p_model]

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
for (k in sort(unique(trl$fold))) {
  idx <- which(trl$fold == k)
  e <- encode_resid(trl[fold != k], trl[fold == k])
  for (cc in RES_COLS) set(trl, idx, cc, e[[cc]])
}
e <- encode_resid(trl, tel)
for (cc in RES_COLS) set(tel, seq_len(nrow(tel)), cc, e[[cc]])

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

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
set.seed(123)
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
cat(">>> CONTROL (residuals from", BASE, ") OOF logloss:",
    round(logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)])), 5), "\n")
cat("    xgb_lw2 incumbent          1.14152\n")
cat("    main run (xgb_lw2 resids)  1.09962   <- suspected leakage\n")
cat("CONTROL_DONE\n")

# ---------------------------------------------------------------------------
# APPENDED: persist artifacts + refit on all training data for test predictions.
# The control measured 1.13721 but never saved anything, so the candidate could
# not be blend-tested. Named xgb_resenc2 to keep it distinct from the leaky
# xgb_resenc artifacts, which are retained only as the record of the measurement.
#
# CAVEAT worth carrying: this is leak-free by ARGUMENT, not by construction. The
# baseline mnl_pw has ~150 global coefficients and no design-level features, so it
# cannot memorise an individual choice set and the leak path that destroyed run.R
# is closed in practice. A fully airtight version needs a nested (double) OOF
# baseline. Treat any gain here as plausible-but-unproven until it survives a
# leaderboard test.
# ---------------------------------------------------------------------------
saveRDS(oof, "model/artifacts/oof_xgb_resenc2.rds")
set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_resenc2.rds")
cat("OK -- wrote oof_xgb_resenc2.rds and test_xgb_resenc2.rds\n")