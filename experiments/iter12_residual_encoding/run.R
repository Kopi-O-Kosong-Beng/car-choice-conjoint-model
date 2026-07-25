# =============================================================================
# ITERATION 12 — Residual-based design encoding.
#
# HYPOTHESIS. This is a DESIGNED conjoint experiment: only a few hundred distinct
# choice-set designs exist per task position, each shown to ~4.7 respondents, and
# 98.5% of TEST rows reuse a design that appears in training. Iteration 01
# exploited that by encoding the shrunk empirical choice SHARE per (design, alt)
# (+0.0046, z = 2.9). But a raw share is a noisy statistic: with ~3.8 tasks per
# design most of the signal has to be shrunk away, so the encoding arrives at the
# model badly attenuated.
#
# Encode the RESIDUAL instead. For each (design, alternative), average
#     (observed chosen indicator - model predicted probability)
# over the REFERENCE respondents, where the model is our best single model
# (xgb_lw2, listwise xgboost, OOF 1.14152). Residuals have much smaller variance
# than raw choice indicators, because the model already explains most of the
# variation. With the same shrinkage strength, a lower-variance statistic keeps a
# larger share of its signal. And what survives is precisely what the
# attribute-based model FAILS to capture about a design.
#
# A useful structural property: within one task the four residuals sum exactly to
# zero (sum of chosen = 1 = sum of p). Since every reference task contributes all
# four of its rows, the support count n is identical across the four alternatives
# of a design, so the shrunk residual encoding sums to EXACTLY zero within a
# design. It is therefore a pure listwise correction — it can only reallocate
# probability between alternatives, never inflate or deflate a whole choice set.
# That is the right shape for a metric that is a softmax over the four rows.
#
# LEAKAGE CONTROL (non-negotiable). Respondents live entirely inside one fold.
# For a training row in fold k the residual encoding is built ONLY from
# respondents in folds != k, so a person never contributes to their own encoding.
# Test rows use all 1,135 training respondents. Baseline p_model is the OOF
# prediction of xgb_lw2, so the reference respondent's own prediction was itself
# made by a model that never saw that respondent.
# (Known second-order caveat, recorded honestly: a reference respondent in fold j
# has p_model from a model trained on folds != j, which includes fold k. That path
# is discussed in EXPERIMENTS.md; it can only ATTENUATE the encoding, since it
# makes the residual anti-correlated with the held-out fold's choices.)
#
# CONTROLLED COMPARISON. Everything else is copied verbatim from
# model/03_xgb_listwise.R — same feature list, same share encoding (ENC_COLS),
# same custom listwise objective, same hyperparameters (eta 0.03, depth 8,
# min_child_weight 20, subsample/colsample 0.8), same seeds, same early-stopping
# split carved out by Case. The ONLY change under test is the added residual
# features. nthread = 4 (shared machine).
#
# Artifacts: model/artifacts/{oof,test}_xgb_resenc.rds
# Compare:   model/compare.R xgb_lw2 xgb_resenc
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter12_residual_encoding/run.R
# Runtime: ~20 min.
#
# -----------------------------------------------------------------------------
# RESULT (written after running) — DO NOT PUT xgb_resenc IN THE BLEND.
#
# OOF 1.09962 vs 1.14152, paired z = 9.52. That is nine times the entire
# project's accumulated gain from one derived feature, and it is LEAKAGE.
#
# Three things gave it away before the control was run:
#   1. The direct-correction diagnostic below says the opposite. Adding the
#      encoding to p_model makes every configuration WORSE (best 1.14515 at
#      alpha=20, w=0.25). A feature with no usable positive signal cannot
#      legitimately be worth 0.042 to a tree ensemble.
#   2. cor(encoding, the target's own held-out residual) is NEGATIVE, -0.071.
#      Genuine design signal would make it positive.
#   3. resid_a1 -- the LEAST shrunk feature, the one retaining the most raw
#      memorisation -- is the model's #2 feature by gain, above share_a1.
#
# THE MECHANISM. The stated fold rule is honoured: the encoding for a fold-k row
# uses only folds != k. But each reference respondent i (fold j) contributes
# chosen_i - p_model_i, and p_model_i is xgb_lw2's OOF prediction, produced by a
# model trained on all folds EXCEPT j -- a training set that INCLUDES fold k. A
# depth-8 xgboost holding the design-share encoding can partially memorise an
# individual choice set, so p_model_i on design d already embeds what fold-k
# respondents chose on design d, the target included. Subtracting it hands the
# target's own label back, sign-flipped. Fold-honest reference SETS are not
# enough; the reference respondents' PREDICTIONS must also be honest w.r.t. the
# scored fold, which requires a nested (double) out-of-fold baseline.
#
# THE CONTROL (control_leakfree.R): identical script, identical folds, features,
# hyperparameters and seeds, with p_model taken from mnl_pw instead -- a
# part-worth conditional logit with ~150 global coefficients and no design-level
# features, structurally unable to memorise a choice set, so the leak path is
# closed. It scores 1.13721. Same idea, same everything, leak removed:
#
#      xgb_lw2 incumbent                        1.14152
#      leak-free residual encoding (control)    1.13721   (+0.0043, plausible)
#      this script                              1.09962   (+0.0419, ~90% leak)
#
# Supporting dose-response, cor(encoding, own held-out residual) by baseline:
#      mnl_pw +0.014   mnl +0.044   mixl2 +0.020    (no design features)
#      xgb_de -0.066   xgb_lw -0.062   xgb_lw2 -0.071  (design-share encoding)
# The sign flips exactly when the baseline gains the ability to memorise designs.
#
# VERDICT: hypothesis not supported as specified. The honest value of the idea is
# ~+0.004, the same order as the share encoding of iteration 01 that it partly
# duplicates. The artifacts below are kept only as the record of the measurement.
# -----------------------------------------------------------------------------
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

# ---- share encoding, identical to production --------------------------------
apply_design_encoding(trl, tel)

# ---- baseline model probabilities, attached to the long training rows -------
base_oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(base_oof, No)
stopifnot(nrow(base_oof) == 21565, identical(base_oof$No, sort(unique(trl$No))))
pm <- melt(base_oof, id.vars = "No", measure.vars = c("p1","p2","p3","p4"),
           variable.name = "altf", value.name = "p_model")
pm[, alt := as.integer(sub("p", "", as.character(altf)))][, altf := NULL]
trl <- merge(trl, pm, by = c("No","alt"), all.x = TRUE)
setorder(trl, No, alt)
stopifnot(!anyNA(trl$p_model))
trl[, resid := as.numeric(chosen) - p_model]
# within-task residuals must sum to zero
stopifnot(max(abs(trl[, .(s = sum(resid)), by = No]$s)) < 1e-9)
cat("residual sd:", round(sd(trl$resid), 4),
    "   chosen sd:", round(sd(as.numeric(trl$chosen)), 4), "\n")

# ---- the residual encoder ---------------------------------------------------
# shrunk toward ZERO (not toward a prior share): sum(resid) / (n + alpha).
RES_ALPHAS <- c(1, 5, 20)                    # same strengths as the share encoder
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
stopifnot(!anyNA(trl[, ..RES_COLS]), !anyNA(tel[, ..RES_COLS]))

# structural check: residual encoding must sum to exactly 0 within each task
for (a in RES_ALPHAS) {
  s <- trl[, .(s = sum(get(paste0("resid_a", a)))), by = No]$s
  cat(sprintf("resid_a%-3d within-task sum  max|.| = %.2e   sd of feature = %.4f\n",
              a, max(abs(s)), sd(trl[[paste0("resid_a", a)]])))
}
cat("mean support n per (design,alt), train rows:", round(mean(trl$resid_n), 2),
    "  test rows:", round(mean(tel$resid_n), 2), "\n")
cat("rows with zero support: train",
    round(100 * mean(trl$resid_n == 0), 2), "%  test",
    round(100 * mean(tel$resid_n == 0), 2), "%\n\n")

# ---- DIAGNOSTIC: the encoding alone, used as a direct correction ------------
# p_corrected = normalise(max(p_model + resid_enc, eps)). If the residual carried
# real signal this beats 1.14152 with no learning at all.
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
cat("baseline xgb_lw2 OOF:", round(logloss(ytr, as.matrix(base_oof[, .(p1,p2,p3,p4)])), 5), "\n")
for (a in RES_ALPHAS) {
  for (w in c(0.25, 0.5, 1.0)) {
    pc <- pmax(trl$p_model + w * trl[[paste0("resid_a", a)]], 1e-6)
    P  <- matrix(pc, ncol = 4, byrow = TRUE)
    P  <- P / rowSums(P)
    cat(sprintf("  correction alpha=%-3d weight=%.2f  OOF = %.5f\n",
                a, w, logloss(ytr, P)))
  }
}
cat("\n")

# ---- model: production listwise xgboost + the residual features -------------
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, ENC_COLS,
          RES_COLS)

softmax_by_task <- function(scores) {
  M <- matrix(scores, ncol = 4, byrow = TRUE)
  M <- M - apply(M, 1, max)
  E <- exp(M)
  E / rowSums(E)
}
obj_listwise <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  p <- as.vector(t(softmax_by_task(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
eval_tasklogloss <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  P <- softmax_by_task(preds)
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
  dm <- xgb.DMatrix(as.matrix(d[, ..feat]))
  bi <- get_best_iter(fit)
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

set.seed(123)
oof_list <- list()
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  d <- trl[fold == k]
  P <- softmax_by_task(raw_pred(fit, d))
  oof_list[[k]] <- data.table(No = unique(d$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
  cat("fold", k, "best_iter", get_best_iter(fit), "\n"); flush.console()
}
oof <- rbindlist(oof_list); setorder(oof, No)
stopifnot(nrow(oof) == 21565, identical(oof$No, base_oof$No),
          all(abs(rowSums(as.matrix(oof[, -1])) - 1) < 1e-9))
cat(">>> xgb_resenc OOF logloss:",
    round(logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)])), 5),
    "   (xgb_lw2 baseline: 1.14152)\n")
saveRDS(oof, "model/artifacts/oof_xgb_resenc.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_resenc.rds")

imp <- as.data.frame(xgb.importance(model = fit_full))
print(head(imp[, c("Feature", "Gain")], 12))
cat("\nresidual-feature ranks / gains:\n")
print(imp[imp$Feature %in% RES_COLS, c("Feature", "Gain")])
cat("rank of each residual feature:",
    paste(RES_COLS, match(RES_COLS, imp$Feature), collapse = "  "), "\n")
cat("OK\n")
