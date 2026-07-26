# =============================================================================
# ITERATION 16 — bundle-level encoding instead of choice-set-level encoding.
#
# HYPOTHESIS AS BRIEFED. Iteration 01 encodes the empirical choice share per
# (design, alternative), where a design is the whole 3-bundle choice set: 5,681
# designs with ~3.8 observations each, thin support, which is why almost all of the
# signal had to be shrunk away for only +0.0046. But 17,043 distinct BUNDLES appear
# across 79,686 bundle slots, and a bundle should recur in DIFFERENT choice sets.
# Its win rate across varying contexts is then a different statistic with different
# support: "this bundle is attractive" as opposed to "this bundle beat those
# particular rivals". Encode it, add the appearance count and the rival strength it
# faced so the model can discount easy wins, and encode the none option by the set
# it was offered against.
#
# -----------------------------------------------------------------------------
# WHY THE BRIEFED VERSION CANNOT WORK — three structural collapses, all measured in
# diagnostics.R and diagnostics2.R BEFORE any model was fitted.
#
# (1) A bundle never recurs in a different choice set. There are 5,681 designs and
#     exactly 5,681 x 3 = 17,043 distinct real bundles, and the map
#     bundle -> (design, position) is a BIJECTION: 0 bundles appear in more than one
#     design, 0 appear at more than one position. So a bundle-keyed win rate is not
#     a new statistic with better support -- it is the SAME statistic. Building it
#     literally and differencing it against the existing (design, alt) encoding
#     gives max absolute difference 0.000e+00 on alternatives 1-3, for every
#     shrinkage strength and for the count. Support per bundle is 4.68 observations,
#     identical to the design key's 4.68, because they are the same partition.
#
# (2) The natural repair -- back off to the bundle's PRESENCE PATTERN (which of the
#     19 non-price features it carries, ignoring tier) -- is a property of the
#     CHOICE SET, not of the bundle. In 26,562 of 26,562 tasks (100.00%) all three
#     real bundles share one presence pattern: a choice set offers the same 9
#     features at three different tier/price configurations. The pattern is
#     therefore constant within a task, its within-task contrast is identically
#     zero, and under a softmax over the four alternatives it cannot move a single
#     prediction. The direct-correction detector confirmed it: logloss unchanged to
#     five decimals at every weight, for both baselines.
#
# (3) The presence pattern does not even pool across designs: 5,511 patterns for
#     5,681 designs, only 167 patterns span more than one design, 94.07% of designs
#     own their pattern outright. Tasks per pattern 9.64 versus tasks per design
#     9.35 -- a 3% support gain. Encoding the none option "by the set it was offered
#     against" is thus, to within 3%, exactly the (design, alt=4) share already in
#     the model.
#
# The conjoint design nests bundles inside choice sets bijectively. There is no
# bundle-level statistic between the attribute level and the choice-set level.
#
# -----------------------------------------------------------------------------
# WHAT IS ACTUALLY TESTED HERE. The highest-order key that BOTH recurs across
# choice sets AND varies within a task is the marginal (attribute, level) cell: 91
# cells with a median 12,460 observations each, versus 4.68 for the design key.
# A bundle is encoded by COMPOSING its own cells -- the empirical log-odds of being
# chosen, summed over its 20 attribute levels, at three shrinkage strengths. That is
# a bundle-level statistic in the sense intended: a property of the alternative
# itself, pooled over every context it ever appeared in, well supported, and it
# varies within a task. It is naive-Bayes-shaped: it ignores interactions between
# attributes, which is precisely what buys it the support.
#
# The "discount easy wins" term becomes the strength of the rivals actually faced --
# max, mean, and the alternative's edge over the strongest rival, plus rival price
# and level-sum. Rival RICHNESS cannot serve: richness is 9 on every real bundle and
# 0 on the none option, so the production feature rich_task_mean is the constant
# 6.75 and carries no information at all.
#
# PRIOR EXPECTATION, RECORDED BEFORE THE FIT: near zero. The direct-signal detector
# put cor(encoding, own held-out residual) at +0.0063 against xgb_lw2 and -0.0025
# against mnl_pw, and the best direct correction was 1.14149 versus a 1.14152
# baseline -- an improvement of 0.00003. Both models already receive the raw
# attribute levels, so a naive-Bayes composite of them should be fully absorbed.
# This run measures that rather than asserting it.
#
# LEAKAGE CONTROL. Only raw `chosen` indicators from reference respondents enter the
# encoding -- no model prediction of any kind, at any point. The iteration-12
# failure mode (reference respondents' baseline predictions were themselves fitted
# on the scored fold, so subtracting them handed the target its own label back
# sign-flipped) is closed BY CONSTRUCTION here, not by argument. Respondents live
# entirely inside one fold; a fold-k row is encoded from folds != k only, so nobody
# contributes to their own feature. Test rows use all 1,135 training respondents.
# Both prescribed detectors were run before fitting; results quoted above.
#
# CONTROLLED COMPARISON. Everything else is copied verbatim from
# model/03_xgb_listwise.R -- same feature list, same design-share encoding
# (ENC_COLS, kept as instructed), same custom listwise objective, same
# hyperparameters, same seeds (123 / 7), same early-stopping split carved by Case.
# The only change under test is the added bundle-encoding columns. nthread = 4.
#
# Artifacts: model/artifacts/{oof,test}_xgb_bundle.rds
# Compare:   model/compare.R xgb_lw2 xgb_bundle
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter16_bundle_encoding/run.R
# Runtime: ~20 min.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")
source("experiments/iter16_bundle_encoding/marginal_encoder.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")

long <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)

apply_design_encoding(trl, tel)                      # unchanged, kept as instructed
add_marg_context(trl); add_marg_context(tel)
apply_marginal_encoding(trl, tel)                    # the change under test

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo,
          ENC_COLS, MRG_COLS)
cat("features:", length(feat), " (production has", length(feat) - length(MRG_COLS), ")\n")

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

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
set.seed(123)
oof_list <- list()
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  d <- trl[fold == k]
  P <- softmax_by_task(raw_pred(fit, d))
  oof_list[[k]] <- data.table(No = unique(d$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
  cat("fold", k, "best_iter", get_best_iter(fit), "\n")
}
oof <- rbindlist(oof_list); setorder(oof, No)
stopifnot(nrow(oof) == 21565, all(abs(rowSums(as.matrix(oof[, -1])) - 1) < 1e-9))
cat(">>> xgb_bundle OOF logloss:",
    round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5),
    "  (incumbent xgb_lw2: 1.14152)\n")
saveRDS(oof, "model/artifacts/oof_xgb_bundle.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_bundle.rds")

imp <- as.data.frame(xgb.importance(model = fit_full))
cat("\ntop 15 features by gain:\n"); print(head(imp[, c("Feature", "Gain")], 15))
cat("\nrank of each added bundle feature:\n")
for (cc in MRG_COLS) {
  r <- which(imp$Feature == cc)
  cat(sprintf("  %-16s %s\n", cc, if (length(r)) sprintf("rank %d / %d, gain %.5f", r, nrow(imp), imp$Gain[r]) else "unused"))
}
cat("OK\n")
