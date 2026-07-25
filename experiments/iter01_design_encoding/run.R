# =============================================================================
# ITERATION 01 — Design-level empirical choice shares as features.
#
# HYPOTHESIS: 98.5% of test rows reuse a choice-set design seen in training.
# The empirical share of each alternative within its design is a nonparametric
# estimate of the population choice probability -- information no attribute-based
# model can recover, because it captures whatever the attributes fail to encode.
# Support is thin (~3.8 rows/design), so shares must be shrunk toward a prior;
# we hand xgboost several shrinkage strengths plus the support count and let it
# learn how far to trust them.
#
# LEAKAGE CONTROL: for a training row in fold k, shares are computed ONLY from
# respondents in folds != k. Respondents live entirely inside one fold, so a
# person can never contribute to their own encoding. Test rows use all 1,135
# training respondents.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter01_design_encoding/run.R
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")

# ---- design key: the 3 real alternatives, in the order shown ----------------
acols3 <- unlist(lapply(1:3, function(j) paste0(ATTRS, j)))
dk <- data.table(No = wide$No, dkey = do.call(paste, c(wide[, ..acols3], sep = "_")))
long <- merge(long, dk, by = "No")
setorder(long, No, alt)

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)

# ---- encoder: shrunk empirical share of each alternative within its design ---
ALPHAS <- c(1, 5, 20)          # weak / medium / strong shrinkage toward the prior
encode <- function(ref, target) {
  prior <- ref[, .(pr = sum(chosen) / .N), by = alt]
  cnt   <- ref[, .(n_ch = sum(chosen), n = .N), by = .(dkey, alt)]
  out   <- merge(target[, .(dkey, alt)], cnt, by = c("dkey", "alt"), all.x = TRUE, sort = FALSE)
  out   <- merge(out, prior, by = "alt", all.x = TRUE, sort = FALSE)
  out[is.na(n), `:=`(n = 0, n_ch = 0)]
  for (a in ALPHAS) out[, paste0("share_a", a) := (n_ch + a * pr) / (n + a)]
  out[, design_n := n]
  out[, .SD, .SDcols = c(paste0("share_a", ALPHAS), "design_n")]
}
# merge() reorders; rebuild alignment explicitly instead of trusting row order
encode_aligned <- function(ref, target) {
  t2 <- copy(target)[, .rid := .I]
  prior <- ref[, .(pr = sum(chosen) / .N), by = alt]
  cnt   <- ref[, .(n_ch = sum(chosen), n = .N), by = .(dkey, alt)]
  t2 <- merge(t2[, .(.rid, dkey, alt)], cnt, by = c("dkey", "alt"), all.x = TRUE)
  t2 <- merge(t2, prior, by = "alt", all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, n_ch = 0)]
  for (a in ALPHAS) t2[, paste0("share_a", a) := (n_ch + a * pr) / (n + a)]
  t2[, design_n := n]
  setorder(t2, .rid)
  t2[, .SD, .SDcols = c(paste0("share_a", ALPHAS), "design_n")]
}

enc_cols <- c(paste0("share_a", ALPHAS), "design_n")
for (cc in enc_cols) { trl[, (cc) := NA_real_]; tel[, (cc) := NA_real_] }
for (k in 1:5) {
  idx <- which(trl$fold == k)
  e <- encode_aligned(trl[fold != k], trl[fold == k])
  for (cc in enc_cols) set(trl, idx, cc, e[[cc]])
}
e <- encode_aligned(trl, tel)
for (cc in enc_cols) set(tel, seq_len(nrow(tel)), cc, e[[cc]])

# sanity: encoded shares must sum to ~1 within a task for the low-shrinkage version
chk <- trl[, .(s = sum(share_a1)), by = No]$s
cat("share_a1 within-task sum: mean", round(mean(chk), 4), " range",
    round(range(chk), 4), "(should be ~1)\n")

# ---- diagnostic: how good is the encoding ALONE, with no model? -------------
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
for (a in ALPHAS) {
  P <- matrix(trl[[paste0("share_a", a)]], ncol = 4, byrow = TRUE)
  P <- P / rowSums(P)
  cat(sprintf("  empirical shares alone (alpha=%2d): OOF %.5f\n", a, logloss(ytr, P)))
}
cat("  (benchmark 1.38629; best single model so far 1.15516)\n")

# ---- xgboost with the encoding added ---------------------------------------
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, enc_cols)

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
                             error = function(e) predict(fit, dm)) else predict(fit, dm)
}
train_one <- function(dtr, des) {
  a <- list(params = params,
            data = xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen)),
            nrounds = 3000, early_stopping_rounds = 100, verbose = 0)
  m_es <- xgb.DMatrix(as.matrix(des[, ..feat]), label = as.numeric(des$chosen))
  if (use_evals) a$evals <- list(es = m_es) else a$watchlist <- list(es = m_es)
  do.call(xgb.train, a)
}

set.seed(123)
oof_list <- list()
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  d <- trl[fold == k]
  p <- pred_best(fit, xgb.DMatrix(as.matrix(d[, ..feat])))
  oof_list[[k]] <- data.table(No = d$No, alt = d$alt, p = norm_by_group(p, d$No))
  cat("fold", k, "best_iter", get_best_iter(fit), "\n")
}
oof <- dcast(rbindlist(oof_list), No ~ alt, value.var = "p")
setnames(oof, c("No", "p1", "p2", "p3", "p4")); setorder(oof, No)
cat(">>> XGB+design-encoding OOF logloss:",
    round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5),
    "  (xgb_long without encoding: 1.15516)\n")
saveRDS(oof, "model/artifacts/oof_xgb_de.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
p <- pred_best(fit_full, xgb.DMatrix(as.matrix(tel[, ..feat])))
tp <- dcast(data.table(No = tel$No, alt = tel$alt, p = norm_by_group(p, tel$No)),
            No ~ alt, value.var = "p")
setnames(tp, c("No", "p1", "p2", "p3", "p4")); setorder(tp, No)
saveRDS(tp, "model/artifacts/test_xgb_de.rds")
imp <- as.data.frame(xgb.importance(model = fit_full))
cat("\nimportance rank of encoding features:\n")
print(imp[imp$Feature %in% enc_cols, c("Feature", "Gain")])
cat("top 8 overall:\n"); print(head(imp[, c("Feature", "Gain")], 8))
cat("OK\n")
