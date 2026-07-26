# =============================================================================
# ITERATION 15 — settling the residual-encoding question with a properly NESTED
# (double out-of-fold) baseline.   Artifacts: {oof,test}_xgb_resenc3.rds
#
# THE QUESTION.  iter12's residual design encoding scored 1.09962 and was pure
# leakage.  The control that fixed it (control_leakfree_full.R -> xgb_resenc2,
# OOF 1.13721, +0.0043 over xgb_lw2, paired z = 2.03) swaps the tree baseline for
# mnl_pw, a part-worth conditional logit with ~150 global coefficients and no
# design-level features.  That closes the leak by ARGUMENT — a global-coefficient
# model cannot memorise an individual choice set — but not by CONSTRUCTION:
# mnl_pw's OOF prediction for a reference respondent in fold j still comes from a
# fit on folds != j, a set that includes the scored fold k.  So a sliver of fold-k
# label information can in principle still ride into the residual, sign-flipped.
#
# THE AIRTIGHT VERSION.  For scoring fold k, every reference respondent's baseline
# must come from a model that saw neither their own fold nor fold k:
#     for k in 1:5:  for j != k:  fit on folds \ {k, j}; predict fold j
# 20 fits, done in experiments/iter15_nested_resenc/nested_baseline.R and cached
# in nested_base.rds.  Column k of that matrix is the baseline honest w.r.t. fold
# k.  The encoding applied to fold-k rows is then built from residuals that cannot
# contain any fold-k choice, at any remove.
#
# For TEST rows there is no scored fold to protect, so the ordinary mnl_pw OOF
# baseline is used (exactly as in the control).
#
# CONTROLLED COMPARISON.  Everything downstream — feature list, share encoding
# (ENC_COLS), residual shrinkage alphas, custom listwise objective,
# hyperparameters (eta 0.03 / depth 8 / min_child_weight 20 / subsample 0.8 /
# colsample 0.8), early-stopping split carved by Case, seeds 123 and 7 — is copied
# verbatim from control_leakfree_full.R.  The ONLY change under test is the
# HONESTY OF THE BASELINE.
#
#   gain survives  -> xgb_resenc2's +0.0043 is real and shippable
#   gain collapses -> xgb_resenc2 was still leaking and must not be shipped
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter15_nested_resenc/run.R
# Runtime: ~25 min with the cache present, ~65 min without.
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
apply_design_encoding(trl, tel)              # production share encoding, unchanged

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]

# ---- 1. the nested baseline -------------------------------------------------
CACHE <- "experiments/iter15_nested_resenc/nested_base.rds"
if (!file.exists(CACHE))
  stop("run experiments/iter15_nested_resenc/nested_baseline.R first (~40 min)")
nb <- readRDS(CACHE)
stopifnot(identical(as.numeric(nb$No), as.numeric(trl$No)),
          identical(as.numeric(nb$alt), as.numeric(trl$alt)),
          identical(as.numeric(nb$fold), as.numeric(trl$fold)))
NB <- nb$NB

# ---- 2. the ordinary (single) OOF baseline, for the test rows + diagnostics --
base_oof <- readRDS("model/artifacts/oof_mnl_pw.rds"); setorder(base_oof, No)
stopifnot(nrow(base_oof) == 21565, identical(base_oof$No, sort(unique(trl$No))))
pm <- melt(base_oof, id.vars = "No", measure.vars = c("p1","p2","p3","p4"),
           variable.name = "altf", value.name = "p_model")
pm[, alt := as.integer(sub("p", "", as.character(altf)))][, altf := NULL]
trl <- merge(trl, pm, by = c("No","alt"), all.x = TRUE)
setorder(trl, No, alt)
stopifnot(!anyNA(trl$p_model))
trl[, resid_oof := as.numeric(chosen) - p_model]

cat(sprintf("baseline mnl_pw OOF                 : %.5f\n",
            logloss(ytr, matrix(trl$p_model, ncol = 4, byrow = TRUE))))
cat(sprintf("sd(residual)  single-OOF baseline   : %.5f\n", sd(trl$resid_oof)))
nested_sd <- sapply(1:5, function(k) sd(as.numeric(trl$chosen[trl$fold != k]) - NB[trl$fold != k, k]))
cat(sprintf("sd(residual)  nested baseline       : %.5f  (per k: %s)\n",
            mean(nested_sd), paste(round(nested_sd, 4), collapse = " ")))
ylk <- setNames(ytr, as.character(sort(unique(trl$No))))   # No -> y lookup
nested_ll <- sapply(1:5, function(k) {
  ii <- which(trl$fold != k)
  nos <- trl$No[ii][seq(1, length(ii), 4)]
  logloss(unname(ylk[as.character(nos)]), matrix(NB[ii, k], ncol = 4, byrow = TRUE))
})
cat(sprintf("nested baseline logloss (per k)     : %s\n\n", paste(round(nested_ll, 5), collapse = " ")))

# ---- 3. the residual encoder ------------------------------------------------
RES_ALPHAS <- c(1, 5, 20)
RES_COLS   <- c(paste0("resid_a", RES_ALPHAS), "resid_n")
encode_resid <- function(ref, target) {          # ref needs a column `resid`
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
  ref <- copy(trl[ii])
  ref[, resid := as.numeric(chosen) - NB[ii, k]]     # <- NESTED, honest w.r.t. k
  e <- encode_resid(ref, trl[fold == k])
  for (cc in RES_COLS) set(trl, which(trl$fold == k), cc, e[[cc]])
}
# test rows: no scored fold to protect, ordinary OOF baseline is fine
ref_all <- copy(trl)[, resid := resid_oof]
e <- encode_resid(ref_all, tel)
for (cc in RES_COLS) set(tel, seq_len(nrow(tel)), cc, e[[cc]])
stopifnot(!anyNA(trl[, ..RES_COLS]), !anyNA(tel[, ..RES_COLS]))

for (a in RES_ALPHAS) {
  s <- trl[, .(s = sum(get(paste0("resid_a", a)))), by = No]$s
  cat(sprintf("resid_a%-3d within-task sum max|.| = %.2e   sd of feature = %.4f\n",
              a, max(abs(s)), sd(trl[[paste0("resid_a", a)]])))
}
cat("mean support n per (design,alt): train", round(mean(trl$resid_n), 2),
    " test", round(mean(tel$resid_n), 2), "\n\n")

# =============================================================================
# 4. THE TWO LEAK DETECTORS, nested encoding vs the single-OOF encoding of iter13
# =============================================================================
# (a) cor(encoding, the row's OWN held-out residual).  Positive = genuine design
#     signal;  negative = the encoding is handing the label back sign-flipped.
# (b) direct correction  p = normalise(p_mnl_pw + w * encoding).  A genuine signal
#     improves logloss with no learning at all;  a leak cannot.
# iter13 reference numbers come from experiments/iter13_resenc_validation/.
DIAG_ALPHAS <- c(1, 5, 20, 50)
enc_single <- function(alpha) {              # iter13's construction, single OOF
  out <- numeric(nrow(trl))
  for (k in 1:5) {
    ref <- copy(trl[fold != k]); ref[, resid := resid_oof]
    agg <- ref[, .(rs = sum(resid), n = .N), by = .(dkey, alt)]
    t2 <- copy(trl[fold == k])[, .rid := .I][, .(.rid, dkey, alt)]
    t2 <- merge(t2, agg, by = c("dkey","alt"), all.x = TRUE)
    t2[is.na(n), `:=`(n = 0, rs = 0)][, enc := rs / (n + alpha)]
    setorder(t2, .rid)
    out[which(trl$fold == k)] <- t2$enc
  }
  out
}
enc_nested <- function(alpha) {               # this iteration's construction
  out <- numeric(nrow(trl))
  for (k in 1:5) {
    ii <- which(trl$fold != k)
    ref <- copy(trl[ii]); ref[, resid := as.numeric(chosen) - NB[ii, k]]
    agg <- ref[, .(rs = sum(resid), n = .N), by = .(dkey, alt)]
    t2 <- copy(trl[fold == k])[, .rid := .I][, .(.rid, dkey, alt)]
    t2 <- merge(t2, agg, by = c("dkey","alt"), all.x = TRUE)
    t2[is.na(n), `:=`(n = 0, rs = 0)][, enc := rs / (n + alpha)]
    setorder(t2, .rid)
    out[which(trl$fold == k)] <- t2$enc
  }
  out
}
base_ll <- logloss(ytr, matrix(trl$p_model, ncol = 4, byrow = TRUE))
cat("LEAK DETECTORS  (baseline mnl_pw OOF =", sprintf("%.5f", base_ll), ")\n")
cat("  detector (a): cor(encoding, own held-out residual) -- positive = genuine\n")
cat("  detector (b): p_mnl_pw + w*encoding, renormalised  -- '*' beats the baseline\n\n")
for (lab in c("single-OOF (= xgb_resenc2 construction)", "NESTED     (= xgb_resenc3)")) {
  cat(lab, "\n")
  for (a in DIAG_ALPHAS) {
    enc <- if (grepl("NESTED", lab)) enc_nested(a) else enc_single(a)
    cr  <- cor(enc, trl$resid_oof)
    line <- sprintf("   alpha=%-3d cor=%+.4f  ", a, cr)
    for (w in c(0.25, 0.5, 0.75, 1.0)) {
      P  <- matrix(pmax(trl$p_model + w * enc, 1e-9), ncol = 4, byrow = TRUE)
      P  <- P / rowSums(P)
      ll <- logloss(ytr, P)
      line <- paste0(line, sprintf("w=%.2f:%.5f%s ", w, ll, if (ll < base_ll) "*" else " "))
    }
    cat(line, "\n")
  }
}
cat("\n")

# =============================================================================
# 5. the model — byte-identical to control_leakfree_full.R except for the features
# =============================================================================
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
cat(sprintf("\n>>> xgb_resenc3 (NESTED baseline) OOF logloss: %.5f\n",
            logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)]))))
cat("    xgb_resenc2 (single-OOF baseline)         1.13721\n")
cat("    xgb_lw2     (no residual encoding)        1.14152\n")
saveRDS(oof, "model/artifacts/oof_xgb_resenc3.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-9))
saveRDS(tp, "model/artifacts/test_xgb_resenc3.rds")

imp <- as.data.frame(xgb.importance(model = fit_full))
print(head(imp[, c("Feature", "Gain")], 12))
cat("\nresidual-feature gains and ranks:\n")
print(imp[imp$Feature %in% RES_COLS, c("Feature", "Gain")])
cat("rank of each residual feature:",
    paste(RES_COLS, match(RES_COLS, imp$Feature), collapse = "  "), "\n")
cat("RUN_DONE\n")
