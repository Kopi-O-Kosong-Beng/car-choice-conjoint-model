# =============================================================================
# ITERATION 26 — Seed-bagging the monotone listwise xgboost.
#
# HYPOTHESIS (stated before running)
# ---------------------------------
# Every xgboost artifact in this repository is a SINGLE seed. With subsample 0.8
# and colsample_bytree 0.8, plus a randomly drawn 10% early-stopping holdout, the
# fitted predictor carries real Monte-Carlo variance that has nothing to do with
# the data. Averaging predictions over S independent seeds removes it.
#
# WHY THIS ONE IS DIFFERENT FROM EVERY OTHER GAIN IN THE LOG. Bagging involves no
# model selection: no hyperparameter is chosen by looking at a fold, no feature is
# picked because it scored well. It cannot overfit the fold structure, because it
# never consults it. Every other improvement here was selected by inspecting the
# seed-42 folds, which is why the local->public transfer rate has decayed from
# ~58% to ~1/3 across eighteen experiments. A variance reduction should transfer
# at close to 100%. It is therefore worth having even if it is small.
#
# THE SECOND OUTPUT MATTERS AS MUCH AS THE FIRST. The spread of single-seed OOF
# logloss across S seeds is a number this repository has never measured, and it
# calibrates the whole log. `xgb_mono` beat `xgb_lw2` by 0.00172. If the seed sd
# is of that order, then that margin -- and several other accepted results -- were
# partly seed noise, and the honest reading of the experiment log changes. If the
# seed sd is far smaller, the accepted results stand. Either answer is worth the
# compute.
#
# EXPECTATION. Bagging S predictors reduces the variance component of the loss by
# roughly 1/S. Boosted trees with early stopping are already partly self-averaging,
# so expect a modest gain -- +0.001 to +0.004 -- and expect it to be LARGER in
# log-probability space than in probability space, since logloss is a function of
# log p and the geometric mean is the natural average for it. Also expect the
# bagged model to earn MORE blend weight than the single-seed one, because the
# blend's other members are not averaging away the same noise.
#
# ONE CHANGE UNDER TEST: the number of seeds averaged. Features, folds,
# hyperparameters, objective and the monotone constraint are byte-identical to
# experiments/iter08_mono_tuned/run.R, from which the setup below is copied.
#
# LEAKAGE: none possible. No target-derived feature is constructed; each seed's
# fold-k prediction comes from a fit that never saw fold k, exactly as before.
#
# RESUMABILITY (the lesson from iteration 17, which finished but looked like a
# failure because a caller died before assembling it): every seed writes its own
# artifact as its last act. Re-running skips seeds already on disk, and `combine`
# averages whatever is present. A crash costs one seed, not the run.
#
#   Rscript experiments/iter26_seedbag/run.R <seed>     # one seed
#   Rscript experiments/iter26_seedbag/run.R combine    # average what exists
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) || length(args) > 2) stop("usage: run.R <seed|combine> [mono|lw2]")
JOB <- args[1]
# CFG was added after the first four mono seeds had been drawn. It defaults to
# mono so those artifacts keep their filenames and stay valid.
#
# WHY THE SECOND CONFIG EXISTS. The first four seeds averaged 1.14411 against a
# stored xgb_mono of 1.13980 and an xgb_lw2 of 1.14152 -- i.e. the monotone
# constraint's entire +0.00172 margin (iteration 08, z = 1.35) sits inside a seed
# spread of 0.0088. But four of MY seeds against ONE stored seed is not a fair
# test: the stored artifacts were fitted with nthread = 4 and these with 3, which
# changes float summation order. To say anything about the constraint we need the
# seed DISTRIBUTION of both configurations under identical conditions.
CFG <- if (length(args) == 2) args[2] else "mono"
if (!CFG %in% c("mono", "lw2")) stop("config must be mono or lw2")
OUTDIR <- "experiments/iter26_seedbag"
seedfile <- function(s) file.path(OUTDIR,
  if (CFG == "mono") sprintf("seed_%03d.rds", s) else sprintf("seed_lw2_%03d.rds", s))
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

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

mono <- rep(0, length(feat))
if (CFG == "mono") {
  mono[match("Price",   feat)] <- -1
  mono[match("Price_c", feat)] <- -1
}   # CFG == "lw2" leaves every entry 0, i.e. the unconstrained production config

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
train_one <- function(dtr, des, params) {
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

# =============================================================================
# combine: average whatever seeds are on disk
# =============================================================================
if (JOB == "combine") {
  fs <- sort(Sys.glob(file.path(OUTDIR,
         if (CFG == "mono") "seed_[0-9][0-9][0-9].rds" else "seed_lw2_*.rds")))
  if (!length(fs)) stop("no seed artifacts found for config ", CFG)
  ps <- lapply(fs, readRDS)
  cat("averaging", length(ps), "seeds:", paste(sapply(ps, `[[`, "seed"), collapse = ", "), "\n\n")

  lls <- sapply(ps, function(p) logloss(ytr, p$oof))
  cat("--- single-seed OOF logloss ---\n")
  for (i in seq_along(ps)) cat(sprintf("  seed %-3d %.5f\n", ps[[i]]$seed, lls[i]))
  cat(sprintf("  mean %.5f   sd %.5f   min %.5f   max %.5f   range %.5f\n",
              mean(lls), sd(lls), min(lls), max(lls), max(lls) - min(lls)))
  cat(sprintf("\n  stored single-seed xgb_mono (seed 123): 1.13980\n"))
  cat(sprintf("  xgb_lw2 (the model xgb_mono beat by 0.00172): 1.14152\n"))
  cat(sprintf("  >>> is that 0.00172 margin larger than the seed sd of %.5f?\n\n", sd(lls)))

  # arithmetic mean in probability space
  A <- Reduce(`+`, lapply(ps, `[[`, "oof")) / length(ps)
  # geometric mean in log space, renormalised -- the natural average for logloss
  G <- exp(Reduce(`+`, lapply(ps, function(p) log(pmax(p$oof, 1e-15)))) / length(ps))
  G <- G / rowSums(G)
  cat(sprintf("--- bagged OOF ---\n  arithmetic (probability space): %.5f\n", logloss(ytr, A)))
  cat(sprintf("  geometric  (log space):        %.5f\n", logloss(ytr, G)))
  cat(sprintf("  best single seed:              %.5f\n", min(lls)))
  cat(sprintf("  mean single seed:              %.5f\n", mean(lls)))
  cat(sprintf("  >>> variance removed: %.5f (mean single -> bagged)\n\n",
              mean(lls) - min(logloss(ytr, A), logloss(ytr, G))))

  best <- if (logloss(ytr, G) <= logloss(ytr, A)) G else A
  cat("keeping", if (identical(best, G)) "geometric" else "arithmetic", "\n")

  TA <- Reduce(`+`, lapply(ps, `[[`, "test")) / length(ps)
  TG <- exp(Reduce(`+`, lapply(ps, function(p) log(pmax(p$test, 1e-15)))) / length(ps))
  TG <- TG / rowSums(TG)
  tbest <- if (identical(best, G)) TG else TA

  No_tr <- sort(unique(trl$No)); No_te <- sort(unique(tel$No))
  stopifnot(length(No_tr) == 21565, length(No_te) == 4997)
  stopifnot(!anyNA(best), !anyNA(tbest),
            all(abs(rowSums(best) - 1) < 1e-9), all(abs(rowSums(tbest) - 1) < 1e-9))
  NM <- if (CFG == "mono") "xgb_monobag" else "xgb_lw2bag"
  saveRDS(data.table(No = No_tr, p1 = best[,1], p2 = best[,2], p3 = best[,3], p4 = best[,4]),
          sprintf("model/artifacts/oof_%s.rds", NM))
  saveRDS(data.table(No = No_te, p1 = tbest[,1], p2 = tbest[,2], p3 = tbest[,3], p4 = tbest[,4]),
          sprintf("model/artifacts/test_%s.rds", NM))
  cat(sprintf("wrote %s  (predicted test none-rate %.3f vs train %.3f)\n",
              NM, mean(tbest[, 4]), mean(ytr == 4)))
  cat("next: Rscript model/compare.R xgb_mono ", NM, "\n")
  cat("      Rscript experiments/iter17_hb/blend_probe.R mnl_pw,xgb_lw2,",
      NM, ",lcmnl3_both\n", sep = "")
  cat("OK\n"); quit(save = "no")
}

# =============================================================================
# one seed
# =============================================================================
S <- as.integer(JOB)
outf <- seedfile(S)
if (file.exists(outf)) { cat(CFG, "seed", S, "already done, skipping\n"); quit(save = "no") }

# xgboost keeps its OWN RNG for subsample/colsample -- varying only R's seed would
# change the early-stopping split but leave every tree identical. Both are varied.
params <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = 3,
               monotone_constraints = mono, seed = S)

t0 <- Sys.time()
set.seed(S)
oof_list <- list()
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases], params)
  d <- trl[fold == k]
  P <- softmax_by_task(raw_pred(fit, d))
  oof_list[[k]] <- data.table(No = unique(d$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
  cat("  seed", S, "fold", k, "best_iter", get_best_iter(fit), "\n")
}
oof <- rbindlist(oof_list); setorder(oof, No)
Moof <- as.matrix(oof[, .(p1,p2,p3,p4)])
cat(sprintf(">>> seed %d OOF: %.5f\n", S, logloss(ytr, Moof)))

es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases], params)
P <- softmax_by_task(raw_pred(fit_full, tel))
tp <- data.table(No = unique(tel$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
setorder(tp, No)
Mtest <- as.matrix(tp[, .(p1,p2,p3,p4)])
stopifnot(nrow(tp) == 4997, all(abs(rowSums(Mtest) - 1) < 1e-9))

saveRDS(list(seed = S, oof = Moof, test = Mtest,
             mins = as.numeric(difftime(Sys.time(), t0, units = "mins"))), outf)
cat(sprintf("seed %d done in %.1f min -> %s\n", S,
            as.numeric(difftime(Sys.time(), t0, units = "mins")), outf))
cat("OK\n")
