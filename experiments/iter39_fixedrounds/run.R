# =============================================================================
# ITERATION 39 — KILL THE EARLY-STOPPING CARVE (copy of iteration 26, ONE change)
#
# HYPOTHESIS (stated before running)
# ---------------------------------
# model/03_xgb_listwise.R carves a random 10% of respondents out of EVERY fit to
# serve as an early-stopping validation set. Across the 50 production fits logged
# in experiments/iter26_seedbag/log_seeds_lw2.txt, best_iter ranges 104 to 1068
# while the per-fold MEANS span only 477-577: between-fold sd of the mean is ~40
# against a within-fold sd of ~220. The carve is therefore close to a random
# number generator with mean ~540 -- it carries almost no information about the
# fold's true optimum, and for that non-information the protocol pays 10% of the
# independent units on every single fit.
#
# The part no local metric can see: the SHIPPED test refit trains on 1,021
# respondents instead of 1,135. Those 114 people are never used at all.
#
# This is NOT the variance argument seed-bagging already handles. Bagging averages
# models fitted at 104 rounds and at 1068; that is not the same object as a model
# fitted at 540, and no amount of bagging can manufacture respondents that were
# never in the training set. The data-recovery half involves NO model selection,
# so it should transfer at ~100% rather than the ~50% the leaderboard shows.
#
# ONE CHANGE UNDER TEST: the early-stopping protocol. Everything else -- features,
# folds, hyperparameters, objective, seeds -- is a byte-identical copy of
# experiments/iter26_seedbag/run.R.
#
# THE HONEST WEAKNESS. NR = 540 is the mean best_iter read off the production
# folds, so the scalar was chosen with knowledge of this fold structure. It is ONE
# number rather than a per-fit choice, so the exposure is far smaller than the
# carve's, but it is not zero. Set ITER39_NR to 450 or 650 for the sensitivity
# arm: if the result is flat across that range, 540 is not doing the work.
#
# MATCHED CONTROL. xgb_lw2bag3 is a 3-seed bag of this SAME config WITH early
# stopping (OOF 1.13856). Compare against THAT, not against the 10-seed
# xgb_lw2bag (1.13682) -- comparing to the 10-seed bag would confound the carve
# change with bag depth, the exact error iteration 26 caught.
#
# DECISION RULE (fixed before running). Adopt only if: paired respondent-clustered
# z >= 2 vs xgb_lw2bag3 (model-level seed sd is 0.00283, so eyeballing is not
# admissible); the gain is stable across NR in {450, 540, 650}; the nested BLEND
# improves by more than the blend seed sd of 0.00048 when this replaces
# xgb_lw2bag in the 5-member free-sign pool; and the shipped test none-rate does
# not move further from the measured 0.2665. Reject otherwise -- a null is
# informative, it would mean the carve costs nothing.
#
# LEAKAGE: none possible. No target-derived feature is constructed; each seed's
# fold-k prediction comes from a fit that never saw fold k.
#
#   Rscript experiments/iter39_fixedrounds/run.R <seed> lw2
#   Rscript experiments/iter39_fixedrounds/run.R combine lw2
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
OUTDIR <- "experiments/iter39_fixedrounds"
seedfile <- function(s) file.path(OUTDIR,
  if (CFG == "mono") sprintf("fr_%03d.rds", s) else sprintf("fr_lw2_%03d.rds", s))
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
# ===== THE ONE CHANGE UNDER TEST (iteration 39) ==============================
# No early-stopping carve. train_one runs a FIXED number of rounds and ignores
# its `des` argument, which is retained only so the signature is unchanged.
#
# CAUTION, learned the hard way while writing this: changing train_one is NOT
# sufficient. Iteration 26's CALL SITES pass dtr_all[!Case %in% es_cases] -- the
# 90% subset -- as the training data. Leaving them alone would have kept the
# carve intact while the code LOOKED changed, and the experiment would have
# measured nothing while appearing to succeed. That is exactly how iteration 30
# burned 45 minutes on a mislabelled run. Both call sites below now pass the
# FULL data, and the greps to confirm it are:
#     grep -n "es_cases" run.R   -> must return nothing
#     grep -n "train_one(" run.R -> must show train_one(dtr_all, ...) and
#                                   train_one(trl, ...)
NR <- as.integer(Sys.getenv("ITER39_NR", "540"))
train_one <- function(dtr, des, params) {
  xgb.train(params = params,
            data = xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen)),
            nrounds = NR, verbose = 0, obj = obj_listwise, maximize = FALSE)
}
# get_best_iter() now returns NULL for these fits (no best_iteration attribute
# is ever set), so raw_pred() falls through to predicting over all NR trees --
# which is what we want, and requires no edit there.
# =============================================================================

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]

# =============================================================================
# combine: average whatever seeds are on disk
# =============================================================================
if (JOB == "combine") {
  fs <- sort(Sys.glob(file.path(OUTDIR,
         if (CFG == "mono") "fr_[0-9][0-9][0-9].rds" else "fr_lw2_*.rds")))
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
  # ITERATION 39: distinct artifact names. This line previously read
  #     NM <- if (CFG == "mono") "xgb_monobag" else "xgb_lw2bag"
  # inherited from iteration 26, and on the first run it OVERWROTE the PRODUCTION
  # artifacts oof_xgb_lw2bag.rds / test_xgb_lw2bag.rds (a live blend member).
  # Recovered with `git checkout --` because artifacts are tracked; the value is
  # back at 1.13682. The name is built from a VARIABLE, which is why a search for
  # the literal string "oof_xgb_lw2bag.rds" when retargeting this copy found
  # nothing. CLAUDE.md rule 5 -- one artifact name, one producing script.
  NM <- if (CFG == "mono") "xgb_monofr" else "xgb_lw2fr"
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
  # iteration 39: NO carve. dtr_all in full -- 908 respondents, not 817.
  fit <- train_one(dtr_all, NULL, params)
  d <- trl[fold == k]
  P <- softmax_by_task(raw_pred(fit, d))
  oof_list[[k]] <- data.table(No = unique(d$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
  cat("  seed", S, "fold", k, "best_iter", get_best_iter(fit), "\n")
}
oof <- rbindlist(oof_list); setorder(oof, No)
Moof <- as.matrix(oof[, .(p1,p2,p3,p4)])
cat(sprintf(">>> seed %d OOF: %.5f\n", S, logloss(ytr, Moof)))

# iteration 39: NO carve on the SHIPPED refit either. This is the half no local
# metric can ever see -- 1,135 respondents instead of 1,021.
fit_full <- train_one(trl, NULL, params)
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
