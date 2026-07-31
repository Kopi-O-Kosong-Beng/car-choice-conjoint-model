# =============================================================================
# ITERATION 81 — REBUILD THE PRODUCTION TREE HONESTLY
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# WHY
# -----------------------------------------------------------------------------
# Two facts sit in CLAUDE.md next to each other and have never been reconciled:
#
#   (a) Iteration 48 established that `apply_design_encoding()` is called ONCE,
#       before the CV loop, so a training row in fold j carries an encoding built
#       from a set containing the scored fold k. Its HONEST value is -0.00596
#       (z -3.53), i.e. the encoding actively HURTS once the leak is removed. The
#       honest depth optimum is 4-5; production ships depth 8.
#   (b) `model/03_xgb_listwise.R:37` and `experiments/iter26_seedbag/run.R` both
#       still call it before the loop, at depth 8. Kept on the grounds that
#       removing it is "board-neutral".
#
# "Board-neutral" was never tested on the board. It was inferred. Meanwhile the
# second modelling track -- which does NOT carry this encoder and runs max_depth 3
# with 100 models averaged -- scores 0.009 better than ours once both are put at
# the same margin. That is the whole remaining gap, and it points straight at the
# one defect we have documented and not fixed.
#
# -----------------------------------------------------------------------------
# WHAT IS UNDER TEST, AND WHY THIS IS NOT A SEARCH
# -----------------------------------------------------------------------------
# This does not hunt for a new gain. It applies two ALREADY-MEASURED corrections
# to a member that never received them, and measures the result honestly. Per
# CLAUDE.md's own reasoning that is a correction to a mis-specified fit, not a
# selection event, so it does not spend the replication budget.
#
# Four arms, each seed-bagged. Bagging involves no model selection (iteration 26),
# so it cannot overfit the folds.
#
#   ctrl     encoding ON (leaky, as production), max_depth 8   <- reproduces xgb_lw2bag
#   noenc8   encoding OFF,                        max_depth 8   <- isolates the leak
#   noenc5   encoding OFF,                        max_depth 5   <- + honest depth
#   noenc4   encoding OFF,                        max_depth 4   <- + honest depth
#
# ctrl minus noenc8 is the leak's contribution at fixed capacity.
# noenc8 minus noenc5/noenc4 is the depth correction at fixed (zero) leak exposure.
# Both contrasts have MATCHED leak exposure on each side, which is the condition
# iteration 54-59 violated and paid 1.205 for.
#
# -----------------------------------------------------------------------------
# DECISION RULE, PRE-REGISTERED
# -----------------------------------------------------------------------------
# The plain OOF of an arm WITH the encoding is not comparable to one without it --
# the leaky arm's OOF is inflated by construction. So plain OOF is reported for
# the record but is NOT the decision measure. The decision measure is:
#
#   1. the SEGMENT-REWEIGHTED nested blend OOF (reweight training respondents by
#      p_test(segmentind)/p_train(segmentind), unclipped), computed by
#      blend_eval.R, comparing {arm + lcmnl3_both} against {xgb_lw2bag + lcmnl3_both};
#   2. paired, respondent-clustered, so a z is available;
#   3. an arm ships only if it wins on (1) by more than the paired SE, AND the
#      win does not depend on the encoding being present.
#
# If no arm wins, that is a real result and the production tree stands. Write it up.
#
# LEAKAGE: none in noenc* by construction -- no target-derived feature exists.
# ctrl reproduces the known leak deliberately, as the control.
#
# RESUMABILITY (iteration 17's lesson): every seed writes its own artifact as its
# last act; re-running skips seeds already on disk.
#
#   Rscript experiments/iter81_honesttree/run.R <seed> <arm>
#   Rscript experiments/iter81_honesttree/run.R combine <arm>
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: run.R <seed|combine> <ctrl|noenc8|noenc5|noenc4>")
JOB <- args[1]
ARM <- args[2]
stopifnot(ARM %in% c("ctrl", "noenc8", "noenc5", "noenc4"))

USE_ENC <- (ARM == "ctrl")
DEPTH   <- switch(ARM, ctrl = 8L, noenc8 = 8L, noenc5 = 5L, noenc4 = 4L)
NTHREAD <- as.integer(Sys.getenv("ITER81_NTHREAD", "3"))

DIR <- "experiments/iter81_honesttree"
dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
f_oof  <- function(s) file.path(DIR, sprintf("oof_%s_s%s.rds",  ARM, s))
f_test <- function(s) file.path(DIR, sprintf("test_%s_s%s.rds", ARM, s))

# ---- data, byte-identical to model/03_xgb_listwise.R up to the encoding call ----
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")

long <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
if (USE_ENC) { apply_design_encoding(trl, tel) }

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo)
if (USE_ENC) { feat <- c(feat, ENC_COLS) }
stopifnot(all(feat %in% names(trl)), all(feat %in% names(tel)))

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

params <- list(eta = 0.03, max_depth = DEPTH, min_child_weight = 20,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0,
               nthread = NTHREAD)
new_api <- packageVersion("xgboost") >= "2.1.0"
get_best_iter <- function(fit) {
  bi <- tryCatch(xgb.attr(fit, "best_iteration"), error = function(e) NULL)
  if (is.null(bi)) { bi <- tryCatch(fit$best_iteration, error = function(e) NULL) }
  if (is.null(bi) || is.na(suppressWarnings(as.numeric(bi)))) { NULL } else { as.integer(bi) }
}
raw_pred <- function(fit, d) {
  dm <- xgb.DMatrix(as.matrix(d[, ..feat]))
  bi <- get_best_iter(fit)
  p <- if (!is.null(bi)) {
    tryCatch(predict(fit, dm, iterationrange = c(1, bi + 1), outputmargin = TRUE),
             error = function(e) predict(fit, dm, outputmargin = TRUE))
  } else {
    predict(fit, dm, outputmargin = TRUE)
  }
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

# ---------------------------------------------------------------- one seed ----
if (JOB != "combine") {
  SEED <- as.integer(JOB)
  if (file.exists(f_oof(SEED)) && file.exists(f_test(SEED))) {
    cat(sprintf("[%s seed %d] already on disk, skipping\n", ARM, SEED)); quit(status = 0)
  }
  t0 <- Sys.time()
  set.seed(SEED)
  oof_list <- list()
  for (k in 1:5) {
    dtr_all <- trl[fold != k]
    es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
    fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
    d <- trl[fold == k]
    P <- softmax_by_task(raw_pred(fit, d))
    oof_list[[k]] <- data.table(No = unique(d$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
    cat(sprintf("[%s seed %d] fold %d best_iter %s\n", ARM, SEED, k,
                as.character(get_best_iter(fit))))
    flush.console()
  }
  oof <- rbindlist(oof_list); setorder(oof, No)
  ll <- logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)]))
  cat(sprintf("[%s seed %d] plain OOF %.5f\n", ARM, SEED, ll))

  set.seed(SEED + 100000L)
  es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
  fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
  P <- softmax_by_task(raw_pred(fit_full, tel))
  tp <- data.table(No = unique(tel$No), p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])
  setorder(tp, No)
  stopifnot(nrow(oof) == 21565L, nrow(tp) == 4997L,
            max(abs(rowSums(as.matrix(oof[, -1])) - 1)) < 1e-9,
            max(abs(rowSums(as.matrix(tp[,  -1])) - 1)) < 1e-9)
  # write as the LAST act, both together
  saveRDS(oof, f_oof(SEED)); saveRDS(tp, f_test(SEED))
  cat(sprintf("[%s seed %d] done in %.1f min  (mean p4 test %.5f)\n",
              ARM, SEED, as.numeric(difftime(Sys.time(), t0, units = "mins")), mean(tp$p4)))
  quit(status = 0)
}

# ---------------------------------------------------------------- combine ----
gm <- function(mats) {
  L <- Reduce(`+`, lapply(mats, function(M) log(pmax(M, 1e-12)))) / length(mats)
  Q <- exp(L - apply(L, 1, max)); Q / rowSums(Q)
}
seeds <- sort(as.integer(gsub(sprintf("^oof_%s_s|\\.rds$", ARM), "",
        list.files(DIR, pattern = sprintf("^oof_%s_s[0-9]+\\.rds$", ARM)))))
seeds <- seeds[file.exists(f_test(seeds))]
if (!length(seeds)) stop("no seeds on disk for arm ", ARM)
cat(sprintf("combining arm %s over %d seeds: %s\n", ARM, length(seeds), paste(seeds, collapse = ",")))

O <- lapply(seeds, function(s) { d <- readRDS(f_oof(s));  setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
Tt <- lapply(seeds, function(s) { d <- readRDS(f_test(s)); setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
No_o <- readRDS(f_oof(seeds[1]))[order(No), No]
No_t <- readRDS(f_test(seeds[1]))[order(No), No]

singles <- sapply(O, function(M) logloss(ytr, M))
cat(sprintf("  single-seed OOF: %s\n", paste(sprintf("%.5f", singles), collapse = " ")))
cat(sprintf("  mean %.5f   sd %.5f\n", mean(singles), if (length(singles) > 1) sd(singles) else NA_real_))

Ob <- gm(O); Tb <- gm(Tt)
cat(sprintf("  BAGGED plain OOF %.5f   (gain vs mean single %+.5f)\n",
            logloss(ytr, Ob), logloss(ytr, Ob) - mean(singles)))

nm <- sprintf("xgbh_%s", ARM)
saveRDS(data.table(No = No_o, p1 = Ob[,1], p2 = Ob[,2], p3 = Ob[,3], p4 = Ob[,4]),
        sprintf("model/artifacts/oof_%s.rds", nm))
saveRDS(data.table(No = No_t, p1 = Tb[,1], p2 = Tb[,2], p3 = Tb[,3], p4 = Tb[,4]),
        sprintf("model/artifacts/test_%s.rds", nm))
cat(sprintf("  wrote model/artifacts/{oof,test}_%s.rds   test mean p4 %.5f\n", nm, mean(Tb[,4])))
cat("done\n")
