# =============================================================================
# ITERATION 09 — two-stage decomposition of the choice.  Artifacts: xgb_2stage
#
# HYPOTHESIS (stated before running).
#   Every model so far treats the task as one flat 4-way softmax.  But alternative 4
#   is not a product, it is "I decline to buy".  The decision *whether to buy at all*
#   plausibly keys on properties of the whole choice set (how expensive is the
#   cheapest thing on offer, how much content is available, best content-per-dollar)
#   and on the respondent (income, age, attachment to the current car).  The decision
#   *which bundle, given that I buy* keys on RELATIVE price and RELATIVE feature
#   content within the set.  Forcing one softmax to serve both may cost accuracy on
#   the single largest class (alt 4, 30.2%).
#
#   Prediction if true: the combined model beats xgb_lw2 (1.14152), and the gain is
#   visible in the "none-vs-buy" part of the loss decomposition.
#   Prediction if false: the decomposition wins nothing, because the flat listwise
#   model already has an alt-4 dummy and can carve the same split internally, while
#   the two-stage version pays for it by (a) fitting stage B on only 15,046 of the
#   21,565 tasks and (b) losing all parameter sharing between the two decisions.
#
# DESIGN.
#   Stage A: binary xgboost, ONE ROW PER TASK (21,565), target = "alt 4 was chosen".
#            Features are set-level summaries only — min/mean/max of every attribute
#            across the three real bundles, price spread, content spread, best
#            content-per-dollar, Task number, all 15 demographics, and a fold-aware
#            shrunk empirical none-rate for the design.
#            NOTE: the `richness` column in long.rds is a constant 9 on every real
#            bundle (it counts non-zero attributes), so it carries no information.
#            `lvlsum` (range 9..31) is the actual content measure and is used instead.
#   Stage B: 3-way listwise softmax xgboost over alt 1..3, fitted ONLY on the 15,046
#            tasks where a real bundle was chosen.  Same custom objective as
#            model/03_xgb_listwise.R but with groups of 3 consecutive rows, and every
#            "task-relative" feature (centering, price rank, rival price) RECOMPUTED
#            over the three real bundles rather than over all four alternatives.
#   Combine: p4 = pA ;  p_j = (1 - pA) * pB_j  for j = 1..3.
#
# HONESTY CONSTRAINTS.
#   - Fixed folds from model/artifacts/folds.rds, grouped by respondent (Case).
#   - For fold k, BOTH stages are trained only on folds != k, and both target
#     encoders are built only from folds != k.
#   - Training rows get leave-one-inner-fold-out encodings, so no row's own outcome
#     ever enters its own feature.
#   - Early-stopping splits are carved out by Case, never by row, in both stages.
#   - Stage A's max_depth is chosen per fold on that fold's inner early-stopping
#     split only — never on the fold being scored.
#
# Run:  & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter09_two_stage/run.R
# Runtime: ~30-45 min.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")

NTHREAD <- 4
ENC_A   <- c(1, 5, 20)          # shrinkage strengths, same scheme as encode_design.R

long <- readRDS("model/artifacts/long.rds")
wide <- readRDS("model/artifacts/wide.rds")
fold <- readRDS("model/artifacts/folds.rds")

long <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)
stopifnot(all(long[, .N, by = No]$N == 4))

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")

# =============================================================================
# 1. BUILD THE TWO TABLES
# =============================================================================

# ---- 1a. task-level table for stage A ---------------------------------------
b <- long[alt <= 3]                                   # the three real bundles
setorder(b, No, alt)
b[, vpp := lvlsum / Price]                            # content per dollar

agg_min  <- b[, lapply(.SD, min),  by = No, .SDcols = ATTRS]
agg_mean <- b[, lapply(.SD, mean), by = No, .SDcols = ATTRS]
agg_max  <- b[, lapply(.SD, max),  by = No, .SDcols = ATTRS]
setnames(agg_min,  ATTRS, paste0(ATTRS, "_min3"))
setnames(agg_mean, ATTRS, paste0(ATTRS, "_mean3"))
setnames(agg_max,  ATTRS, paste0(ATTRS, "_max3"))

extra <- b[, .(
  price_sd3     = sd(Price),
  price_rng3    = max(Price) - min(Price),
  price_ndist3  = uniqueN(Price),
  lvl_sd3       = sd(lvlsum),
  lvl_rng3      = max(lvlsum) - min(lvlsum),
  vpp_max3      = max(vpp),
  vpp_mean3     = mean(vpp),
  lvl_at_minp   = lvlsum[which.min(Price)],   # content you get for the cheapest option
  price_at_maxl = Price[which.max(lvlsum)],   # what the richest option costs
  Task          = Task[1],
  dkey          = dkey[1]
), by = No]

tsk  <- Reduce(function(x, y) merge(x, y, by = "No"),
               list(agg_min, agg_mean, agg_max, extra))
meta <- unique(long[, c("No", "Case", "is_test", "y", demo), with = FALSE])
stopifnot(nrow(meta) == uniqueN(long$No))
tsk <- merge(tsk, meta, by = "No")
setorder(tsk, No)
tsk[, is_none := as.integer(y == 4)]                  # NA on test rows, never used there

ENCA_COLS   <- c(paste0("none_a", ENC_A), "designA_n")
FEAT_A <- c(paste0(ATTRS, "_min3"), paste0(ATTRS, "_mean3"), paste0(ATTRS, "_max3"),
            "price_sd3","price_rng3","price_ndist3","lvl_sd3","lvl_rng3",
            "vpp_max3","vpp_mean3","lvl_at_minp","price_at_maxl","Task", demo, ENCA_COLS)

# ---- 1b. long table (alt 1..3 only) for stage B ------------------------------
lb <- copy(b)
lb[, `:=`(price_mean3 = mean(Price), price_min3 = min(Price), price_max3 = max(Price),
          lvl_mean3   = mean(lvlsum), lvl_min3  = min(lvlsum), lvl_max3  = max(lvlsum),
          vpp_mean3   = mean(vpp)), by = No]
# task-relative features RECOMPUTED over the three real bundles (not over all four)
C3 <- paste0(ATTRS, "_c3")
lb[, (C3) := lapply(.SD, function(z) z - mean(z)), by = No, .SDcols = ATTRS]
lb[, `:=`(lvlsum_c3 = lvlsum - lvl_mean3,
          vpp_c3    = vpp - vpp_mean3,
          price_c3  = Price - price_mean3)]
lb[, price_rank3   := frank(Price, ties.method = "min"), by = No]
lb[, price_minriv3 := vapply(seq_len(.N), function(i) min(Price[-i]),  0), by = No]
lb[, lvl_maxriv3   := vapply(seq_len(.N), function(i) max(lvlsum[-i]), 0), by = No]
lb[, `:=`(alt1 = as.integer(alt == 1), alt2 = as.integer(alt == 2), alt3 = as.integer(alt == 3))]
setorder(lb, No, alt)
stopifnot(all(lb[, .N, by = No]$N == 3))

ENCB_COLS <- c(paste0("bshare_a", ENC_A), "designB_n")
FEAT_B <- c(ATTRS, C3, "lvlsum","lvlsum_c3","vpp","vpp_c3","price_c3",
            "price_rank3","price_minriv3","lvl_maxriv3",
            "price_mean3","price_min3","price_max3","lvl_mean3","lvl_min3","lvl_max3",
            "Task","alt1","alt2","alt3", demo, ENCB_COLS)

# =============================================================================
# 2. FOLD-AWARE TARGET ENCODERS (same shrinkage scheme as model/encode_design.R)
# =============================================================================

# Stage A: shrunk P(none) per design key, learned from a reference set of TASKS.
encA <- function(ref, target) {
  pr  <- mean(ref$is_none)
  cnt <- ref[, .(n_none = sum(is_none), n = .N), by = dkey]
  t2  <- merge(data.table(.rid = seq_len(nrow(target)), dkey = target$dkey),
               cnt, by = "dkey", all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, n_none = 0)]
  for (a in ENC_A) t2[, paste0("none_a", a) := (n_none + a * pr) / (n + a)]
  t2[, designA_n := n]
  setorder(t2, .rid)
  t2[, .SD, .SDcols = ENCA_COLS]
}

# Stage B: shrunk CONDITIONAL share of alt j GIVEN a real bundle was chosen.
# The reference is restricted to tasks with y != 4, which is what makes it conditional.
encB <- function(ref, target) {
  ref <- ref[y != 4]
  pr  <- ref[, .(pr = sum(chosen) / .N), by = alt]
  cnt <- ref[, .(n_ch = sum(chosen), n = .N), by = .(dkey, alt)]
  t2  <- merge(data.table(.rid = seq_len(nrow(target)), dkey = target$dkey, alt = target$alt),
               cnt, by = c("dkey", "alt"), all.x = TRUE)
  t2  <- merge(t2, pr, by = "alt", all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, n_ch = 0)]
  for (a in ENC_A) t2[, paste0("bshare_a", a) := (n_ch + a * pr) / (n + a)]
  t2[, designB_n := n]
  setorder(t2, .rid)
  t2[, .SD, .SDcols = ENCB_COLS]
}

# Training rows: leave-one-inner-fold-out, so a row's own outcome never feeds its
# own feature and nothing outside `dt` (i.e. nothing from the scored fold) is used.
encode_train <- function(dt, encfn, cols) {
  out <- data.table(matrix(NA_real_, nrow(dt), length(cols)))
  setnames(out, cols)
  for (j in sort(unique(dt$fold))) {
    ii <- which(dt$fold == j)
    e  <- encfn(dt[fold != j], dt[ii])
    for (cc in cols) set(out, ii, cc, as.numeric(e[[cc]]))
  }
  out
}
attach_cols <- function(dt, vals) {
  for (cc in names(vals)) set(dt, j = cc, value = as.numeric(vals[[cc]]))
  invisible(dt)
}

# =============================================================================
# 3. MODEL PLUMBING
# =============================================================================
best_iter <- function(fit) {
  bi <- tryCatch(xgb.attr(fit, "best_iteration"), error = function(e) NULL)
  if (is.null(bi)) bi <- tryCatch(fit$best_iteration, error = function(e) NULL)
  if (is.null(bi) || is.na(suppressWarnings(as.numeric(bi)))) NULL else as.integer(bi)
}
pred_at_best <- function(fit, X, margin) {
  dm <- xgb.DMatrix(X, nthread = NTHREAD)
  bi <- best_iter(fit)
  p  <- if (!is.null(bi))
          tryCatch(predict(fit, dm, iterationrange = c(1, bi + 1), outputmargin = margin),
                   error = function(e) predict(fit, dm, outputmargin = margin))
        else predict(fit, dm, outputmargin = margin)
  as.vector(p)
}

# --- stage A: plain binary logistic on task-level rows ---
paramsA_for <- function(depth) list(objective = "binary:logistic", eval_metric = "logloss",
                                    eta = 0.03, max_depth = depth, min_child_weight = 20,
                                    subsample = 0.8, colsample_bytree = 0.8, nthread = NTHREAD)
fitA <- function(dtr, des, depth) {
  xgb.train(params = paramsA_for(depth),
            data  = xgb.DMatrix(as.matrix(dtr[, ..FEAT_A]), label = dtr$is_none, nthread = NTHREAD),
            evals = list(es = xgb.DMatrix(as.matrix(des[, ..FEAT_A]), label = des$is_none, nthread = NTHREAD)),
            nrounds = 4000, early_stopping_rounds = 150, verbose = 0, maximize = FALSE)
}
# depth chosen by logloss on the inner ES split, which lives entirely inside the
# training folds — the scored fold is never touched.
fitA_select <- function(dtr, des, depths = c(4, 6, 8)) {
  fits <- lapply(depths, function(d) fitA(dtr, des, d))
  sc <- vapply(fits, function(f) {
    p <- pmin(pmax(pred_at_best(f, as.matrix(des[, ..FEAT_A]), margin = FALSE), 1e-9), 1 - 1e-9)
    -mean(log(ifelse(des$is_none == 1, p, 1 - p)))
  }, 0)
  i <- which.min(sc)
  list(fit = fits[[i]], depth = depths[i], all = paste(round(sc, 5), collapse = "/"))
}

# --- stage B: listwise softmax over groups of 3 consecutive rows ---
softmax3 <- function(s) {
  M <- matrix(s, ncol = 3, byrow = TRUE)
  M <- M - apply(M, 1, max)
  E <- exp(M)
  E / rowSums(E)
}
obj_lw3 <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  p <- as.vector(t(softmax3(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
eval_lw3 <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  P <- softmax3(preds); Y <- matrix(y, ncol = 3, byrow = TRUE)
  list(metric = "lw3logloss", value = -mean(log(pmax(P[Y == 1], 1e-15))))
}
paramsB <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
                subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = NTHREAD)
fitB <- function(dtr, des) {
  stopifnot(nrow(dtr) %% 3 == 0, nrow(des) %% 3 == 0,
            all(dtr[, .N, by = No]$N == 3), all(des[, .N, by = No]$N == 3))
  xgb.train(params = paramsB,
            data  = xgb.DMatrix(as.matrix(dtr[, ..FEAT_B]), label = as.numeric(dtr$chosen), nthread = NTHREAD),
            evals = list(es = xgb.DMatrix(as.matrix(des[, ..FEAT_B]), label = as.numeric(des$chosen), nthread = NTHREAD)),
            nrounds = 4000, early_stopping_rounds = 150, verbose = 0,
            obj = obj_lw3, custom_metric = eval_lw3, maximize = FALSE)
}

# =============================================================================
# 4. OUT-OF-FOLD LOOP
# =============================================================================
tsk_tr <- merge(tsk[is_test == FALSE], fold[, .(No, fold)], by = "No"); setorder(tsk_tr, No)
tsk_te <- copy(tsk[is_test == TRUE]);                                   setorder(tsk_te, No)
lb_tr  <- merge(lb[is_test == FALSE],  fold[, .(No, fold)], by = "No"); setorder(lb_tr, No, alt)
lb_te  <- copy(lb[is_test == TRUE]);                                    setorder(lb_te, No, alt)
stopifnot(nrow(tsk_tr) == 21565, nrow(tsk_te) == 4997,
          nrow(lb_tr) == 3 * 21565, nrow(lb_te) == 3 * 4997,
          identical(tsk_tr$No, unique(lb_tr$No)), identical(tsk_te$No, unique(lb_te$No)))

ytr <- tsk_tr$y
oof <- data.table(No = tsk_tr$No, p1 = NA_real_, p2 = NA_real_, p3 = NA_real_, p4 = NA_real_)
diagA <- numeric(5); diagB <- numeric(5)

set.seed(2609)
for (k in 1:5) {
  t0 <- Sys.time()
  trk_A <- tsk_tr[fold != k]; tek_A <- tsk_tr[fold == k]
  trk_B <- lb_tr[fold != k];  tek_B <- lb_tr[fold == k]

  # ---- encoders: reference is folds != k, and training rows are leave-inner-out ----
  attach_cols(trk_A, encode_train(trk_A, encA, ENCA_COLS))
  attach_cols(tek_A, encA(tsk_tr[fold != k], tek_A))
  attach_cols(trk_B, encode_train(trk_B, encB, ENCB_COLS))
  attach_cols(tek_B, encB(lb_tr[fold != k], tek_B))

  # ---- inner early-stopping split, carved by respondent ----------------------
  es_cases <- sample(unique(trk_A$Case), round(0.1 * uniqueN(trk_A$Case)))

  selA <- fitA_select(trk_A[!Case %in% es_cases], trk_A[Case %in% es_cases])
  pA   <- pred_at_best(selA$fit, as.matrix(tek_A[, ..FEAT_A]), margin = FALSE)

  fB <- fitB(trk_B[y != 4 & !(Case %in% es_cases)], trk_B[y != 4 & Case %in% es_cases])
  PB <- softmax3(pred_at_best(fB, as.matrix(tek_B[, ..FEAT_B]), margin = TRUE))

  stopifnot(identical(tek_A$No, unique(tek_B$No)), nrow(PB) == nrow(tek_A))
  pA <- pmin(pmax(pA, 1e-6), 1 - 1e-6)
  P  <- cbind((1 - pA) * PB[, 1], (1 - pA) * PB[, 2], (1 - pA) * PB[, 3], pA)
  idx <- match(tek_A$No, oof$No)
  for (j in 1:4) set(oof, idx, paste0("p", j), P[, j])

  yk <- tek_A$y; rl <- yk != 4
  diagA[k] <- -mean(log(ifelse(yk == 4, pA, 1 - pA)))
  diagB[k] <- -mean(log(pmax(PB[cbind(which(rl), yk[rl])], 1e-15)))
  cat(sprintf("fold %d  depthA=%d (ES %s)  iterA=%s iterB=%s | A(bin)=%.5f B(3way)=%.5f [%.1f min]\n",
              k, selA$depth, selA$all, best_iter(selA$fit), best_iter(fB),
              diagA[k], diagB[k], as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  flush.console()
}

setorder(oof, No)
Pm <- as.matrix(oof[, .(p1, p2, p3, p4)])
stopifnot(!anyNA(Pm), all(abs(rowSums(Pm) - 1) < 1e-8), nrow(oof) == 21565)
ll <- logloss(ytr, Pm)
cat(sprintf("\n>>> xgb_2stage OOF logloss: %.5f   (xgb_lw2 = 1.14152, blend = 1.13556)\n", ll))
cat(sprintf("    stage A mean binary logloss : %.5f\n", mean(diagA)))
cat(sprintf("    stage B mean 3-way logloss  : %.5f\n", mean(diagB)))

# ---- decompose the incumbent the same way, to locate where any difference lives ----
f_lw2 <- "model/artifacts/oof_xgb_lw2.rds"
if (file.exists(f_lw2)) {
  I <- readRDS(f_lw2); setorder(I, No); stopifnot(identical(I$No, oof$No))
  Q  <- as.matrix(I[, .(p1, p2, p3, p4)]); Q <- Q / rowSums(Q)
  qA <- pmin(pmax(Q[, 4], 1e-6), 1 - 1e-6)
  QB <- Q[, 1:3] / rowSums(Q[, 1:3])
  rl <- ytr != 4
  cat(sprintf("    xgb_lw2 implied A (binary)  : %.5f\n", -mean(log(ifelse(ytr == 4, qA, 1 - qA)))))
  cat(sprintf("    xgb_lw2 implied B (3-way)   : %.5f\n",
              -mean(log(pmax(QB[cbind(which(rl), ytr[rl])], 1e-15)))))
}
saveRDS(oof, "model/artifacts/oof_xgb_2stage.rds")

# =============================================================================
# 5. REFIT ON ALL TRAINING DATA -> TEST PREDICTIONS
# =============================================================================
set.seed(77)
attach_cols(tsk_tr, encode_train(tsk_tr, encA, ENCA_COLS))
attach_cols(tsk_te, encA(tsk_tr, tsk_te))
attach_cols(lb_tr,  encode_train(lb_tr, encB, ENCB_COLS))
attach_cols(lb_te,  encB(lb_tr, lb_te))

es_cases <- sample(unique(tsk_tr$Case), round(0.1 * uniqueN(tsk_tr$Case)))
selA <- fitA_select(tsk_tr[!Case %in% es_cases], tsk_tr[Case %in% es_cases])
cat(sprintf("\nfull refit: depthA = %d (ES %s)\n", selA$depth, selA$all))
fB <- fitB(lb_tr[y != 4 & !(Case %in% es_cases)], lb_tr[y != 4 & Case %in% es_cases])

pA <- pmin(pmax(pred_at_best(selA$fit, as.matrix(tsk_te[, ..FEAT_A]), margin = FALSE), 1e-6), 1 - 1e-6)
PB <- softmax3(pred_at_best(fB, as.matrix(lb_te[, ..FEAT_B]), margin = TRUE))
stopifnot(identical(tsk_te$No, unique(lb_te$No)), nrow(PB) == nrow(tsk_te))
tp <- data.table(No = tsk_te$No,
                 p1 = (1 - pA) * PB[, 1], p2 = (1 - pA) * PB[, 2],
                 p3 = (1 - pA) * PB[, 3], p4 = pA)
setorder(tp, No)
stopifnot(nrow(tp) == 4997, !anyNA(tp), all(abs(rowSums(as.matrix(tp[, -1])) - 1) < 1e-8))
saveRDS(tp, "model/artifacts/test_xgb_2stage.rds")

cat("\nstage A top features:\n")
print(head(as.data.frame(xgb.importance(model = selA$fit))[, c("Feature", "Gain")], 12))
cat("\nstage B top features:\n")
print(head(as.data.frame(xgb.importance(model = fB))[, c("Feature", "Gain")], 12))
cat(sprintf("\nmean predicted p4 on test: %.4f   (train none rate 0.3023)\n", mean(tp$p4)))
cat("OK\n")
