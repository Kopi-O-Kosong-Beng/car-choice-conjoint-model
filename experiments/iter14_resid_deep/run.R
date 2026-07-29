# =============================================================================
# ITERATION 13 — residual-based design encoding. HYPOTHESIS WRITTEN BEFORE RUNNING.
#
# HYPOTHESIS: iteration 01 encoded the RAW empirical choice share per design
# (+0.0046, z = 2.94). But the raw share re-encodes information the attribute
# model already captures; only the part the model MISSES is new. Encoding the
# average RESIDUAL (observed - predicted) per design isolates exactly that, and
# residuals have smaller variance than raw 0/1 choice indicators, so the same
# shrinkage retains more signal per observation. Support is thin (~3.8 rows per
# design), so variance reduction in the thing being averaged is the whole game.
#
# WHY THIS IS THE PRINCIPLED CHOICE, not just another feature. The pool's loss
# gradient in score space is exactly (p - y). So the ONLY functional direction
# that can improve a log-opinion pool is a model of its own residual -- one
# boosting stage on the pool itself. In respondent space that is blocked
# (residual unpredictable from demographics, OOF R^2 = -0.012, measured). In
# design space it is open, and this is that construction. One shot; iterating it
# on the same folds is the textbook route to OOF overfitting and is forbidden.
#
# LEAKAGE CONTROL — the subtle part, and the reason for stage 1.
# The naive version takes residuals from mnl_pw's ordinary OOF predictions. That
# leaks: a respondent in fold j has an OOF prediction from a model trained on all
# folds EXCEPT j -- which INCLUDES fold k. Using their residual to build a feature
# for fold-k rows therefore lets fold k's own labels influence fold k's features,
# via the fitted coefficients. It is second-order and small, but it is exactly the
# kind of thing that silently inflates OOF, so we do not rely on it being small.
# Stage 1 fits mnl_pw on folds NOT IN {j, k} for all 20 ordered pairs, giving, for
# each target fold k, predictions for every other respondent from a model that
# never saw fold k. Those residuals are clean with respect to fold k.
# Test rows use the ordinary 5-fold OOF residuals (no target fold to protect).
#
# ONE CHANGE UNDER TEST: the three residual-encoding columns are ADDED to the
# feature set. Everything else -- folds, objective, bagging protocol, seeds, the
# existing ENC_COLS -- is identical to experiments/iter12_bagged_configs/ bagA.
# Judged bag-vs-bag against oof_xgb_bagA (single-fit tree comparisons are
# inadmissible: the early-stopping carve alone has SD 0.0032).
#
# Emits: oof_xgb_bagResB.rds / test_xgb_bagResB.rds
# Run: Rscript experiments/iter13_residual_encoding/run.R
# Runtime: stage 1 ~20 mnl fits, stage 2 ~30 xgb fits. Background it.
# =============================================================================
suppressMessages({ library(data.table); library(mlogit); library(dfidx); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
dir.create("experiments/iter13_residual_encoding", showWarnings = FALSE, recursive = TRUE)

long <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)

# ---------- stage 1: fold-pair-clean predictions from the part-worth logit -----
# Rebuild mnl_pw's design exactly as model/02_mnl_partworth.R does.
pw_cols <- character(0)
for (a in ATTRS) {
  lv <- sort(unique(long[alt != 4][[a]])); ref <- lv[1]
  for (l in setdiff(lv, ref)) {
    nm <- sprintf("%s_L%s", a, l); long[, (nm) := as.numeric(get(a) == l)]
    pw_cols <- c(pw_cols, nm)
  }
}
cand <- c("asc2", "asc3", "asc4", pw_cols)
X  <- as.matrix(long[, ..cand])
Xc <- X - as.matrix(long[, lapply(.SD, mean), by = No, .SDcols = cand][
  match(long$No, unique(long$No)), ..cand])
qrx  <- qr(Xc, tol = 1e-7)
keep <- cand[sort(qrx$pivot[seq_len(qrx$rank)])]
pw_cols <- setdiff(keep, c("asc2", "asc3", "asc4"))
xvars <- c(pw_cols, "Price_x_age", "Price_x_ppark", "Price_x_inc",
           grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 1"))
cat("stage 1: part-worth params:", length(pw_cols), "\n"); flush.console()

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
setorder(trl, No, alt)
tel <- long[is_test == TRUE]; setorder(tel, No, alt)

fit_predict <- function(dtr, dva) {
  dx  <- dfidx(as.data.frame(dtr), idx = c("No", "alt"), choice = "chosen")
  fit <- mlogit(fml, data = dx)
  d2 <- copy(dva); d2[, chosen := FALSE]; d2[alt == 1, chosen := TRUE]
  P <- predict(fit, newdata = dfidx(as.data.frame(d2), idx = c("No", "alt"), choice = "chosen"))
  list(P = P, nos = sort(unique(dva$No)))
}

PAIR <- "experiments/iter13_residual_encoding/pairpred.rds"
if (file.exists(PAIR)) {
  pairpred <- readRDS(PAIR); cat("stage 1: reusing cached fold-pair predictions\n")
} else {
  pairpred <- list()          # pairpred[[k]] = data.table(No, p1..p4) for folds != k
  for (k in 1:5) {
    acc <- list()
    for (j in setdiff(1:5, k)) {
      r <- fit_predict(trl[!fold %in% c(j, k)], trl[fold == j])
      acc[[length(acc) + 1]] <- data.table(No = r$nos, p1 = r$P[,1], p2 = r$P[,2],
                                           p3 = r$P[,3], p4 = r$P[,4])
    }
    pairpred[[k]] <- rbindlist(acc); setorder(pairpred[[k]], No)
    cat(sprintf("  target fold %d: clean predictions for %d tasks\n",
                k, nrow(pairpred[[k]]))); flush.console()
  }
  saveRDS(pairpred, PAIR)
}
# ordinary OOF (for the TEST-row encoding only; no target fold to protect there)
oof_mnl <- readRDS("model/artifacts/oof_mnl_pw.rds"); setorder(oof_mnl, No)

# ---------- residual encoder --------------------------------------------------
RES_ALPHAS <- c(5, 20, 50)          # residuals are lower-variance; shrink less hard
RES_COLS   <- c(paste0("resid_a", RES_ALPHAS), "resid_n")

# ref: long-format rows WITH a `resid` column; target: rows to encode
encode_resid <- function(ref, target) {
  t2 <- copy(target)[, .rid := .I]
  cnt <- ref[, .(rsum = sum(resid), n = .N), by = .(dkey, alt)]
  t2 <- merge(t2[, .(.rid, dkey, alt)], cnt, by = c("dkey", "alt"), all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, rsum = 0)]
  for (a in RES_ALPHAS) t2[, paste0("resid_a", a) := rsum / (n + a)]   # prior mean 0
  t2[, resid_n := n]
  setorder(t2, .rid)
  t2[, .SD, .SDcols = RES_COLS]
}
attach_resid <- function(rows, pred) {           # pred: data.table(No, p1..p4)
  m <- melt(pred, id.vars = "No", variable.name = "alt", value.name = "p")
  m[, alt := as.integer(sub("^p", "", as.character(alt)))]
  r <- merge(rows[, .(No, alt, chosen, dkey)], m, by = c("No", "alt"))
  r[, resid := as.numeric(chosen) - p]
  setorder(r, No, alt); r
}

for (cc in RES_COLS) { trl[, (cc) := NA_real_]; tel[, (cc) := NA_real_] }
for (k in 1:5) {
  ref <- attach_resid(trl[fold != k], pairpred[[k]])
  e   <- encode_resid(ref, trl[fold == k])
  idx <- which(trl$fold == k)
  for (cc in RES_COLS) set(trl, idx, cc, e[[cc]])
}
ref_all <- attach_resid(trl, oof_mnl)
e <- encode_resid(ref_all, tel)
for (cc in RES_COLS) set(tel, seq_len(nrow(tel)), cc, e[[cc]])
cat("stage 1 done. residual encoding attached.\n"); flush.console()

# ---------- stage 2: bagged xgboost, bagA config + residual columns -----------
apply_design_encoding(trl, tel)
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo,
          ENC_COLS, RES_COLS)

softmax_by_task <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE)
  M <- M - apply(M, 1, max); E <- exp(M); E / rowSums(E) }
obj_listwise <- function(preds, dtrain) { y <- getinfo(dtrain, "label")
  p <- as.vector(t(softmax_by_task(preds))); list(grad = p - y, hess = pmax(2*p*(1-p), 1e-6)) }
eval_tl <- function(preds, dtrain) { y <- getinfo(dtrain, "label")
  P <- softmax_by_task(preds); Y <- matrix(y, ncol = 4, byrow = TRUE)
  list(metric = "tasklogloss", value = -mean(log(pmax(P[Y == 1], 1e-15)))) }
new_api <- packageVersion("xgboost") >= "2.1.0"
gbi <- function(f) { b <- tryCatch(xgb.attr(f, "best_iteration"), error = function(e) NULL)
  if (is.null(b)) b <- tryCatch(f$best_iteration, error = function(e) NULL)
  if (is.null(b) || is.na(suppressWarnings(as.numeric(b)))) NULL else as.integer(b) }
rp <- function(f, d) { dm <- xgb.DMatrix(as.matrix(d[, ..feat])); b <- gbi(f)
  as.vector(if (!is.null(b)) tryCatch(predict(f, dm, iterationrange = c(1, b+1), outputmargin = TRUE),
    error = function(e) predict(f, dm, outputmargin = TRUE)) else predict(f, dm, outputmargin = TRUE)) }
PAR <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
            subsample = 0.8, colsample_bytree = 0.8, base_score = 0, nthread = 0)
tr1 <- function(dtr, des) {
  a <- list(params = PAR, data = xgb.DMatrix(as.matrix(dtr[, ..feat]),
            label = as.numeric(dtr$chosen)), nrounds = 5000,
            early_stopping_rounds = 150, verbose = 0, obj = obj_listwise, maximize = FALSE)
  m <- xgb.DMatrix(as.matrix(des[, ..feat]), label = as.numeric(des$chosen))
  if (new_api) { a$evals <- list(es = m); a$custom_metric <- eval_tl }
  else         { a$watchlist <- list(es = m); a$feval <- eval_tl }
  do.call(xgb.train, a)
}

K <- 5L; SEEDS <- 1000L + seq_len(K)      # identical to iter12 => paired with bagA
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
nos_tr <- unique(long[is_test == FALSE, .(No)])[order(No), No]
nos_te <- sort(unique(tel$No))
ps <- array(NA_real_, dim = c(length(nos_tr), 4, K))
for (k in 1:5) {
  d <- trl[fold == k]; rk <- match(unique(d$No), nos_tr)
  da <- trl[fold != k]; cs <- unique(da$Case); it <- integer(K)
  for (b in seq_len(K)) {
    set.seed(SEEDS[b] + 17L * k)
    es <- sample(cs, round(0.1 * length(cs)))
    f <- tr1(da[!Case %in% es], da[Case %in% es])
    ps[rk, , b] <- softmax_by_task(rp(f, d))
    bb <- gbi(f); it[b] <- if (is.null(bb)) NA_integer_ else bb
  }
  cat(sprintf("  fold %d  iters %s\n", k, paste(it, collapse = "/"))); flush.console()
}
lls <- apply(ps, 3, function(P) logloss(ytr, P))
Pa  <- apply(ps, c(1, 2), mean); Pa <- Pa / rowSums(Pa)
cat(sprintf("  per-seed: %s\n  seed mean %.5f  SD %.5f\n  BAGGED %.5f\n",
            paste(sprintf("%.5f", lls), collapse = " "), mean(lls), sd(lls), logloss(ytr, Pa)))
saveRDS(data.table(No = nos_tr, p1 = Pa[,1], p2 = Pa[,2], p3 = Pa[,3], p4 = Pa[,4]),
        "model/artifacts/oof_xgb_bagResB.rds")

cs <- unique(trl$Case); Pt <- array(NA_real_, dim = c(length(nos_te), 4, K))
for (b in seq_len(K)) {
  set.seed(SEEDS[b] + 977L)
  es <- sample(cs, round(0.1 * length(cs)))
  f <- tr1(trl[!Case %in% es], trl[Case %in% es])
  Pt[, , b] <- softmax_by_task(rp(f, tel))
}
Pte <- apply(Pt, c(1, 2), mean); Pte <- Pte / rowSums(Pte)
stopifnot(nrow(Pte) == 4997, all(abs(rowSums(Pte) - 1) < 1e-9))
saveRDS(data.table(No = nos_te, p1 = Pte[,1], p2 = Pte[,2], p3 = Pte[,3], p4 = Pte[,4]),
        "model/artifacts/test_xgb_bagResB.rds")
cat("\nNOT promoted. Judge with: Rscript model/compare.R xgb_bagA xgb_bagRes\n")
cat("then Rscript model/predict_lb.R mnl_pw xgb_bagRes\nOK\n")
