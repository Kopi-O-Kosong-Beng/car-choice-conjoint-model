# =============================================================================
# ITERATION 15, STEP 1 — the properly NESTED (double) out-of-fold part-worth
# baseline.  Produces experiments/iter15_nested_resenc/nested_base.rds.
#
# WHY THIS EXISTS.  iter12 established that a fold-honest reference SET is not
# enough: the reference respondents' PREDICTIONS must also be honest w.r.t. the
# scored fold.  The control (control_leakfree_full.R, artifact xgb_resenc2,
# OOF 1.13721) closes that leak by ARGUMENT — mnl_pw has ~150 global coefficients
# and no design-level features, so it cannot memorise an individual choice set —
# but not by CONSTRUCTION: mnl_pw's OOF prediction for a reference respondent in
# fold j still comes from a model fitted on folds != j, a set that includes the
# scored fold k.
#
# THE NESTED CONSTRUCTION.  For every ordered pair (k, j), k != j:
#     fit the part-worth conditional logit on folds NOT IN {k, j}
#     predict fold j
# Column k of the output matrix therefore holds, for every training row outside
# fold k, a baseline prediction from a model that saw NEITHER that row's own fold
# NOR fold k.  20 fits, ~2 min each.
#
# The design matrix, reference-level choice, rank-revealing QR on the
# task-demeaned matrix and model formula are copied verbatim from
# model/02_mnl_partworth.R so that the only difference from the incumbent
# baseline is which folds each fit sees.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter15_nested_resenc/nested_baseline.R
# Runtime: ~40 min.  Launch in the background.
# =============================================================================
suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

# ---- part-worth dummies (verbatim from model/02_mnl_partworth.R) -------------
pw_cols <- character(0)
for (a in ATTRS) {
  lv_real <- sort(unique(long[alt != 4][[a]]))
  ref <- lv_real[1]
  for (l in setdiff(lv_real, ref)) {
    nm <- sprintf("%s_L%s", a, l)
    long[, (nm) := as.numeric(get(a) == l)]
    pw_cols <- c(pw_cols, nm)
  }
}
cand <- c("asc2", "asc3", "asc4", pw_cols)
X  <- as.matrix(long[, ..cand])
Xc <- X - as.matrix(long[, lapply(.SD, mean), by = No, .SDcols = cand][
  match(long$No, unique(long$No)), ..cand])
qrx <- qr(Xc, tol = 1e-7)
keep <- cand[sort(qrx$pivot[seq_len(qrx$rank)])]
dropped <- setdiff(cand, keep)
if (length(dropped)) cat("dropped as unidentified:", paste(dropped, collapse = ", "), "\n")
pw_cols <- setdiff(keep, c("asc2", "asc3", "asc4"))
cat("part-worth parameters:", length(pw_cols), "\n")

xvars <- c(pw_cols, "Price_x_age", "Price_x_ppark", "Price_x_inc",
           grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 1"))

setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
setorder(trl, No, alt)
stopifnot(nrow(trl) == 21565 * 4)

fit_predict <- function(dtr, dva) {
  dtr_x <- dfidx(as.data.frame(dtr), idx = c("No", "alt"), choice = "chosen")
  fit <- mlogit(fml, data = dtr_x)
  dva2 <- copy(dva); dva2[, chosen := FALSE]; dva2[alt == 1, chosen := TRUE]
  dva_x <- dfidx(as.data.frame(dva2), idx = c("No", "alt"), choice = "chosen")
  list(P = predict(fit, newdata = dva_x), nos = sort(unique(dva$No)))
}

ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)

NB <- matrix(NA_real_, nrow = nrow(trl), ncol = 5)   # col k = baseline honest w.r.t. fold k
t0 <- Sys.time()
for (k in 1:5) {
  for (j in setdiff(1:5, k)) {
    idx <- which(trl$fold == j)
    d   <- trl[fold == j]
    r   <- fit_predict(trl[!fold %in% c(k, j)], d)
    stopifnot(identical(as.numeric(r$nos), as.numeric(unique(d$No))),
              nrow(r$P) == length(idx) / 4)
    NB[idx, k] <- as.vector(t(r$P))
    # per-task sanity: the four predictions must sum to 1
    s <- rowSums(matrix(NB[idx, k], ncol = 4, byrow = TRUE))
    stopifnot(max(abs(s - 1)) < 1e-8)
    ll <- logloss(ymap[No %in% d$No][order(No), y], matrix(NB[idx, k], ncol = 4, byrow = TRUE))
    cat(sprintf("k=%d j=%d  fold-%d logloss %.5f   elapsed %.1f min\n",
                k, j, j, ll, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
    flush.console()
  }
}

# every cell outside the diagonal fold must be filled
for (k in 1:5) stopifnot(!anyNA(NB[trl$fold != k, k]), all(is.na(NB[trl$fold == k, k])))

saveRDS(list(No = trl$No, alt = trl$alt, fold = trl$fold, NB = NB),
        "experiments/iter15_nested_resenc/nested_base.rds")
cat("NESTED_BASELINE_DONE\n")
