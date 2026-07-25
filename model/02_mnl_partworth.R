# =============================================================================
# PRODUCTION MODEL 1 of 2 — part-worth conditional logit.  OOF 1.15686
# Contributes ~38% of the final blend. Originally experiments/iter02_partworth/.
# Writes: model/artifacts/{oof,test}_mnl_pw.rds
#
# HYPOTHESIS: we currently code every attribute as a linear numeric, so utility
# is forced to move by a constant amount per level step. But these are 3-7 level
# qualitative tiers (and Price has 12 codes). Standard conjoint practice is to
# estimate a separate part-worth utility per level. Freeing the shape should help
# the linear-utility models most -- they are the ones the constraint binds on.
# Trees can already approximate step functions, so we expect little there.
#
# Cost: 71 part-worth parameters vs 20 linear, on 21,565 choice sets. Comfortable.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/02_mnl_partworth.R
# Runtime: ~10 min. Requires 00_load.R and 01_folds.R to have run first.
# =============================================================================
suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

# ---- build part-worth dummies -----------------------------------------------
# Reference level = the LOWEST level that actually occurs on a real bundle
# (alternatives 1-3). For most attributes that is 0. For Price it is 1, because
# Price is 0 only on the all-zero "none" option -- using 0 as the reference makes
# the price dummies sum to (1 - none), which is collinear with the none-constant.
pw_cols <- character(0)
for (a in ATTRS) {
  lv_real <- sort(unique(long[alt != 4][[a]]))
  ref <- lv_real[1]
  for (l in setdiff(lv_real, ref)) {
    nm <- sprintf("%s_L%s", a, l)
    long[, (nm) := as.numeric(get(a) == l)]
    pw_cols <- c(pw_cols, nm)
  }
  if (ref != 0) cat(sprintf("  note: %s reference level = %s (0 never occurs on a real bundle)\n", a, ref))
}
cat("part-worth parameters before rank check:", length(pw_cols), "\n")

# ---- identification check ----------------------------------------------------
# A conditional logit can only identify columns that VARY WITHIN a choice set.
# Task-demean every candidate column, then use a rank-revealing QR to drop any
# that are linearly dependent. ASCs go first so pivoting keeps them.
cand <- c("asc2", "asc3", "asc4", pw_cols)
X <- as.matrix(long[, ..cand])
Xc <- X - as.matrix(long[, lapply(.SD, mean), by = No, .SDcols = cand][
  match(long$No, unique(long$No)), ..cand])
qrx <- qr(Xc, tol = 1e-7)
keep <- cand[sort(qrx$pivot[seq_len(qrx$rank)])]
dropped <- setdiff(cand, keep)
if (length(dropped)) cat("dropped as unidentified:", paste(dropped, collapse = ", "), "\n")
pw_cols <- setdiff(keep, c("asc2", "asc3", "asc4"))
cat("part-worth parameters:", length(pw_cols), " (vs", length(ATTRS), "linear)\n")

xvars <- c(pw_cols, "Price_x_age", "Price_x_ppark", "Price_x_inc",
           grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 1"))

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]

fit_predict <- function(dtr, dva) {
  dtr_x <- dfidx(as.data.frame(dtr), idx = c("No", "alt"), choice = "chosen")
  fit <- mlogit(fml, data = dtr_x)
  dva2 <- copy(dva); dva2[, chosen := FALSE]; dva2[alt == 1, chosen := TRUE]
  dva_x <- dfidx(as.data.frame(dva2), idx = c("No", "alt"), choice = "chosen")
  list(P = predict(fit, newdata = dva_x), nos = sort(unique(dva$No)), fit = fit)
}

oof <- data.table(No = sort(unique(trl$No)), p1 = NA_real_, p2 = NA_real_,
                  p3 = NA_real_, p4 = NA_real_)
for (k in 1:5) {
  r <- fit_predict(trl[fold != k], trl[fold == k])
  oof[match(r$nos, No), `:=`(p1 = r$P[, 1], p2 = r$P[, 2], p3 = r$P[, 3], p4 = r$P[, 4])]
  cat("fold", k, "done\n")
}
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
cat(">>> MNL part-worth OOF logloss:",
    round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5),
    "  (linear-coded MNL: 1.17731)\n")
saveRDS(oof, "model/artifacts/oof_mnl_pw.rds")

rf <- fit_predict(trl, tel)
saveRDS(data.table(No = rf$nos, p1 = rf$P[, 1], p2 = rf$P[, 2],
                   p3 = rf$P[, 3], p4 = rf$P[, 4]), "model/artifacts/test_mnl_pw.rds")

# ---- report material: is the level response actually nonlinear? -------------
cf <- coef(rf$fit)
cat("\nPrice part-worths by level (utility relative to level 1) -- linearity check:\n")
pr <- cf[grep("^Price_L", names(cf))]
print(round(pr, 4))
cat("\nlargest positive feature part-worths (what buyers actually value):\n")
fw <- cf[grep("^(CC|GN|NS|BU|FA|LD|BZ|FC|FP|RP|PP|KA|SC|TS|NV|MA|LB|AF|HU)_L", names(cf))]
print(round(sort(fw, decreasing = TRUE)[1:10], 4))
saveRDS(cf, "experiments/iter02_partworth/coefs.rds")
cat("OK\n")
