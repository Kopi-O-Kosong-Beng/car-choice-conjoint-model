# =============================================================================
# ITERATION 05 — Mixed logit, respecified.
#
# WHY v1 FAILED (1.18743, worse than plain MNL): we put a NORMAL prior on the
# price coefficient. Estimated mean -0.19 with sd 0.46 puts roughly a third of the
# simulated population at a POSITIVE price coefficient -- people who prefer paying
# more. That is not heterogeneity, it is a mis-specified sign constraint, and it
# corrupts the population-averaged probabilities we actually predict with.
#
# THREE FIXES, all motivated by earlier iterations:
#   1. Log-normal on NEGATIVE price => coefficient is negative for everyone,
#      by construction. Heterogeneity in magnitude, not in sign.
#   2. log(price) rather than price. Iteration 02's part-worths showed a clearly
#      CONCAVE price response (steps of -0.52, -0.24, -0.13, ... saturating at the
#      top). log-price captures that shape with one parameter instead of eleven.
#   3. Part-worth coding for the other attributes (iteration 02, z = 11.8).
#
# The random none-constant is kept: v1 estimated its sd at ~2.1, which is large and
# credible -- people genuinely differ in willingness to buy anything at all.
#
# Usage: pass a fold 1-5 to fit that fold, or 0 to combine + fit on all data.
#   & "C:\...\Rscript.exe" experiments/iter05_mixl_v2/run.R 1
# =============================================================================
suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")
args <- commandArgs(trailingOnly = TRUE)
K <- as.integer(if (length(args)) args[1] else stop("need fold 1..5, or 0 to combine"))

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

# concave price term; 0 on the all-zero option, whose ASC absorbs the difference
long[, neglprice := ifelse(Price > 0, -log(Price), 0)]

pw_cols <- character(0)
for (a in setdiff(ATTRS, "Price")) {
  lv_real <- sort(unique(long[alt != 4][[a]]))
  for (l in setdiff(lv_real, lv_real[1])) {
    nm <- sprintf("%s_L%s", a, l)
    long[, (nm) := as.numeric(get(a) == l)]
    pw_cols <- c(pw_cols, nm)
  }
}
# identification: drop anything constant within every choice set
cand <- c("asc2", "asc3", "asc4", "neglprice", pw_cols)
X <- as.matrix(long[, ..cand])
Xc <- X - as.matrix(long[, lapply(.SD, mean), by = No, .SDcols = cand][
  match(long$No, unique(long$No)), ..cand])
qrx <- qr(Xc, tol = 1e-7)
keep <- cand[sort(qrx$pivot[seq_len(qrx$rank)])]
if (length(setdiff(cand, keep))) cat("dropped:", paste(setdiff(cand, keep), collapse = ", "), "\n")
xvars <- c(keep, "Price_x_age", "Price_x_inc")
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 0"))

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
setorder(trl, No, alt)

fit_mixl <- function(d) {
  dx <- dfidx(as.data.frame(d), idx = list(c("No", "Case"), "alt"), choice = "chosen")
  mlogit(fml, data = dx, rpar = c(neglprice = "ln", asc4 = "n"),
         R = 100, halton = NA, panel = TRUE)
}
# Population-averaged probabilities: what we need for respondents never seen before.
predict_uncond <- function(fit, d, R = 500, seed = 99) {
  co <- coef(fit)
  stopifnot(all(c("neglprice", "sd.neglprice", "asc4", "sd.asc4") %in% names(co)))
  fixed <- co[setdiff(names(co), c("sd.neglprice", "sd.asc4"))]
  # log-normal coefficient: exp(mu + sigma*z); mu is the coefficient on neglprice
  fixed_nolp <- fixed[setdiff(names(fixed), "neglprice")]
  base_u <- as.vector(as.matrix(as.data.frame(d)[, names(fixed_nolp)]) %*% fixed_nolp)
  set.seed(seed); Z <- matrix(rnorm(R * 2), R)
  acc <- numeric(nrow(d))
  for (r in seq_len(R)) {
    b_lp <- exp(co["neglprice"] + Z[r, 1] * abs(co["sd.neglprice"]))
    u <- base_u + d$neglprice * b_lp + d$asc4 * (Z[r, 2] * abs(co["sd.asc4"]))
    U <- matrix(u, ncol = 4, byrow = TRUE)
    E <- exp(U - pmax(U[,1], U[,2], U[,3], U[,4]))
    acc <- acc + as.vector(t(E / rowSums(E)))
  }
  data.table(No = d$No, alt = d$alt, p = acc / R)
}

if (K >= 1 && K <= 5) {
  t0 <- Sys.time()
  fit <- fit_mixl(trl[fold != K])
  va <- trl[fold == K]; setorder(va, No, alt)
  saveRDS(list(fold = K, pred = predict_uncond(fit, va), coefs = coef(fit)),
          sprintf("model/artifacts/mixl2_fold%d.rds", K))
  cat("fold", K, "done in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")
} else {
  parts <- lapply(1:5, function(k) readRDS(sprintf("model/artifacts/mixl2_fold%d.rds", k))$pred)
  oof <- dcast(rbindlist(parts), No ~ alt, value.var = "p")
  setnames(oof, c("No", "p1", "p2", "p3", "p4")); setorder(oof, No)
  ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
  cat(">>> MIXL v2 OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)])), 5),
      "  (v1: 1.18743, part-worth MNL: 1.15686)\n")
  saveRDS(oof, "model/artifacts/oof_mixl2.rds")
  fit <- fit_mixl(trl)
  tel <- long[is_test == TRUE]; tel[, chosen := FALSE]; setorder(tel, No, alt)
  tp <- dcast(predict_uncond(fit, tel), No ~ alt, value.var = "p")
  setnames(tp, c("No", "p1", "p2", "p3", "p4")); setorder(tp, No)
  saveRDS(tp, "model/artifacts/test_mixl2.rds")
  co <- coef(fit)
  cat("\nprice coefficient is log-normal: median exp(mu) =",
      round(exp(co["neglprice"]), 4), " sigma =", round(abs(co["sd.neglprice"]), 4), "\n")
  cat("none-constant: mean", round(co["asc4"], 3), " sd", round(abs(co["sd.asc4"]), 3), "\n")
  saveRDS(co, "experiments/iter05_mixl_v2/coefs.rds")
  cat("OK\n")
}
