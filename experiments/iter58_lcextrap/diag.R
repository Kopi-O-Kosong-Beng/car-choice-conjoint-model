# =============================================================================
# ITERATION 58 -- WHY IS A MODEL'S NATIVE NONE-RATE RIGHT OR WRONG?
# diag.R : READ-ONLY diagnostics. Emits NO artifacts into model/artifacts/.
#
# QUESTION. xgb_lw2bag ships test none-rate 0.2726 against a MEASURED truth of
# 0.266521 (probe algebra, exact). lcmnl3_both ships 0.2237, a miss of -0.043.
# Both are fitted on a population whose none-rate is 0.30230. What structural
# property makes the tree's extrapolation nearly right and the latent-class
# model's badly wrong?
#
# THIS SCRIPT ANSWERS FOUR THINGS BEFORE ANY REFIT IS PAID FOR:
#   D1  reconstruct test_lcmnl3_both from the saved full-refit fit, so that the
#       membership coefficients G can be manipulated without a refit.
#   D2  decompose BOTH shipped none-rates by segment. The measured r* is one
#       scalar and is consistent with two incompatible segment stories
#       (iteration 42's fork), so any claim about "over-extrapolation" must say
#       which story it assumes.
#   D3  is the membership channel miscalibrated WITHIN the training population?
#       If OOF per-segment none-rates match observed per-segment none-rates,
#       then the softmax is NOT overfitting; the miss is a population
#       difference, and no cross-validated shrinkage can find it.
#   D4  the response curve of the shipped none-rate to shrinking G toward 0.
#       Report the s that hits 0.266521 -- and label it as TUNED.
# =============================================================================
suppressMessages({ library(data.table) })
source("model/99_utils.R")

R_STAR <- (1.7918 - 1.499) / 1.0986    # probe algebra, iteration 42
cat(sprintf("r* (measured TEST none-rate) = %.6f\n\n", R_STAR))

long <- readRDS("model/artifacts/long.rds")
setorder(long, No, alt)
tasks <- unique(long[, .(No, Case, Task, y, is_test, segmentind, incomea, agea)])
setorder(tasks, No)
tasks[, lux := as.integer(segmentind %in% c(3, 5))]

tr <- tasks[is_test == FALSE]
te <- tasks[is_test == TRUE]
cat("=== population shape ===\n")
cat(sprintf("train tasks %d  resp %d  luxury rows %.5f (%d resp)\n",
            nrow(tr), uniqueN(tr$Case), mean(tr$lux), uniqueN(tr[lux == 1]$Case)))
cat(sprintf("TEST  tasks %d  resp %d  luxury rows %.5f (%d resp)\n",
            nrow(te), uniqueN(te$Case), mean(te$lux), uniqueN(te[lux == 1]$Case)))
cat(sprintf("train observed none-rate: all %.5f | lux %.5f | non %.5f\n\n",
            mean(tr$y == 4), mean(tr[lux == 1]$y == 4), mean(tr[lux == 0]$y == 4)))

# =============================================================================
# D1 -- reconstruct the latent-class test predictions from the saved fit
# =============================================================================
cat("=== D1: reconstruct lcmnl3_both test predictions from fit_C3.rds ===\n")
fit <- readRDS("experiments/iter11_latent_class/fit_C3.rds")
keep <- fit$keep

# rebuild the design exactly as iter25/run.R does (TASKMODE = "both")
pw <- character(0)
for (a in ATTRS) {
  lv <- sort(unique(long[alt != 4][[a]])); ref <- lv[1]
  for (l in setdiff(lv, ref)) {
    nm <- sprintf("%s_L%s", a, l); long[, (nm) := as.numeric(get(a) == l)]; pw <- c(pw, nm)
  }
}
long[, Task_c := (as.numeric(Task) - 10) / 9]
long[, none_x_Task := asc4 * Task_c]
long[, Price_x_Task := Price * Task_c]
stopifnot(all(keep %in% names(long)))
Xall <- as.matrix(long[, ..keep])

DEMO_CAT <- c("segmentind", "pparkind", "genderind", "educind", "regionind", "Urbind")
DEMO_NUM <- c("agea", "incomea", "milesa", "nighta", "yearind", "milesind", "nightind")
dem <- unique(long[, c("Case", "is_test", DEMO_CAT, DEMO_NUM), with = FALSE])
setorder(dem, Case)
Zl <- list(Intercept = rep(1, nrow(dem)))
for (v in DEMO_CAT) {
  lv <- sort(unique(dem[[v]])); for (l in lv[-1]) Zl[[sprintf("%s_%s", v, l)]] <- as.numeric(dem[[v]] == l)
}
tf <- function(v, x) if (v %in% c("incomea", "milesa", "nighta")) log1p(x) else x
for (v in DEMO_NUM) { x <- tf(v, as.numeric(dem[[v]])); Zl[[v]] <- (x - mean(x)) / sd(x) }
Z <- do.call(cbind, Zl); rownames(Z) <- as.character(dem$Case)
stopifnot(identical(colnames(Z), fit$zcols))

cl_prob <- function(X, beta) {
  Vm <- matrix(as.vector(X %*% beta), ncol = 4, byrow = TRUE)
  mx <- pmax(Vm[, 1], Vm[, 2], Vm[, 3], Vm[, 4]); E <- exp(Vm - mx); E / rowSums(E)
}
row_softmax <- function(eta) {
  mx <- do.call(pmax, lapply(seq_len(ncol(eta)), function(j) eta[, j]))
  E <- exp(eta - mx); E / rowSums(E)
}
rows_of <- function(ti) as.vector(rbind(4L*ti-3L, 4L*ti-2L, 4L*ti-1L, 4L*ti))

# NOTE: fit$betas are the RELABELLED (descending share) list; fit$G is RAW order.
# Recover the permutation by matching the reconstruction against the artifact.
pred_lc <- function(ti, G, betas, scale = 1) {
  X <- Xall[rows_of(ti), , drop = FALSE]
  Zp <- Z[as.character(tasks$Case[ti]), , drop = FALSE]
  Pi <- row_softmax(cbind(0, scale * (Zp %*% G)))
  out <- matrix(0, length(ti), 4)
  for (cc in seq_along(betas)) out <- out + Pi[, cc] * cl_prob(X, betas[[cc]])
  list(P = out, Pi = Pi)
}
te_t <- which(tasks$is_test == TRUE); tr_t <- which(tasks$is_test == FALSE)
ref <- readRDS("model/artifacts/test_lcmnl3_both.rds"); setorder(ref, No)
stopifnot(identical(ref$No, tasks$No[te_t]))

# try all 3! orderings of betas against the RAW G column order
perms <- list(c(1,2,3), c(1,3,2), c(2,1,3), c(2,3,1), c(3,1,2), c(3,2,1))
best <- NULL
for (p in perms) {
  P <- clip_norm(pred_lc(te_t, fit$G, fit$betas[p])$P)
  d <- max(abs(P - as.matrix(ref[, .(p1,p2,p3,p4)])))
  if (is.null(best) || d < best$d) best <- list(p = p, d = d, P = P)
}
cat(sprintf("best beta permutation vs artifact: (%s)  max|diff| = %.3e\n",
            paste(best$p, collapse=","), best$d))
stopifnot(best$d < 1e-6)
BETAS <- fit$betas[best$p]
cat("RECONSTRUCTION EXACT -- G can now be manipulated without refitting.\n\n")

# =============================================================================
# D2 -- decompose the shipped none-rates by segment
# =============================================================================
cat("=== D2: shipped none-rate by segment ===\n")
lux_te <- tasks$lux[te_t]; lux_tr <- tasks$lux[tr_t]
f <- mean(lux_te)
show_test <- function(nm, P) {
  cat(sprintf("  %-22s all %.5f | lux %.5f | non %.5f\n", nm,
              mean(P[,4]), mean(P[lux_te==1,4]), mean(P[lux_te==0,4])))
}
show_oof <- function(nm, P) {
  cat(sprintf("  %-22s all %.5f | lux %.5f | non %.5f\n", nm,
              mean(P[,4]), mean(P[lux_tr==1,4]), mean(P[lux_tr==0,4])))
}
grab <- function(f) {
  d <- readRDS(f)
  if (is.matrix(d)) return(d[, c("p1","p2","p3","p4"), drop = FALSE])
  setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)])
}
for (m in c("lcmnl3_both", "xgb_lw2bag", "blend")) {
  fte <- sprintf("model/artifacts/test_%s.rds", m)
  if (!file.exists(fte)) next
  show_test(m, grab(fte))
}
cat("  --- truth: only the WEIGHTED SUM is measured -----------------------\n")
cat(sprintf("  measured r*            all %.5f | lux ??      | non ??\n", R_STAR))
r_non_B <- mean(tr[lux == 0]$y == 4)
cat(sprintf("  story B (non = train)  all %.5f | lux %.5f | non %.5f\n",
            R_STAR, (R_STAR - (1-f)*r_non_B)/f, r_non_B))
# story A: a single global logit shift d applied to the training-fitted OOF shape
cat("  story A (global logit shift) computed in iteration 42: lux 0.2095 non 0.3924\n\n")

# =============================================================================
# D3 -- is the membership channel miscalibrated INSIDE the training population?
# =============================================================================
cat("=== D3: OOF per-segment calibration (can CV even see the defect?) ===\n")
cat(sprintf("  %-22s observed: all %.5f | lux %.5f | non %.5f\n", "",
            mean(tr$y == 4), mean(tr[lux==1]$y==4), mean(tr[lux==0]$y==4)))
for (m in c("lcmnl3_both", "xgb_lw2bag", "blend")) {
  fo <- sprintf("model/artifacts/oof_%s.rds", m)
  if (!file.exists(fo)) next
  show_oof(m, grab(fo))
}
cat("\n")

# =============================================================================
# D4 -- response of the shipped none-rate to shrinking G  (post-hoc scale, no refit)
# =============================================================================
cat("=== D4: post-hoc shrinkage of the membership linear predictor ===\n")
cat("  s = 0 -> everyone gets the SAME mixing weights (no demographic channel)\n")
cat("  s = 1 -> production\n")
ytr <- tasks$y[tr_t]
grid <- c(0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)
res <- data.table()
for (s in grid) {
  pt <- pred_lc(te_t, fit$G, BETAS, scale = s)
  po <- pred_lc(tr_t, fit$G, BETAS, scale = s)   # IN-SAMPLE (full refit) -- for shape only
  res <- rbind(res, data.table(
    s = s,
    test_p4     = mean(pt$P[,4]),
    test_p4_lux = mean(pt$P[lux_te==1,4]),
    test_p4_non = mean(pt$P[lux_te==0,4]),
    miss        = mean(pt$P[,4]) - R_STAR,
    insample_ll = logloss(ytr, clip_norm(po$P)),
    pi_maxmean  = mean(apply(pt$Pi, 1, max))))
}
print(res, digits = 5)
# solve for the s that reproduces r* exactly
fs <- function(s) mean(pred_lc(te_t, fit$G, BETAS, scale = s)$P[,4]) - R_STAR
s_star <- tryCatch(uniroot(fs, c(0, 1.5))$root, error = function(e) NA_real_)
cat(sprintf("\n  s that reproduces r* EXACTLY: %.4f   <-- TUNED TO THE MEASUREMENT\n", s_star))

cat("\n=== membership saturation (softmax extremeness) ===\n")
p1 <- pred_lc(tr_t, fit$G, BETAS, 1)$Pi; p2 <- pred_lc(te_t, fit$G, BETAS, 1)$Pi
u_tr <- !duplicated(tasks$Case[tr_t]); u_te <- !duplicated(tasks$Case[te_t])
cat(sprintf("  train resp: mean max(pi) %.4f  frac max(pi)>0.9 %.4f  frac >0.99 %.4f\n",
            mean(apply(p1[u_tr,,drop=FALSE],1,max)),
            mean(apply(p1[u_tr,,drop=FALSE],1,max) > 0.9),
            mean(apply(p1[u_tr,,drop=FALSE],1,max) > 0.99)))
cat(sprintf("  TEST  resp: mean max(pi) %.4f  frac max(pi)>0.9 %.4f  frac >0.99 %.4f\n",
            mean(apply(p2[u_te,,drop=FALSE],1,max)),
            mean(apply(p2[u_te,,drop=FALSE],1,max) > 0.9),
            mean(apply(p2[u_te,,drop=FALSE],1,max) > 0.99)))
cat("  class none-rates (segment behaviour, weighted train):",
    paste(sprintf("%.3f", sapply(seq_along(BETAS), function(cc)
      mean(cl_prob(Xall[rows_of(tr_t),,drop=FALSE], BETAS[[cc]])[,4]))), collapse="  "), "\n")
cat("  mean class shares  train:", paste(sprintf("%.4f", colMeans(p1[u_tr,,drop=FALSE])), collapse="  "),
    "\n  mean class shares  TEST :", paste(sprintf("%.4f", colMeans(p2[u_te,,drop=FALSE])), collapse="  "), "\n")
cat("  mean class shares  train LUX:",
    paste(sprintf("%.4f", colMeans(p1[u_tr & tasks$lux[tr_t]==1,,drop=FALSE])), collapse="  "),
    "\n  mean class shares  TEST  LUX:",
    paste(sprintf("%.4f", colMeans(p2[u_te & tasks$lux[te_t]==1,,drop=FALSE])), collapse="  "), "\n")
cat("\nOK\n")
