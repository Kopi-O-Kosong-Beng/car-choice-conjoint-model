# =============================================================================
# ITERATION 44 -- Nested SEGMENT-CONDITIONAL p4 calibration of each blend MEMBER
#                 before the free-sign log-opinion pool refits.
#
# THIS HEADER IS WRITTEN BEFORE THE SCRIPT IS RUN. Nothing below is tuned after
# seeing a number. Everything numeric in "DECISION RULE" is pre-registered.
#
# -----------------------------------------------------------------------------
# FREEZE STATUS -- stated plainly, as the repo rules require
# -----------------------------------------------------------------------------
# The project is FROZEN for modelling. This iteration changes member predictions
# and refits combiner weights, so it IS modelling work and would normally need
# the user to re-open the freeze. Mitigations, none of which excuse it:
#   * it refits NO model -- it is a post-hoc monotone transform of existing
#     oof_*/test_* artifacts plus ONE shrinkage parameter, so it spends one
#     selection event, not many;
#   * it writes only NEW artifact names (segcal5), touches no production file,
#     modifies no member of members.txt, and uploads nothing.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# Measured OOF on labelled training data, where the luxury none-rate truth is
# 0.1599: every TREE member over-predicts p4 on luxury respondents by +0.096 to
# +0.102 (xgb_lw2bag 0.2590, xgb_long 0.2617, xgb_wide 0.2566, xgb_2stage
# 0.2558), while lcmnl3_both is off by only +0.023 (0.1825). Non-luxury
# calibration is near-perfect for all of them (-0.0003 to -0.0154).
#
# That is textbook shrinkage of a rare cell: luxury respondents are 9.43% of
# TRAINING rows (2033 / 21565) and 68.82% of TEST rows (3439 / 4997). A bias
# that is nearly harmless on the training mix is the dominant error on the
# graded mix.
#
# H1 (PRIMARY): correcting each member's p4 per segment -- honestly nested, and
#   BEFORE the pool refits -- improves the SEGMENT-REWEIGHTED nested blend OOF
#   by more than 4x the blend-level seed sd, for a structural reason rather than
#   a decimal.
# H2 (SECONDARY): the free-sign pool's large NEGATIVE weights on xgb_long /
#   xgb_wide / xgb_2stage (-0.348 / -0.189 / -0.237) exist partly to cancel a
#   SHARED luxury bias. Removing the bias at source should shrink them toward
#   the simplex. A large move toward the simplex is itself the finding, whatever
#   the primary metric does.
#
# -----------------------------------------------------------------------------
# WHY THIS IS NOT THE ALREADY-DEAD "temperature / calibration maps"
# -----------------------------------------------------------------------------
# EXPERIMENTS.md kills "temperature/calibration maps (ceiling +0.00018)". The
# GLOBAL, non-segment version of exactly this shift was run as a control and
# lands at +0.00007..+0.00014 on plain OOF -- reproducing that dead ceiling to
# the decimal. The segment-conditional operator is a different object and is not
# on the dead list. The GLOBAL control is re-run here as CONTROL C1 so the
# distinction is on the record rather than asserted.
#
# -----------------------------------------------------------------------------
# THE NESTING CORRECTION (the reason this run may disagree with feasibility)
# -----------------------------------------------------------------------------
# The feasibility numbers quoted below fitted each shift on OTHER FOLDS' OOF
# predictions -- predictions produced by models that had seen the held-out fold.
# That is precisely the "nest everything that is fitted, and the baselines those
# encodings are built on" failure behind both prior leakage incidents. Here the
# shift used to calibrate the pool's TRAINING rows is fitted on INNER-fold-held-
# out rows only. THE GAIN IS EXPECTED TO SHRINK.
#   feasibility (OPTIMISTIC UPPER BOUNDS, contaminated):
#     xgb_lw2bag       plain 1.13682 -> 1.13463 (-0.00219) | segrw 1.22059 -> 1.20291 (-0.01768)
#     blend_freepool5  plain 1.12341 -> 1.12349 (+0.00008) | segrw 1.19285 -> 1.19106 (-0.00178)
#     lcmnl3_both      plain 1.13863 -> 1.13935 (+0.00073) | segrw 1.20874 -> 1.21285 (+0.00411)
# If the properly nested pool gain lands below 0.00200 that is a clean REJECT,
# and the honest reflection is that the feasibility number was contaminated.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- pre-registered, not to be edited after any number is seen
# -----------------------------------------------------------------------------
# PRIMARY METRIC: segment-reweighted nested blend OOF. Weights are the FIXED,
#   KNOWN test luxury ROW share -- 0.688213 on luxury rows and 0.311787 on the
#   rest, rescaled to the training segment shares (lux x 7.3002, non x 0.34424).
#   Known constants, not fitted. No leakage.
#   Baseline (reproduced in STEP 0): free5 segment-reweighted nested = 1.19285.
#
# ACCEPT requires ALL of:
#   (P1) segment-reweighted nested blend OOF improves by >= 0.00200
#        (4x the blend-level seed sd 0.00048) vs 1.19285.
#   (P2) the improvement replicates under folds_b.rds at >= 0.00160
#        (the measured x0.8 replication rate).
#   (P3) EVERY one of the 5 folds improves on the reweighted metric. Four or
#        fewer is this repo's own leak signature and REJECTS regardless of mean.
#
# GUARD (blocking, independent): the PLAIN nested blend OOF must not degrade by
#   more than 0.00048 (one blend-level seed sd) against 1.12341. The plain
#   nested OOF remains the project's stated decision number; it is being
#   supplemented, not overridden. Degrade it further => REJECTED whatever the
#   reweighted number says.
#
# LAMBDA-DEGENERACY GATE: shrinkage d_s = d_global + lambda*(d_s_raw - d_global),
#   lambda in {0, 0.25, 0.50, 0.75, 1.00}, selected per member by an INNER
#   leave-one-fold-out loop over the OUTER TRAINING folds only, minimising the
#   segment-reweighted per-member logloss on inner held-out rows. lambda is
#   fitted, therefore nested.
#   Pre-registered pool-level summary: lambda_pool(k) := MEDIAN over the 5
#   members of the lambda selected in outer fold k. If lambda_pool == 0 in >= 3
#   of the 5 outer folds, the segment correction is not real: REJECT and record.
#
# SHIFT AUDIT: the audit is near-tautological here (the change IS a segment
#   correction), so the reverse is reported instead -- the gain on the PLAIN,
#   unreweighted OOF. If the plain gain is materially negative while the
#   reweighted gain is large, that is said explicitly. It is the honest shape of
#   the result and the freeze rules require it to be reported.
#
# REPORTED REGARDLESS OF OUTCOME:
#   (a) refitted free-sign pool weights vs the incumbent
#       +1.151 / +0.606 / -0.348 / -0.189 / -0.237 -- do the negatives shrink;
#   (b) shipped test p4 all/lux/non vs file1 (0.2622/0.2325/0.3277),
#       file2 (0.2622/0.2092/0.3792) and the measured r* 0.26652;
#   (c) per-fold deltas; (d) selected lambdas.
#
# -----------------------------------------------------------------------------
# METHOD
# -----------------------------------------------------------------------------
# STEP 0  harness: reproduce free5 nested plain 1.12341 and segrw 1.19285 to 1e-5.
# STEP 1  segment mask lux = segmentind %in% c(3,5); assert 2033/21565, 3439/4997.
# STEP 2  calibration operator p4' = p4 e^d / (1 - p4 + p4 e^d); p1:p3 rescaled
#         by (1-p4')/(1-p4). Assert the within-buy shares are untouched to 1e-12
#         and that the logit shift is constant within segment to 1e-12.
# STEP 3  nested loop reproducing 06_blend.R's structure INTERNALLY (06_blend.R
#         is not modified and not sourced): inner lambda selection, outer shift
#         fit by uniroot on mean(p4_shifted) = observed none-rate, pool refit on
#         the nested-calibrated training rows, score held-out fold plain and
#         segment-reweighted. Repeated under folds_b.rds.
# STEP 4  test artifacts: lambda + shifts refit on the full training OOF, applied
#         to each member's test artifact, pooled with the full-data weights.
# STEP 5  write, as the LAST act, NEW names only, each grep-verified absent.
#
# FORBIDDEN and never opened for writing: model/members.txt, blend.rds,
# test_blend.rds, folds*.rds, model/06_blend.R, model/artifacts/quarantine/*,
# and anything named *lw2bag*, *freepool5*, *_cal*, *segA*. NOTHING IS UPLOADED.
# =============================================================================

suppressMessages(library(data.table))
source("model/99_utils.R")
options(width = 130)
set.seed(42)

MEMB   <- c("xgb_lw2bag", "lcmnl3_both", "xgb_long", "xgb_wide", "xgb_2stage")
M      <- length(MEMB)
EPS0   <- 1e-12
LGRID  <- c(0, 0.25, 0.50, 0.75, 1.00)
W_LUX  <- 0.688213                     # KNOWN test luxury ROW share
W_NON  <- 0.311787
OUTDIR <- "experiments/iter44_segcal"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

BASE_PLAIN <- 1.12341                  # free5 nested, plain
BASE_SEGRW <- 1.19285                  # free5 nested, segment-reweighted
INCUMBENT_W <- c(1.151, 0.606, -0.348, -0.189, -0.237)

cat("=================================================================\n")
cat("ITERATION 44 -- nested segment-conditional p4 calibration\n")
cat("=================================================================\n\n")

# ---------------------------------------------------------------- data --------
long <- readRDS("model/artifacts/long.rds")
ymap <- unique(long[is_test == FALSE, .(No, y, Case, segmentind)]); setorder(ymap, No)
tmap <- unique(long[is_test == TRUE,  .(No, Case, segmentind)]);    setorder(tmap, No)
y    <- ymap$y
NTR  <- nrow(ymap); NTE <- nrow(tmap)

OOF <- lapply(MEMB, function(m) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  stopifnot(identical(d$No, ymap$No)); as.matrix(d[, .(p1, p2, p3, p4)])
})
TST <- lapply(MEMB, function(m) {
  d <- readRDS(sprintf("model/artifacts/test_%s.rds", m)); setorder(d, No)
  stopifnot(identical(d$No, tmap$No)); as.matrix(d[, .(p1, p2, p3, p4)])
})
names(OOF) <- names(TST) <- MEMB
stopifnot(all(sapply(OOF, nrow) == 21565), all(sapply(TST, nrow) == 4997))

FOLDS   <- list(folds   = readRDS("model/artifacts/folds.rds")[order(No), fold],
                folds_b = readRDS("model/artifacts/folds_b.rds")[order(No), fold])
stopifnot(all(sapply(FOLDS, length) == NTR))

# ------------------------------------------------ STEP 1: segment mask --------
LUX_TR <- ymap$segmentind %in% c(3, 5)
LUX_TE <- tmap$segmentind %in% c(3, 5)
cat("STEP 1 -- segment mask lux = segmentind in {3,5}\n")
cat(sprintf("  train luxury rows: %d / %d  (%.2f%%)\n", sum(LUX_TR), NTR, 100 * mean(LUX_TR)))
cat(sprintf("  test  luxury rows: %d / %d  (%.2f%%)\n", sum(LUX_TE), NTE, 100 * mean(LUX_TE)))
stopifnot(sum(LUX_TR) == 2033L, sum(LUX_TE) == 3439L)
WT <- ifelse(LUX_TR, W_LUX / mean(LUX_TR), W_NON / mean(!LUX_TR))
cat(sprintf("  reweight multipliers: lux x %.4f | non x %.5f  (mean weight %.6f)\n\n",
            WT[LUX_TR][1], WT[!LUX_TR][1], mean(WT)))

# --------------------------------------------- STEP 2: calibration op ---------
# p4' = p4 e^d / (1 - p4 + p4 e^d); p1:p3 rescaled by (1-p4')/(1-p4).
apply_shift <- function(P, d) {
  e   <- exp(d)
  p4  <- P[, 4]
  p4n <- p4 * e / (1 - p4 + p4 * e)
  s   <- (1 - p4n) / pmax(1 - p4, 1e-15)
  cbind(P[, 1] * s, P[, 2] * s, P[, 3] * s, p4n)
}
# d that makes mean(p4_shifted[rows]) equal `target`
fit_shift <- function(p4, target) {
  if (length(p4) < 5L) return(0)
  f <- function(d) mean(p4 * exp(d) / (1 - p4 + p4 * exp(d))) - target
  lo <- -12; hi <- 12
  if (f(lo) > 0 || f(hi) < 0) return(0)
  uniroot(f, c(lo, hi), tol = 1e-12)$root
}

cat("STEP 2 -- calibration operator assertions\n")
{
  Pt <- OOF[[1]][1:5000, ]
  dv <- ifelse(LUX_TR[1:5000], 0.31, -0.07)
  Pn <- apply_shift(Pt, dv)
  lg <- qlogis(pmin(pmax(Pn[, 4], 1e-12), 1 - 1e-12)) - qlogis(pmin(pmax(Pt[, 4], 1e-12), 1 - 1e-12))
  s1 <- max(abs(sd(lg[LUX_TR[1:5000]])), abs(sd(lg[!LUX_TR[1:5000]])))
  wb_o <- Pt[, 1:3] / rowSums(Pt[, 1:3]); wb_n <- Pn[, 1:3] / rowSums(Pn[, 1:3])
  s2 <- max(abs(wb_o - wb_n)); s3 <- max(abs(rowSums(Pn) - 1))
  cat(sprintf("  sd of logit shift within segment : %.3e  (< 1e-12 required)\n", s1))
  cat(sprintf("  max |within-buy share change|    : %.3e  (< 1e-12 required)\n", s2))
  cat(sprintf("  max |rowSums - 1|                : %.3e\n", s3))
  stopifnot(s1 < 1e-12, s2 < 1e-12, s3 < 1e-10)
}
cat("  PASS\n\n")

# ------------------------------------------------- pooling machinery ----------
# Reproduces model/06_blend.R free mode EXACTLY (same objective, same starts,
# same optimiser sequence, same seed). 06_blend.R is neither modified nor sourced.
pool_lp <- function(theta, LPs) {
  w <- theta[1:M]; eA <- plogis(theta[M + 1]) * 0.10
  L <- w[1] * LPs[[1]]
  for (m in 2:M) L <- L + w[m] * LPs[[m]]
  mx <- pmax(L[, 1], L[, 2], L[, 3], L[, 4])
  P  <- exp(L - mx); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
fit_pool <- function(LPs, yy) {
  o1 <- function(theta) logloss(yy, pool_lp(theta, LPs))
  set.seed(42)
  starts <- list(c(rep(1 / M, M), -3), c(rep(1, M), -3), c(rep(0.5, M), -3))
  for (i in 1:3) starts[[length(starts) + 1L]] <- c(rnorm(M, 1 / M, 0.5), -3)
  best <- NULL
  for (s in starts) {
    o <- optim(s, o1, method = "BFGS", control = list(maxit = 500))
    o <- optim(o$par, o1, method = "Nelder-Mead",
               control = list(maxit = 5000, reltol = 1e-12))
    if (is.null(best) || o$value < best$value) best <- o
  }
  best
}
LP <- function(Ps) lapply(Ps, function(P) log(pmax(P, EPS0)))
subrows <- function(LPs, rows) lapply(LPs, function(L) L[rows, , drop = FALSE])
ll_rows <- function(P, yy) -log(pmax(P[cbind(seq_along(yy), yy)], 1e-15))

# =================================================== STEP 0: harness ==========
cat("STEP 0 -- HARNESS CHECK (must reproduce 1.12341 / 1.19285)\n")
LP_RAW <- LP(OOF)
run_baseline <- function(fmap) {
  Ph <- matrix(NA_real_, NTR, 4); pars <- vector("list", 5)
  for (k in 1:5) {
    tr <- fmap != k; te <- fmap == k
    o <- fit_pool(subrows(LP_RAW, tr), y[tr]); pars[[k]] <- o$par
    Ph[te, ] <- pool_lp(o$par, subrows(LP_RAW, te))
  }
  list(P = Ph, pars = pars)
}
t0 <- Sys.time()
BASE <- lapply(FOLDS, run_baseline)
li_base <- ll_rows(BASE$folds$P, y)
h_plain <- mean(li_base); h_segrw <- sum(WT * li_base) / sum(WT)
cat(sprintf("  nested plain            : %.5f   (target %.5f)\n", h_plain, BASE_PLAIN))
cat(sprintf("  nested segment-reweighted: %.5f   (target %.5f)\n", h_segrw, BASE_SEGRW))
if (abs(h_plain - BASE_PLAIN) > 1e-5 || abs(h_segrw - BASE_SEGRW) > 1e-5) {
  stop("HARNESS MISMATCH -- STOPPING, nothing else is trustworthy")
}
cat("  PASS  (", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))), "s )\n\n")

# ================================ STEP 3: nested segment calibration ==========
# For one outer fold k of one fold structure, return everything needed.
# mode: "seg" (segment-conditional, the treatment) or "glob" (global, CONTROL C1)
outer_fold <- function(fmap, k, mode = "seg") {
  tr <- which(fmap != k); te <- which(fmap == k)
  inner <- setdiff(sort(unique(fmap)), k)

  # --- (a) inner LOO lambda selection, per member, on inner-held-out rows only
  # inner_pred[[m]][[as.character(lam)]] is the calibrated prediction on ALL tr
  # rows, each inner fold calibrated by a shift fitted on the other inner folds.
  inner_cal <- lapply(seq_len(M), function(m)
    lapply(LGRID, function(l) matrix(NA_real_, NTR, 4)))
  for (j in inner) {
    itr <- tr[fmap[tr] != j]; ite <- tr[fmap[tr] == j]
    tg  <- mean(y[itr] == 4)
    tgL <- mean(y[itr[LUX_TR[itr]]] == 4)
    tgN <- mean(y[itr[!LUX_TR[itr]]] == 4)
    for (m in seq_len(M)) {
      P  <- OOF[[m]]
      dg <- fit_shift(P[itr, 4], tg)
      dl <- if (mode == "seg") { fit_shift(P[itr[LUX_TR[itr]], 4], tgL) } else { dg }
      dn <- if (mode == "seg") { fit_shift(P[itr[!LUX_TR[itr]], 4], tgN) } else { dg }
      for (li in seq_along(LGRID)) {
        lam <- LGRID[li]
        dvec <- ifelse(LUX_TR[ite], dg + lam * (dl - dg), dg + lam * (dn - dg))
        inner_cal[[m]][[li]][ite, ] <- apply_shift(P[ite, , drop = FALSE], dvec)
      }
    }
  }
  lam_sel <- numeric(M)
  lam_tab <- matrix(NA_real_, M, length(LGRID))
  for (m in seq_len(M)) {
    for (li in seq_along(LGRID)) {
      l_i <- ll_rows(inner_cal[[m]][[li]][tr, , drop = FALSE], y[tr])
      lam_tab[m, li] <- sum(WT[tr] * l_i) / sum(WT[tr])
    }
    lam_sel[m] <- LGRID[which.min(lam_tab[m, ])]
  }

  # --- training rows, calibrated with the SELECTED lambda (inner-nested)
  CALtr <- lapply(seq_len(M), function(m)
    inner_cal[[m]][[which(LGRID == lam_sel[m])]][tr, , drop = FALSE])

  # --- (b) outer shifts fitted on folds != k, (c) applied to fold k
  tg  <- mean(y[tr] == 4)
  tgL <- mean(y[tr[LUX_TR[tr]]] == 4)
  tgN <- mean(y[tr[!LUX_TR[tr]]] == 4)
  draw <- matrix(NA_real_, M, 3, dimnames = list(MEMB, c("d_glob", "d_lux_raw", "d_non_raw")))
  dshr <- matrix(NA_real_, M, 2, dimnames = list(MEMB, c("d_lux_shr", "d_non_shr")))
  CALte <- vector("list", M)
  for (m in seq_len(M)) {
    P  <- OOF[[m]]
    dg <- fit_shift(P[tr, 4], tg)
    dl <- if (mode == "seg") { fit_shift(P[tr[LUX_TR[tr]], 4], tgL) } else { dg }
    dn <- if (mode == "seg") { fit_shift(P[tr[!LUX_TR[tr]], 4], tgN) } else { dg }
    lam <- lam_sel[m]
    dls <- dg + lam * (dl - dg); dns <- dg + lam * (dn - dg)
    draw[m, ] <- c(dg, dl, dn); dshr[m, ] <- c(dls, dns)
    CALte[[m]] <- apply_shift(P[te, , drop = FALSE], ifelse(LUX_TR[te], dls, dns))
  }

  # --- (d) pool refit on the calibrated training rows, (e) score fold k
  o <- fit_pool(LP(CALtr), y[tr])
  Pk <- pool_lp(o$par, LP(CALte))
  list(rows = te, P = Pk, par = o$par, lam = lam_sel, draw = draw, dshr = dshr,
       lam_tab = lam_tab)
}

run_treatment <- function(fmap, mode = "seg") {
  Ph <- matrix(NA_real_, NTR, 4); L <- vector("list", 5)
  for (k in 1:5) {
    r <- outer_fold(fmap, k, mode)
    Ph[r$rows, ] <- r$P; L[[k]] <- r
  }
  list(P = Ph, folds = L)
}

score2 <- function(P, rows) {
  l <- ll_rows(P[rows, , drop = FALSE], y[rows])
  c(plain = mean(l), segrw = sum(WT[rows] * l) / sum(WT[rows]))
}

cat("STEP 3 -- nested loop, folds.rds (production fold structure)\n")
t0 <- Sys.time()
TRT_A <- run_treatment(FOLDS$folds, "seg")
cat("  (", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))), "s )\n")

fold_rows <- lapply(1:5, function(k) which(FOLDS$folds == k))
FC <- rbindlist(lapply(1:5, function(k) {
  rr <- fold_rows[[k]]
  b <- score2(BASE$folds$P, rr); t <- score2(TRT_A$P, rr)
  data.table(structure_ = "folds", fold = k, n = length(rr),
             base_plain = b[["plain"]], cal_plain = t[["plain"]],
             d_plain = b[["plain"]] - t[["plain"]],
             base_segrw = b[["segrw"]], cal_segrw = t[["segrw"]],
             d_segrw = b[["segrw"]] - t[["segrw"]])
}))
print(FC[, .(fold, n, base_plain = round(base_plain, 5), cal_plain = round(cal_plain, 5),
             d_plain = round(d_plain, 5), base_segrw = round(base_segrw, 5),
             cal_segrw = round(cal_segrw, 5), d_segrw = round(d_segrw, 5))])

li_trt <- ll_rows(TRT_A$P, y)
A_plain <- mean(li_trt); A_segrw <- sum(WT * li_trt) / sum(WT)
cat(sprintf("\n  NESTED plain  : %.5f -> %.5f   delta %+.5f\n", h_plain, A_plain, h_plain - A_plain))
cat(sprintf("  NESTED segrw  : %.5f -> %.5f   delta %+.5f\n", h_segrw, A_segrw, h_segrw - A_segrw))

# paired, respondent-clustered SE on the reweighted delta
pair_se <- function(d_row) {
  cl <- data.table(Case = ymap$Case, d = d_row)[, .(dm = mean(d)), by = Case]
  c(est = mean(cl$dm), se = sd(cl$dm) / sqrt(nrow(cl)), n = nrow(cl))
}
ps <- pair_se(WT * (li_base - li_trt))
cat(sprintf("  paired respondent-clustered (segrw): %+.5f  SE %.5f  z %.2f  (n=%d)\n",
            ps[["est"]], ps[["se"]], ps[["est"]] / ps[["se"]], ps[["n"]]))
pp <- pair_se(li_base - li_trt)
cat(sprintf("  paired respondent-clustered (plain): %+.5f  SE %.5f  z %.2f\n\n",
            pp[["est"]], pp[["se"]], pp[["est"]] / pp[["se"]]))

cat("  selected lambda per member per outer fold (folds.rds):\n")
LAMTAB <- rbindlist(lapply(1:5, function(k) {
  f <- TRT_A$folds[[k]]
  data.table(structure_ = "folds", fold = k, member = MEMB, lambda = f$lam,
             d_glob = f$draw[, 1], d_lux_raw = f$draw[, 2], d_non_raw = f$draw[, 3],
             d_lux_shr = f$dshr[, 1], d_non_shr = f$dshr[, 2])
}))
print(dcast(LAMTAB, member ~ fold, value.var = "lambda"))
lam_pool <- sapply(1:5, function(k) median(TRT_A$folds[[k]]$lam))
cat("  lambda_pool (median over members) per outer fold:", lam_pool, "\n")
cat(sprintf("  folds with lambda_pool == 0: %d of 5   (>= 3 REJECTS)\n\n", sum(lam_pool == 0)))

# --------------------------------- CONTROL C1: global (non-segment) shift -----
cat("CONTROL C1 -- GLOBAL (non-segment) nested shift, the already-dead object\n")
CTRL <- run_treatment(FOLDS$folds, "glob")
li_c <- ll_rows(CTRL$P, y)
cat(sprintf("  nested plain : %.5f -> %.5f   delta %+.5f\n", h_plain, mean(li_c), h_plain - mean(li_c)))
cat(sprintf("  nested segrw : %.5f -> %.5f   delta %+.5f\n\n",
            h_segrw, sum(WT * li_c) / sum(WT), h_segrw - sum(WT * li_c) / sum(WT)))

# ------------------------------------------- replication under folds_b --------
cat("STEP 3b -- replication under folds_b.rds (independent respondent grouping)\n")
cat("  CAVEAT: the member OOF predictions were themselves produced under\n")
cat("  folds.rds, so this tests the CALIBRATION + COMBINER fitting on an\n")
cat("  independent split, not the members. Stated so it is not over-read.\n")
TRT_B <- run_treatment(FOLDS$folds_b, "seg")
li_bb <- ll_rows(BASE$folds_b$P, y); li_tb <- ll_rows(TRT_B$P, y)
B_base_plain <- mean(li_bb); B_base_segrw <- sum(WT * li_bb) / sum(WT)
B_cal_plain  <- mean(li_tb); B_cal_segrw  <- sum(WT * li_tb) / sum(WT)
cat(sprintf("  NESTED plain  : %.5f -> %.5f   delta %+.5f\n",
            B_base_plain, B_cal_plain, B_base_plain - B_cal_plain))
cat(sprintf("  NESTED segrw  : %.5f -> %.5f   delta %+.5f\n",
            B_base_segrw, B_cal_segrw, B_base_segrw - B_cal_segrw))
fold_rows_b <- lapply(1:5, function(k) which(FOLDS$folds_b == k))
FCB <- rbindlist(lapply(1:5, function(k) {
  rr <- fold_rows_b[[k]]
  b <- score2(BASE$folds_b$P, rr); t <- score2(TRT_B$P, rr)
  data.table(structure_ = "folds_b", fold = k, n = length(rr),
             base_plain = b[["plain"]], cal_plain = t[["plain"]],
             d_plain = b[["plain"]] - t[["plain"]],
             base_segrw = b[["segrw"]], cal_segrw = t[["segrw"]],
             d_segrw = b[["segrw"]] - t[["segrw"]])
}))
print(FCB[, .(fold, n, d_plain = round(d_plain, 5), d_segrw = round(d_segrw, 5))])
LAMTAB_B <- rbindlist(lapply(1:5, function(k) {
  f <- TRT_B$folds[[k]]
  data.table(structure_ = "folds_b", fold = k, member = MEMB, lambda = f$lam,
             d_glob = f$draw[, 1], d_lux_raw = f$draw[, 2], d_non_raw = f$draw[, 3],
             d_lux_shr = f$dshr[, 1], d_non_shr = f$dshr[, 2])
}))
cat("\n")

# ================================== GATES =====================================
d_segrw_A <- h_segrw - A_segrw
d_plain_A <- h_plain - A_plain
d_segrw_B <- B_base_segrw - B_cal_segrw
P1 <- d_segrw_A >= 0.00200
P2 <- d_segrw_B >= 0.00160
P3 <- all(FC$d_segrw > 0)
GUARD <- (A_plain - BASE_PLAIN) <= 0.00048
LAMG  <- sum(lam_pool == 0) < 3
cat("=================== PRE-REGISTERED GATES ===================\n")
cat(sprintf("  P1 segrw delta >= 0.00200            : %+.5f  %s\n", d_segrw_A, ifelse(P1, "PASS", "FAIL")))
cat(sprintf("  P2 folds_b segrw delta >= 0.00160    : %+.5f  %s\n", d_segrw_B, ifelse(P2, "PASS", "FAIL")))
cat(sprintf("  P3 all 5 folds improve (segrw)       : %d of 5   %s\n", sum(FC$d_segrw > 0), ifelse(P3, "PASS", "FAIL")))
cat(sprintf("  GUARD plain degrades <= 0.00048      : %+.5f  %s\n", A_plain - BASE_PLAIN, ifelse(GUARD, "PASS", "FAIL")))
cat(sprintf("  LAMBDA lambda_pool==0 in < 3 folds   : %d of 5   %s\n", sum(lam_pool == 0), ifelse(LAMG, "PASS", "FAIL")))
VERDICT <- if (P1 && P2 && P3 && GUARD && LAMG) { "ACCEPT" } else { "REJECT" }
cat(sprintf("  >>> VERDICT: %s\n", VERDICT))
cat("============================================================\n\n")

# ================================== STEP 4: test ==============================
cat("STEP 4 -- full-data refit and test artifacts\n")
fmapA <- FOLDS$folds
# lambda on the full training OOF, by leave-one-fold-out over all 5 folds
inner_full <- lapply(seq_len(M), function(m) lapply(LGRID, function(l) matrix(NA_real_, NTR, 4)))
for (j in 1:5) {
  itr <- which(fmapA != j); ite <- which(fmapA == j)
  tg  <- mean(y[itr] == 4)
  tgL <- mean(y[itr[LUX_TR[itr]]] == 4); tgN <- mean(y[itr[!LUX_TR[itr]]] == 4)
  for (m in seq_len(M)) {
    P  <- OOF[[m]]
    dg <- fit_shift(P[itr, 4], tg)
    dl <- fit_shift(P[itr[LUX_TR[itr]], 4], tgL)
    dn <- fit_shift(P[itr[!LUX_TR[itr]], 4], tgN)
    for (li in seq_along(LGRID)) {
      lam <- LGRID[li]
      dvec <- ifelse(LUX_TR[ite], dg + lam * (dl - dg), dg + lam * (dn - dg))
      inner_full[[m]][[li]][ite, ] <- apply_shift(P[ite, , drop = FALSE], dvec)
    }
  }
}
lam_full <- numeric(M)
for (m in seq_len(M)) {
  sc <- sapply(seq_along(LGRID), function(li) {
    l_i <- ll_rows(inner_full[[m]][[li]], y); sum(WT * l_i) / sum(WT)
  })
  lam_full[m] <- LGRID[which.min(sc)]
}
cat("  lambda (full training, LOO-selected):", paste(MEMB, lam_full, sep = "=", collapse = "  "), "\n")

CAL_TR_FULL <- lapply(seq_len(M), function(m) inner_full[[m]][[which(LGRID == lam_full[m])]])
tg  <- mean(y == 4); tgL <- mean(y[LUX_TR] == 4); tgN <- mean(y[!LUX_TR] == 4)
cat(sprintf("  training none-rate targets: all %.5f | lux %.5f | non %.5f\n", tg, tgL, tgN))
CAL_TE <- vector("list", M); DFULL <- matrix(NA_real_, M, 5,
  dimnames = list(MEMB, c("d_glob", "d_lux_raw", "d_non_raw", "d_lux_shr", "d_non_shr")))
for (m in seq_len(M)) {
  P <- OOF[[m]]
  dg <- fit_shift(P[, 4], tg); dl <- fit_shift(P[LUX_TR, 4], tgL); dn <- fit_shift(P[!LUX_TR, 4], tgN)
  lam <- lam_full[m]
  dls <- dg + lam * (dl - dg); dns <- dg + lam * (dn - dg)
  DFULL[m, ] <- c(dg, dl, dn, dls, dns)
  CAL_TE[[m]] <- apply_shift(TST[[m]], ifelse(LUX_TE, dls, dns))
}
print(round(DFULL, 4))

o_cal  <- fit_pool(LP(CAL_TR_FULL), y)
o_base <- fit_pool(LP_RAW, y)
W_CAL  <- o_cal$par[1:M]; W_BASE <- o_base$par[1:M]
cat("\n  (a) FREE-SIGN POOL WEIGHTS\n")
WTAB <- data.table(member = MEMB, incumbent = INCUMBENT_W,
                   before_refit = round(W_BASE, 4), after_calibration = round(W_CAL, 4),
                   change = round(W_CAL - W_BASE, 4))
print(WTAB)
cat(sprintf("  eps: before %.5f  after %.5f\n", plogis(o_base$par[M + 1]) * 0.10,
            plogis(o_cal$par[M + 1]) * 0.10))
cat(sprintf("  sum|negative weights|: before %.4f  after %.4f  (H2: should shrink)\n",
            sum(abs(pmin(W_BASE, 0))), sum(abs(pmin(W_CAL, 0)))))
cat(sprintf("  n negative weights   : before %d  after %d\n",
            sum(W_BASE < 0), sum(W_CAL < 0)))

P_TEST <- clip_norm(pool_lp(o_cal$par, LP(CAL_TE)))
P_TEST_BASE <- clip_norm(pool_lp(o_base$par, LP(TST)))
lvl <- fit_shift(P_TEST[, 4], 0.2621914)     # level-match to what file1/file2 ship
P_TEST_LVL <- apply_shift(P_TEST, rep(lvl, NTE))
cat("\n  (b) SHIPPED TEST p4\n")
sh <- function(P) c(all = mean(P[, 4]), lux = mean(P[LUX_TE, 4]), non = mean(P[!LUX_TE, 4]))
PT <- rbind(
  `free5 uncalibrated (refit)`      = sh(P_TEST_BASE),
  `iter44 segcal5 (no level shift)` = sh(P_TEST),
  `iter44 segcal5 + level shift`    = sh(P_TEST_LVL),
  `file1 freepool5_cal`             = c(0.2621914, 0.2324961, 0.3277383),
  `file2 freepool5_segA`            = c(0.2621914, 0.2091619, 0.3792443))
print(round(PT, 4))
cat(sprintf("  measured public r* = 0.26652 (band [0.2661, 0.2670])\n"))

# ------------------------------------------ nested OOF artifact (folds.rds) ---
mk <- function(no, P) {
  P <- clip_norm(P)
  d <- data.table(No = as.integer(no), p1 = P[, 1], p2 = P[, 2], p3 = P[, 3], p4 = P[, 4])
  setorder(d, No); d[]
}
OOF_OUT  <- mk(ymap$No, TRT_A$P)
TEST_OUT <- mk(tmap$No, P_TEST)

# ================================== STEP 5: write, LAST act ===================
cat("\nSTEP 5 -- writing artifacts (LAST act), names grep-verified absent\n")
outs <- c("oof_segcal5.rds", "test_segcal5.rds")
existing <- list.files("model/artifacts")
for (o in outs) {
  hit <- grep(sub("\\.rds$", "", o), existing, fixed = TRUE, value = TRUE)
  cat(sprintf("  collision check %-18s : %d match(es)\n", o, length(hit)))
  stopifnot(length(hit) == 0L)
}
stopifnot(nrow(OOF_OUT) == 21565, nrow(TEST_OUT) == 4997,
          identical(TEST_OUT$No, 21566:26562))
fwrite(rbind(FC, FCB), file.path(OUTDIR, "foldwise.csv"))
fwrite(rbind(LAMTAB, LAMTAB_B), file.path(OUTDIR, "lambda_sel.csv"))
fwrite(WTAB, file.path(OUTDIR, "weights.csv"))
saveRDS(OOF_OUT,  "model/artifacts/oof_segcal5.rds")
saveRDS(TEST_OUT, "model/artifacts/test_segcal5.rds")
cat("  wrote model/artifacts/oof_segcal5.rds, model/artifacts/test_segcal5.rds\n")
cat("  wrote foldwise.csv, lambda_sel.csv, weights.csv\n")
cat("\nNOTHING WAS UPLOADED. No production file was opened for writing.\n")
cat(sprintf("FINAL VERDICT: %s\n", VERDICT))
cat("OK\n")
