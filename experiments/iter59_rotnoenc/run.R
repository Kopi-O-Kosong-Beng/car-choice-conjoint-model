# =============================================================================
# ITERATION 59 -- IS THE ROTATION GAIN REAL, OR IS IT AMPLIFIED ENCODING LEAK?
#
# Written BEFORE any number here is looked at. THIS BLOCKS A SUBMISSION.
#
# -----------------------------------------------------------------------------
# THE THREAT
# -----------------------------------------------------------------------------
# Iteration 48 established that model/encode_design.R's ENC_COLS carry a
# structural leak: apply_design_encoding() is called ONCE, BEFORE the CV loop,
# so a training row in fold j is encoded from folds != j -- a set containing the
# scored fold k. n_ch = C - c_j exactly, the 4-column tuple is invertible, and
# with a MEDIAN of 1 row per (cell, fold) the leave-own-fold-out rule degenerates
# to leave-own-ROW-out for 46.3% of rows. Honest value of the encoding is
# NEGATIVE (-0.00596, z -3.53); the +0.0218 it appears to be worth is the
# complement identity.
#
# The leak's size is MONOTONE IN TREE CAPACITY with no saturation:
#     d4 -0.0004 | d6 +0.0066 | d8 +0.0224 | d10 +0.0394 | d10/mcw10/1400 +0.0777
#
# Iteration 54's rotation ADDS EFFECTIVE CAPACITY -- that is its stated
# mechanism: oblique cuts reach partitions that axis-aligned splits cannot reach
# inside 540 rounds. Cell isolation is exactly such a partition. And the
# rotation gain grew monotonically in k (dsvd8 -> 16 -> 32 -> 72 -> random),
# which is the same signature. xgb_rot's 1.11363 sits in the same range as the
# maximally leak-exploiting config (1.11799-1.11980), which scores 1.19632 once
# the encoding is removed.
#
# NONE OF ITERATION 54-57'S GATES COULD HAVE CAUGHT THIS.
#   * folds_b re-runs the SAME one-shot encoding, so it replicates the leak.
#   * the leak is uniform across folds (5/5, every depth, every seed), so this
#     repo's "a leak concentrates in one fold" heuristic is blind to it.
#   * segment reweighting is blind to it -- the leak is population-independent.
# A 91% folds_b retention and 105% reweighted retention are therefore consistent
# with the gain being entirely leak. Passing five gates means nothing here.
#
# ONE PIECE OF COUNTER-EVIDENCE, recorded so it is not forgotten if the verdict
# goes the other way: the rotation blend ships mean p4 = 0.27145 against the
# probe-MEASURED 0.26652, while production ships 0.24800. A pure leak-exploiter
# has no reason to land closer to a quantity measured on the graded population.
#
# -----------------------------------------------------------------------------
# THE TEST
# -----------------------------------------------------------------------------
# Four arms, one factor at a time, everything else identical (depth 8, mcw 20,
# eta .03, 540 fixed rounds, subsample/colsample .8, same seeds, same folds):
#     base_enc    rot_enc      <- iteration 54's comparison, reproduced
#     base_noenc  rot_noenc    <- the same comparison with ENC_COLS DELETED
#
# DECISION RULE -- fixed before running. Let
#     G_enc   = base_enc   - rot_enc      (gain with the leaky encoding present)
#     G_noenc = base_noenc - rot_noenc    (gain with the leak channel removed)
#   1. G_noenc >= 0.60 * G_enc  -> the rotation is REAL and mostly independent of
#      the encoding. The candidate may ship, on the no-encoding arm's numbers.
#   2. 0.20 * G_enc <= G_noenc < 0.60 * G_enc -> partly leak. Do NOT ship the
#      current candidate; rebuild without ENC_COLS and re-gate from scratch.
#   3. G_noenc < 0.20 * G_enc -> the rotation is a leak amplifier. REJECT, and
#      the correct read is that iteration 54-58 measured the leak, not a model.
#   ALSO REPORTED, because it is the number that would actually ship: the
#   absolute OOF of rot_noenc. If rot_noenc is worse than production's HONEST
#   ~1.1359, there is nothing here regardless of the ratio.
#
# NO ARTIFACTS to model/artifacts. Measurement only. Nothing is uploaded.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R"); source("model/encode_design.R")

DIR <- "experiments/iter59_rotnoenc"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
NR <- 540L; SEEDS <- 1:3; K <- 71L
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
long <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)          # exactly as production does it -- leak included

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
FEAT_ENC <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
              "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, ENC_COLS)
FEAT_NOENC <- setdiff(FEAT_ENC, ENC_COLS)
cat(sprintf("  with encoding %d features | without %d (dropped: %s)\n",
            length(FEAT_ENC), length(FEAT_NOENC), paste(ENC_COLS, collapse = ", ")))

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
NOS <- sort(unique(trl$No))
lev <- lapply(ATTRS, function(a) sort(unique(rbind(trl[, ..ATTRS], tel[, ..ATTRS])[[a]])))
names(lev) <- ATTRS
OH <- do.call(cbind, lapply(ATTRS, function(a) {
  M <- outer(trl[[a]], lev[[a]], "==") * 1.0; colnames(M) <- paste0("oh_", a, "_L", lev[[a]]); M }))
rot_cols <- paste0("rot", seq_len(K))

sbt <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
                     E <- exp(M); E / rowSums(E) }
obj_lw <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(sbt(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
fitpred <- function(d, v, fs, seed) {
  prm <- list(eta = 0.03, max_depth = 8, min_child_weight = 20, subsample = 0.8,
              colsample_bytree = 0.8, base_score = 0, nthread = 4, seed = seed)
  f <- xgb.train(params = prm,
                 data = xgb.DMatrix(as.matrix(d[, ..fs]), label = as.numeric(d$chosen)),
                 nrounds = NR, verbose = 0, obj = obj_lw, maximize = FALSE)
  sbt(as.vector(predict(f, xgb.DMatrix(as.matrix(v[, ..fs])), outputmargin = TRUE)))
}
one <- function(rot, enc, seed) {
  fs <- if (enc) { FEAT_ENC } else { FEAT_NOENC }
  P <- matrix(NA_real_, 21565L, 4L)
  for (k in 1:5) {
    ins <- which(trl$fold != k); out <- which(trl$fold == k)
    if (!rot) {
      P[match(unique(trl$No[out]), NOS), ] <- fitpred(trl[ins], trl[out], fs, seed)
    } else {
      set.seed(1000L + seed)
      Q <- qr.Q(qr(matrix(rnorm(ncol(OH) * K), ncol(OH), K)))
      Z <- sweep(OH, 2, colMeans(OH[ins, , drop = FALSE]), "-") %*% Q
      d <- copy(trl[ins]); v <- copy(trl[out])
      for (j in seq_len(K)) { set(d, NULL, rot_cols[j], Z[ins, j]); set(v, NULL, rot_cols[j], Z[out, j]) }
      P[match(unique(trl$No[out]), NOS), ] <- fitpred(d, v, c(fs, rot_cols), seed)
    }
  }
  logloss(ytr, P)
}

rule("FOUR ARMS x 3 SEEDS -- one factor at a time")
res <- rbindlist(lapply(SEEDS, function(s) {
  be <- one(FALSE, TRUE,  s); re <- one(TRUE, TRUE,  s)
  bn <- one(FALSE, FALSE, s); rn <- one(TRUE, FALSE, s)
  cat(sprintf("  seed %d | ENC base %.5f rot %.5f (gain %+.5f) | NOENC base %.5f rot %.5f (gain %+.5f)\n",
              s, be, re, re - be, bn, rn, rn - bn))
  data.table(seed = s, base_enc = be, rot_enc = re, base_noenc = bn, rot_noenc = rn)
}))
fwrite(res, file.path(DIR, "arms.csv")); saveRDS(res, file.path(DIR, "arms.rds"))

rule("VERDICT")
G_enc   <- mean(res$base_enc   - res$rot_enc)
G_noenc <- mean(res$base_noenc - res$rot_noenc)
ratio   <- G_noenc / G_enc
cat(sprintf("  mean base_enc   %.5f   mean rot_enc   %.5f   G_enc   %+.5f\n",
            mean(res$base_enc), mean(res$rot_enc), -G_enc))
cat(sprintf("  mean base_noenc %.5f   mean rot_noenc %.5f   G_noenc %+.5f\n",
            mean(res$base_noenc), mean(res$rot_noenc), -G_noenc))
cat(sprintf("\n  SURVIVAL RATIO G_noenc / G_enc = %.0f%%\n", 100 * ratio))
cat(sprintf("  -> %s\n", if (ratio >= 0.60) {
      "ROTATION IS REAL -- mostly independent of the encoding. Candidate may ship."
    } else if (ratio >= 0.20) {
      "PARTLY LEAK -- do NOT ship the current candidate; rebuild without ENC_COLS."
    } else {
      "LEAK AMPLIFIER -- REJECT. Iterations 54-58 measured the leak, not a model." }))
cat(sprintf("\n  Absolute check: rot_noenc %.5f vs production's HONEST member ~1.1359\n",
            mean(res$rot_noenc)))
cat(sprintf("  -> %s\n", if (mean(res$rot_noenc) < 1.1359 - 0.00283) {
      "rot_noenc still beats the honest production member" } else {
      "rot_noenc does NOT beat the honest production member -- nothing here" }))
cat("\ndone\n")
