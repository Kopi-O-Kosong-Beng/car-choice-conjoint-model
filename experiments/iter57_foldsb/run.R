# =============================================================================
# ITERATION 57 -- THE LAST GATE: DOES THE ROTATION REPLICATE ON AN INDEPENDENT
#                 RESPONDENT GROUPING?
#
# Written BEFORE any number here is looked at.
#
# WHY THIS GATE EXISTS
# -----------------------------------------------------------------------------
# Every number so far -- the 10/10 seed sweep, the 105% reweighted retention,
# the 33-sd blend gate -- was measured on ONE fold structure, folds.rds (seed
# 42), the same structure against which 56 iterations of decisions have been
# made. Iteration 21 measured what that costs: gains selected on the production
# folds replicate on an independent respondent grouping at only ~80% (member
# 79%, blend 81%). folds_b.rds (seed 43) has been built and verified independent
# (22.0% fold agreement against 20% by chance) and is the correct instrument.
#
# The rotation SHOULD replicate near 100%, and the reason is mechanical rather
# than hopeful: it introduces no new information and fits no parameter on the
# outcome. The one-hot basis is a re-encoding of ATTRS, which base_feat already
# carried as ordinal numerics, and the rotation is a random orthonormal mix of
# that basis drawn from a seed -- never fitted. There is nothing here that CAN
# be tuned to a particular fold structure. If it nonetheless drops sharply, the
# mechanism is not what iteration 54 concluded and the result must not ship.
#
# DECISION RULE -- fixed before running
#   retention = (folds_b gain) / (folds gain), member level, matched seeds.
#     >= 90%  replicates. Clear to build a submission candidate.
#     60-90%  the iteration-21 norm. Proceed, discount the expected public gain.
#     < 60%   fold-structure-specific. DO NOT SHIP on the folds number.
#   3 seeds, paired. Not 10: the folds number is already 10/10 at sd 0.00180, so
#   the question here is location, not noise, and 3 paired seeds resolve a shift
#   of this size (~0.020) with room to spare.
#
# NO ARTIFACTS are written to model/artifacts. Read-only measurement.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R"); source("model/encode_design.R")

DIR <- "experiments/iter57_foldsb"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
NR <- 540L; SEEDS <- 1:3; K <- 71L
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long <- readRDS("model/artifacts/long.rds")
wide <- readRDS("model/artifacts/wide.rds")
long <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
sbt <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
                     E <- exp(M); E / rowSums(E) }
obj_lw <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(sbt(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}

# Everything downstream of the fold column is rebuilt per structure, because the
# design encoding is FOLD-AWARE: reusing folds.rds's encoding under folds_b
# would leak across the new partition. This is the mistake that would quietly
# invalidate the whole gate.
run_structure <- function(foldfile) {
  folds <- readRDS(foldfile)
  trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
  tel <- long[is_test == TRUE]
  setorder(trl, No, alt); setorder(tel, No, alt)
  apply_design_encoding(trl, tel)
  base_feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
                 "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, ENC_COLS)
  NOS <- sort(unique(trl$No))
  lev <- lapply(ATTRS, function(a) sort(unique(rbind(trl[, ..ATTRS], tel[, ..ATTRS])[[a]])))
  names(lev) <- ATTRS
  onehot <- function(D) do.call(cbind, lapply(ATTRS, function(a) {
    M <- outer(D[[a]], lev[[a]], "==") * 1.0; colnames(M) <- paste0("oh_", a, "_L", lev[[a]]); M }))
  OH <- onehot(trl[, ..ATTRS])
  rot_cols <- paste0("rot", seq_len(K))
  fitpred <- function(d, v, fs, seed) {
    prm <- list(eta = 0.03, max_depth = 8, min_child_weight = 20, subsample = 0.8,
                colsample_bytree = 0.8, base_score = 0, nthread = 4, seed = seed)
    f <- xgb.train(params = prm,
                   data = xgb.DMatrix(as.matrix(d[, ..fs]), label = as.numeric(d$chosen)),
                   nrounds = NR, verbose = 0, obj = obj_lw, maximize = FALSE)
    sbt(as.vector(predict(f, xgb.DMatrix(as.matrix(v[, ..fs])), outputmargin = TRUE)))
  }
  one <- function(mode, seed) {
    P <- matrix(NA_real_, 21565L, 4L)
    for (k in 1:5) {
      ins <- which(trl$fold != k); out <- which(trl$fold == k)
      if (mode == "base") {
        P[match(unique(trl$No[out]), NOS), ] <- fitpred(trl[ins], trl[out], base_feat, seed)
      } else {
        set.seed(1000L + seed)
        Q <- qr.Q(qr(matrix(rnorm(ncol(OH) * K), ncol(OH), K)))
        Z <- sweep(OH, 2, colMeans(OH[ins, , drop = FALSE]), "-") %*% Q
        d <- copy(trl[ins]); v <- copy(trl[out])
        for (j in seq_len(K)) { set(d, NULL, rot_cols[j], Z[ins, j]); set(v, NULL, rot_cols[j], Z[out, j]) }
        P[match(unique(trl$No[out]), NOS), ] <- fitpred(d, v, c(base_feat, rot_cols), seed)
      }
    }
    logloss(ytr, P)
  }
  r <- rbindlist(lapply(SEEDS, function(s) {
    b <- one("base", s); o <- one("rot", s)
    cat(sprintf("    seed %d  base %.5f  rot %.5f  (delta %+.5f)\n", s, b, o, o - b))
    data.table(seed = s, base = b, rot = o)
  }))
  r
}

rule("PRODUCTION FOLDS (folds.rds, seed 42) -- matched 3-seed reference")
A <- run_structure("model/artifacts/folds.rds")
saveRDS(A, file.path(DIR, "folds_a.rds"))

rule("INDEPENDENT FOLDS (folds_b.rds, seed 43) -- the replication")
B <- run_structure("model/artifacts/folds_b.rds")
saveRDS(B, file.path(DIR, "folds_b.rds"))

rule("VERDICT")
dA <- mean(A$rot - A$base); dB <- mean(B$rot - B$base)
cat(sprintf("  folds   (seed 42)  base %.5f  rot %.5f  gain %+.5f  wins %d/%d\n",
            mean(A$base), mean(A$rot), dA, sum(A$rot < A$base), nrow(A)))
cat(sprintf("  folds_b (seed 43)  base %.5f  rot %.5f  gain %+.5f  wins %d/%d\n",
            mean(B$base), mean(B$rot), dB, sum(B$rot < B$base), nrow(B)))
ret <- dB / dA
cat(sprintf("\n  REPLICATION RETENTION %.0f%%  -> %s\n", 100 * ret,
            if (ret >= 0.90) { "REPLICATES -- clear to build a candidate" } else
            if (ret >= 0.60) { "iteration-21 norm (~80%) -- proceed, discount the expected gain" } else {
              "FOLD-STRUCTURE-SPECIFIC -- do not ship on the folds number" }))
cat(sprintf("  (iteration 21 measured the historical norm at member 79%%, blend 81%%)\n"))
fwrite(rbind(A[, .(structure = "folds",   seed, base, rot)],
             B[, .(structure = "folds_b", seed, base, rot)]), file.path(DIR, "replication.csv"))
cat("\ndone\n")
