# =============================================================================
# ITERATION 59 -- IS THE ROTATION GAIN A GENUINE ENCODING WIN, OR IS IT
#                 AMPLIFYING THE DESIGN-SHARE LEAK?
#
# Everything in this header is written BEFORE any result is looked at.
# WRITES NOTHING TO model/artifacts. Screen + this directory only.
#
# -----------------------------------------------------------------------------
# WHY THIS HAS TO BE RUN BEFORE ANY SLOT IS SPENT
# -----------------------------------------------------------------------------
# Iterations 49/51/54 replaced the tree's ordinal attribute codes with 71 dense
# oblique coordinates of the attribute one-hot space and moved the member OOF
# 1.13555 -> 1.11363 (10-seed bag, 10/10 seeds). A candidate CSV has already
# been built from it (submissions/sub_20260729_rotblend_cal*.csv, nested 1.11210
# against production 1.12819).
#
# TWO FACTS MAKE THAT NUMBER UNSAFE:
#
#  (1) iteration 51's OWN pre-registered correctness check FAILED and was
#      overridden. Its header states: "dsvd72 must land within one seed sd of
#      onehot. If it does not, this script has a bug and every number in it is
#      void." Measured |dsvd72 - onehot| = 0.01795. The script printed
#      "FAIL -- script bug, all numbers void" and then printed "ROTATION
#      SURVIVES" on the next line, because the verdict branch never read the
#      check. The discrepancy the check flagged IS the entire claimed gain
#      (dsvd72 1.11868 vs onehot 1.13663 = 0.01795).
#
#  (2) base_feat contains ENC_COLS. The concurrent leak investigation
#      (iter48/iter54_enchonest) established that ENC_COLS is an exactly
#      invertible encoding of (C - c_own_fold), that 46.3% of training rows sit
#      in a (cell,fold) block of size 1 so leave-own-fold-out degenerates to
#      leave-own-ROW-out, and -- decisively -- that the size of the extracted
#      leak is MONOTONE IN THE TREE'S CELL-ISOLATION POWER with no saturation:
#        depth 4 -0.0004 | depth 6 +0.0066 | depth 8 +0.0224 | depth 10 +0.0394
#        depth10/mcw10/eta.02/1400 +0.0777.
#      71 dense linear combinations of the design indicators are a LARGE
#      increase in cell-isolation power at fixed depth: isolating one of 5,624
#      profiles needs ~20 axis-aligned splits on dummies but ~log2(5624) ~ 12.5
#      on generic dense directions. So "give the tree oblique coordinates" is,
#      mechanically, "give the tree a cheaper way to find the cell whose label
#      ENC_COLS is holding".
#
# The rotation is label-free, so iteration 54's checks A (nested basis) and B
# (random rotation) CANNOT detect this. A leak that lives in a different feature
# block is invisible to every check that only perturbs the basis.
#
# -----------------------------------------------------------------------------
# THE TEST -- the same 2x2 ablation that killed the depth-10 sweep
# -----------------------------------------------------------------------------
#   base_enc    base_feat (with ENC_COLS)            expect ~1.1376 (reproduces iter54)
#   base_noenc  base_feat minus ENC_COLS             expect ~1.1588 (leak report)
#   rot_enc     base_feat + 71 random rot cols       expect ~1.1152 (reproduces iter54)
#   rot_noenc   base_feat minus ENC_COLS + rot cols  <- THE UNKNOWN
#   oh_noenc    base_feat minus ENC_COLS + 92 sparse dummies
#               (separates "part-worth information" from "oblique geometry")
#
# DECISION RULE, fixed before running. Let
#     G_enc   = rot_enc   - base_enc     (the advertised gain, ~ -0.0225)
#     G_noenc = rot_noenc - base_noenc   (the same gain with the leak removed)
#
#   A. G_noenc <= -0.010                 -> the rotation carries real signal
#                                           independent of the leak. The
#                                           candidate is worth a slot.
#   B. -0.010 < G_noenc <= -0.003        -> PARTIAL. Most of the headline is
#                                           leak amplification. Do not ship the
#                                           headline number; re-derive honestly.
#   C. G_noenc > -0.003 (one seed sd)    -> THE ENTIRE GAIN IS LEAK
#                                           AMPLIFICATION. Do not upload
#                                           sub_20260729_rotblend_cal*.csv.
#
# Secondary, reported either way: rot_noenc vs oh_noenc. If they agree within a
# seed sd the mechanism is part-worth ENCODING (already known to be worth ~0.020
# to the MNL, iteration 02) and "rotation"/"factorization" is the wrong name.
#
# ARTIFACTS: experiments/iter59_rotleak/zz59_arms.csv only. All filenames are
# string literals. model/artifacts is not written to. members.txt, blend.rds,
# folds*.rds untouched.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R"); source("model/encode_design.R")

DIR <- "experiments/iter59_rotleak"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
NR <- 540L
SEEDS <- as.integer(strsplit(Sys.getenv("ZZ59_SEEDS", "1"), ",")[[1]])
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
long <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
base_feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
               "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, ENC_COLS)
noenc_feat <- setdiff(base_feat, ENC_COLS)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
NOS <- sort(unique(trl$No))

allA <- rbind(trl[, ..ATTRS], tel[, ..ATTRS])
lev <- lapply(ATTRS, function(a) sort(unique(allA[[a]]))); names(lev) <- ATTRS
onehot <- function(D) {
  out <- vector("list", length(ATTRS))
  for (i in seq_along(ATTRS)) {
    a <- ATTRS[i]; L <- lev[[a]]
    M <- outer(D[[a]], L, "==") * 1.0; colnames(M) <- paste0("oh_", a, "_L", L); out[[i]] <- M
  }
  do.call(cbind, out)
}
OH_tr <- onehot(trl[, ..ATTRS]); OH_te <- onehot(tel[, ..ATTRS])
K <- 71L
rot_cols <- paste0("rot", seq_len(K))
oh_cols  <- colnames(OH_tr)

sbt <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
                     E <- exp(M); E / rowSums(E) }
obj_lw <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(sbt(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
fit_fold <- function(d, v, fs, seed) {
  prm <- list(eta = 0.03, max_depth = 8, min_child_weight = 20, subsample = 0.8,
              colsample_bytree = 0.8, base_score = 0, nthread = 4, seed = seed)
  fit <- xgb.train(params = prm,
                   data = xgb.DMatrix(as.matrix(d[, ..fs]), label = as.numeric(d$chosen)),
                   nrounds = NR, verbose = 0, obj = obj_lw, maximize = FALSE)
  sbt(as.vector(predict(fit, xgb.DMatrix(as.matrix(v[, ..fs])), outputmargin = TRUE)))
}

# identical to iteration 54's "rand" mode, so the arms are comparable
make_rand <- function(rows_for_fit, seed) {
  set.seed(1000L + seed)
  Q <- qr.Q(qr(matrix(rnorm(ncol(OH_tr) * K), ncol(OH_tr), K)))
  list(ctr = colMeans(OH_tr[rows_for_fit, , drop = FALSE]), V = Q)
}
apply_basis <- function(B, OH) sweep(OH, 2, B$ctr, "-") %*% B$V

# arm: "base" | "rot" | "oh" ; enc: TRUE/FALSE
oof_arm <- function(arm, enc, seed) {
  P <- matrix(NA_real_, 21565L, 4L)
  fbase <- if (enc) { base_feat } else { noenc_feat }
  for (k in 1:5) {
    ins <- which(trl$fold != k); out <- which(trl$fold == k)
    d <- copy(trl[ins]); v <- copy(trl[out])
    fs <- fbase
    if (arm == "rot") {
      B <- make_rand(ins, seed); Z <- apply_basis(B, OH_tr)
      for (j in seq_len(K)) { set(d, NULL, rot_cols[j], Z[ins, j]); set(v, NULL, rot_cols[j], Z[out, j]) }
      fs <- c(fbase, rot_cols)
    } else if (arm == "oh") {
      for (j in seq_along(oh_cols)) { set(d, NULL, oh_cols[j], OH_tr[ins, j]); set(v, NULL, oh_cols[j], OH_tr[out, j]) }
      fs <- c(fbase, oh_cols)
    }
    P[match(unique(trl$No[out]), NOS), ] <- fit_fold(d, v, fs, seed)
  }
  P
}

ARMS <- list(
  list(id = "base_enc",   arm = "base", enc = TRUE),
  list(id = "base_noenc", arm = "base", enc = FALSE),
  list(id = "rot_enc",    arm = "rot",  enc = TRUE),
  list(id = "rot_noenc",  arm = "rot",  enc = FALSE),
  list(id = "oh_noenc",   arm = "oh",   enc = FALSE)
)

rule(sprintf("2x2 (+1) ABLATION -- seeds %s", paste(SEEDS, collapse = ",")))
res <- list()
for (s in SEEDS) for (a in ARMS) {
  t0 <- Sys.time()
  P <- oof_arm(a$arm, a$enc, s)
  ll <- logloss(ytr, P)
  res[[length(res) + 1L]] <- data.table(seed = s, id = a$id, oof = ll,
                                        mins = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  cat(sprintf("  seed %d  %-11s OOF %.5f   (%.1f min)\n", s, a$id, ll, tail(res, 1)[[1]]$mins))
  fwrite(rbindlist(res), file.path(DIR, "zz59_arms.csv"))
}

R <- rbindlist(res)
M <- dcast(R, seed ~ id, value.var = "oof")
rule("PER-SEED CONTRASTS")
print(M)
M[, G_enc   := rot_enc - base_enc]
M[, G_noenc := rot_noenc - base_noenc]
M[, leak_base := base_noenc - base_enc]
M[, leak_rot  := rot_noenc  - rot_enc]
cat("\n")
print(M[, .(seed, G_enc, G_noenc, leak_base, leak_rot)])

rule("VERDICT")
ge <- mean(M$G_enc); gn <- mean(M$G_noenc)
cat(sprintf("  advertised gain  G_enc   = %+.5f  (rotation, WITH the leaky encoding)\n", ge))
cat(sprintf("  honest gain      G_noenc = %+.5f  (rotation, encoding REMOVED)\n", gn))
cat(sprintf("  share of the headline that is leak amplification: %.0f%%\n",
            100 * (1 - gn / ge)))
cat(sprintf("  leak extracted by BASE tree = %+.5f ; by ROT tree = %+.5f  (ratio %.2fx)\n",
            mean(M$leak_base), mean(M$leak_rot), mean(M$leak_rot) / mean(M$leak_base)))
verd <- if (gn <= -0.010) { "A: rotation carries real signal independent of the leak -- candidate is worth a slot" } else if (gn <= -0.003) { "B: PARTIAL -- most of the headline is leak amplification; re-derive honestly" } else { "C: THE ENTIRE GAIN IS LEAK AMPLIFICATION -- DO NOT UPLOAD the rotblend CSV" }
cat(sprintf("\n  PRE-REGISTERED VERDICT -> %s\n", verd))
if ("oh_noenc" %in% names(M)) {
  cat(sprintf("\n  mechanism check: rot_noenc %.5f vs oh_noenc %.5f  (diff %+.5f, one seed sd 0.00283)\n",
              mean(M$rot_noenc), mean(M$oh_noenc), mean(M$rot_noenc) - mean(M$oh_noenc)))
  cat(sprintf("  -> %s\n", if (abs(mean(M$rot_noenc) - mean(M$oh_noenc)) < 0.00283) {
    "sparse dummies MATCH the rotation: the mechanism is PART-WORTH ENCODING, not geometry" } else if (mean(M$rot_noenc) < mean(M$oh_noenc)) {
    "the rotation beats sparse dummies: oblique geometry is doing real work" } else {
    "sparse dummies BEAT the rotation" }))
}
fwrite(M, file.path(DIR, "zz59_contrasts.csv"))
cat("\n  wrote experiments/iter59_rotleak/zz59_arms.csv and zz59_contrasts.csv\n")
cat("  model/artifacts NOT written. members.txt / blend.rds / folds*.rds untouched.\n")
cat("\ndone\n")
