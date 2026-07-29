# =============================================================================
# ITERATION 51 -- IS THE SVD GAIN FACTORIZATION, OR JUST PART-WORTH ENCODING?
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# THE CLAIM UNDER TEST
# -----------------------------------------------------------------------------
# Iteration 49 found that appending a truncated SVD of the attribute indicator
# matrix improves the tree, and improves it MONOTONICALLY in k:
#
#     base 1.13761 | dsvd8 1.13369 | dsvd16 1.13082 | dsvd32 1.12655
#
# Iteration 49's header pre-registered monotone-in-k as a REJECT condition, on
# the grounds that a real rotation effect should saturate early. It did not
# saturate. So the gain needs an explanation that is not "factorization helped",
# and there is an obvious one:
#
#   The base feature set codes each attribute as an ORDINAL NUMERIC (ATTRS and
#   ATTRS_c). It contains no level dummies. The SVD components are linear
#   combinations OF THE ONE-HOT INDICATOR MATRIX. So appending them smuggles
#   PART-WORTH information into a model that previously had none -- and
#   part-worth coding is separately the single largest modelling win in this
#   project (iteration 02, worth 0.020). As k grows, the components span more of
#   the 72-dimensional one-hot space, so the gain grows. That is exactly the
#   monotone shape observed.
#
# If that is what is happening, the credit belongs to the ENCODING, and the
# rotation is incidental -- an orthogonal basis for a space is worth no more
# than the space. This matters because iteration 49's win would otherwise be
# written up as "matrix factorization worked", which would be false, and because
# the cheaper and more interpretable fix (just add the dummies) would be missed.
#
# This is the same ablation that killed the depth-10 sweep: a result that
# reproduced exactly on an independent re-fit and still turned out to be the
# encoding rather than the mechanism claimed for it.
#
# -----------------------------------------------------------------------------
# THE ARITHMETIC THAT SETS THE GRID
# -----------------------------------------------------------------------------
# 92 level-dummies across 20 attributes. Within each attribute the dummies sum
# to 1 on every row, so column-centring removes one dimension per attribute:
# rank(Xc) = 92 - 20 = 72. k = 72 therefore spans the SAME column space as the
# raw dummies, and "dsvd72" and "onehot" must agree up to the tree's own
# axis-alignment preference. That equality is a CORRECTNESS CHECK on this
# script, not a result.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
#   Let B = base + raw one-hot dummies, C = base + dsvd32.
#   1. If B <= C + 0.00283 (one model-level seed sd), the SVD adds NOTHING
#      beyond the encoding. VERDICT: encoding, not factorization. Report it that
#      way, ship the dummies (simpler, interpretable, no SVD to nest).
#   2. Only if C beats B by MORE than 0.00283 is the rotation itself doing work,
#      in which case the factorization claim survives and goes to the 10-seed
#      confirmation.
#   3. Either way the winner still faces the BLEND GATE (> 0.00048 on the nested
#      2-member blend), the SHIFT AUDIT (>= ~100%), and folds_b replication.
#      Iteration 39 gained +0.00252 at member level and +0.00020 at blend level.
#   4. dsvd72 must land within one seed sd of onehot. If it does not, this
#      script has a bug and every number in it is void.
#
# NOTE ON WHAT "WINNING" WOULD MEAN. Even under outcome 1 this is a real gain to
# the tree member -- the tree genuinely lacked part-worth coding. It is simply
# not a factorization result, and the report must not call it one.
#
# ARTIFACTS: none written to model/artifacts. Screen only. Emission happens in a
# separate step once the blend gate is known.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R"); source("model/encode_design.R")

DIR <- "experiments/iter51_encablation"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
NR <- 540L; SEED <- 1L
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
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
NOS <- sort(unique(trl$No))

rule("SECTION 1 -- BUILD BOTH ENCODINGS FROM ONE INDICATOR MATRIX")
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
prof <- unique(allA)
Xp <- onehot(prof); Xc <- scale(Xp, center = TRUE, scale = FALSE)
rk <- qr(Xc, tol = 1e-8)$rank
cat(sprintf("  dummies %d over %d attributes -> predicted rank %d, measured rank %d\n",
            ncol(Xp), length(ATTRS), ncol(Xp) - length(ATTRS), rk))
K_MAX <- as.integer(min(72L, rk))
sv <- svd(Xc, nu = 0, nv = K_MAX)

oh_cols  <- colnames(Xp)
svd_cols <- paste0("dsv", seq_len(K_MAX))
for (D in list(trl, tel)) NULL   # explicit: both tables get both blocks below
add_blocks <- function(D) {
  M <- onehot(D[, ..ATTRS])
  for (j in seq_along(oh_cols)) set(D, NULL, oh_cols[j], M[, j])
  Z <- scale(M, center = attr(Xc, "scaled:center"), scale = FALSE) %*% sv$v
  for (j in seq_len(K_MAX)) set(D, NULL, svd_cols[j], Z[, j])
}
add_blocks(trl); add_blocks(tel)
cat(sprintf("  one-hot block %d cols   svd block %d cols\n", length(oh_cols), K_MAX))

rule("SECTION 2 -- ABLATION")
sbt <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
                     E <- exp(M); E / rowSums(E) }
obj_lw <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(sbt(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
oof_for <- function(fs, seed) {
  prm <- list(eta = 0.03, max_depth = 8, min_child_weight = 20, subsample = 0.8,
              colsample_bytree = 0.8, base_score = 0, nthread = 4, seed = seed)
  P <- matrix(NA_real_, 21565L, 4L)
  for (k in 1:5) {
    d <- trl[fold != k]
    fit <- xgb.train(params = prm,
                     data = xgb.DMatrix(as.matrix(d[, ..fs]), label = as.numeric(d$chosen)),
                     nrounds = NR, verbose = 0, obj = obj_lw, maximize = FALSE)
    v <- trl[fold == k]
    P[match(unique(v$No), NOS), ] <-
      sbt(as.vector(predict(fit, xgb.DMatrix(as.matrix(v[, ..fs])), outputmargin = TRUE)))
  }
  P
}
num_attr <- c(ATTRS, paste0(ATTRS, "_c"))
CFG <- list(
  list(nm = "base",           fs = base_feat),
  list(nm = "onehot",         fs = c(base_feat, oh_cols)),
  list(nm = "onehot_nonum",   fs = c(setdiff(base_feat, num_attr), oh_cols)),
  list(nm = "dsvd32",         fs = c(base_feat, svd_cols[1:32])),
  list(nm = "dsvd72",         fs = c(base_feat, svd_cols[seq_len(K_MAX)])),
  list(nm = "onehot_dsvd32",  fs = c(base_feat, oh_cols, svd_cols[1:32]))
)
res <- list()
for (i in seq_along(CFG)) {
  t0 <- Sys.time(); ll <- logloss(ytr, oof_for(CFG[[i]]$fs, SEED))
  res[[i]] <- data.table(cfg = CFG[[i]]$nm, nfeat = length(CFG[[i]]$fs), oof = ll,
                         mins = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  cat(sprintf("  %-16s nfeat %3d  OOF %.5f   (%.1f min)\n",
              CFG[[i]]$nm, length(CFG[[i]]$fs), ll, res[[i]]$mins))
  saveRDS(rbindlist(res), file.path(DIR, "ablation.rds"))
}
R <- rbindlist(res); fwrite(R, file.path(DIR, "ablation.csv"))

rule("VERDICT")
g <- function(n) R$oof[R$cfg == n]
B <- g("onehot"); C <- g("dsvd32"); SD <- 0.00283
print(R[order(oof)])
cat(sprintf("\n  base            %.5f\n  onehot   (B)    %.5f\n  dsvd32   (C)    %.5f\n", g("base"), B, C))
cat(sprintf("  B - C = %+.5f   (one model-level seed sd = %.5f)\n", B - C, SD))
cat(sprintf("\n  CORRECTNESS CHECK  |dsvd72 - onehot| = %.5f  -> %s\n",
            abs(g("dsvd72") - B),
            if (abs(g("dsvd72") - B) < SD) { "PASS (same column space, as predicted)" } else {
              "FAIL -- script bug, all numbers void" }))
cat(sprintf("\n  PRE-REGISTERED VERDICT: %s\n",
            if (B <= C + SD) {
              "ENCODING, NOT FACTORIZATION. The dummies capture it; ship those."
            } else { "ROTATION SURVIVES -- the SVD beats the raw dummies. Go to 10 seeds." }))
cat("\n  Whatever wins still faces: 10 seeds -> compare.R -> BLEND GATE (>0.00048)\n")
cat("  -> shift_audit.R -> folds_b. Member gains do not automatically reach the blend.\n")
