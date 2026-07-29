# =============================================================================
# ITERATION 54 -- IS THE ROTATION GAIN REAL? THREE ADVERSARIAL CHECKS
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# WHAT IS BEING DOUBTED
# -----------------------------------------------------------------------------
# Iteration 51 found that replacing the 92 attribute-level dummies with a
# 71-component SVD of the SAME centred indicator matrix -- identical column
# space, identical information -- moves the fixed-rounds tree from 1.13663 to
# 1.11868. That is -0.019, or 6.7 model-level seed sds, and it is far larger
# than anything this project has adopted. Two prior results of this shape were
# wrong: the depth-10 sweep reproduced exactly on an independent re-fit and
# still died on ablation, and the free-sign blend passed seven checks and then
# scored 1.209 on the board. Size is a reason for suspicion, not belief.
#
# THREE THINGS COULD PRODUCE THIS NUMBER WITHOUT A REAL GAIN:
#
#   (A) TRANSDUCTIVE BASIS. The SVD in iteration 51 is fitted on the unique
#       profiles of train AND test together. It uses no labels, and design
#       assignment was measured random w.r.t. demographics (iteration 18, min
#       p = 0.1228), so it should be clean. "Should be" is not a measurement.
#       CHECK: refit the basis inside each fold, on that fold's TRAINING rows
#       only. If the gain survives, the basis is not the mechanism.
#
#   (B) IT IS NOT THE SVD, IT IS DENSITY. A tree may simply prefer dense
#       real-valued columns to sparse binary ones, in which case ANY rotation
#       of the same space would do and "matrix factorization" is the wrong name
#       for the finding -- it would be "give the tree oblique coordinates".
#       CHECK: a RANDOM orthonormal rotation of the same centred matrix, same
#       rank. If random matches SVD, the finding is real but is about geometry,
#       not about the principal directions, and must be reported that way.
#
#   (C) SINGLE-SEED LUCK. One seed is worth nothing at this project's noise
#       floor. CHECK: 10 seeds, paired.
#
# -----------------------------------------------------------------------------
# PRE-REGISTERED READING OF EACH OUTCOME
# -----------------------------------------------------------------------------
#   nested basis holds  -> not leakage; proceed.
#   nested basis dies   -> the transductive fit was doing the work. REJECT and
#                          write it up as a near-miss; do not ship.
#   random ~= SVD       -> report as "oblique coordinates", NOT as factorization.
#                          Still shippable, but the report must name it correctly.
#   random << SVD       -> the principal directions matter; genuinely a
#                          factorization result.
#   10-seed delta < 0.00283 -> single-seed luck. REJECT.
#
# THE GATE THAT ACTUALLY DECIDES: the nested 2-member blend must improve by more
# than the blend-level seed sd 0.00048. Iteration 39 gained +0.00252 at member
# level and only +0.00020 at blend level. A member gain is not a result.
# Then shift_audit.R (>= ~100%), then folds_b.
#
# ARTIFACTS: oof_xgb_rot.rds / test_xgb_rot.rds, STRING LITERALS, written last
# and only if the checks pass. Iteration 39 overwrote a live member by
# inheriting a name through a variable and had to be recovered from git.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R"); source("model/encode_design.R")

DIR <- "experiments/iter54_rotverify"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
NR <- 540L
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

# basis modes -------------------------------------------------------------
#  "trans"  SVD on unique profiles of train+test        (iteration 51's version)
#  "nested" SVD on the FOLD-TRAINING rows only          (check A)
#  "rand"   random orthonormal rotation, same rank      (check B)
make_basis <- function(mode, rows_for_fit, seed = 1L) {
  if (mode == "rand") {
    set.seed(1000L + seed)
    Q <- qr.Q(qr(matrix(rnorm(ncol(OH_tr) * K), ncol(OH_tr), K)))
    return(list(ctr = colMeans(OH_tr[rows_for_fit, , drop = FALSE]), V = Q))
  }
  M <- if (mode == "trans") {
    as.matrix(unique(rbind(as.data.table(OH_tr), as.data.table(OH_te))))
  } else {
    unique(OH_tr[rows_for_fit, , drop = FALSE])
  }
  ctr <- colMeans(M)
  sv <- svd(sweep(M, 2, ctr, "-"), nu = 0, nv = K)
  list(ctr = ctr, V = sv$v)
}
apply_basis <- function(B, OH) sweep(OH, 2, B$ctr, "-") %*% B$V

oof_rot <- function(mode, seed) {
  P <- matrix(NA_real_, 21565L, 4L)
  for (k in 1:5) {
    ins <- which(trl$fold != k); out <- which(trl$fold == k)
    B <- make_basis(mode, ins, seed)
    Ztr <- apply_basis(B, OH_tr)
    d <- copy(trl[ins]); v <- copy(trl[out])
    for (j in seq_len(K)) { set(d, NULL, rot_cols[j], Ztr[ins, j]); set(v, NULL, rot_cols[j], Ztr[out, j]) }
    fs <- c(base_feat, rot_cols)
    P[match(unique(trl$No[out]), NOS), ] <- fit_fold(d, v, fs, seed)
  }
  P
}
oof_base <- function(seed) {
  P <- matrix(NA_real_, 21565L, 4L)
  for (k in 1:5) {
    ins <- which(trl$fold != k); out <- which(trl$fold == k)
    P[match(unique(trl$No[out]), NOS), ] <- fit_fold(trl[ins], trl[out], base_feat, seed)
  }
  P
}

rule("CHECKS A + B  (1 seed each -- ranking only)")
res <- list()
for (m in c("base", "trans", "nested", "rand")) {
  t0 <- Sys.time()
  P <- if (m == "base") { oof_base(1L) } else { oof_rot(m, 1L) }
  ll <- logloss(ytr, P)
  res[[m]] <- data.table(mode = m, oof = ll,
                         mins = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  cat(sprintf("  %-7s OOF %.5f   (%.1f min)\n", m, ll, res[[m]]$mins))
  saveRDS(rbindlist(res), file.path(DIR, "checks.rds"))
}
R <- rbindlist(res); fwrite(R, file.path(DIR, "checks.csv"))
g <- function(n) R$oof[R$mode == n]
SD <- 0.00283
rule("VERDICT ON A AND B")
cat(sprintf("  base                 %.5f\n", g("base")))
cat(sprintf("  transductive basis   %.5f  (delta %+.5f)\n", g("trans"),  g("trans")  - g("base")))
cat(sprintf("  NESTED basis         %.5f  (delta %+.5f)   <- check A\n", g("nested"), g("nested") - g("base")))
cat(sprintf("  RANDOM rotation      %.5f  (delta %+.5f)   <- check B\n", g("rand"),   g("rand")   - g("base")))
cat(sprintf("\n  A: nested vs transductive = %+.5f  -> %s\n", g("nested") - g("trans"),
            if (g("nested") - g("base") < -SD) { "SURVIVES nesting -- not a transduction artifact" } else {
              "DIES under nesting -- the transductive basis was the mechanism. REJECT." }))
cat(sprintf("  B: random vs nested       = %+.5f  -> %s\n", g("rand") - g("nested"),
            if (abs(g("rand") - g("nested")) < SD) {
              "RANDOM MATCHES SVD -- report as oblique coordinates, NOT factorization" } else if (g("rand") > g("nested")) {
              "SVD beats random -- the principal directions carry it" } else {
              "random BEATS svd -- report honestly; the basis choice is not what matters" }))

rule("CHECK C -- 10 SEEDS on the best surviving mode, paired by respondent")
BESTM <- R[mode != "base"][which.min(oof), mode]
if (g(BESTM) - g("base") >= -SD) {
  cat("  no mode beats base by more than one seed sd at 1 seed. Stopping before spending 10 seeds.\n")
} else {
  cat(sprintf("  carrying forward: %s\n", BESTM))
  accB <- matrix(0, 21565L, 4L); accR <- matrix(0, 21565L, 4L); per <- list()
  for (s in 1:10) {
    Pb <- oof_base(s); Pr <- oof_rot(BESTM, s)
    accB <- accB + log(pmax(Pb, 1e-15)); accR <- accR + log(pmax(Pr, 1e-15))
    per[[s]] <- data.table(seed = s, base = logloss(ytr, Pb), rot = logloss(ytr, Pr))
    cat(sprintf("    seed %2d  base %.5f  rot %.5f  (delta %+.5f)\n",
                s, per[[s]]$base, per[[s]]$rot, per[[s]]$rot - per[[s]]$base))
    saveRDS(rbindlist(per), file.path(DIR, "seeds.rds"))
  }
  PER <- rbindlist(per); fwrite(PER, file.path(DIR, "seeds.csv"))
  nrm <- function(A) { E <- exp(A / 10 - apply(A / 10, 1, max)); E / rowSums(E) }
  Pb10 <- nrm(accB); Pr10 <- nrm(accR)
  llb <- logloss(ytr, Pb10); llr <- logloss(ytr, Pr10)
  cat(sprintf("\n  10-seed bag   base %.5f   rot %.5f   delta %+.5f\n", llb, llr, llr - llb))
  cat(sprintf("  per-seed paired: mean %+.5f  sd %.5f  wins %d/10\n",
              mean(PER$rot - PER$base), sd(PER$rot - PER$base), sum(PER$rot < PER$base)))
  # emit for the blend gate
  oof_dt <- data.table(No = NOS, p1 = Pr10[,1], p2 = Pr10[,2], p3 = Pr10[,3], p4 = Pr10[,4])
  accT <- matrix(0, 4997L, 4L)
  for (s in 1:10) {
    B <- make_basis(BESTM, seq_len(nrow(trl)), s)
    Ztr <- apply_basis(B, OH_tr); Zte <- apply_basis(B, OH_te)
    d <- copy(trl); v <- copy(tel)
    for (j in seq_len(K)) { set(d, NULL, rot_cols[j], Ztr[, j]); set(v, NULL, rot_cols[j], Zte[, j]) }
    accT <- accT + log(pmax(fit_fold(d, v, c(base_feat, rot_cols), s), 1e-15))
  }
  Pt <- nrm(accT)
  te_dt <- data.table(No = sort(unique(tel$No)), p1 = Pt[,1], p2 = Pt[,2], p3 = Pt[,3], p4 = Pt[,4])
  stopifnot(nrow(oof_dt) == 21565L, nrow(te_dt) == 4997L,
            !file.exists("model/artifacts/oof_xgb_rot.rds"))
  saveRDS(oof_dt, "model/artifacts/oof_xgb_rot.rds")
  saveRDS(te_dt,  "model/artifacts/test_xgb_rot.rds")
  cat("\n  wrote oof_xgb_rot.rds / test_xgb_rot.rds\n")
  cat("  NEXT AND DECISIVE: BLEND_MEMBERS='xgb_rot lcmnl3_both' and\n")
  cat("  'xgb_lw2bag lcmnl3_both xgb_rot' through model/06_blend.R, then shift_audit.R, then folds_b.\n")
}
cat("\ndone\n")
