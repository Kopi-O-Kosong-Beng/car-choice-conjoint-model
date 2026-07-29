# =============================================================================
# ITERATION 49 -- MATRIX FACTORIZATION (Topic 8) AS FEATURES
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# WHY THIS IS BEING RUN
# -----------------------------------------------------------------------------
# Our second modelling track reports a gain from matrix factorization. This
# track has NEVER run SVD, PCA, NMF or any factorization -- grep for
# svd/prcomp/princomp/NMF/irlba/softImpute across every .R file returns nothing.
# The only eigendecomposition anywhere is the 4x4 blend error-correlation matrix
# in iteration 19. Vault/Topics/Topic 8 deprioritised it on the argument that the
# feature columns are "low-cardinality and few -- no dimensionality pressure".
#
# That argument is about COMPRESSION. It does not cover the two things a
# factorization actually buys a tree model, and those are what this tests:
#
#   (1) ROTATION. xgboost splits on axis-aligned thresholds. Correlated
#       demographic columns (incomeind/incomea, ageind/agea, milesind/milesa are
#       each near-duplicate pairs) force the tree to spend many shallow splits
#       approximating an oblique boundary. Principal components hand it that
#       boundary as a single axis. This costs nothing and needs no new signal --
#       it changes the GEOMETRY the learner searches.
#   (2) DENSE RE-CODING OF A SPARSE ONE-HOT. The 20 ordinal attributes expand to
#       ~100 level-dummies. A truncated SVD of that indicator matrix gives dense
#       coordinates in which a single split can separate groups of levels that
#       currently need one split each.
#
# Neither creates information. Both change what the learner can reach in a fixed
# number of rounds -- which is exactly the regime we are in, since iteration 39
# fixed the round count at 540.
#
# NO LABELS ARE USED BY ANY FACTORIZATION HERE. PCA and SVD are unsupervised, so
# they cannot leak the outcome. The demographic PCA is nevertheless fitted
# FOLD-WISE (rule 7: nest everything that is fitted) even though an unsupervised
# transform would survive transductive fitting; the conservative choice costs
# nothing and cannot be argued with later. The design SVD is fitted on the union
# of train and test PROFILES, which is legitimate transduction of the same kind
# encode_design.R already performs -- design assignment carries no label.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# Appending principal components of the demographic block and/or a truncated SVD
# of the attribute-level indicator matrix beats the incumbent fixed-rounds member
# (xgb_lw2fr10, 10-seed OOF 1.13551; 1-seed equivalent 1.13740).
#
# DIRECTIONAL PRE-REGISTRATION, so a null cannot be reread as a win: if the
# rotation mechanism is real, the gain must be LARGER at small k (a few strong
# oblique directions) and must NOT keep growing monotonically with k. A win that
# only appears at the largest k is the model re-deriving the raw columns and is
# to be treated as noise.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
#   1. SCREEN at 1 seed. A 1-seed number is worth NOTHING on its own (model-level
#      seed sd 0.00283) and is used ONLY to rank. Incumbent 1-seed ref = 1.13740.
#   2. CONFIRM the top config at 10 seeds, paired respondent-clustered z >=
#      qnorm(1 - 0.025/N) over the N configs screened. Multiplicity is CUMULATIVE
#      with iteration 45's 16 points and iteration 47's 27.
#   3. BLEND GATE. It must improve the production 2-member nested blend by more
#      than the blend-level seed sd 0.00048. Iteration 39 gained +0.00252 at
#      member level and only +0.00020 at blend level -- member gains do not
#      automatically reach the blend, and the blend is the decision number.
#   4. SHIFT AUDIT. model/shift_audit.R must retain >= ~100%. The design encoding
#      retained 77% and a fatigue term 64%; both were rejected for that reason.
#   5. REPLICATE under folds_b before any production change.
# ADOPT only if all five hold. A null closes the factorization channel and is
# reported as such -- iteration 18 is the template for a useful negative result.
#
# ARTIFACTS: oof_xgb_mf.rds / test_xgb_mf.rds -- NEW names. Built from STRING
# LITERALS, never from a variable: iteration 39 overwrote a live blend member by
# inheriting an artifact name through a variable, and it had to be recovered
# from git. Verified absent before writing.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R"); source("model/encode_design.R")

DIR <- "experiments/iter49_matfact"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
NR   <- 540L      # the iteration-39 fixed-rounds protocol
SEED <- 1L        # screening seed
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

# ============================================================== SECTION 1 =====
rule("SECTION 1 -- DATA, exactly as iteration 47 builds it")

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

ytr  <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
NOS  <- sort(unique(trl$No))
stopifnot(nrow(trl) == 21565L * 4L, nrow(tel) == 4997L * 4L, length(ytr) == 21565L)
cat(sprintf("  trl %d rows (%d tasks)   tel %d rows (%d tasks)   base features %d\n",
            nrow(trl), nrow(trl) / 4L, nrow(tel), nrow(tel) / 4L, length(base_feat)))

# ============================================================== SECTION 2 =====
# MF BLOCK A -- principal components of the demographic block, FOLD-NESTED.
# Demographics are constant within respondent, so the PCA is fitted on the
# respondent-level matrix (one row per person), not the task-level one, which
# would silently weight people by their task count (identical here at 19 each,
# but wrong in principle and wrong if a person were ever dropped).
rule("SECTION 2 -- MF BLOCK A: demographic PCA (fold-nested, unsupervised)")

K_DEMO_MAX <- 8L
demo_pc_cols <- paste0("dpc", seq_len(K_DEMO_MAX))
for (cc in demo_pc_cols) { trl[, (cc) := NA_real_]; tel[, (cc) := NA_real_] }

resp_tr <- unique(trl[, c("Case", "fold", demo), with = FALSE])
resp_te <- unique(tel[, c("Case", demo), with = FALSE])
cat(sprintf("  respondents: train %d  test %d   demographic columns %d\n",
            nrow(resp_tr), nrow(resp_te), length(demo)))

fit_pca <- function(M) prcomp(M, center = TRUE, scale. = TRUE)
proj    <- function(p, M) scale(M, center = p$center, scale = p$scale) %*% p$rotation[, seq_len(K_DEMO_MAX)]

for (k in sort(unique(trl$fold))) {
  ptr <- fit_pca(as.matrix(resp_tr[fold != k, ..demo]))
  # apply to the held-out fold's respondents
  hv  <- resp_tr[fold == k]
  Z   <- proj(ptr, as.matrix(hv[, ..demo]))
  map <- data.table(Case = hv$Case); for (j in seq_len(K_DEMO_MAX)) map[[demo_pc_cols[j]]] <- Z[, j]
  idx <- trl[, .I[fold == k]]
  mm  <- map[match(trl$Case[idx], map$Case)]
  for (j in seq_len(K_DEMO_MAX)) set(trl, idx, demo_pc_cols[j], mm[[demo_pc_cols[j]]])
}
# test rows: rotation from ALL training respondents
pall <- fit_pca(as.matrix(resp_tr[, ..demo]))
Zt   <- proj(pall, as.matrix(resp_te[, ..demo]))
mapt <- data.table(Case = resp_te$Case); for (j in seq_len(K_DEMO_MAX)) mapt[[demo_pc_cols[j]]] <- Zt[, j]
mmt  <- mapt[match(tel$Case, mapt$Case)]
for (j in seq_len(K_DEMO_MAX)) set(tel, seq_len(nrow(tel)), demo_pc_cols[j], mmt[[demo_pc_cols[j]]])

stopifnot(!anyNA(trl[, ..demo_pc_cols]), !anyNA(tel[, ..demo_pc_cols]))
sdev <- pall$sdev^2 / sum(pall$sdev^2)
cat(sprintf("  variance explained by PC1..PC8: %s\n",
            paste(sprintf("%.3f", head(sdev, 8)), collapse = " ")))
cat(sprintf("  cumulative at k=4: %.3f   at k=8: %.3f\n", sum(sdev[1:4]), sum(sdev[1:8])))

# ============================================================== SECTION 3 =====
# MF BLOCK B -- truncated SVD of the attribute-level indicator matrix.
# Built on the UNIQUE alternative profiles across train and test (no labels), so
# it is transduction of the same kind encode_design.R already does.
rule("SECTION 3 -- MF BLOCK B: truncated SVD of the one-hot attribute matrix")

K_SVD_MAX <- 32L
svd_cols  <- paste0("dsv", seq_len(K_SVD_MAX))

allA <- rbind(trl[, ..ATTRS], tel[, ..ATTRS])
prof <- unique(allA)
cat(sprintf("  unique alternative profiles: %d   (of %d rows)\n", nrow(prof), nrow(allA)))

lev <- lapply(ATTRS, function(a) sort(unique(allA[[a]])))
names(lev) <- ATTRS
cat(sprintf("  level counts: %s\n", paste(sprintf("%s=%d", ATTRS, lengths(lev)), collapse = " ")))

onehot <- function(D) {
  out <- vector("list", length(ATTRS))
  for (i in seq_along(ATTRS)) {
    a <- ATTRS[i]; L <- lev[[a]]
    M <- outer(D[[a]], L, "==") * 1.0
    colnames(M) <- paste0(a, "_L", L)
    out[[i]] <- M
  }
  do.call(cbind, out)
}
Xp <- onehot(prof)
cat(sprintf("  indicator matrix: %d profiles x %d dummies\n", nrow(Xp), ncol(Xp)))

# centre columns before SVD -- an uncentred SVD spends its first component on the
# grand mean, which is the same for every row and therefore useless to a tree.
Xc <- scale(Xp, center = TRUE, scale = FALSE)
sv <- svd(Xc, nu = 0, nv = K_SVD_MAX)
ev <- sv$d^2 / sum(sv$d^2)
cat(sprintf("  spectrum (first 12): %s\n", paste(sprintf("%.3f", head(ev, 12)), collapse = " ")))
cat(sprintf("  cumulative at k=8: %.3f  k=16: %.3f  k=32: %.3f\n",
            sum(ev[1:8]), sum(ev[1:16]), sum(ev[1:32])))

score_rows <- function(D) {
  M <- scale(onehot(D), center = attr(Xc, "scaled:center"), scale = FALSE)
  M %*% sv$v
}
Ztr <- score_rows(trl[, ..ATTRS]); Zte <- score_rows(tel[, ..ATTRS])
for (j in seq_len(K_SVD_MAX)) {
  set(trl, NULL, svd_cols[j], Ztr[, j]); set(tel, NULL, svd_cols[j], Zte[, j])
}
stopifnot(!anyNA(trl[, ..svd_cols]), !anyNA(tel[, ..svd_cols]))

# ============================================================== SECTION 4 =====
rule("SECTION 4 -- SCREEN")

sbt <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
                     E <- exp(M); E / rowSums(E) }
obj_lw <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(sbt(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}

oof_for <- function(fs, seed) {
  prm <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
              subsample = 0.8, colsample_bytree = 0.8, base_score = 0,
              nthread = 4, seed = seed)
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

CFG <- list(
  list(nm = "base",              fs = base_feat),
  list(nm = "demoPC4",           fs = c(base_feat, demo_pc_cols[1:4])),
  list(nm = "demoPC8",           fs = c(base_feat, demo_pc_cols[1:8])),
  list(nm = "demoPC8_noraw",     fs = c(setdiff(base_feat, demo), demo_pc_cols[1:8])),
  list(nm = "dsvd8",             fs = c(base_feat, svd_cols[1:8])),
  list(nm = "dsvd16",            fs = c(base_feat, svd_cols[1:16])),
  list(nm = "dsvd32",            fs = c(base_feat, svd_cols[1:32])),
  list(nm = "demoPC4_dsvd8",     fs = c(base_feat, demo_pc_cols[1:4], svd_cols[1:8])),
  list(nm = "demoPC8_dsvd16",    fs = c(base_feat, demo_pc_cols[1:8], svd_cols[1:16]))
)
cat(sprintf("  %d configs, 1 seed, fixed %d rounds\n", length(CFG), NR))
cat("  incumbent xgb_lw2fr10 1-seed equivalent = 1.13740\n")
cat("  NOTE 'base' here is the SAME feature set as the incumbent, so base's own\n")
cat("  number is the honest within-run control -- compare challengers to IT, not\n")
cat("  to 1.13740, which came from a different machine and BLAS.\n\n")

res <- list(); Ps <- list()
for (i in seq_along(CFG)) {
  t0 <- Sys.time()
  P  <- oof_for(CFG[[i]]$fs, SEED)
  ll <- logloss(ytr, P)
  Ps[[CFG[[i]]$nm]] <- P
  res[[i]] <- data.table(cfg = CFG[[i]]$nm, nfeat = length(CFG[[i]]$fs), oof = ll,
                         mins = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  cat(sprintf("  %-16s nfeat %3d  OOF %.5f   (%.1f min)\n",
              CFG[[i]]$nm, length(CFG[[i]]$fs), ll, res[[i]]$mins))
  saveRDS(rbindlist(res), file.path(DIR, "screen.rds"))
  saveRDS(Ps, file.path(DIR, "screen_preds.rds"))
}

R <- rbindlist(res); setorder(R, oof)
rule("SCREEN RESULT (1 seed -- ranking only, NOT evidence)")
print(R)
fwrite(R, file.path(DIR, "screen.csv"))

b <- R$oof[R$cfg == "base"]
cat(sprintf("\n  within-run control 'base' = %.5f\n", b))
cat(sprintf("  best config: %-16s %.5f   (delta vs base %+.5f)\n", R$cfg[1], R$oof[1], R$oof[1] - b))
cat(sprintf("  configs beating base: %d of %d\n", sum(R$oof < b), nrow(R)))
cat(sprintf("  model-level seed sd is 0.00283 -- a delta smaller than that is NOTHING.\n"))
cat(sprintf("  |delta| of best vs base as a multiple of seed sd: %.2f\n", abs(R$oof[1] - b) / 0.00283))

# directional pre-registration check
dsv <- R[grepl("^dsvd", cfg)][order(cfg)]
if (nrow(dsv) == 3L) {
  cat(sprintf("\n  DIRECTIONAL CHECK (gain must not grow monotonically with k):\n"))
  d8  <- R$oof[R$cfg == "dsvd8"]; d16 <- R$oof[R$cfg == "dsvd16"]; d32 <- R$oof[R$cfg == "dsvd32"]
  cat(sprintf("    dsvd8 %.5f  dsvd16 %.5f  dsvd32 %.5f\n", d8, d16, d32))
  mono <- (d32 < d16) && (d16 < d8)
  cat(sprintf("    monotone-in-k? %s%s\n", if (mono) { "YES" } else { "NO" },
              if (mono) { "  -- pre-registered as NOISE, not a win" } else { "  -- consistent with rotation" }))
}
cat("\n  NEXT (only if a config beats base by > 1 seed sd): confirm at 10 seeds,\n")
cat("  then compare.R, then the blend gate, then shift_audit.R, then folds_b.\n")
saveRDS(list(screen = R, preds = Ps), file.path(DIR, "screen_full.rds"))
cat("\n  wrote", file.path(DIR, "screen_full.rds"), "\n")
