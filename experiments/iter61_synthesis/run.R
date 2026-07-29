# =============================================================================
# ITERATION 61 -- THE SYNTHESIS MODEL
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# WHY A NEW MODEL RATHER THAN ANOTHER TUNE
# -----------------------------------------------------------------------------
# Three submissions this week improved the plain nested OOF and got WORSE on the
# board: 1.12341 -> 1.209, 1.12341 -> 1.211, 1.11210 -> 1.205. Iteration 48
# found the cause. model/encode_design.R is called ONCE, before the CV loop, so
# a training row in fold j carries an encoding built from a set that contains
# the scored fold k. With a median of 1 row per (cell, fold), leave-own-fold-out
# degenerates to leave-own-ROW-out for 46.3% of rows. The encoding's honest
# value is NEGATIVE (-0.00596, z -3.53). Its apparent +0.0218 is a complement
# identity, and the size of that identity GROWS WITH TREE CAPACITY:
#     d4 -0.0004 | d6 +0.0066 | d8 +0.0224 | d10 +0.0394
# The entire capacity axis of this project was therefore tuned backwards. The
# honest no-encoding curve gets monotonically WORSE with depth, optimum at 4-5,
# and production ships depth 8.
#
# There is no version of "tune the current model" that survives that. The
# incumbent's depth, its blend weight (0.528 fitted on leaked OOF; honest wants
# 0.194-0.243) and every hyperparameter above depth 6 are artefacts.
#
# -----------------------------------------------------------------------------
# THE SECOND PROBLEM, WHICH IS INDEPENDENT OF THE FIRST
# -----------------------------------------------------------------------------
# Train is 9.4% luxury respondents; the graded test rows are 68.8% luxury.
# Iteration 18 measured alternative 4 at 33% BETWEEN-RESPONDENT variance against
# 5-7% for alternatives 1-3. So the choice AMONG bundles is a property of the
# bundle and transfers to any crowd; the decision to WALK AWAY is a property of
# the person and does not. Plain OOF grades mostly the crowd we have; Kaggle
# grades mostly the crowd we do not.
#
# Reweighting our OOF to the test segment mix moves 1.12819 -> 1.19610 against
# an actual public 1.197 -- it closes 98.7% of the offset. THEREFORE THE
# SEGMENT-REWEIGHTED NESTED OOF IS THE DECISION NUMBER IN THIS ITERATION, and
# the plain one is reported only to show the difference.
#
# CAUTION, so this is not confused with iteration 07: that iteration fitted the
# BLEND OBJECTIVE on a reweighted target -- 10 parameters against a noisier
# signal -- and lost on its own metric because variance beat bias. Weighting the
# BASE MODEL FIT is a different operation on 86,260 rows, and it is the one
# thing in the team's second track that attacks the composition problem directly.
# We have never tried it. Reweighting is used here to WEIGHT A FIT and to JUDGE,
# never to fit the combiner.
#
# -----------------------------------------------------------------------------
# WHAT IS BEING COMBINED, AND WHOSE EVIDENCE SUPPORTS EACH PIECE
# -----------------------------------------------------------------------------
#   listwise softmax objective   iteration 03; both pipelines converged on it
#   NO design-share encoding     iteration 48; the honest value is negative
#   shallow trees, many seeds    honest optimum depth 4-5, not 8
#   one-hot attribute dummies    part-worth coding is worth 0.020 (iteration 02)
#   segment importance weights   attacks the composition problem inside the fit
#   MF cold-start features       rank-3 SVD of revealed preference -> demographic
#                                bridge; a person-level taste signal that raw
#                                demographics cannot supply (iteration 18 put
#                                demographics at 11.2% of true heterogeneity)
#   latent-class MNL 2nd member  OURS. Iteration 19: 93% of blend error variance
#                                lies on the tree-vs-logit axis. lcmnl3_both has
#                                no encoding and is therefore uncontaminated.
#   log-opinion pool             ours, with temperature and a uniform floor
#   probe-anchored margin        r* = 0.26652, the only quantity measured ON the
#                                graded population. Applied at DEPLOY, not here.
#
# -----------------------------------------------------------------------------
# STAGED, SO EACH ADDITION IS ATTRIBUTABLE
# -----------------------------------------------------------------------------
#   S1  honest baseline: no encoding, depth sweep, one-hot dummies
#   S2  + MF cold-start features
#   S3  + segment importance weighting, tau swept
#   S4  blend the winner with lcmnl3_both
# Each stage is judged on the SEGMENT-REWEIGHTED nested OOF. A stage that does
# not improve it is dropped even if it improves the plain number -- that is the
# exact mistake this iteration exists to avoid.
#
# HONEST REFERENCE POINTS, fixed before running:
#   production blend  plain 1.12819  segment-reweighted 1.19610  public 1.197
#   honest production member (no encoding, d8) ~1.1592
# A result is only interesting if it beats 1.19610 on the reweighted metric.
#
# ARTIFACTS: oof_xgb_syn.rds / test_xgb_syn.rds -- NEW names, string literals,
# written last. Iteration 39 overwrote a live member by inheriting a name.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")

DIR <- "experiments/iter61_synthesis"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
S_SCREEN <- as.integer(Sys.getenv("SYN_SEEDS", "3"))
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
NOS <- sort(unique(trl$No))
AP  <- setdiff(ATTRS, "Price")
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")

# ---- THE DECISION METRIC ----------------------------------------------------
mkw <- function(v) {
  resp <- unique(wide[, c("Case","is_test",v), with = FALSE])
  a <- resp[is_test == FALSE, .(ptr = .N/sum(!resp$is_test)), by = v]
  b <- resp[is_test == TRUE,  .(pte = .N/sum(resp$is_test)),  by = v]
  t2 <- merge(a, b, by = v, all.x = TRUE); t2[is.na(pte), pte := 0]
  t2[, wt := pte/ptr]
  merge(unique(wide[is_test == FALSE, c("No", v), with = FALSE]),
        t2[, c(v,"wt"), with = FALSE], by = v)[order(No), wt]
}
W_SEG <- mkw("segmentind")
rwll <- function(P) { l <- -log(pmax(P[cbind(seq_along(ytr), ytr)], 1e-15)); sum(W_SEG*l)/sum(W_SEG) }
cat(sprintf("  segment reweighting: ESS %.0f of %d tasks; reference 1.19610 -> public 1.197\n",
            sum(W_SEG)^2/sum(W_SEG^2), length(ytr)))

# ---- one-hot attribute level dummies (part-worth coding for the tree) -------
allA <- rbind(trl[, ..ATTRS], tel[, ..ATTRS])
lev <- lapply(ATTRS, function(a) sort(unique(allA[[a]]))); names(lev) <- ATTRS
onehot <- function(D) do.call(cbind, lapply(ATTRS, function(a) {
  M <- outer(D[[a]], lev[[a]], "==") * 1.0
  colnames(M) <- paste0("oh_", a, "_L", lev[[a]]); M }))
OH_tr <- onehot(trl[, ..ATTRS]); OH_te <- onehot(tel[, ..ATTRS])
oh_cols <- colnames(OH_tr)
for (j in seq_along(oh_cols)) { set(trl, NULL, oh_cols[j], OH_tr[, j])
                                set(tel, NULL, oh_cols[j], OH_te[, j]) }
cat(sprintf("  one-hot block: %d level dummies over %d attributes\n", length(oh_cols), length(ATTRS)))

# ---- MF cold-start person-taste bridge, FOLD-NESTED -------------------------
# Person signal: for each attribute, the mean (chosen level - offered-menu mean)
# over tasks where a REAL bundle was picked; plus the price analogue; plus the
# person's raw decline rate. SVD that person x item matrix, then regress each
# latent score on DEMOGRAPHICS ONLY. Every respondent -- fit, held-out or test --
# receives a demographics-only prediction, so the feature distribution is
# identical everywhere. That symmetry is what makes the cold-start test honest.
MF_RANK <- 3L
mf_cols <- c("mf_fit", "mf_outside")
# A respondent who declined on EVERY task contributes no (chosen - menu mean)
# delta at all, so their taste row is undefined rather than zero. They are given
# a zero row, which is the population mean after centring -- i.e. "no evidence
# of atypical taste" -- and their decline rate still carries their real signal.
person_signal <- function(D) {
  d3 <- D[alt %in% 1:3]
  mm <- d3[, lapply(.SD, mean), by = No, .SDcols = ATTRS]
  setnames(mm, ATTRS, paste0(ATTRS, "__m"))
  ch <- D[chosen == TRUE & alt %in% 1:3, c("No", ATTRS), with = FALSE]
  del <- merge(ch, mm, by = "No")
  for (a in ATTRS) set(del, NULL, a, del[[a]] - del[[paste0(a, "__m")]])
  key <- unique(D[, .(No, Case)])
  del <- merge(del[, c("No", ATTRS), with = FALSE], key, by = "No")
  ps <- del[, lapply(.SD, mean), by = Case, .SDcols = ATTRS]
  y4 <- unique(D[, .(No, Case, out = as.integer(y == 4L))])
  orate <- y4[, .(out = mean(out)), by = Case]
  ps <- merge(data.table(Case = sort(unique(D$Case))), ps, by = "Case", all.x = TRUE)
  for (a in ATTRS) set(ps, which(!is.finite(ps[[a]])), a, 0)
  merge(ps, orate, by = "Case")
}
fit_mf <- function(tr_rows) {
  P <- person_signal(trl[tr_rows])
  R <- as.matrix(P[, ..ATTRS]); ctr <- colMeans(R)
  Rc <- sweep(R, 2, ctr, "-")
  sv <- svd(Rc, nu = MF_RANK, nv = MF_RANK)
  Z <- Rc %*% sv$v
  dtab <- unique(trl[, c("Case", demo), with = FALSE])
  dz <- dtab[match(P$Case, dtab$Case)]
  X <- cbind(1, as.matrix(dz[, ..demo]))
  cf <- qr.solve(crossprod(X) + diag(1e-3, ncol(X)), crossprod(X, cbind(Z, P$out)))
  stopifnot(all(is.finite(cf)))
  list(ctr = ctr, V = sv$v, cf = cf)
}
apply_mf <- function(m, D) {
  dz <- unique(D[, c("Case", demo), with = FALSE])
  X <- cbind(1, as.matrix(dz[, ..demo]))
  pred <- X %*% m$cf                                   # nCase x (RANK + 1)
  taste <- sweep(pred[, seq_len(MF_RANK), drop = FALSE] %*% t(m$V), 2, m$ctr, "+")
  idx <- match(D$Case, dz$Case)
  fitv <- rowSums(as.matrix(D[, ..ATTRS]) * taste[idx, , drop = FALSE])
  fitv[D$alt == 4L] <- 0                               # the outside option has no attributes
  data.table(mf_fit = fitv, mf_outside = pred[idx, MF_RANK + 1L])
}

# ---- listwise objective -----------------------------------------------------
sbt <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
                     E <- exp(M); E / rowSums(E) }
obj_lw <- function(preds, dtrain) {
  yy <- getinfo(dtrain, "label"); p <- as.vector(t(sbt(preds)))
  list(grad = p - yy, hess = pmax(2*p*(1-p), 1e-6))
}
FEAT_CORE <- c(ATTRS, paste0(ATTRS,"_c"), "richness","lvlsum","price_rank","price_min_rival",
               "price_task_mean","rich_task_mean","Task", paste0("alt",1:4), demo, oh_cols)

# segment importance weight per TASK, raised to tau (tau=0 unweighted, 1 full)
segw <- function(tau) {
  w <- W_SEG^tau
  rep(w, each = 4L)
}

oof_run <- function(depth, nr, use_mf, tau, seeds) {
  acc <- matrix(0, 21565L, 4L)
  for (s in seq_len(seeds)) {
    P <- matrix(NA_real_, 21565L, 4L)
    for (k in 1:5) {
      ins <- which(trl$fold != k); out <- which(trl$fold == k)
      d <- trl[ins]; v <- trl[out]; fs <- FEAT_CORE
      if (use_mf) {
        m  <- fit_mf(ins)                    # fitted on this fold's TRAINING rows only
        fd <- apply_mf(m, d); fv <- apply_mf(m, v)
        for (cc in mf_cols) { set(d, NULL, cc, fd[[cc]]); set(v, NULL, cc, fv[[cc]]) }
        fs <- c(fs, mf_cols)
      }
      dm <- xgb.DMatrix(as.matrix(d[, ..fs]), label = as.numeric(d$chosen))
      if (tau > 0) setinfo(dm, "weight", segw(tau)[ins])
      fit <- xgb.train(params = list(eta = 0.05, max_depth = depth, min_child_weight = 5,
                                     subsample = 0.8, colsample_bytree = 0.8,
                                     base_score = 0, nthread = 4, seed = s),
                       data = dm, nrounds = nr, verbose = 0, obj = obj_lw, maximize = FALSE)
      P[match(unique(v$No), NOS), ] <-
        sbt(as.vector(predict(fit, xgb.DMatrix(as.matrix(v[, ..fs])), outputmargin = TRUE)))
    }
    acc <- acc + log(pmax(P, 1e-15))
  }
  E <- exp(acc/seeds - apply(acc/seeds, 1, max)); E / rowSums(E)
}

rec <- function(tag, P, t0) {
  r <- data.table(cfg = tag, plain = logloss(ytr, P), seg = rwll(P),
                  mins = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  cat(sprintf("  %-28s plain %.5f   SEG %.5f   (%.1f min)\n", tag, r$plain, r$seg, r$mins))
  r
}

res <- list()
rule(sprintf("S1 -- HONEST BASELINE: no encoding, one-hot, depth sweep (%d seeds)", S_SCREEN))
cat("  reference to beat on SEG: 1.19610 (production blend)\n")
for (cfg in list(c(3,400), c(4,400), c(5,300), c(6,250))) {
  t0 <- Sys.time()
  P <- oof_run(cfg[1], cfg[2], FALSE, 0, S_SCREEN)
  res[[length(res)+1]] <- rec(sprintf("d%d_r%d", cfg[1], cfg[2]), P, t0)
  saveRDS(rbindlist(res), file.path(DIR, "stages.rds"))
}
R <- rbindlist(res); best1 <- R[which.min(seg)]
cat(sprintf("\n  S1 winner on SEG: %s at %.5f\n", best1$cfg, best1$seg))
bd <- as.integer(sub("d(\\d+)_.*", "\\1", best1$cfg)); br <- as.integer(sub(".*_r(\\d+)", "\\1", best1$cfg))

rule("S2 -- + MF COLD-START FEATURES")
t0 <- Sys.time(); P2 <- oof_run(bd, br, TRUE, 0, S_SCREEN)
res[[length(res)+1]] <- rec(sprintf("d%d_r%d+mf", bd, br), P2, t0)
saveRDS(rbindlist(res), file.path(DIR, "stages.rds"))
use_mf <- rbindlist(res)[.N, seg] < best1$seg
cat(sprintf("  MF %s (SEG %+.5f)\n", if (use_mf) "KEPT" else "DROPPED",
            rbindlist(res)[.N, seg] - best1$seg))

rule("S3 -- + SEGMENT IMPORTANCE WEIGHTING IN THE FIT")
base3 <- if (use_mf) rbindlist(res)[.N, seg] else best1$seg
for (tau in c(0.25, 0.50, 1.00)) {
  t0 <- Sys.time(); P3 <- oof_run(bd, br, use_mf, tau, S_SCREEN)
  res[[length(res)+1]] <- rec(sprintf("d%d_r%d%s_tau%.2f", bd, br,
                                      if (use_mf) "+mf" else "", tau), P3, t0)
  saveRDS(rbindlist(res), file.path(DIR, "stages.rds"))
}
R <- rbindlist(res); setorder(R, seg)
rule("ALL STAGES, ranked by the SEGMENT-REWEIGHTED metric")
print(R)
fwrite(R, file.path(DIR, "stages.csv"))
cat(sprintf("\n  best %s: plain %.5f  SEG %.5f  vs production SEG 1.19610 (delta %+.5f)\n",
            R$cfg[1], R$plain[1], R$seg[1], R$seg[1] - 1.19610))
cat("\n  NEXT: rebuild the winner at 20 seeds, emit oof_xgb_syn/test_xgb_syn,\n")
cat("  blend with lcmnl3_both, then apply the probe margin at deploy.\n")
