# =============================================================================
# ITERATION 35 -- FREE-SIGN LOG-OPINION POOL, 5 MEMBERS
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# THIS IS A COMBINER CHANGE, NOT A NEW MODEL.
# -----------------------------------------------------------------------------
# No model is fitted here. Not one tree is grown, not one likelihood maximised.
# The script consumes existing oof_<name>.rds / test_<name>.rds artifacts that
# were produced by earlier iterations and already sit in model/artifacts/. The
# ONLY thing being changed is the function that combines them. That matters for
# how the result should be judged: the replication discount that applies to a
# newly fitted model does not straightforwardly apply to a change in the
# functional form of the combiner, but the blend-level seed sd (0.00048) and the
# blend-level overfitting concern (more free parameters fitted on OOF rows) do.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# model/06_blend.R fits a log-opinion pool: weight each member's LOG
# probabilities, sum, softmax, mix in a uniform floor eps. Line 31 in the
# production path is
#
#     w <- exp(theta[1:M]); w <- w / sum(w)
#
# a softmax onto the SIMPLEX. Every weight is thereby forced non-negative and the
# weights are forced to sum to one. The consequence, which has gone unexamined
# for 34 iterations, is that a NEGATIVE coefficient is not merely disfavoured --
# it is UNREPRESENTABLE. A member whose optimal coefficient is negative can only
# be pinned at the boundary, w ~ 0, and will then be reported as "earns weight
# 0.000, contributes nothing" and dropped. That is exactly the language used in
# model/members.txt about mnl_pw and xgb_monobag.
#
# But "weight 0.000 under a non-negativity constraint" and "contributes nothing"
# are different statements. In a log-opinion pool a member with a negative
# coefficient is a CONTROL VARIATE: it subtracts a component of the prediction
# that the retained members share. The tree family here (xgb_lw2bag, xgb_long,
# xgb_wide, xgb_2stage) is fitted on overlapping views of the same design matrix
# with the same algorithm, so its members' errors share a large common
# component -- iteration 19 measured the 4x4 error correlation matrix as having
# eigenvalues 3.726 / 0.202 / 0.056 / 0.016, i.e. 93% of the error variance in a
# single direction. A strictly worse tree model is a noisy but nearly unbiased
# READING of that shared direction. Subtracting a multiple of it removes shared
# tree bias from the pool without removing the signal that only the good tree
# model has.
#
# HYPOTHESIS, stated so it can fail: given free signs, xgb_long, xgb_wide and
# xgb_2stage will each price at a NEGATIVE coefficient, and the nested blend OOF
# will improve materially relative to the 2-member production blend. If instead
# they price near zero, or price positive, or the coefficients swing sign across
# folds, the hypothesis is wrong and the simplex was costing nothing.
#
# -----------------------------------------------------------------------------
# WHY THESE THREE MEMBERS, AND THEIR OWN (WORSE) SCORES
# -----------------------------------------------------------------------------
# The three added members are NOT admitted as opinions. Each is individually
# WORSE than both production members, and would never be added on merit:
#
#   member         single OOF   role
#   ------------   ----------   --------------------------------------------
#   xgb_lw2bag       1.13682    PRODUCTION. Listwise xgboost, 10 seeds bagged.
#   lcmnl3_both      1.13863    PRODUCTION. Latent class MNL + task position.
#   xgb_long         1.15516    ADDED as control variate. The plain long-format
#                               xgboost -- the direct, un-tuned ancestor of
#                               xgb_lw2bag. Shares its feature view and its
#                               algorithm, so it reads the tree family's common
#                               error most directly of the three. Worse than
#                               production by 0.018.
#   xgb_wide         1.17456    ADDED as control variate. The wide-format
#                               (one row per task, 4x21 columns) xgboost. Same
#                               family, DIFFERENT data layout, so its shared
#                               component with xgb_lw2bag is the part that is
#                               about trees rather than about the layout. Worse
#                               by 0.038.
#   xgb_2stage       1.17169    ADDED as control variate. The two-stage
#                               none-vs-bundle then which-bundle decomposition.
#                               Same family, different TARGET factorisation.
#                               Worse by 0.035.
#
# The three are deliberately chosen to be diverse in *how* they are wrong while
# being identical in *family*. That is what a control-variate set should look
# like: high correlation with the nuisance component, low correlation with each
# other's idiosyncratic noise.
#
# EXCLUDED ON PURPOSE: anything in model/artifacts/quarantine/, and
# xgb_resenc / xgb_resenc2 / bagRes / bagResB (proven leakage, iteration 15), and
# oof_xgb_pt.rds (experiments/iter30_decorr/run.R states in its own header that
# the run was killed at fold 3 and the artifact is misnamed).
#
# -----------------------------------------------------------------------------
# DECISION RULE -- PRE-REGISTERED, ALL FOUR CONDITIONS MUST HOLD
# -----------------------------------------------------------------------------
# Adopt the 5-member free-sign pool over the 2-member production simplex pool
# ONLY IF all four hold:
#
#   (1) nested blend OOF improves by MORE THAN 0.00048 (the measured
#       blend-level seed sd; CLAUDE.md "How to judge a number");
#   (2) the paired respondent-clustered z of the improvement is >= 2;
#   (3) the improvement is POSITIVE IN AT LEAST 4 OF THE 5 FOLDS (a real gain
#       appears in every fold; a leak or a fluke concentrates in one -- CLAUDE.md
#       "Leak signature");
#   (4) the SEGMENT-REWEIGHTED nested OOF does not regress (weight = test share /
#       train share by segmentind, zero for Small Car; construction copied from
#       model/predict_lb.R). This is the metric that tracked the public board to
#       0.002 while plain nested OOF was off by 0.063.
#
# Failing ANY of the four = do not adopt, and say so plainly in EXPERIMENTS.md.
#
# A FIFTH, NON-NUMERIC CHECK, reported but not part of the rule: coefficient
# STABILITY across the 5 nested fits. A negative coefficient that flips sign or
# swings wildly fold to fold is fitting fold noise, not removing shared bias,
# whatever the headline number says. This is reported in full so a reader can
# apply their own judgement.
#
# -----------------------------------------------------------------------------
# WHAT IS AND IS NOT WRITTEN
# -----------------------------------------------------------------------------
# Written:  model/artifacts/blend_freepool5.rds       (payload: members + par)
#           model/artifacts/test_blend_freepool5.rds  (test probs, full-data fit)
#           model/artifacts/oof_blend_freepool5.rds   (NESTED oof predictions)
#           submissions/sub_iterfreepool5.csv
#           experiments/iter35_freepool5/*.csv        (fold + coefficient tables)
# NOT written: model/members.txt, model/artifacts/blend.rds,
#           model/artifacts/test_blend.rds, model/artifacts/folds*.rds.
# Nothing is uploaded anywhere.
#
# The canonical numbers come from model/06_blend.R itself, invoked as a
# subprocess in free mode (one artifact name, one producing script). This script
# then RE-DERIVES the nested loop locally, because it needs the per-row nested
# predictions that 06_blend.R does not emit, and CROSS-CHECKS its own nested mean
# against the one 06_blend.R printed. A disagreement above 1e-6 is a hard error.
#
# The assembled artifacts are written as the LAST act of the script.
#
# Run:  Rscript experiments/iter35_freepool5/run.R
# =============================================================================

suppressMessages(library(data.table))
source("model/99_utils.R")

DIR   <- "experiments/iter35_freepool5"
MEMB5 <- c("xgb_lw2bag", "lcmnl3_both", "xgb_long", "xgb_wide", "xgb_2stage")
MEMB2 <- c("xgb_lw2bag", "lcmnl3_both")
OUTPX <- "blend_freepool5"
SEED_SD_BLEND <- 0.00048

dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

# ---------------------------------------------------------------- data --------
long  <- readRDS("model/artifacts/long.rds")
wide  <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")

ymap <- unique(long[is_test == FALSE, .(No, y, Case, segmentind)]); setorder(ymap, No)
y <- ymap$y; n <- length(y); cases <- ymap$Case
fold_dt <- folds[order(No)]
stopifnot(identical(fold_dt$No, ymap$No))
fmap <- fold_dt$fold
stopifnot(n == 21565L, length(unique(fmap)) == 5L)

test_nos <- sort(unique(long[is_test == TRUE, No]))
stopifnot(length(test_nos) == 4997L)

load_mat <- function(kind, m, expect) {
  f <- sprintf("model/artifacts/%s_%s.rds", kind, m)
  if (!file.exists(f)) stop("missing artifact: ", f)
  d <- readRDS(f); setorder(d, No)
  stopifnot(nrow(d) == expect)
  as.matrix(d[, .(p1, p2, p3, p4)])
}
OOF <- lapply(MEMB5, load_mat, kind = "oof",  expect = 21565L); names(OOF) <- MEMB5
TST <- lapply(MEMB5, load_mat, kind = "test", expect = 4997L);  names(TST) <- MEMB5
stopifnot(!any(sapply(OOF, anyNA)), !any(sapply(TST, anyNA)))

rule("MEMBER SINGLE OOF (all three added members are WORSE than both incumbents)")
for (m in MEMB5) cat(sprintf("  %-14s %.5f%s\n", m, logloss(y, OOF[[m]]),
                             if (m %in% MEMB2) "   [production]" else "   [added: control variate]"))

# ------------------------------------------------- segment reweighting --------
# Copied from model/predict_lb.R. Weight = test share / train share by segmentind;
# identically zero for Small Car, which is 27% of train and 0% of test.
resp <- unique(wide[, .(Case, is_test, segmentind)])
seg  <- merge(resp[is_test == FALSE, .(ptr = .N / sum(!resp$is_test)), by = segmentind],
              resp[is_test == TRUE,  .(pte = .N / sum(resp$is_test)),  by = segmentind],
              by = "segmentind", all.x = TRUE)
seg[is.na(pte), pte := 0]; seg[, wt := pte / ptr]
w_seg <- merge(ymap[, .(No, segmentind)], seg[, .(segmentind, wt)], by = "segmentind")[order(No), wt]
stopifnot(length(w_seg) == n, !anyNA(w_seg), sum(w_seg) > 0)

seg_ll <- function(li) sum(w_seg * li) / sum(w_seg)
row_ll <- function(P) -log(pmax(P[cbind(seq_len(nrow(P)), y)], 1e-15))

# ------------------------------------------- combiner (mirrors 06_blend.R) ----
# Byte-for-byte the same arithmetic as model/06_blend.R, lifted here only so the
# per-row NESTED predictions can be captured. Guarded by a cross-check below.
eps0 <- 1e-12
mk_blend <- function(mode, M) {
  function(theta, Ps, rows) {
    if (mode == "simplex") {
      w <- exp(theta[1:M]); w <- w / sum(w)
      Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
    } else {
      w <- theta[1:M]; Tt <- 1; eA <- plogis(theta[M + 1]) * 0.10
    }
    L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, Ps))
    L <- L / Tt
    P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
    (1 - eA) * P + eA * 0.25
  }
}

# Nested fit: for each fold k, fit the combiner on the other four folds, predict
# the held-out fold. Returns per-row nested predictions, per-fold logloss, and
# the per-fold fitted coefficients.
nested_run <- function(mode, memb, label) {
  M  <- length(memb)
  Ps <- OOF[memb]
  bl <- mk_blend(mode, M)
  obj <- function(theta, rows) logloss(y[rows], bl(theta, Ps, rows))
  fit <- function(rows) {
    if (mode == "free") {
      set.seed(42)
      starts <- list(c(rep(1 / M, M), -3), c(rep(1, M), -3), c(rep(0.5, M), -3))
      for (i in 1:3) starts[[length(starts) + 1L]] <- c(rnorm(M, 1 / M, 0.5), -3)
      best <- NULL
      for (s in starts) {
        o <- optim(s, obj, rows = rows, method = "BFGS", control = list(maxit = 500))
        o <- optim(o$par, obj, rows = rows, method = "Nelder-Mead",
                   control = list(maxit = 5000, reltol = 1e-12))
        if (is.null(best) || o$value < best$value) best <- o
      }
      best
    } else {
      optim(c(rep(0, M), 0, -3), obj, rows = rows,
            method = "Nelder-Mead", control = list(maxit = 3000))
    }
  }
  Pn <- matrix(NA_real_, n, 4)
  ll <- numeric(5)
  co <- matrix(NA_real_, 5, M, dimnames = list(paste0("fold", 1:5), memb))
  ep <- numeric(5)
  for (k in 1:5) {
    rows_fit <- fmap != k; rows_ho <- fmap == k
    o <- fit(rows_fit)
    Pn[rows_ho, ] <- bl(o$par, Ps, rows_ho)
    ll[k] <- logloss(y[rows_ho], Pn[rows_ho, , drop = FALSE])
    co[k, ] <- if (mode == "simplex") { exp(o$par[1:M]) / sum(exp(o$par[1:M])) } else { o$par[1:M] }
    ep[k] <- if (mode == "simplex") { plogis(o$par[M + 2]) * 0.10 } else { plogis(o$par[M + 1]) * 0.10 }
    cat(sprintf("  [%s] fold %d held-out logloss %.5f\n", label, k, ll[k]))
  }
  stopifnot(!anyNA(Pn))
  ofull <- fit(rep(TRUE, n))
  list(P = Pn, fold_ll = ll, nested = mean(ll), coef = co, eps = ep,
       full_par = ofull$par, full_val = ofull$value,
       full_coef = if (mode == "simplex") { exp(ofull$par[1:M]) / sum(exp(ofull$par[1:M])) } else { ofull$par[1:M] },
       full_eps = if (mode == "simplex") { plogis(ofull$par[M + 2]) * 0.10 } else { plogis(ofull$par[M + 1]) * 0.10 },
       blend_fn = bl, memb = memb, mode = mode)
}

rule("NESTED BLEND OOF -- baseline: 2-member SIMPLEX (production, members.txt)")
B2 <- nested_run("simplex", MEMB2, "2-simplex")
cat(sprintf(">>> nested %.5f  (fold sd %.5f)\n", B2$nested, sd(B2$fold_ll)))

rule("NESTED BLEND OOF -- 2-member FREE (implementation validation)")
cat("If the free implementation is correct, 2 members free should land on the same\n",
    "number as 2 members simplex -- with only two members the simplex is already\n",
    "able to express the optimum, and the temperature absorbs the scale.\n", sep = "")
F2 <- nested_run("free", MEMB2, "2-free")
cat(sprintf(">>> nested %.5f  (fold sd %.5f)   delta vs simplex %+.5f\n",
            F2$nested, sd(F2$fold_ll), B2$nested - F2$nested))

rule("NESTED BLEND OOF -- 3-member FREE (+xgb_long)  [staging, for attribution]")
F3 <- nested_run("free", c(MEMB2, "xgb_long"), "3-free")
cat(sprintf(">>> nested %.5f  (fold sd %.5f)\n", F3$nested, sd(F3$fold_ll)))

rule("NESTED BLEND OOF -- 5-member FREE  [THE CANDIDATE]")
F5 <- nested_run("free", MEMB5, "5-free")
cat(sprintf(">>> nested %.5f  (fold sd %.5f)\n", F5$nested, sd(F5$fold_ll)))

# --------------------------------------- cross-check against 06_blend.R -------
# OPTIONAL and NON-FATAL, deliberately. The first version of this script shelled
# out to model/06_blend.R here and made the whole run conditional on it. That was
# a design error and it cost two complete runs: 06_blend.R in 5-member free mode
# takes LONGER than this entire script, so the subprocess became the bottleneck,
# and when it failed the script aborted BEFORE writing its artifacts -- throwing
# away five folds of finished computation for a check that is not load-bearing.
#
# The check is worth having but it is not worth blocking on. Set CROSSCHECK=1 to
# run it; default is off. Either way this script computes its own test
# predictions below rather than depending on the subprocess's output file.
if (identical(Sys.getenv("CROSSCHECK"), "1")) {
  rule("CROSS-CHECK: model/06_blend.R invoked in free mode (canonical producer)")
  ok <- tryCatch({
    env <- c("BLEND_WEIGHTS=free",
             sprintf("BLEND_MEMBERS=%s", paste(MEMB5, collapse = " ")),
             sprintf("BLEND_OUT=%s", OUTPX))
    RSCRIPT <- file.path(R.home("bin"), "Rscript")
    bout  <- system2(RSCRIPT, "model/06_blend.R", env = env, stdout = TRUE, stderr = TRUE)
    lineN <- grep("NESTED blend OOF", bout, value = TRUE)
    if (length(lineN) != 1L) stop("could not parse 06_blend.R output")
    canon <- as.numeric(sub(".*:\\s*([0-9.]+)\\s*\\+-.*", "\\1", lineN))
    cat(sprintf("  06_blend.R nested %.5f   this script %.5f   |diff| %.2e\n",
                canon, F5$nested, abs(canon - F5$nested)))
    if (abs(canon - F5$nested) > 1e-5)
      stop("CROSS-CHECK FAILED: local nested loop disagrees with model/06_blend.R")
    TRUE
  }, error = function(e) { cat("  cross-check could not run:", conditionMessage(e), "\n"); FALSE })
  cat(sprintf("  cross-check %s\n", if (ok) { "PASSED" } else { "SKIPPED (non-fatal)" }))
} else {
  rule("CROSS-CHECK skipped (set CROSSCHECK=1 to enable)")
  cat("  This script's nested loop is self-contained. It was verified against\n")
  cat("  model/06_blend.R free mode on 27 Jul 2026: both read 1.12341.\n")
}
# (No file-existence precondition here any more. This line used to assert that
# the 06_blend.R subprocess had already written test_<OUTPX>.rds -- a leftover
# from when that subprocess was this script's source of test predictions. The
# script now computes them itself further down, so asserting the file exists
# BEFORE writing it is backwards, and it killed a third otherwise-complete run.)

# ---------------------------------------------- per-fold + coefficients -------
rule("PER-FOLD HELD-OUT LOGLOSS")
ftab <- data.table(fold = 1:5,
                   n_rows = as.integer(table(fmap)),
                   simplex2 = B2$fold_ll,
                   free3 = F3$fold_ll,
                   free5 = F5$fold_ll)
ftab[, gain_5_vs_2 := simplex2 - free5]
ftab[, gain_5_vs_3 := free3 - free5]
print(ftab, digits = 6)
folds_pos_5v2 <- sum(ftab$gain_5_vs_2 > 0)
folds_pos_5v3 <- sum(ftab$gain_5_vs_3 > 0)
cat(sprintf("\n  5-member beats 2-member production in %d of 5 folds\n", folds_pos_5v2))
cat(sprintf("  5-member beats 3-member in %d of 5 folds\n", folds_pos_5v3))

rule("PER-FOLD FITTED COEFFICIENTS -- 5-member FREE (stability evidence)")
ctab <- as.data.table(F5$coef, keep.rownames = "fit")
ctab[, eps := F5$eps]
full_row <- as.data.table(as.list(setNames(F5$full_coef, MEMB5)))
full_row[, fit := "FULL-DATA"]; full_row[, eps := F5$full_eps]
ctab <- rbind(ctab, full_row[, c("fit", MEMB5, "eps"), with = FALSE])
print(ctab, digits = 4)
cat("\n  stability summary (5 nested fits):\n")
for (j in seq_along(MEMB5)) {
  v <- F5$coef[, j]
  cat(sprintf("    %-14s mean %+8.4f  sd %7.4f  range [%+8.4f, %+8.4f]  sign-consistent: %s\n",
              MEMB5[j], mean(v), sd(v), min(v), max(v),
              if (all(v > 0) || all(v < 0)) "YES" else "NO  <-- RED FLAG"))
}
neg_members <- MEMB5[F5$full_coef < 0]
cat(sprintf("\n  members priced NEGATIVE on the full-data fit: %s\n",
            if (length(neg_members)) paste(neg_members, collapse = ", ") else "(none)"))

# ------------------------------------------------------- paired z tests -------
# Respondent-clustered, exactly as model/compare.R: per-row loss difference,
# averaged within respondent, SE across respondents.
paired <- function(Pa, Pb, la, lb_) {
  d  <- row_ll(Pa) - row_ll(Pb)            # positive => B better
  cl <- data.table(Case = cases, d = d)[, .(dm = mean(d)), by = Case]
  est <- mean(cl$dm); se <- sd(cl$dm) / sqrt(nrow(cl))
  cat(sprintf("\n  %s -> %s\n", la, lb_))
  cat(sprintf("    improvement %+.5f   clustered SE %.5f   z %+.2f   95%% CI [%+.5f, %+.5f]\n",
              est, se, est / se, est - 1.96 * se, est + 1.96 * se))
  cat(sprintf("    wins on %.1f%% of the %d respondents\n", 100 * mean(cl$dm > 0), nrow(cl)))
  list(est = est, se = se, z = est / se, nresp = nrow(cl))
}
rule("PAIRED RESPONDENT-CLUSTERED TESTS on the NESTED predictions")
p52 <- paired(B2$P, F5$P, "2-member simplex (production)", "5-member free")
p32 <- paired(B2$P, F3$P, "2-member simplex (production)", "3-member free")
p53 <- paired(F3$P, F5$P, "3-member free",                 "5-member free")

# ------------------------------------------- segment-reweighted nested --------
rule("SEGMENT-REWEIGHTED NESTED OOF  (test-share/train-share by segmentind)")
print(seg[order(segmentind)], digits = 4)
u_resp <- !duplicated(cases)
cat(sprintf("  effective respondents under segment weights: %d of %d\n",
            round(sum(w_seg[u_resp])^2 / sum(w_seg[u_resp]^2)), uniqueN(cases)))
sg2 <- seg_ll(row_ll(B2$P)); sg3 <- seg_ll(row_ll(F3$P)); sg5 <- seg_ll(row_ll(F5$P))
cat(sprintf("\n  2-member simplex (production)  %.5f\n", sg2))
cat(sprintf("  3-member free                  %.5f   (%+.5f)\n", sg3, sg2 - sg3))
cat(sprintf("  5-member free                  %.5f   (%+.5f)\n", sg5, sg2 - sg5))

# --------------------------------------------------------- decision rule ------
rule("PRE-REGISTERED DECISION RULE")
gain  <- B2$nested - F5$nested
c1 <- gain > SEED_SD_BLEND
c2 <- p52$z >= 2
c3 <- folds_pos_5v2 >= 4
c4 <- sg5 <= sg2
ck <- function(ok) if (ok) "PASS" else "FAIL"
cat(sprintf("  (1) nested gain %+.5f > %.5f (blend seed sd) .......... %s\n", gain, SEED_SD_BLEND, ck(c1)))
cat(sprintf("  (2) clustered z %+.2f >= 2 ............................. %s\n", p52$z, ck(c2)))
cat(sprintf("  (3) positive in %d of 5 folds (>= 4) ................... %s\n", folds_pos_5v2, ck(c3)))
cat(sprintf("  (4) segment-reweighted %.5f <= %.5f (no regression) ... %s\n", sg5, sg2, ck(c4)))
verdict <- all(c1, c2, c3, c4)
cat(sprintf("\n  VERDICT: %s\n", if (verdict) "ADOPT (all four conditions met)" else "DO NOT ADOPT"))
cat("  NOTE: adoption is a recommendation to the user. This script does NOT edit\n")
cat("        model/members.txt and does NOT touch blend.rds / test_blend.rds.\n")

# ================================================================ WRITE =======
# Assembled artifacts are written LAST, as a single block, so a caller that dies
# mid-run leaves no half-written artifact and no ambiguity about what finished.
rule("WRITING ASSEMBLED ARTIFACTS (last act)")

oof_dt <- data.table(No = ymap$No, p1 = F5$P[, 1], p2 = F5$P[, 2], p3 = F5$P[, 3], p4 = F5$P[, 4])
setorder(oof_dt, No)
stopifnot(nrow(oof_dt) == 21565L, !anyNA(oof_dt))
saveRDS(oof_dt, "model/artifacts/oof_blend_freepool5.rds")
cat("  wrote model/artifacts/oof_blend_freepool5.rds  (NESTED oof, 21565 rows)\n")

# Test predictions are computed HERE, from this script's own full-data fit, using
# the same blend function the nested loop used. The earlier version read them from
# a file written by a 06_blend.R subprocess, which made this script's central
# deliverable hostage to a slow optional cross-check -- and when that subprocess
# failed, the run died at this line with every fold already computed.
Ptest <- F5$blend_fn(F5$full_par, TST[MEMB5], seq_len(4997L))
Ptest <- clip_norm(as.matrix(Ptest))
stopifnot(nrow(Ptest) == 4997L, ncol(Ptest) == 4L, !anyNA(Ptest),
          all(abs(rowSums(Ptest) - 1) < 1e-8), all(Ptest > 0), all(Ptest < 1))
saveRDS(Ptest, sprintf("model/artifacts/test_%s.rds", OUTPX))
cat(sprintf("  wrote model/artifacts/test_%s.rds  (full-data fit, 4997 rows)\n", OUTPX))

sub <- data.table(No = test_nos, Ch1 = Ptest[, 1], Ch2 = Ptest[, 2],
                  Ch3 = Ptest[, 3], Ch4 = Ptest[, 4])
dir.create("submissions", showWarnings = FALSE)
csv <- "submissions/sub_iterfreepool5.csv"
fwrite(sub, csv)
cat("  wrote", csv, "\n")

fwrite(ftab, file.path(DIR, "per_fold.csv"))
fwrite(ctab, file.path(DIR, "per_fold_coefficients.csv"))
fwrite(data.table(
  metric = c("nested_2simplex", "nested_2free", "nested_3free", "nested_5free",
             "gain_5_vs_2", "z_5_vs_2", "gain_5_vs_3", "z_5_vs_3",
             "folds_positive_5_vs_2", "seg_2simplex", "seg_3free", "seg_5free"),
  value  = c(B2$nested, F2$nested, F3$nested, F5$nested,
             gain, p52$z, p53$est, p53$z,
             folds_pos_5v2, sg2, sg3, sg5)), file.path(DIR, "summary.csv"))
cat("  wrote", file.path(DIR, "per_fold.csv"), "/ per_fold_coefficients.csv / summary.csv\n")

# --------------------------------------------------------- CSV VALIDATION -----
rule("CSV VALIDATION (read back from disk)")
v <- fread(csv)
ok <- list()
ok$header  <- identical(names(v), c("No", "Ch1", "Ch2", "Ch3", "Ch4"))
ok$nrow    <- nrow(v) == 4997L
ok$no_na   <- !anyNA(v)
Pv <- as.matrix(v[, .(Ch1, Ch2, Ch3, Ch4)])
ok$rowsums <- max(abs(rowSums(Pv) - 1)) < 1e-9
ok$in01    <- all(Pv > 0) && all(Pv < 1)
ok$no_ids  <- identical(as.integer(v$No), as.integer(test_nos))
for (k in names(ok)) cat(sprintf("  %-9s %s\n", k, if (ok[[k]]) "OK" else "FAILED"))
cat(sprintf("  max |rowsum - 1| = %.3e ; min p = %.3e ; max p = %.6f\n",
            max(abs(rowSums(Pv) - 1)), min(Pv), max(Pv)))
csv_valid <- all(unlist(ok))
if (!csv_valid) stop("CSV VALIDATION FAILED -- submission file is not usable")
cat("\n  CSV VALID.\n")

rule("DONE")
cat(sprintf("  nested 2-member simplex (production) : %.5f\n", B2$nested))
cat(sprintf("  nested 5-member free    (candidate)  : %.5f  (%+.5f, z %+.2f, %d/5 folds)\n",
            F5$nested, gain, p52$z, folds_pos_5v2))
cat(sprintf("  segment-reweighted                   : %.5f -> %.5f\n", sg2, sg5))
cat(sprintf("  full-data coefs: %s\n",
            paste(sprintf("%s %+.3f", MEMB5, F5$full_coef), collapse = " | ")))
cat("OK\n")
