# =============================================================================
# ITERATION 34 -- freepool3: the 3-member UNCONSTRAINED-SIGN log-opinion pool.
#
# THIS IS A COMBINER CHANGE, NOT A NEW MODEL.
# Nothing is fitted here except blend coefficients. No xgboost, no mlogit, no
# feature engineering, no retune. The script consumes existing oof_*/test_*
# artifacts only and never touches folds.rds. It therefore spends one selection
# event on the *combiner*, not on the model search space that CLAUDE.md declares
# measured-exhausted.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS (stated before any result was seen)
# -----------------------------------------------------------------------------
# model/06_blend.R fits a log-opinion pool: weight each member's log-probabilities,
# sum, softmax, mix in a uniform floor eps. Line 31 in the historical version reads
#
#     w <- exp(theta[1:M]); w <- w / sum(w)
#
# which is a softmax onto the SIMPLEX. Every weight is forced non-negative and the
# weights are forced to sum to 1. A NEGATIVE coefficient is therefore not merely
# disfavoured, it is UNREPRESENTABLE.
#
# That constraint encodes an assumption: that every member is an *opinion* to be
# averaged, and that the only question is how much to trust it. But a pool of
# log-probabilities is a linear model in log space, and a linear model has a second
# use for a regressor -- as a CONTROL VARIATE. A member that is individually WORSE,
# but whose errors are strongly correlated with the shared bias of the family it
# belongs to, is useful precisely by being SUBTRACTED: it estimates the common
# component so that component can be removed from the members we do trust.
#
# Under the simplex, such a member can only be priced at (approximately) zero and is
# then recorded as "contributing nothing" -- which is exactly what the leave-one-out
# audits of iterations 11 and 19 concluded about several tree variants. The claim
# under test is that that conclusion was an ARTEFACT OF THE PARAMETERISATION, not a
# fact about the members.
#
# The 1 added member here is a strictly worse model individually and is admitted as
# a control variate for the TREE FAMILY's shared bias, not as an opinion.
#
# -----------------------------------------------------------------------------
# MEMBERS, and why each was chosen (single-model OOF logloss, lower is better)
# -----------------------------------------------------------------------------
#   xgb_lw2bag    1.13682   RETAINED. Production member. Listwise xgboost, 10 seeds
#                           averaged. The tree end of the blend's only real axis of
#                           disagreement (iter 19: 93% of error variance in one
#                           component, tree vs logit).
#   lcmnl3_both   1.13863   RETAINED. Production member. Latent-class MNL with task
#                           position. The logit end of that same axis.
#   xgb_long      1.15516   ADDED, and it is WORSE than both production members by
#                           0.018-0.019 -- roughly 6 model-level seed sd. Under the
#                           simplex it has earned weight ~0 since iteration 11 and
#                           was read as redundant. It is admitted here for the
#                           opposite reason: it is a tree model trained on the LONG
#                           (per-alternative) representation, so it shares
#                           xgb_lw2bag's inductive bias while being a demonstrably
#                           weaker predictor. That is the signature of a control
#                           variate -- highly correlated errors, less signal.
#
# A reader should note the ordering: the added member is worse. If the free pool
# improves the blend, it cannot be because a better opinion was added.
#
# -----------------------------------------------------------------------------
# DECISION RULE, pre-registered. Adopt ONLY IF ALL FOUR HOLD:
# -----------------------------------------------------------------------------
#   (1) nested blend OOF improves over the production 2-member simplex blend by
#       MORE THAN the blend-level seed sd of 0.00048;
#   (2) the paired respondent-clustered z on nested held-out losses is >= 2;
#   (3) the gain is positive in at least 4 of the 5 folds
#       (CLAUDE.md leak signature: a real gain appears in every fold, a leak
#        concentrates in one);
#   (4) the segment-reweighted nested OOF does not regress
#       (per MEMORY: segment-reweighted OOF tracks the leaderboard to 0.002 while
#        plain nested OOF is off by 0.063).
# Failing ANY of the four means do not adopt. Additionally, and NOT part of the
# adopt/reject rule but reported as evidence: the per-fold fitted coefficients.
# An unstable negative weight -- one that flips sign or swings wildly across folds
# -- is a red flag that the negative coefficient is fitting fold-specific noise
# rather than a structural shared bias, and should veto adoption on judgement even
# if the four criteria pass.
#
# EXPECTED nested OOF: 1.12521 (recomputed by the main agent from artifacts).
# If this script produces something materially different, THAT IS THE FINDING and
# it must be reported, not quietly accepted.
#
# -----------------------------------------------------------------------------
# WHAT THE DECISION NUMBER IS
# -----------------------------------------------------------------------------
# The nested blend OOF: for each of the 5 respondent-grouped folds, fit the combiner
# on the other four folds and score the held-out fold. Never a plain OOF, never a
# single fold. The combiner is a fitted quantity, so it must be nested (CLAUDE.md
# rule 7). oof_blend_freepool3.rds below therefore stores the NESTED held-out
# predictions, not the full-data ones.
#
# -----------------------------------------------------------------------------
# OUTPUTS (written as the LAST act of the script, see the assembly block)
# -----------------------------------------------------------------------------
#   model/artifacts/oof_blend_freepool3.rds    nested held-out preds, 21565 x (No,p1..p4)
#   model/artifacts/test_blend_freepool3.rds   full-data-fit test preds, 4997 rows
#   model/artifacts/blend_freepool3.rds        the fitted combiner payload
#   submissions/sub_iterfreepool3.csv          No,Ch1,Ch2,Ch3,Ch4
# It does NOT edit model/members.txt and does NOT overwrite blend.rds/test_blend.rds.
#
# Re-runnable from a clean checkout: all randomness inside the optimiser is seeded
# (06_blend.R fit_free sets seed 42), folds.rds is read-only, no state carries over.
# =============================================================================

suppressMessages(library(data.table))
source("model/99_utils.R")

stopifnot(file.exists("model/06_blend.R"))
dir.create("experiments/iter34_freepool3", showWarnings = FALSE, recursive = TRUE)
dir.create("submissions", showWarnings = FALSE)

MEMB_FREE <- c("xgb_lw2bag", "lcmnl3_both", "xgb_long")
MEMB_PROD <- c("xgb_lw2bag", "lcmnl3_both")

# ---------------------------------------------------------------------------
# Run 06_blend.R in a private environment. We reuse ITS optimiser and ITS blend()
# rather than reimplementing them, so there is exactly one implementation of the
# combiner in the repo and this experiment cannot silently diverge from production.
# BLEND_OUT is mandatory here -- omitting it would write to blend.rds/test_blend.rds.
# ---------------------------------------------------------------------------
run_blend <- function(members, mode, out) {
  stopifnot(nzchar(out), mode %in% c("simplex", "free"))
  Sys.setenv(BLEND_WEIGHTS = mode,
             BLEND_MEMBERS = paste(members, collapse = " "),
             BLEND_OUT     = out)
  e <- new.env(parent = globalenv())
  cat("\n================ 06_blend.R [", mode, "] ", paste(members, collapse = " + "),
      " -> ", out, "================\n", sep = "")
  source("model/06_blend.R", local = e)
  Sys.unsetenv(c("BLEND_WEIGHTS", "BLEND_MEMBERS", "BLEND_OUT"))
  e
}

# Nested: refit the combiner on 4 folds, predict the 5th. Uses the sourced
# environment's own fit_w/obj/blend, so mode and member set come along for free.
nested_pred <- function(e, tag) {
  P  <- matrix(NA_real_, length(e$y), 4)
  ll <- numeric(5)
  co <- vector("list", 5)
  for (k in 1:5) {
    o <- e$fit_w(e$fmap != k)
    rows <- e$fmap == k
    P[rows, ] <- e$blend(o$par, e$OOF, rows)
    ll[k] <- e$obj(o$par, rows)
    co[[k]] <- o$par
    cat(sprintf("  [%s] fold %d  held-out %.5f\n", tag, k, ll[k]))
  }
  stopifnot(!anyNA(P), all(abs(rowSums(P) - 1) < 1e-8))
  list(P = P, ll = ll, coefs = co)
}

coef_of <- function(e, par) {
  if (e$MODE == "free") { par[1:e$M] } else { w <- exp(par[1:e$M]); w / sum(w) }
}

# ---------------------------------------------------------------------------
cat("\n### ITERATION 34 -- freepool3 ###\n")
cat("run at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")

long  <- readRDS("model/artifacts/long.rds")
wide  <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y, Case, segmentind)]); setorder(ymap, No)
y     <- ymap$y
stopifnot(nrow(ymap) == 21565)

e_free <- run_blend(MEMB_FREE, "free",    "blend_freepool3")
e_prod <- run_blend(MEMB_PROD, "simplex", "blend_prod2ref")
stopifnot(identical(e_free$y, y), identical(e_prod$y, y))

cat("\n---- nested refits (decision number) ----\n")
nf <- nested_pred(e_free, "free3")
np <- nested_pred(e_prod, "prod2")

nested_free <- mean(nf$ll)
nested_prod <- mean(np$ll)

# ---------------------------------------------------------------------------
cat("\n================ RESULTS ================\n")
cat(sprintf("NESTED blend OOF  production 2-member simplex : %.5f\n", nested_prod))
cat(sprintf("NESTED blend OOF  free-sign 3-member pool     : %.5f\n", nested_free))
cat(sprintf("improvement                                   : %+.5f  (blend seed sd 0.00048)\n",
            nested_prod - nested_free))

cat("\nper-fold held-out logloss:\n")
cat(sprintf("  fold %d   prod2 %.5f   free3 %.5f   delta %+.5f  %s\n",
            1:5, np$ll, nf$ll, np$ll - nf$ll,
            ifelse(np$ll - nf$ll > 0, "free3 better", "PROD BETTER")), sep = "")
folds_pos <- sum(np$ll - nf$ll > 0)
cat(sprintf("  folds where free3 wins: %d of 5\n", folds_pos))

cat("\nper-fold fitted FREE coefficients (stability evidence):\n")
cat(sprintf("  %-14s", "fold"), sprintf(" %10s", MEMB_FREE), "     eps\n")
for (k in 1:5) {
  cf <- coef_of(e_free, nf$coefs[[k]])
  cat(sprintf("  fold %-9d", k), sprintf(" %+10.4f", cf),
      sprintf("  %8.5f\n", plogis(nf$coefs[[k]][e_free$M + 1]) * 0.10))
}
cm <- sapply(1:5, function(k) coef_of(e_free, nf$coefs[[k]]))
cat(sprintf("  %-14s", "mean"), sprintf(" %+10.4f", rowMeans(cm)), "\n")
cat(sprintf("  %-14s", "sd"),   sprintf(" %10.4f", apply(cm, 1, sd)), "\n")
cat(sprintf("  %-14s", "sign consistent"),
    sprintf(" %10s", ifelse(apply(cm, 1, function(v) all(v > 0) || all(v < 0)), "YES", "NO")), "\n")

bl_free <- readRDS("model/artifacts/blend_freepool3.rds")
cat("\nfull-data FREE coefficients:\n")
cat(sprintf("  %-14s %+8.4f\n", MEMB_FREE, bl_free$par[1:length(MEMB_FREE)]), sep = "")
cat(sprintf("  %-14s %8.5f\n", "eps", plogis(bl_free$par[length(MEMB_FREE) + 1]) * 0.10))

cat("\nsingle-model OOF of each member (added member should be WORSE):\n")
for (m in union(MEMB_FREE, MEMB_PROD)) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  cat(sprintf("  %-14s %.5f\n", m, logloss(y, as.matrix(d[, .(p1, p2, p3, p4)]))))
}

# ---- paired respondent-clustered z on NESTED held-out losses ---------------
li_f <- -log(pmax(nf$P[cbind(seq_along(y), y)], 1e-15))
li_p <- -log(pmax(np$P[cbind(seq_along(y), y)], 1e-15))
dv   <- li_p - li_f                                   # positive => free3 better
cl   <- data.table(Case = ymap$Case, d = dv)[, .(dm = mean(d)), by = Case]
est  <- mean(cl$dm); se <- sd(cl$dm) / sqrt(nrow(cl)); zv <- est / se
cat(sprintf("\npaired vs production 2-member (nested, respondent-clustered):\n"))
cat(sprintf("  improvement %+.5f   SE %.5f   z = %.2f   95%% CI [%+.5f, %+.5f]  (n = %d respondents)\n",
            est, se, zv, est - 1.96 * se, est + 1.96 * se, nrow(cl)))

# ---- segment-reweighted nested OOF (predict_lb.R construction) -------------
# weight = test share / train share by segmentind; ZERO for Small Car (0% of test).
resp <- unique(wide[, .(Case, is_test, segmentind)])
seg  <- merge(resp[is_test == FALSE, .(ptr = .N / sum(!resp$is_test)), by = segmentind],
              resp[is_test == TRUE,  .(pte = .N / sum(resp$is_test)),  by = segmentind],
              by = "segmentind", all.x = TRUE)
seg[is.na(pte), pte := 0]; seg[, wt := pte / ptr]
w_seg <- merge(ymap[, .(No, segmentind)], seg[, .(segmentind, wt)], by = "segmentind")[order(No), wt]
stopifnot(length(w_seg) == length(y))
segll <- function(li) sum(w_seg * li) / sum(w_seg)
ess_resp <- function(wv, cs) { u <- !duplicated(cs); w <- wv[u]; round(sum(w)^2 / sum(w^2)) }
cat(sprintf("\nSEGMENT-reweighted NESTED OOF (leaderboard forecaster, ESS %d of %d respondents):\n",
            ess_resp(w_seg, ymap$Case), uniqueN(ymap$Case)))
cat(sprintf("  production 2-member : %.5f\n", segll(li_p)))
cat(sprintf("  free-sign 3-member  : %.5f   delta %+.5f  (%s)\n",
            segll(li_f), segll(li_p) - segll(li_f),
            if (segll(li_f) <= segll(li_p)) { "no regression" } else { "REGRESSION" }))

# ---- pre-registered decision rule, evaluated -------------------------------
c1 <- (nested_prod - nested_free) > 0.00048
c2 <- zv >= 2
c3 <- folds_pos >= 4
c4 <- segll(li_f) <= segll(li_p)
cat("\n---- DECISION RULE (pre-registered) ----\n")
cat(sprintf("  (1) nested gain > 0.00048          : %s  (%+.5f)\n", ifelse(c1, "PASS", "FAIL"), nested_prod - nested_free))
cat(sprintf("  (2) clustered z >= 2               : %s  (z = %.2f)\n", ifelse(c2, "PASS", "FAIL"), zv))
cat(sprintf("  (3) positive in >= 4 of 5 folds    : %s  (%d of 5)\n", ifelse(c3, "PASS", "FAIL"), folds_pos))
cat(sprintf("  (4) segment-reweighted no regress  : %s  (%+.5f)\n", ifelse(c4, "PASS", "FAIL"), segll(li_p) - segll(li_f)))
cat(sprintf("  VERDICT: %s\n", ifelse(c1 && c2 && c3 && c4, "ADOPT (subject to coefficient-stability judgement)", "DO NOT ADOPT")))

# ===========================================================================
# ASSEMBLY -- the LAST act. Everything above is diagnostics; artifacts land here
# so a killed caller can never leave this experiment looking like a failure.
# ===========================================================================
cat("\n---- assembling artifacts ----\n")

oof_dt <- data.table(No = ymap$No, p1 = nf$P[, 1], p2 = nf$P[, 2], p3 = nf$P[, 3], p4 = nf$P[, 4])
setorder(oof_dt, No)
saveRDS(oof_dt, "model/artifacts/oof_blend_freepool3.rds")
cat("wrote model/artifacts/oof_blend_freepool3.rds  (NESTED held-out preds,", nrow(oof_dt), "rows )\n")

# test_blend_freepool3.rds was already written by 06_blend.R from the full-data fit.
Ptest <- readRDS("model/artifacts/test_blend_freepool3.rds")
Ptest <- clip_norm(Ptest)
nos <- sort(unique(long[is_test == TRUE, No]))
stopifnot(nrow(Ptest) == 4997, length(nos) == 4997, min(nos) == 21566, max(nos) == 26562)

sub <- data.table(No = nos, Ch1 = Ptest[, 1], Ch2 = Ptest[, 2], Ch3 = Ptest[, 3], Ch4 = Ptest[, 4])
csv <- "submissions/sub_iterfreepool3.csv"
fwrite(sub, csv)
cat("wrote", csv, "\n")

# ---- CSV VALIDATION (read back from disk, not from memory) ----------------
cat("\n---- CSV validation (read back from disk) ----\n")
chk <- fread(csv)
v <- list()
v$header   <- identical(names(chk), c("No", "Ch1", "Ch2", "Ch3", "Ch4"))
v$nrows    <- nrow(chk) == 4997
v$no_match <- identical(as.integer(chk$No), as.integer(nos))
Pm <- as.matrix(chk[, .(Ch1, Ch2, Ch3, Ch4)])
v$no_na    <- !anyNA(Pm)
v$rowsum1  <- all(abs(rowSums(Pm) - 1) < 1e-9)
v$in01     <- all(Pm > 0) && all(Pm < 1)
for (nm in names(v)) cat(sprintf("  %-10s %s\n", nm, ifelse(v[[nm]], "OK", "*** FAIL ***")))
cat(sprintf("  max |rowsum-1| = %.3e   min p = %.3e   max p = %.5f\n",
            max(abs(rowSums(Pm) - 1)), min(Pm), max(Pm)))
csv_valid <- all(unlist(v))
cat(sprintf("  CSV VALID: %s\n", csv_valid))
stopifnot(csv_valid)

cat(sprintf("\nMACHINE-READABLE\nnested_free3=%.5f\nnested_prod2=%.5f\ngain=%+.6f\nz=%.4f\nfolds_pos=%d\nseg_free3=%.5f\nseg_prod2=%.5f\ncsv=%s\n",
            nested_free, nested_prod, nested_prod - nested_free, zv, folds_pos,
            segll(li_f), segll(li_p), csv))
cat("\nOK -- iteration 34 complete\n")
