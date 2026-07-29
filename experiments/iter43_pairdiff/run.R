# =============================================================================
# ITERATION 43 -- PAIRED-SUBMISSION DIFFERENCING
#
# Everything in this header -- hypothesis, gates, thresholds, anchors, tolerance
# rulings -- is written BEFORE any number produced by this script is looked at.
# Nothing below is tuned after the fact.
#
# THIS ITERATION IS OFFLINE. It uploads NOTHING. It builds one candidate
# submission file and one frozen inversion table; spending the Kaggle slots is
# the user's decision and belongs to a later, separate act.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# Iteration 42's dedicated probe slot is unnecessary, because the measurement it
# buys is obtainable as a BY-PRODUCT of two submissions that are both real
# candidates.
#
# A constant per-segment shift on the alternative-4 LOGIT leaves the within-buy
# 3-simplex untouched to machine precision (iteration 42's GATE 6 asserts this;
# SECTION 2 here re-asserts it against the shipped artifact rather than against
# a rebuild). Therefore if two submissions differ ONLY by such a shift, the
# difference of their returned public scores is exactly the binary none-margin
# difference, which is AFFINE in (r_lux, r_non). Combined with probe 1's
# identity r* = f*r_lux + (1-f)*r_non, one score difference identifies r_non.
#
# With file1 = the existing uniform w=0.85 file (du = +0.15311, the shipped
# model/artifacts/test_blend_freepool5_cal.rds) and file2 = the story-A 2-moment
# file built here (dl ~ -0.00301, dn ~ +0.41831):
#     E[s2 - s1] = -0.13136 * (r_non - 0.3933) - 0.00456
#     inversion:  r_non_hat = 0.3933 - (Delta_s + 0.00456) / 0.13136
# Rounding sd of a difference of two 3-dp scores is 0.000408, so
# sd(r_non_hat) ~ 0.0031 -- 3.9x worse than the dedicated probe's 0.0008, but
# 22x smaller than the 0.069 public->private respondent-drift sd that actually
# dominates the decision. The precision the probe buys is on a channel that is
# not binding.
#
# Three consequences the probe design forfeits:
#  (a) Kaggle scores a team's BEST PUBLIC submission on the private board, so
#      uploading both files makes the ADOPT/ABSTAIN decision automatically and
#      with certainty on the public rows -- no gates, no sigma, no trough. It
#      also genuinely truncates the downside: file2 reaches the private board
#      only if it actually beat file1 publicly.
#  (b) Slot 1 is free5cal85, which submissions/log.md records as built 27 Jul,
#      NEVER uploaded, with a pre-registered forecast (1.187; <=1.192 confirms
#      the free-sign base transfers, >=1.195 means abandon it for the 2-member
#      base). That upload was going to happen anyway, so the marginal cost of
#      the measurement is ONE slot -- the same as the probe -- but spent on a
#      file that can win.
#  (c) r_non_hat is exact enough to build the true 2-moment optimum for a later
#      slot if story A turns out to be the wrong point on the curve.
#
# SECONDARY HYPOTHESIS, free from the same two numbers: the free5 pool's segment
# SHAPE is wrong in the direction the members already indicate. Every tree
# member's raw test luxury p4 (0.259-0.278) implies r_non 0.24-0.28;
# lcmnl3_both (0.190) implies 0.436; the pool (0.210) implies 0.392. NOTHING in
# the family lands in iteration 42's ABSTAIN trough [0.305, 0.360] -- which is
# exactly the pool's own uniform-shift implication (0.33266). Delta_s therefore
# also arbitrates the tree-vs-logit disagreement ON THE GRADED POPULATION.
#
# NO FITTING OCCURS. No seed over model parameters, no optimiser over anything
# but 1-D uniroot calls on monotone moment equations whose right-hand sides are
# MEASURED. There is therefore no selection event charged against the
# replication budget. (set.seed(43) is used for the Monte-Carlo drift integral
# in SECTION 5 only; it integrates a stated distribution, it does not choose.)
#
# =============================================================================
# DECISION RULES -- pre-registered, verbatim
# =============================================================================
#
# GATE A (offline, BLOCKING) -- construction correctness. Reproduce to 1e-5:
#     f = 0.688213
#     raw free5 test p4: all 0.23766 | lux 0.20960 | non 0.29960
#     du(w=0.85) = +0.15311
#     logit(test_blend_freepool5_cal.rds[,4]) - logit(test_blend_freepool5.rds[,4])
#       is CONSTANT at 0.15311 with sd < 1e-12
#   The constancy assertion is the one iteration 42 never made against the
#   shipped artifact, and it is precisely what makes the differencing valid.
#   Any mismatch: STOP.
#
# GATE B (offline, BLOCKING) -- file2 validity. test_blend_freepool5_segA.rds:
#     row sums 1 within 1e-12; max p < 0.95; min p > 1e-3;
#     within-buy shares p_k/(1-p4) identical to test_blend_freepool5.rds
#       within 1e-12;
#     overall mean p4 EQUAL to file1's overall mean p4 within 1e-9
#       (this is what makes Delta_s a pure segment-SHAPE contrast);
#     per-segment mean p4 = the solved targets t_lux / t_non within 1e-9.
#   Any failure: STOP, do not write, do not upload.
#
#   TOLERANCE RULING, made before running and for a purely arithmetic reason.
#   The brief quotes the GATE B targets as literals to 4-5 significant decimals
#   (0.26219 overall, 0.2092 luxury, 0.3792 non-luxury) while asking for 1e-9 /
#   1e-6 agreement. That is unattainable in principle: a 4-dp literal carries
#   +-5e-5 of its own rounding. The gate is therefore implemented in two tiers:
#     TIER 1 (BLOCKING): agreement with the INTERNALLY SOLVED targets, and
#            file2-vs-file1 overall-mean equality, at 1e-9 -- these are exact
#            identities and must hold.
#     TIER 2 (REPORTED, non-blocking): agreement with the quoted literals at
#            5e-4 (luxury/non-luxury, quoted to 4 dp) and 5e-6 (overall, quoted
#            to 5 dp), i.e. each literal's own rounding half-width.
#   No number is relaxed after being seen; the tiering follows from the number
#   of digits in the brief.
#
# GATE C (offline, BLOCKING) -- inversion table frozen. invert.csv over
#   Delta_s in [-0.030, +0.030] step 0.0005, columns Delta_s, r_non, r_lux,
#   G_file1, G_file2, G_exactopt, gain_of_exactopt_over_better_of_the_two.
#   Cross-check three anchors:
#       Delta_s(r_non = 0.3171) = +0.00545
#       Delta_s(r_non = 0.3325) = +0.00343
#       Delta_s(r_non = 0.3933) = -0.00456
#   TOLERANCE RULING, again made before running: the anchors were recomputed by
#   the reviewer at a slightly different r_non lookup precision than this script
#   builds at -- iteration 42's own log shows dl = -0.00297 where the brief
#   quotes -0.00301, a 4e-5 discrepancy from exactly that cause. So:
#     TIER 1 (BLOCKING): all three anchors within 1e-4.
#     TIER 2 (REPORTED, non-blocking): all three within 1e-5.
#   Independently BLOCKING, and not subject to any tolerance judgement:
#     Delta_s must be EXACTLY affine in r_non -- max residual of
#     lm(Delta_s ~ r_non) < 1e-9 -- and the fitted slope within 1e-4 of
#     -0.13136. A non-linearity would mean the within-buy shares are not
#     identical and the whole method is void; this doubles as an independent
#     check on GATE B.
#
# SECTION 6 unit tests (offline). invert_pair() must return
#     Delta_s = -0.010 -> ADOPT-high-exactopt
#     Delta_s = +0.003 -> NO-THIRD-SLOT
#     Delta_s = +0.011 -> ADOPT-low-exactopt
#   and must write nothing under dry_run = TRUE. A unit-test failure does NOT
#   block writing file2 (file2's validity is GATE B's business and does not
#   depend on invert_pair), but it DOES block any deferred use of invert_pair
#   and is printed as a loud FAILED banner.
#
# --- everything below fires only if the user authorises slots ----------------
#
# SLOT 1: upload sub_20260727_2200_free5cal85.csv (already built, unchanged).
#   Resolves the standing pre-registration in submissions/log.md.
#   <=1.192 confirms the free-sign base transfers; >=1.195 means the OOF->test
#   regime drift ate it and remaining slots go to the 2-member base. Record s1
#   to 3 dp.
#   IF s1 >= 1.195: STOP THE WHOLE LINE. Do not upload file2 -- a segment
#   refinement on a base that did not transfer spends a slot on a losing
#   artifact. Fall back to the 2-member base and write the report.
#
# SLOT 2 (only if s1 <= 1.194): upload sub_<UTC>_free5segA.csv. Record s2 to
#   3 dp. Delta_s = s2 - s1.
#
# GATE D -- feasibility.
#   r_non_hat = 0.3933 - (Delta_s + 0.00456)/0.13136
#   r_lux_hat = (r* - (1-f)*r_non_hat)/f
#   Require r_non_hat in [0.15, 0.60] AND r_lux_hat in [0.05, 0.45]. Outside:
#   the two probes contradict each other. Report the contradiction, ship
#   whichever file Kaggle selected, stop.
#
# GATE E -- the ADOPT decision is made by KAGGLE, not by us. Delta_s < 0 =>
#   file2 is the new public leader and is auto-selected; Delta_s > 0 => file1
#   stays selected. Take no action either way. Do NOT re-rank the two files by
#   any local criterion.
#
# GATE F -- is a THIRD slot warranted? Only if ALL of:
#   (i)   |r_non_hat - 0.3933| >= 0.030;
#   (ii)  G_exactopt(r_non_hat) exceeds G of whichever file Kaggle selected by
#         >= 0.00100 under BOTH w = 0.85 and w = 1.00;
#   (iii) that margin survives r_non_hat +- 0.069 (the MEASURED public->private
#         respondent drift sd, NOT 0.0028) in at least one direction, with
#         P(negative private delta) stated explicitly.
#   If (i)-(iii) hold, build test_blend_freepool5_exactopt.rds and upload.
#   Otherwise STOP: the remaining slots buy nothing and the report is binding.
#
# MANDATORY RISK STATEMENT accompanying any ADOPT -- this is the CORRECTION to
# iteration 42's caveat 4, which must never be repeated. The shift is fitted on
# PUBLIC labels; the private half is a different ~79 respondents, of whom only
# ~25 are non-luxury. Respondent-level none-rate sd is 0.2858 (non-luxury) /
# 0.2142 (luxury), giving sd(r_non_priv - r_non_pub) ~ 0.063-0.069. The
# expectation of the private gain is UNBIASED, because logloss is affine in the
# true rate -- but the DOWNSIDE IS NOT ZERO and must never again be reported as
# zero. SECTION 5 tabulates E[G], sd, P(G<0) and the 5th/95th percentiles for
# every candidate point, and every ADOPT recommendation printed by this script
# carries its P(G<0) alongside its G.
#
# HARD CONSTRAINTS OBSERVED: does not touch members.txt, blend.rds,
# test_blend.rds, folds*.rds, 06_blend.R, quarantine/,
# test_blend_freepool5_cal.rds, probe_alt4.csv, probe_seg.csv. Does not
# regenerate any oof_*.rds or test_*.rds MEMBER artifact. New artifact names are
# checked absent before writing. Writes its artifacts as its LAST act. Every
# top-level `else` is braced.
# =============================================================================

suppressMessages(library(data.table))
source("model/99_utils.R")

DIR <- "experiments/iter43_pairdiff"
dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")
ck   <- function(ok) if (isTRUE(ok)) { "PASS" } else { "FAIL" }

# ============================================================== SECTION 1 =====
rule("SECTION 1 -- DATA, MASKS, CONSTANTS")

long <- readRDS("model/artifacts/long.rds")
tr <- unique(long[is_test == FALSE, .(No, y, Case, segmentind)]); setorder(tr, No)
te <- unique(long[is_test == TRUE,  .(No,    Case, segmentind)]); setorder(te, No)
stopifnot(nrow(tr) == 21565L, nrow(te) == 4997L)
stopifnot(identical(te$No, 21566:26562))

lux_tr <- tr$segmentind %in% c(3, 5)
lux_te <- te$segmentind %in% c(3, 5)

R_STAR     <- (1.7918 - 1.499) / 1.0986        # 0.266521 -- probe 1, rounded constants
R_STAR_EX  <- (log(6) - 1.499) / log(3)        # 0.266481 -- exact constants
f <- mean(lux_te)

cat(sprintf("  train tasks %d   test tasks %d   (No %d..%d ascending)\n",
            nrow(tr), nrow(te), min(te$No), max(te$No)))
cat(sprintf("  TEST luxury row share f = %.6f   (%d rows, %d of %d respondents)\n",
            f, sum(lux_te), uniqueN(te$Case[lux_te]), uniqueN(te$Case)))
cat(sprintf("  r*  (rounded constants) = %.6f\n", R_STAR))
cat(sprintf("  r*  (exact log6/log3)   = %.6f    difference %+.6f\n",
            R_STAR_EX, R_STAR_EX - R_STAR))

# ============================================================== SECTION 2 =====
rule("SECTION 2 -- GATE A: CONSTRUCTION CORRECTNESS")

Pt <- as.matrix(readRDS("model/artifacts/test_blend_freepool5.rds"))     # raw pool
P1 <- as.matrix(readRDS("model/artifacts/test_blend_freepool5_cal.rds")) # file1, w=0.85
stopifnot(dim(Pt) == c(4997L, 4L), dim(P1) == c(4997L, 4L))
Pt <- pmax(Pt, 1e-12); Pt <- Pt / rowSums(Pt)
P1 <- pmax(P1, 1e-12); P1 <- P1 / rowSums(P1)

m_all <- mean(Pt[, 4]); m_lux <- mean(Pt[lux_te, 4]); m_non <- mean(Pt[!lux_te, 4])
cat(sprintf("  raw free5 test p4:  all %.5f | lux %.5f | non %.5f\n", m_all, m_lux, m_non))

lgt <- function(p) log(p / (1 - p))
dvec <- lgt(P1[, 4]) - lgt(Pt[, 4])
du_shipped <- mean(dvec); du_sd <- sd(dvec)
cat(sprintf("  logit(file1 p4) - logit(raw p4):  mean %+.9f   sd %.3e   range [%+.9f, %+.9f]\n",
            du_shipped, du_sd, min(dvec), max(dvec)))

# closed-form alt-4 logit shift; identical to a softmax shift on the alt-4 logit
p4shift <- function(p4, d) { a <- p4 * exp(d); a / (1 - p4 + a) }
shift_mat <- function(P, dl, dn) {
  d  <- ifelse(lux_te, dl, dn)
  p4 <- P[, 4]; p4n <- p4shift(p4, d)
  Q <- P
  Q[, 1:3] <- P[, 1:3] * ((1 - p4n) / (1 - p4))
  Q[, 4]   <- p4n
  Q
}
solve_seg <- function(mask, target, P = Pt)
  uniroot(function(d) mean(p4shift(P[mask, 4], d)) - target, c(-8, 8), tol = 1e-12)$root
solve_uni <- function(target, P = Pt)
  uniroot(function(d) mean(p4shift(P[, 4], d)) - target, c(-8, 8), tol = 1e-12)$root

du85 <- solve_uni(m_all + 0.85 * (R_STAR - m_all))
cat(sprintf("  du(w=0.85) solved here = %+.7f   (ships overall p4 %.6f)\n",
            du85, mean(p4shift(Pt[, 4], du85))))

gA <- c(
  f_share      = abs(f - 0.688213) < 1e-5,
  raw_all      = abs(m_all - 0.23766) < 1e-5,
  raw_lux      = abs(m_lux - 0.20960) < 1e-5,
  raw_non      = abs(m_non - 0.29960) < 1e-5,
  du_value     = abs(du85 - 0.15311) < 1e-5,
  du_constant  = du_sd < 1e-12,
  du_matches   = abs(du_shipped - du85) < 1e-9,
  file1_recon  = max(abs(shift_mat(Pt, du85, du85) - P1)) < 1e-9
)
cat("\n  GATE A assertions\n")
for (nm in names(gA)) cat(sprintf("    %-12s %s\n", nm, ck(gA[[nm]])))
if (!all(gA)) stop("GATE A FAILED: ", paste(names(gA)[!gA], collapse = ", "))
cat("  GATE A: PASS (all 8)\n")

# ============================================================== SECTION 3 =====
rule("SECTION 3 -- BUILD FILE2 (story-A 2-moment) AND GATE B")

w <- 0.85
r_non_A <- 0.3933
r_lux_A <- (R_STAR - (1 - f) * r_non_A) / f
t_lux <- m_lux + w * (r_lux_A - m_lux)
t_non <- m_non + w * (r_non_A - m_non)
dl <- solve_seg(lux_te, t_lux)
dn <- solve_seg(!lux_te, t_non)
P2 <- shift_mat(Pt, dl, dn)

cat(sprintf("  story A: r_non %.6f -> r_lux %.6f (via r* identity)\n", r_non_A, r_lux_A))
cat(sprintf("  shrinkage w = %.2f  ->  targets  t_lux %.6f   t_non %.6f\n", w, t_lux, t_non))
cat(sprintf("  solved shifts:  dl = %+.6f   dn = %+.6f\n", dl, dn))
cat(sprintf("  file2 ships p4: all %.6f | lux %.6f | non %.6f\n",
            mean(P2[, 4]), mean(P2[lux_te, 4]), mean(P2[!lux_te, 4])))
cat(sprintf("  file1 ships p4: all %.6f | lux %.6f | non %.6f\n",
            mean(P1[, 4]), mean(P1[lux_te, 4]), mean(P1[!lux_te, 4])))

wb1 <- Pt[, 1:3] / rowSums(Pt[, 1:3])
wb2 <- P2[, 1:3] / rowSums(P2[, 1:3])
gB1 <- c(
  rowsum      = max(abs(rowSums(P2) - 1)) < 1e-12,
  maxp        = max(P2) < 0.95,
  minp        = min(P2) > 1e-3,
  withinbuy   = max(abs(wb1 - wb2)) < 1e-12,
  mean_eq_f1  = abs(mean(P2[, 4]) - mean(P1[, 4])) < 1e-9,
  lux_target  = abs(mean(P2[lux_te, 4])  - t_lux) < 1e-9,
  non_target  = abs(mean(P2[!lux_te, 4]) - t_non) < 1e-9
)
gB2 <- c(
  lit_all = abs(mean(P2[, 4])          - 0.26219) < 5e-6,
  lit_lux = abs(mean(P2[lux_te, 4])    - 0.2092)  < 5e-4,
  lit_non = abs(mean(P2[!lux_te, 4])   - 0.3792)  < 5e-4,
  lit_dl  = abs(dl - (-0.00301)) < 5e-5,
  lit_dn  = abs(dn - ( 0.41831)) < 5e-5
)
cat("\n  GATE B TIER 1 (blocking, exact identities)\n")
for (nm in names(gB1)) cat(sprintf("    %-12s %s\n", nm, ck(gB1[[nm]])))
cat("  GATE B TIER 2 (reported, literals at their own rounding half-width)\n")
for (nm in names(gB2)) cat(sprintf("    %-12s %s\n", nm, ck(gB2[[nm]])))
if (!all(gB1)) stop("GATE B TIER 1 FAILED: ", paste(names(gB1)[!gB1], collapse = ", "))
cat("  GATE B TIER 1: PASS (all 7)\n")
cat(sprintf("  GATE B TIER 2: %d of %d literals matched\n", sum(gB2), length(gB2)))

# ============================================================== SECTION 4 =====
rule("SECTION 4 -- GATE C: THE FROZEN INVERSION TABLE")

mBCE <- function(q, rvec) -mean(rvec * log(q) + (1 - rvec) * log(1 - q))
r_lux_of <- function(r_non, rstar = R_STAR) (rstar - (1 - f) * r_non) / f
rvec_of  <- function(r_non, rstar = R_STAR) ifelse(lux_te, r_lux_of(r_non, rstar), r_non)

q1 <- P1[, 4]; q2 <- P2[, 4]
Delta_s_of <- function(r_non, rstar = R_STAR) {
  rv <- rvec_of(r_non, rstar); mBCE(q2, rv) - mBCE(q1, rv)
}

# linearity check
grid_r <- seq(0.20, 0.55, by = 0.0025)
ds_grid <- vapply(grid_r, Delta_s_of, 0)
fit <- lm(ds_grid ~ grid_r)
slope <- unname(coef(fit)[2]); maxres <- max(abs(residuals(fit)))
cat(sprintf("  lm(Delta_s ~ r_non):  slope %+.7f   max |residual| %.3e\n", slope, maxres))
lin <- c(linear = maxres < 1e-9, slope = abs(slope - (-0.13136)) < 1e-4)
for (nm in names(lin)) cat(sprintf("    %-8s %s\n", nm, ck(lin[[nm]])))
if (!all(lin)) stop("LINEARITY FAILED: ", paste(names(lin)[!lin], collapse = ", "),
                    " -- within-buy shares differ; the method is void.")

anchors <- c(0.3171, 0.3325, 0.3933)
expect  <- c(0.00545, 0.00343, -0.00456)
got     <- vapply(anchors, Delta_s_of, 0)
cat("\n  GATE C anchor cross-check\n")
for (i in seq_along(anchors))
  cat(sprintf("    Delta_s(r_non = %.4f) = %+.7f   expected %+.5f   miss %+.2e   %s / %s\n",
              anchors[i], got[i], expect[i], got[i] - expect[i],
              ck(abs(got[i] - expect[i]) < 1e-4), ck(abs(got[i] - expect[i]) < 1e-5)))
if (!all(abs(got - expect) < 1e-4)) stop("GATE C TIER 1 FAILED")
cat(sprintf("  GATE C TIER 1 (1e-4): PASS (3/3)   TIER 2 (1e-5): %d/3\n",
            sum(abs(got - expect) < 1e-5)))

# inversion coefficients, from the SCRIPT's own affine fit (not the header's)
INV_SLOPE <- slope
INV_R0    <- 0.3933
INV_D0    <- Delta_s_of(0.3933)
r_non_of_ds <- function(ds) INV_R0 + (ds - INV_D0) / INV_SLOPE
cat(sprintf("\n  inversion:  r_non_hat = %.4f + (Delta_s - (%+.7f)) / (%+.7f)\n",
            INV_R0, INV_D0, INV_SLOPE))
cat(sprintf("  sd(Delta_s) from two 3-dp scores = %.6f  ->  sd(r_non_hat) = %.5f\n",
            sqrt(2) * (0.001 / sqrt(12)), sqrt(2) * (0.001 / sqrt(12)) / abs(INV_SLOPE)))

# exact 2-moment optimum at a given r_non, at shrinkage w
exactopt_at <- function(r_non, ww) {
  rl <- r_lux_of(r_non)
  shift_mat(Pt, solve_seg(lux_te, m_lux + ww * (rl - m_lux)),
                solve_seg(!lux_te, m_non + ww * (r_non - m_non)))
}

ds_seq <- seq(-0.030, 0.030, by = 0.0005)
rows <- vector("list", length(ds_seq))
for (i in seq_along(ds_seq)) {
  ds <- ds_seq[i]; rn <- r_non_of_ds(ds); rl <- r_lux_of(rn)
  rv <- rvec_of(rn)
  b1 <- mBCE(q1, rv)
  G_file1 <- 0                       # file1 is the reference
  G_file2 <- b1 - mBCE(q2, rv)       # = -Delta_s by construction
  qe85  <- exactopt_at(rn, 0.85)[, 4]
  qe100 <- exactopt_at(rn, 1.00)[, 4]
  G_ex85  <- b1 - mBCE(qe85,  rv)
  G_ex100 <- b1 - mBCE(qe100, rv)
  better <- max(G_file1, G_file2)
  rows[[i]] <- data.table(
    Delta_s = ds, r_non = rn, r_lux = rl,
    G_file1 = G_file1, G_file2 = G_file2,
    G_exactopt = G_ex85, G_exactopt_w100 = G_ex100,
    gain_of_exactopt_over_better_of_the_two = G_ex85 - better,
    gain_w100 = G_ex100 - better,
    kaggle_selects = if (ds < 0) { "file2" } else { "file1" },
    mBCE_file1 = b1, mBCE_file2 = mBCE(q2, rv), mBCE_exactopt = mBCE(qe85, rv))
}
INV <- rbindlist(rows)
cat(sprintf("\n  built inversion table: %d rows, Delta_s %.4f..%.4f step 0.0005\n",
            nrow(INV), min(ds_seq), max(ds_seq)))
cat("\n  selected rows\n")
sel <- INV[vapply(c(-0.020, -0.010, -0.005, -0.002, 0, 0.003, 0.005, 0.011, 0.020),
                  function(z) which.min(abs(INV$Delta_s - z)), 1L)]
for (i in seq_len(nrow(sel)))
  cat(sprintf("    ds %+0.4f -> r_non %.4f r_lux %.4f | G_f2 %+.5f | G_exopt %+.5f | over-better %+.5f | kaggle picks %s\n",
              sel$Delta_s[i], sel$r_non[i], sel$r_lux[i], sel$G_file2[i],
              sel$G_exactopt[i], sel$gain_of_exactopt_over_better_of_the_two[i],
              sel$kaggle_selects[i]))

# where a THIRD slot would be warranted, in Delta_s terms (gates F(i)+F(ii) only)
INV[, third := (abs(r_non - INV_R0) >= 0.030) &
      (gain_of_exactopt_over_better_of_the_two >= 0.00100) & (gain_w100 >= 0.00100)]
rr3 <- rle(INV$third); e3 <- cumsum(rr3$lengths); s3 <- e3 - rr3$lengths + 1L
cat("\n  THIRD-SLOT bands over Delta_s (GATE F (i) and (ii) only; (iii) applied at lookup)\n")
for (j in seq_along(rr3$values))
  cat(sprintf("    %-13s Delta_s in [%+0.4f, %+0.4f]  ->  r_non [%.4f, %.4f]\n",
              if (rr3$values[j]) { "THIRD SLOT" } else { "no third slot" },
              INV$Delta_s[s3[j]], INV$Delta_s[e3[j]], INV$r_non[e3[j]], INV$r_non[s3[j]]))

# r* sensitivity
ds_ex <- vapply(anchors, function(r) Delta_s_of(r, R_STAR_EX), 0)
cat(sprintf("\n  r* SENSITIVITY (rounded %.6f vs exact %.6f, difference %+.6f)\n",
            R_STAR, R_STAR_EX, R_STAR_EX - R_STAR))
for (i in seq_along(anchors))
  cat(sprintf("    Delta_s(%.4f): %+.7f -> %+.7f   shift %+.2e  (=> r_non shift %+.5f)\n",
              anchors[i], got[i], ds_ex[i], ds_ex[i] - got[i],
              (ds_ex[i] - got[i]) / INV_SLOPE))
cat(sprintf("  max induced r_non shift %.5f -- %.0fx smaller than the 0.0031 rounding sd;\n",
            max(abs(ds_ex - got) / abs(INV_SLOPE)),
            0.0031 / max(abs(ds_ex - got) / abs(INV_SLOPE))))
cat("  it moves no band edge on a 0.0005 Delta_s grid.\n")

# ============================================================= SECTION 4b =====
# SECONDARY HYPOTHESIS. Diagnostic only -- feeds NO gate, changes NO artifact.
# Added to the script AFTER the first run, which produced only display changes
# and this read-only block; every gate, threshold and anchor above is unchanged
# from the pre-registered header and every one of them passed on the first run.
rule("SECTION 4b -- SECONDARY HYPOTHESIS: WHAT EACH MEMBER IMPLIES FOR r_non")

memb_files <- c(xgb_lw2bag = "test_xgb_lw2bag.rds", lcmnl3_both = "test_lcmnl3_both.rds",
                xgb_long = "test_xgb_long.rds", xgb_wide = "test_xgb_wide.rds",
                xgb_2stage = "test_xgb_2stage.rds", xgb_lw2fr = "test_xgb_lw2fr.rds")
cat("  Each member's RAW test luxury p4 is read as its estimate of r_lux; r_non then\n")
cat("  follows from probe 1's identity r_non = (r* - f*r_lux)/(1-f). Delta_s is what\n")
cat("  the paired upload would return if that member were right about the shape.\n\n")
cat(sprintf("    %-14s %9s %9s %9s %11s\n", "member", "lux p4", "-> r_non", "Delta_s", "kaggle picks"))
mrows <- list()
for (nm in names(memb_files)) {
  fp <- file.path("model/artifacts", memb_files[[nm]])
  if (!file.exists(fp)) next
  d <- readRDS(fp); setorder(d, No); Pm <- as.matrix(d[, .(p1, p2, p3, p4)])
  rl <- mean(Pm[lux_te, 4]); rn <- (R_STAR - f * rl) / (1 - f)
  ds <- Delta_s_of(rn)
  cat(sprintf("    %-14s %9.5f %9.4f %+9.5f %11s\n", nm, rl, rn, ds,
              if (ds < 0) { "file2" } else { "file1" }))
  mrows[[nm]] <- data.table(member = nm, lux_p4 = rl, implied_r_non = rn, implied_Delta_s = ds)
}
rl_pool <- m_lux; rn_pool <- (R_STAR - f * rl_pool) / (1 - f)
cat(sprintf("    %-14s %9.5f %9.4f %+9.5f %11s\n", "free5 POOL", rl_pool, rn_pool,
            Delta_s_of(rn_pool), if (Delta_s_of(rn_pool) < 0) { "file2" } else { "file1" }))
rn_unif <- uniroot(function(r) {
  rl <- r_lux_of(r)
  mean(p4shift(Pt[lux_te, 4], du85)) - (m_lux + 1.00 * (rl - m_lux))
}, c(0.05, 0.95), tol = 1e-12)$root
cat(sprintf("\n  the w=1.00 UNIFORM shift's own implication: r_non = %.5f, Delta_s = %+.5f\n",
            rn_unif, Delta_s_of(rn_unif)))
cat(sprintf("  iteration 42's ABSTAIN trough sits at r_non 0.3325 -> Delta_s %+.5f.\n",
            Delta_s_of(0.3325)))
MI <- rbindlist(mrows)
cat(sprintf("\n  tree members span r_non [%.4f, %.4f]; lcmnl3_both sits at %.4f.\n",
            min(MI$implied_r_non[MI$member != "lcmnl3_both"]),
            max(MI$implied_r_non[MI$member != "lcmnl3_both"]),
            MI$implied_r_non[MI$member == "lcmnl3_both"]))
cat("  If the trees are right, Delta_s > 0 and Kaggle keeps file1. If the latent-class\n")
cat("  model is right, Delta_s < 0 and Kaggle switches to file2. One number decides.\n")

# ============================================================== SECTION 5 =====
rule("SECTION 5 -- PUBLIC->PRIVATE DRIFT RISK (the correction to iter42 caveat 4)")

# per-respondent none-rate sd by segment, from TRAIN
rr <- tr[, .(nr = mean(y == 4), lux = segmentind[1] %in% c(3, 5)), by = Case]
s_lux <- sd(rr$nr[rr$lux]); s_non <- sd(rr$nr[!rr$lux])
n_lux_tr <- sum(rr$lux); n_non_tr <- sum(!rr$lux)
n_lux_te <- uniqueN(te$Case[lux_te]); n_non_te <- uniqueN(te$Case[!lux_te])
cat(sprintf("  respondent none-rate sd (train): luxury %.4f (n=%d) | non-luxury %.4f (n=%d) | all %.4f\n",
            s_lux, n_lux_tr, s_non, n_non_tr, sd(rr$nr)))
cat(sprintf("  test respondents: %d luxury, %d non-luxury\n", n_lux_te, n_non_te))

drift_sd <- function(s_seg, n_tot, pub_share) {
  n_pub <- round(pub_share * n_tot); n_priv <- n_tot - n_pub
  c(sd = s_seg * sqrt(1 / n_pub + 1 / n_priv), n_pub = n_pub, n_priv = n_priv)
}
splits <- c(0.5, 0.7)
for (p in splits) {
  a <- drift_sd(s_lux, n_lux_te, p); b <- drift_sd(s_non, n_non_te, p)
  cat(sprintf("  public share %.0f%%:  sd(r_lux_priv - r_lux_pub) = %.4f (n_pub %d, n_priv %d) | sd(r_non diff) = %.4f (n_pub %d, n_priv %d)\n",
              100 * p, a["sd"], a["n_pub"], a["n_priv"], b["sd"], b["n_pub"], b["n_priv"]))
}

# affine kernel of a candidate-vs-baseline margin gain, per segment
kern <- function(qc, qb) {
  A_l <- mean(log(qc[lux_te])      - log(qb[lux_te]))
  B_l <- mean(log(1 - qc[lux_te])  - log(1 - qb[lux_te]))
  A_n <- mean(log(qc[!lux_te])     - log(qb[!lux_te]))
  B_n <- mean(log(1 - qc[!lux_te]) - log(1 - qb[!lux_te]))
  list(A_l = A_l, B_l = B_l, A_n = A_n, B_n = B_n)
}
Gof <- function(K, rl, rn)
  f * (rl * K$A_l + (1 - rl) * K$B_l) + (1 - f) * (rn * K$A_n + (1 - rn) * K$B_n)

set.seed(43)
NDRAW <- 4000L
Zl <- rnorm(NDRAW); Zn <- rnorm(NDRAW)
mc <- function(K, rl_pub, rn_pub, sd_l, sd_n) {
  rl <- pmin(pmax(rl_pub + sd_l * Zl, 1e-4), 1 - 1e-4)
  rn <- pmin(pmax(rn_pub + sd_n * Zn, 1e-4), 1 - 1e-4)
  g <- Gof(K, rl, rn)
  c(EG = mean(g), sdG = sd(g), Pneg = mean(g < 0),
    p05 = unname(quantile(g, 0.05)), p95 = unname(quantile(g, 0.95)))
}

drows <- list()
for (i in seq_len(nrow(INV))) {
  rn <- INV$r_non[i]; rl <- INV$r_lux[i]
  qsel <- if (INV$kaggle_selects[i] == "file2") { q2 } else { q1 }
  Kf2 <- kern(q2, q1)                       # ship file2 instead of file1
  qe  <- exactopt_at(rn, 0.85)[, 4]
  Kex <- kern(qe, qsel)                     # ship exactopt instead of Kaggle's pick
  for (p in splits) {
    sdl <- drift_sd(s_lux, n_lux_te, p)["sd"]; sdn <- drift_sd(s_non, n_non_te, p)["sd"]
    a <- mc(Kf2, rl, rn, sdl, sdn); b <- mc(Kex, rl, rn, sdl, sdn)
    drows[[length(drows) + 1L]] <- data.table(
      Delta_s = INV$Delta_s[i], r_non_pub = rn, r_lux_pub = rl, public_share = p,
      candidate = "file2_vs_file1", EG = a["EG"], sdG = a["sdG"],
      P_G_neg = a["Pneg"], p05 = a["p05"], p95 = a["p95"])
    drows[[length(drows) + 1L]] <- data.table(
      Delta_s = INV$Delta_s[i], r_non_pub = rn, r_lux_pub = rl, public_share = p,
      candidate = "exactopt_vs_selected", EG = b["EG"], sdG = b["sdG"],
      P_G_neg = b["Pneg"], p05 = b["p05"], p95 = b["p95"])
  }
}
DR <- rbindlist(drows)

# the mandatory statement, at story A itself
KA <- kern(q2, q1)
sdl70 <- drift_sd(s_lux, n_lux_te, 0.7)["sd"]; sdn70 <- drift_sd(s_non, n_non_te, 0.7)["sd"]
sdl50 <- drift_sd(s_lux, n_lux_te, 0.5)["sd"]; sdn50 <- drift_sd(s_non, n_non_te, 0.5)["sd"]
mcA70 <- mc(KA, r_lux_A, r_non_A, sdl70, sdn70)
mcA50 <- mc(KA, r_lux_A, r_non_A, sdl50, sdn50)
cat("\n  ---- MANDATORY RISK STATEMENT (story-A ADOPT: ship file2 instead of file1) ----\n")
cat(sprintf("  deterministic public gain at r_non = %.4f: G = %+.5f\n", r_non_A, -Delta_s_of(r_non_A)))
cat(sprintf("  70/30 split:  E[G_private] %+.5f  sd %.5f  P(G<0) %.3f  5th %+.5f  95th %+.5f\n",
            mcA70["EG"], mcA70["sdG"], mcA70["Pneg"], mcA70["p05"], mcA70["p95"]))
cat(sprintf("  50/50 split:  E[G_private] %+.5f  sd %.5f  P(G<0) %.3f  5th %+.5f  95th %+.5f\n",
            mcA50["EG"], mcA50["sdG"], mcA50["Pneg"], mcA50["p05"], mcA50["p95"]))
cat("  The expectation is UNBIASED because logloss is affine in the true rate.\n")
cat("  The downside is NOT zero. Iteration 42's caveat 4 said it was; that was wrong.\n")

# ============================================================== SECTION 6 =====
rule("SECTION 6 -- DEFERRED INVERSION FUNCTION (defined, unit-tested, NOT run for real)")

DRIFT <- 0.069   # measured public->private r_non drift sd at a 70/30 split

invert_pair <- function(s1, s2, dry_run = TRUE) {
  ds <- s2 - s1
  i  <- which.min(abs(INV$Delta_s - ds))
  row <- INV[i]
  out <- list(s1 = s1, s2 = s2, Delta_s = ds, r_non = row$r_non, r_lux = row$r_lux,
              kaggle_selects = row$kaggle_selects, wrote = NA_character_)

  # GATE D -- feasibility
  if (!(row$r_non >= 0.15 && row$r_non <= 0.60 && row$r_lux >= 0.05 && row$r_lux <= 0.45)) {
    out$verdict <- "CONTRADICTION (GATE D)"; return(out)
  }
  # GATE E -- Kaggle has already made the ADOPT decision; we take no action.
  G_sel <- if (row$kaggle_selects == "file2") { row$G_file2 } else { row$G_file1 }
  out$G_selected <- G_sel

  # GATE F
  f_i  <- abs(row$r_non - INV_R0) >= 0.030
  m85  <- row$G_exactopt      - G_sel
  m100 <- row$G_exactopt_w100 - G_sel
  f_ii <- (m85 >= 0.00100) && (m100 >= 0.00100)

  qe   <- exactopt_at(row$r_non, 0.85)[, 4]
  qsel <- if (row$kaggle_selects == "file2") { q2 } else { q1 }
  Kex  <- kern(qe, qsel)
  surv <- vapply(c(-DRIFT, DRIFT), function(dd) {
    rn <- row$r_non + dd
    Gof(Kex, r_lux_of(rn), rn)
  }, 0)
  f_iii <- any(surv >= 0.00100)
  mcx <- mc(Kex, row$r_lux, row$r_non, sdl70, sdn70)
  out$margin_w85 <- m85; out$margin_w100 <- m100
  out$surv_minus <- surv[1]; out$surv_plus <- surv[2]
  out$P_G_neg <- unname(mcx["Pneg"]); out$EG <- unname(mcx["EG"]); out$sdG <- unname(mcx["sdG"])
  out$gateF <- c(i = f_i, ii = f_ii, iii = f_iii)

  if (!(f_i && f_ii && f_iii)) { out$verdict <- "NO-THIRD-SLOT"; return(out) }
  out$verdict <- if (row$r_non > INV_R0) { "ADOPT-high-exactopt" } else { "ADOPT-low-exactopt" }
  if (dry_run) { out$wrote <- "<dry run: write path stubbed>"; return(out) }
  Pe <- exactopt_at(row$r_non, 0.85)
  stopifnot(max(abs(rowSums(Pe) - 1)) < 1e-12, max(Pe) < 0.95, min(Pe) > 1e-3,
            max(abs(Pe[, 1:3] / rowSums(Pe[, 1:3]) - wb1)) < 1e-12)
  saveRDS(Pe, "model/artifacts/test_blend_freepool5_exactopt.rds")
  fn <- sprintf("submissions/sub_%s_free5exactopt.csv", format(Sys.time(), "%Y%m%d", tz = "UTC"))
  fwrite(data.table(No = te$No, Ch1 = Pe[, 1], Ch2 = Pe[, 2], Ch3 = Pe[, 3], Ch4 = Pe[, 4]), fn)
  out$wrote <- paste("model/artifacts/test_blend_freepool5_exactopt.rds", fn)
  out
}

utest <- list(a = invert_pair(1.190, 1.180), b = invert_pair(1.190, 1.193),
              c = invert_pair(1.190, 1.201))
want <- c("ADOPT-high-exactopt", "NO-THIRD-SLOT", "ADOPT-low-exactopt")
for (j in seq_along(utest)) {
  u <- utest[[j]]
  cat(sprintf("\n  unit test %d: Delta_s %+.4f -> r_non %.4f r_lux %.4f | kaggle picks %s\n",
              j, u$Delta_s, u$r_non, u$r_lux, u$kaggle_selects))
  cat(sprintf("    GATE F  (i) %s  (ii) %s [m85 %+.5f m100 %+.5f]  (iii) %s [-drift %+.5f, +drift %+.5f]\n",
              ck(u$gateF[["i"]]), ck(u$gateF[["ii"]]), u$margin_w85, u$margin_w100,
              ck(u$gateF[["iii"]]), u$surv_minus, u$surv_plus))
  cat(sprintf("    verdict %-22s (expected %-22s) %s | E[G] %+.5f sd %.5f P(G<0) %.3f | wrote: %s\n",
              u$verdict, want[j], ck(identical(u$verdict, want[j])),
              u$EG, u$sdG, u$P_G_neg, u$wrote))
}
ut_ok <- all(vapply(seq_along(utest), function(j) identical(utest[[j]]$verdict, want[j]), TRUE))
ut_nowrite <- !file.exists("model/artifacts/test_blend_freepool5_exactopt.rds")
cat(sprintf("\n  unit tests verdicts %s   dry_run wrote nothing %s\n", ck(ut_ok), ck(ut_nowrite)))
if (!(ut_ok && ut_nowrite)) {
  cat("\n  *********************************************************************\n")
  cat("  ** SECTION 6 UNIT TESTS FAILED. invert_pair() MUST NOT BE USED.     **\n")
  cat("  ** file2 remains valid (GATE B is independent) but the deferred     **\n")
  cat("  ** third-slot machinery is BROKEN and is hereby blocked.            **\n")
  cat("  *********************************************************************\n")
}

# ============================================================== SECTION 7 =====
rule("SECTION 7 -- WRITE ARTIFACTS (last act)")

# MANDATORY NAME CHECK
stopifnot(!file.exists("model/artifacts/test_blend_freepool5_segA.rds"))
stopifnot(!file.exists("model/artifacts/test_blend_freepool5_exactopt.rds"))
existing <- list.files("model/artifacts")
cat(sprintf("  name check: %d artifacts scanned; matches for 'segA' -> %d, 'exactopt' -> %d\n",
            length(existing), sum(grepl("segA", existing)), sum(grepl("exactopt", existing))))
stopifnot(sum(grepl("segA", existing)) == 0, sum(grepl("exactopt", existing)) == 0)

fwrite(INV, file.path(DIR, "invert.csv"))
fwrite(DR,  file.path(DIR, "drift_risk.csv"))
saveRDS(P2, "model/artifacts/test_blend_freepool5_segA.rds")
SUB <- sprintf("submissions/sub_%s_free5segA.csv", format(Sys.time(), "%Y%m%d", tz = "UTC"))
fwrite(data.table(No = te$No, Ch1 = P2[, 1], Ch2 = P2[, 2], Ch3 = P2[, 3], Ch4 = P2[, 4]), SUB)

chk <- fread(SUB)
post <- c(nrows = nrow(chk) == 4997L,
          header = identical(names(chk), c("No", "Ch1", "Ch2", "Ch3", "Ch4")),
          No_range = identical(chk$No, 21566:26562),
          rowsum = max(abs(rowSums(as.matrix(chk[, 2:5])) - 1)) < 1e-9)
cat("\n  post-write verification of the submission CSV\n")
for (nm in names(post)) cat(sprintf("    %-9s %s\n", nm, ck(post[[nm]])))
stopifnot(all(post))

cat(sprintf("\n  wrote %s\n", file.path(DIR, "invert.csv")))
cat(sprintf("  wrote %s\n", file.path(DIR, "drift_risk.csv")))
cat("  wrote model/artifacts/test_blend_freepool5_segA.rds\n")
cat(sprintf("  wrote %s\n", SUB))
cat("\n  THIS ITERATION UPLOADED NOTHING. No 'kaggle competitions submit' was run.\n")
cat("  submissions/probe_seg.csv and submissions/probe_alt4.csv are untouched.\n")
cat("  model/artifacts/test_blend_freepool5_cal.rds is untouched and remains file1.\n")
cat("  members.txt, blend.rds, test_blend.rds, folds*.rds, 06_blend.R: untouched.\n")

rule("OK iter43")
cat("OK iter43\n")
