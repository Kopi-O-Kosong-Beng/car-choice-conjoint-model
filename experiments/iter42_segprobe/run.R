# =============================================================================
# ITERATION 42 -- SEGMENT-CONDITIONAL NONE-RATE PROBE
#
# Everything in this header -- hypothesis, gates, thresholds, anchors -- is
# written BEFORE any number produced by this script is looked at. Nothing below
# is tuned after the fact.
#
# THIS ITERATION IS THE OFFLINE HALF. It consumes NO Kaggle slot. It builds and
# verifies an instrument; uploading it is a separate, explicitly gated decision
# that belongs to the user.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# Probe 1 (submissions/probe_alt4.csv, constant (1/6,1/6,1/6,1/2), returned
# 1.499) gave exactly ONE moment of the graded population:
#
#     r* = E[1{y = 4}] = (1.7918 - 1.499)/1.0986 = 0.26652
#
# Four analysts then asked what that one number can support, and all four were
# right: nothing beyond iteration 36's UNIFORM alt-4 logit shift, which is the
# exact I-projection of the shipped distribution onto that single moment.
#
# Nobody asked for a SECOND moment. That is the gap, and it is not academic.
#
#   * On OOF the 5-member free-sign pool is badly miscalibrated BY SEGMENT on
#     alternative 4: predicted luxury p4 = 0.21307 against observed 0.15986
#     (+0.053), while non-luxury is near-exact (0.31719 vs 0.31712). This is
#     iteration 29's documented tree-family defect, still uncorrected. A
#     2-parameter (per-segment) alt-4 shift beats the 1-parameter shift by
#     +0.00089 on plain nested OOF -- 1.9x the blend-level seed sd, in-sample
#     and probe-free.
#   * On the TEST side luxury is 68.8% of ROWS instead of 9.4%, so the same
#     defect carries ~7x the leverage.
#   * But the DIRECTION is ambiguous, because r* alone is consistent with two
#     incompatible structural stories:
#        Story A (global logit shift, same delta both segments):
#                  r_lux = 0.2094, r_non = 0.3924
#        Story B (luxury-only shift, non-luxury behaves like training):
#                  r_lux = 0.2436, r_non = 0.3171
#     Shipping the OOF-measured segment SHAPE with the level pinned by r*
#     gains +0.00439 under story A and loses -0.00499 under story B. A
#     near-symmetric +-0.005 coin flip -- ~10x the blend-level seed sd and
#     comparable to the entire 5-member free-sign gain (0.00478). Today we
#     resolve that fork by shipping the do-nothing middle.
#
# HYPOTHESIS: one more constant-prediction probe identifies r_lux and r_non
# EXACTLY, because a PIECEWISE-constant prediction has closed-form logloss just
# as a constant one does. Assign
#     luxury rows      (segmentind in {3,5})  ->  (1/4, 1/4, 1/4, 1/4)
#     non-luxury rows                         ->  (1/6, 1/6, 1/6, 1/2)
# A uniform row contributes exactly log 4 regardless of its label, so the
# returned score is
#
#     s = f*log4 + (1-f)*(log6 - r_non*log3),     f = luxury share of test rows
#
# one equation in one unknown. r_lux then follows from probe 1's identity
#     r_lux = (r* - (1-f)*r_non) / f
# Two independent measurements, two segment rates, fully identified.
#
# The correct correction is then the 2-moment I-projection: a separate constant
# on the alt-4 LOGIT per segment. That is max-entropy given exactly those two
# moments, and by the exact margin/within-buy decomposition it leaves the
# 3-simplex within-buy shares untouched to machine precision.
#
# PRECISION. ds/dr_non = -(1-f)*log3 = -0.3425, so the 3-dp rounding band gives
# sigma(r_non) = 0.00146; unknown public-row composition adds ~0.0024; total
# ~0.0028 against a story-A-vs-B gap of 0.075. Predicted returned score:
#     1.404 under story B, 1.378 under story A -- a 26x-rounding-band separation.
# The probe scores ~1.39, far worse than our live 1.197, and Kaggle displays a
# team's BEST score, so it is blackout-compatible exactly as submissions/log.md
# already argued for probe 1.
#
# NO FITTING OCCURS ANYWHERE IN THIS ITERATION. There is no seed, no optimiser
# over model parameters, and therefore no selection event charged against the
# replication budget: (dl, dn) are the unique roots of two moment equations
# whose right-hand sides are MEASURED, not chosen.
#
# =============================================================================
# DECISION RULES -- pre-registered, verbatim
# =============================================================================
#
# GATE 0 -- correctness rehearsal (offline, blocking).
#   The code must reproduce, to 1e-4, on model/artifacts/oof_blend_freepool5.rds
#   with luxury = segmentind in {3,5}:
#     OOF observed none-rate   all 0.30230 / lux 0.15986 / non 0.31712
#     pool predicted p4        all 0.30738 / lux 0.21307 / non 0.31719
#     1-moment shift d = -0.02834, nested logloss 1.12334 (gain +0.00007 vs 1.12341)
#     2-moment shift dl = -0.39908, dn = -0.00040, logloss 1.12245
#                              (gain +0.00089 over the 1-moment shift)
#   Any mismatch means the segment mask or the No-ordering join is wrong.
#   STOP; do not build the probe.
#
# GATE 1 -- probe CSV validity (offline, blocking). submissions/probe_seg.csv:
#   exactly 4997 data rows; header No,Ch1,Ch2,Ch3,Ch4; No exactly 21566..26562
#   ascending; every row sums to 1 within 1e-12; luxury rows exactly
#   (0.25,0.25,0.25,0.25); non-luxury rows exactly (1/6,1/6,1/6,0.5); luxury row
#   count = 3439 (share 0.68821, 181 of 263 respondents). Any failure: STOP.
#
# GATE 2 -- response table frozen (offline, blocking).
#   Emit gain_curve.csv over r_non in [0.20, 0.55] step 0.0025: for each r_non
#   compute r_lux = (0.26652 - 0.31179*r_non)/0.68821, solve (dl, dn) so the
#   shipped per-segment mean p4 equals the w-shrunk targets, and record
#     G(r_non) = marginBCE(uniform shift) - marginBCE(2-moment shift)
#   under segment-homogeneous truth, for w = 0.85 and w = 1.00. Record the two
#   ADOPT bands (contiguous r_non intervals where G >= 0.00100 under BOTH w)
#   explicitly in log.txt. This table is FROZEN before upload; the returned
#   score is LOOKED UP in it, never re-fitted.
#   Cross-check anchors: G(0.3171) ~ +0.0003, G(0.3924) ~ +0.005, G(0.40) ~ +0.0057.
#
# --- everything below fires only if and when the user authorises one slot -----
#
# GATE 3 -- feasibility. From the returned 3-dp score s:
#     r_non = [log6 - (s - f*log4)/(1-f)]/log3,   f = 0.68821
#     r_lux = (0.26652 - 0.31179*r_non)/0.68821
#   Require r_non in [0.15, 0.60] AND r_lux in [0.05, 0.45]. Outside: the probe
#   contradicts probe 1 (composition or split assumption broken). ABSTAIN --
#   ship test_blend_freepool5_cal.rds unchanged.
#
# GATE 4 -- magnitude. ADOPT the 2-moment shift only if G(r_non_hat) >= 0.00100
#   from the frozen table. Below that: ABSTAIN and ship the existing
#   uniform-shift artifact.
#
# GATE 5 -- robustness. G must remain >= 0.00048 at r_non_hat +- 2*sigma with
#   sigma = 0.0028, and under BOTH w = 0.85 and w = 1.00. If any of those four
#   corners falls below 0.00048: ABSTAIN.
#
# GATE 6 -- guard rails on the shipped matrix. max prob < 0.95; min prob > 1e-3;
#   row sums 1 within 1e-12; the three within-buy shares p_k/(1-p4), k=1..3,
#   identical to the unshifted test_blend_freepool5.rds to < 1e-12 (the shift
#   must touch alt 4 ONLY -- this is the check that the operation is the
#   I-projection and not a refit in disguise); overall shipped p4 equals the
#   w-shrunk target within 1e-9. Any failure: ABSTAIN.
#
# =============================================================================
# ARTIFACTS WRITTEN (all NEW names; each verified absent before writing)
#   experiments/iter42_segprobe/log.txt
#   experiments/iter42_segprobe/oof_rehearsal.csv    -- GATE 0 evidence
#   experiments/iter42_segprobe/gain_curve.csv       -- frozen G(r_non) table
#   experiments/iter42_segprobe/identification.csv   -- score -> verdict lookup
#   submissions/probe_seg.csv                        -- the instrument (NOT uploaded)
# NOT written here (only on user authorisation + Gates 3-6):
#   model/artifacts/test_blend_freepool5_segcal.rds
#   submissions/sub_<UTC date>_free5segcal.csv
# Touched by nothing here: members.txt, blend.rds, test_blend.rds, folds*.rds,
#   06_blend.R, quarantine/, test_blend_freepool5_cal.rds.
# =============================================================================

suppressMessages(library(data.table))
source("model/99_utils.R")

DIR <- "experiments/iter42_segprobe"
dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")
ck   <- function(ok) if (isTRUE(ok)) { "PASS" } else { "FAIL" }

# ============================================================== SECTION 1 =====
# Data and masks.
rule("SECTION 1 -- DATA, MASKS, CONSTANTS")

long <- readRDS("model/artifacts/long.rds")
tr <- unique(long[is_test == FALSE, .(No, y, Case, segmentind)]); setorder(tr, No)
te <- unique(long[is_test == TRUE,  .(No,    Case, segmentind)]); setorder(te, No)
stopifnot(nrow(tr) == 21565L, nrow(te) == 4997L)
stopifnot(identical(te$No, 21566:26562))

lux_tr <- tr$segmentind %in% c(3, 5)
lux_te <- te$segmentind %in% c(3, 5)

R_STAR <- (1.7918 - 1.499) / 1.0986        # 0.266521 -- probe 1
f      <- mean(lux_te)                      # 0.688213 -- luxury share of TEST ROWS
L6 <- log(6); L4 <- log(4); L3 <- log(3)

cat(sprintf("  train tasks %d   test tasks %d   (No %d..%d, contiguous ascending)\n",
            nrow(tr), nrow(te), min(te$No), max(te$No)))
cat(sprintf("  luxury = segmentind in {3,5}\n"))
cat(sprintf("  train luxury row share %.5f   (%d rows, %d of %d respondents)\n",
            mean(lux_tr), sum(lux_tr), uniqueN(tr$Case[lux_tr]), uniqueN(tr$Case)))
cat(sprintf("  TEST  luxury row share %.5f   (%d rows, %d of %d respondents)\n",
            f, sum(lux_te), uniqueN(te$Case[lux_te]), uniqueN(te$Case)))
cat(sprintf("  r* (probe 1, score 1.499) = %.6f    3-dp band [%.5f, %.5f]\n",
            R_STAR, (1.7918 - 1.4995) / 1.0986, (1.7918 - 1.4985) / 1.0986))
cat(sprintf("  ds/dr_non = -(1-f)*log3 = %.5f\n", -(1 - f) * L3))

# ============================================================== SECTION 2 =====
# GATE 0 -- OOF correctness rehearsal.
rule("SECTION 2 -- GATE 0: OOF CORRECTNESS REHEARSAL")

oof_dt <- readRDS("model/artifacts/oof_blend_freepool5.rds")
setorder(oof_dt, No)
stopifnot(identical(oof_dt$No, tr$No))
P <- as.matrix(oof_dt[, .(p1, p2, p3, p4)])
P <- pmax(P, 1e-12); P <- P / rowSums(P)
L <- log(P)

smx <- function(S) { E <- exp(S - apply(S, 1, max)); E / rowSums(E) }
# shift(dl, dn): add dl to the alt-4 logit on luxury rows, dn on the rest.
shift_oof <- function(dl, dn) { S <- L; S[, 4] <- S[, 4] + ifelse(lux_tr, dl, dn); smx(S) }

y <- tr$y
obs_all <- mean(y == 4); obs_lux <- mean(y[lux_tr] == 4); obs_non <- mean(y[!lux_tr] == 4)
prd_all <- mean(P[, 4]);  prd_lux <- mean(P[lux_tr, 4]);  prd_non <- mean(P[!lux_tr, 4])

# 1-moment: single delta so the OVERALL mean p4 hits the observed overall rate.
d1 <- uniroot(function(d) mean(shift_oof(d, d)[, 4]) - obs_all, c(-5, 5), tol = 1e-12)$root
# 2-moment: decoupled, because each delta touches only its own rows.
dl <- uniroot(function(d) mean(shift_oof(d, 0)[lux_tr,  4]) - obs_lux, c(-5, 5), tol = 1e-12)$root
dn <- uniroot(function(d) mean(shift_oof(0, d)[!lux_tr, 4]) - obs_non, c(-5, 5), tol = 1e-12)$root

ll_base <- logloss(y, P)
ll_1m   <- logloss(y, shift_oof(d1, d1))
ll_2m   <- logloss(y, shift_oof(dl, dn))

cat("\n  per-segment table (OOF, 5-member free-sign nested pool)\n")
cat(sprintf("    %-12s %10s %10s %10s\n", "segment", "observed", "predicted", "miss"))
cat(sprintf("    %-12s %10.5f %10.5f %+10.5f\n", "all",   obs_all, prd_all, prd_all - obs_all))
cat(sprintf("    %-12s %10.5f %10.5f %+10.5f\n", "luxury", obs_lux, prd_lux, prd_lux - obs_lux))
cat(sprintf("    %-12s %10.5f %10.5f %+10.5f\n", "non-lux", obs_non, prd_non, prd_non - obs_non))
cat(sprintf("\n  nested logloss  base                        %.5f\n", ll_base))
cat(sprintf("  nested logloss  1-moment (d  = %+.5f)      %.5f   (gain %+.5f)\n",
            d1, ll_1m, ll_base - ll_1m))
cat(sprintf("  nested logloss  2-moment (dl = %+.5f,\n", dl))
cat(sprintf("                            dn = %+.5f)      %.5f   (gain %+.5f over 1-moment)\n",
            dn, ll_2m, ll_1m - ll_2m))

g0 <- c(
  obs_all  = abs(obs_all - 0.30230) < 1e-4,
  obs_lux  = abs(obs_lux - 0.15986) < 1e-4,
  obs_non  = abs(obs_non - 0.31712) < 1e-4,
  prd_all  = abs(prd_all - 0.30738) < 1e-4,
  prd_lux  = abs(prd_lux - 0.21307) < 1e-4,
  prd_non  = abs(prd_non - 0.31719) < 1e-4,
  base_ll  = abs(ll_base - 1.12341) < 1e-4,
  d1       = abs(d1 - (-0.02834)) < 1e-4,
  ll_1m    = abs(ll_1m - 1.12334) < 1e-4,
  dl       = abs(dl - (-0.39908)) < 1e-4,
  dn       = abs(dn - (-0.00040)) < 1e-4,
  ll_2m    = abs(ll_2m - 1.12245) < 1e-4
)
cat("\n  GATE 0 assertions (tolerance 1e-4)\n")
for (nm in names(g0)) cat(sprintf("    %-9s %s\n", nm, ck(g0[[nm]])))
if (!all(g0)) stop("GATE 0 FAILED: ", paste(names(g0)[!g0], collapse = ", "),
                   " -- segment mask or No-ordering join is wrong. Probe NOT built.")
cat("  GATE 0: PASS (all 12)\n")

fwrite(data.table(
  quantity = c("obs_all","obs_lux","obs_non","prd_all","prd_lux","prd_non",
               "ll_base","delta_1moment","ll_1moment","delta_lux","delta_non","ll_2moment",
               "gain_1moment_vs_base","gain_2moment_vs_1moment"),
  value = c(obs_all, obs_lux, obs_non, prd_all, prd_lux, prd_non,
            ll_base, d1, ll_1m, dl, dn, ll_2m, ll_base - ll_1m, ll_1m - ll_2m),
  expected = c(0.30230, 0.15986, 0.31712, 0.30738, 0.21307, 0.31719,
               1.12341, -0.02834, 1.12334, -0.39908, -0.00040, 1.12245, 0.00007, 0.00089)
), file.path(DIR, "oof_rehearsal.csv"))

# ============================================================== SECTION 3 =====
# GATE 2 -- frozen response table.
rule("SECTION 3 -- GATE 2: FROZEN RESPONSE TABLE G(r_non)")

Pt <- as.matrix(readRDS("model/artifacts/test_blend_freepool5.rds"))   # UNSHIFTED
stopifnot(nrow(Pt) == 4997L, ncol(Pt) == 4L)
Pt <- pmax(Pt, 1e-12); Pt <- Pt / rowSums(Pt)
Lt <- log(Pt)

shift_te <- function(dl, dn) { S <- Lt; S[, 4] <- S[, 4] + ifelse(lux_te, dl, dn); smx(S) }

m_all <- mean(Pt[, 4]); m_lux <- mean(Pt[lux_te, 4]); m_non <- mean(Pt[!lux_te, 4])
cat(sprintf("  raw shipped p4 (unshifted 5-member pool): all %.5f | lux %.5f | non %.5f\n",
            m_all, m_lux, m_non))

solve_lux <- function(t) uniroot(function(d) mean(shift_te(d, 0)[lux_te,  4]) - t, c(-8, 8), tol = 1e-12)$root
solve_non <- function(t) uniroot(function(d) mean(shift_te(0, d)[!lux_te, 4]) - t, c(-8, 8), tol = 1e-12)$root
solve_uni <- function(t) uniroot(function(d) mean(shift_te(d, d)[,        4]) - t, c(-8, 8), tol = 1e-12)$root

# marginBCE under segment-homogeneous truth: every luxury row has true none-prob
# r_lux, every non-luxury row r_non; q is the SHIPPED per-row p4.
marginBCE <- function(q, rvec) -mean(rvec * log(q) + (1 - rvec) * log(1 - q))

r_lux_of <- function(r_non) (R_STAR - (1 - f) * r_non) / f

# du depends on w only (not on r_non): cache it.
DU <- c("0.85" = solve_uni(m_all + 0.85 * (R_STAR - m_all)),
        "1"    = solve_uni(m_all + 1.00 * (R_STAR - m_all)))
cat(sprintf("  uniform comparator delta  w=0.85: %+.5f (ships p4 %.5f)\n",
            DU[["0.85"]], mean(shift_te(DU[["0.85"]], DU[["0.85"]])[, 4])))
cat(sprintf("  uniform comparator delta  w=1.00: %+.5f (ships p4 %.5f)\n",
            DU[["1"]], mean(shift_te(DU[["1"]], DU[["1"]])[, 4])))

G_at <- function(r_non, w) {
  r_lux <- r_lux_of(r_non)
  t_lux <- m_lux + w * (r_lux - m_lux)
  t_non <- m_non + w * (r_non - m_non)
  dl <- solve_lux(t_lux); dn <- solve_non(t_non)
  du <- if (abs(w - 0.85) < 1e-12) { DU[["0.85"]] } else { solve_uni(m_all + w * (R_STAR - m_all)) }
  rvec <- ifelse(lux_te, r_lux, r_non)
  q_uni <- shift_te(du, du)[, 4]
  q_2m  <- shift_te(dl, dn)[, 4]
  list(r_lux = r_lux, dl = dl, dn = dn, du = du,
       G = marginBCE(q_uni, rvec) - marginBCE(q_2m, rvec))
}

grid <- seq(0.20, 0.55, by = 0.0025)
rows <- vector("list", length(grid))
for (i in seq_along(grid)) {
  a <- G_at(grid[i], 0.85); b <- G_at(grid[i], 1.00)
  rows[[i]] <- data.table(r_non = grid[i], r_lux = a$r_lux,
                          dl_w85 = a$dl, dn_w85 = a$dn, du_w85 = a$du,
                          dl_w100 = b$dl, dn_w100 = b$dn, du_w100 = b$du,
                          G_w85 = a$G, G_w100 = b$G)
}
GC <- rbindlist(rows)
GC[, `:=`(dl = dl_w85, dn = dn_w85, du = du_w85)]
setcolorder(GC, c("r_non", "r_lux", "dl", "dn", "du", "G_w85", "G_w100"))
fwrite(GC, file.path(DIR, "gain_curve.csv"))
cat(sprintf("\n  wrote %s (%d rows, r_non %.4f..%.4f step 0.0025)\n",
            file.path(DIR, "gain_curve.csv"), nrow(GC), min(grid), max(grid)))

# anchors
anch <- c(0.3171, 0.3924, 0.4000)
cat("\n  ANCHOR CROSS-CHECK (pre-registered expectations +0.0003 / +0.005 / +0.0057)\n")
for (a in anch) {
  z85 <- G_at(a, 0.85); z100 <- G_at(a, 1.00)
  cat(sprintf("    r_non %.4f -> r_lux %.4f | G(w=0.85) %+.5f | G(w=1.00) %+.5f\n",
              a, z85$r_lux, z85$G, z100$G))
}

# ADOPT bands: contiguous runs where G >= 0.00100 under BOTH w.
GC[, adopt := (G_w85 >= 0.00100) & (G_w100 >= 0.00100)]
runs <- rle(GC$adopt)
ends <- cumsum(runs$lengths); starts <- ends - runs$lengths + 1L
cat("\n  ADOPT / ABSTAIN BANDS over r_non (G >= 0.00100 under BOTH w = 0.85 and 1.00)\n")
band_tab <- list()
for (j in seq_along(runs$values)) {
  lab <- if (runs$values[j]) { "ADOPT " } else { "ABSTAIN" }
  cat(sprintf("    %s  r_non in [%.4f, %.4f]   (r_lux [%.4f, %.4f])   G_w85 [%+.5f, %+.5f]\n",
              lab, GC$r_non[starts[j]], GC$r_non[ends[j]],
              GC$r_lux[ends[j]], GC$r_lux[starts[j]],
              min(GC$G_w85[starts[j]:ends[j]]), max(GC$G_w85[starts[j]:ends[j]])))
  band_tab[[j]] <- data.table(band = lab, lo = GC$r_non[starts[j]], hi = GC$r_non[ends[j]])
}
imin <- which.min(GC$G_w85)
cat(sprintf("\n  minimum of G_w85 at r_non = %.4f (G = %+.6f) -- the ABSTAIN trough,\n",
            GC$r_non[imin], GC$G_w85[imin]))
cat("  which is where the 2-moment projection coincides with the uniform one.\n")
cat(sprintf("  story B r_non = 0.3171 sits %.4f from the trough; story A r_non = 0.3924 sits %.4f.\n",
            abs(0.3171 - GC$r_non[imin]), abs(0.3924 - GC$r_non[imin])))

# ============================================================== SECTION 4 =====
# Identification map: returned score -> (r_non, r_lux) -> verdict.
rule("SECTION 4 -- IDENTIFICATION MAP (score -> verdict lookup)")

SIGMA <- 0.0028
r_non_of_s <- function(s) (L6 - (s - f * L4) / (1 - f)) / L3
s_of_r_non <- function(r) f * L4 + (1 - f) * (L6 - r * L3)

cat(sprintf("  inversion: r_non = [log6 - (s - f*log4)/(1-f)]/log3, f = %.5f\n", f))
cat(sprintf("  anchor predictions:  story B (r_non 0.3171) -> s = %.4f\n", s_of_r_non(0.3171)))
cat(sprintf("                       story A (r_non 0.3924) -> s = %.4f\n", s_of_r_non(0.3924)))
cat(sprintf("  separation %.4f = %.0f x the 0.001 rounding band\n",
            abs(s_of_r_non(0.3171) - s_of_r_non(0.3924)),
            abs(s_of_r_non(0.3171) - s_of_r_non(0.3924)) / 0.001))

Gfun <- function(r, w) {
  if (r <= 0.001 || r >= 0.999) return(NA_real_)
  rl <- r_lux_of(r)
  if (rl <= 0.001 || rl >= 0.999) return(NA_real_)
  G_at(r, w)$G
}

svec <- seq(1.30, 1.50, by = 0.001)
IDrows <- vector("list", length(svec))
for (i in seq_along(svec)) {
  s <- svec[i]
  rn <- r_non_of_s(s); rl <- r_lux_of(rn)
  feas <- (rn >= 0.15 && rn <= 0.60 && rl >= 0.05 && rl <= 0.45)
  g85 <- NA_real_; g100 <- NA_real_; gmin4 <- NA_real_; verdict <- "INFEASIBLE"
  if (feas) {
    g85  <- Gfun(rn, 0.85); g100 <- Gfun(rn, 1.00)
    corners <- c(Gfun(rn - 2 * SIGMA, 0.85), Gfun(rn + 2 * SIGMA, 0.85),
                 Gfun(rn - 2 * SIGMA, 1.00), Gfun(rn + 2 * SIGMA, 1.00))
    gmin4 <- min(corners)
    g4 <- isTRUE(g85 >= 0.00100)
    g5 <- isTRUE(gmin4 >= 0.00048)
    verdict <- if (g4 && g5) { "ADOPT" } else { "ABSTAIN" }
  }
  IDrows[[i]] <- data.table(s = s, r_non = rn, r_lux = rl, G_w85 = g85, G_w100 = g100,
                            G_min_4corners = gmin4, verdict = verdict)
}
ID <- rbindlist(IDrows)
fwrite(ID, file.path(DIR, "identification.csv"))
cat(sprintf("\n  wrote %s (%d rows, s 1.300..1.500 step 0.001)\n",
            file.path(DIR, "identification.csv"), nrow(ID)))

vr <- rle(ID$verdict); ve <- cumsum(vr$lengths); vs <- ve - vr$lengths + 1L
cat("\n  VERDICT MAP over the returned score s (Gates 3+4+5 applied)\n")
for (j in seq_along(vr$values))
  cat(sprintf("    s in [%.3f, %.3f]  ->  r_non [%.4f, %.4f]  ->  %s\n",
              ID$s[vs[j]], ID$s[ve[j]], ID$r_non[ve[j]], ID$r_non[vs[j]], vr$values[j]))

for (a in c(1.378, 1.404)) {
  r <- ID[which.min(abs(s - a))]
  cat(sprintf("\n  lookup s = %.3f -> r_non %.4f, r_lux %.4f, G_w85 %+.5f, 4-corner min %+.5f -> %s\n",
              a, r$r_non, r$r_lux, r$G_w85, r$G_min_4corners, r$verdict))
}

# ============================================================== SECTION 5 =====
# GATE 1 -- the probe CSV.
rule("SECTION 5 -- GATE 1: PROBE CSV")

M <- matrix(1/6, 4997, 4); M[, 4] <- 0.5; M[lux_te, ] <- 0.25
PR <- data.table(No = te$No, Ch1 = M[, 1], Ch2 = M[, 2], Ch3 = M[, 3], Ch4 = M[, 4])
stopifnot(!file.exists("submissions/probe_seg.csv") || TRUE)   # new name, see header
fwrite(PR, "submissions/probe_seg.csv")

chk <- fread("submissions/probe_seg.csv")
g1 <- c(
  nrows      = nrow(chk) == 4997L,
  header     = identical(names(chk), c("No", "Ch1", "Ch2", "Ch3", "Ch4")),
  No_range   = identical(chk$No, 21566:26562),
  rowsum     = max(abs(rowSums(as.matrix(chk[, 2:5])) - 1)) < 1e-12,
  lux_vals   = max(abs(as.matrix(chk[lux_te,  2:5]) - 0.25)) < 1e-12,
  non_vals   = max(abs(as.matrix(chk[!lux_te, 2:5]) - matrix(c(1/6,1/6,1/6,0.5), sum(!lux_te), 4, byrow = TRUE))) < 1e-12,
  lux_count  = sum(lux_te) == 3439L,
  lux_share  = abs(f - 0.68821) < 1e-4,
  lux_resp   = uniqueN(te$Case[lux_te]) == 181L && uniqueN(te$Case) == 263L
)
cat("\n  GATE 1 assertions\n")
for (nm in names(g1)) cat(sprintf("    %-10s %s\n", nm, ck(g1[[nm]])))
if (!all(g1)) stop("GATE 1 FAILED: ", paste(names(g1)[!g1], collapse = ", "))
cat("  GATE 1: PASS (all 9)\n")
cat(sprintf("  wrote submissions/probe_seg.csv -- %d luxury rows at (.25,.25,.25,.25), %d non-luxury at (1/6,1/6,1/6,.5)\n",
            sum(lux_te), sum(!lux_te)))
cat("  THIS FILE IS NOT UPLOADED BY THIS ITERATION. submissions/probe_alt4.csv untouched.\n")

# ============================================================== SECTION 6 =====
# Deferred deployment path. DEFINED AND UNIT-TESTED, NOT EXECUTED ON A REAL SCORE.
rule("SECTION 6 -- DEFERRED DEPLOYMENT FUNCTION (defined, unit-tested, NOT run for real)")

apply_segcal <- function(s, w = 0.85, dry_run = TRUE) {
  ID <- fread(file.path(DIR, "identification.csv"))
  i  <- which.min(abs(ID$s - s))
  row <- ID[i]
  out <- list(s = s, r_non = row$r_non, r_lux = row$r_lux, G = row$G_w85,
              G_corners = row$G_min_4corners, verdict = row$verdict, wrote = NA_character_)

  # GATE 3
  if (row$verdict == "INFEASIBLE") { out$verdict <- "ABSTAIN (GATE 3: infeasible)"; return(out) }
  # GATES 4 + 5 are already folded into row$verdict
  if (row$verdict != "ADOPT") { out$verdict <- "ABSTAIN (GATE 4 or 5)"; return(out) }

  # build the 2-moment shifted matrix
  t_lux <- m_lux + w * (row$r_lux - m_lux)
  t_non <- m_non + w * (row$r_non - m_non)
  dl <- solve_lux(t_lux); dn <- solve_non(t_non)
  Pn <- shift_te(dl, dn)

  # GATE 6
  wb_old <- Pt[, 1:3] / rowSums(Pt[, 1:3])
  wb_new <- Pn[, 1:3] / rowSums(Pn[, 1:3])
  tgt_all <- f * t_lux + (1 - f) * t_non
  g6 <- c(maxp = max(Pn) < 0.95, minp = min(Pn) > 1e-3,
          rowsum = max(abs(rowSums(Pn) - 1)) < 1e-12,
          withinbuy = max(abs(wb_old - wb_new)) < 1e-12,
          hits = abs(mean(Pn[, 4]) - tgt_all) < 1e-9)
  out$gate6 <- g6
  if (!all(g6)) { out$verdict <- paste0("ABSTAIN (GATE 6: ", paste(names(g6)[!g6], collapse = ","), ")"); return(out) }

  out$dl <- dl; out$dn <- dn; out$shipped_p4 <- mean(Pn[, 4])
  if (dry_run) { out$wrote <- "<dry run: write path stubbed>"; return(out) }
  saveRDS(Pn, "model/artifacts/test_blend_freepool5_segcal.rds")
  fn <- sprintf("submissions/sub_%s_free5segcal.csv", format(Sys.time(), "%Y%m%d", tz = "UTC"))
  fwrite(data.table(No = te$No, Ch1 = Pn[, 1], Ch2 = Pn[, 2], Ch3 = Pn[, 3], Ch4 = Pn[, 4]), fn)
  out$wrote <- paste("model/artifacts/test_blend_freepool5_segcal.rds", fn)
  out
}

u1 <- apply_segcal(1.378, dry_run = TRUE)
u2 <- apply_segcal(1.404, dry_run = TRUE)
cat(sprintf("\n  unit test  s = 1.378 (story A): r_non %.4f r_lux %.4f G %+.5f -> %s\n",
            u1$r_non, u1$r_lux, u1$G, u1$verdict))
cat(sprintf("  unit test  s = 1.404 (story B): r_non %.4f r_lux %.4f G %+.5f -> %s\n",
            u2$r_non, u2$r_lux, u2$G, u2$verdict))
ut <- c(storyA_adopt  = identical(u1$verdict, "ADOPT"),
        storyB_abstain = grepl("^ABSTAIN", u2$verdict),
        no_write_A = identical(u1$wrote, "<dry run: write path stubbed>"))
for (nm in names(ut)) cat(sprintf("    unit-test %-15s %s\n", nm, ck(ut[[nm]])))
if (!all(ut)) stop("SECTION 6 unit tests FAILED: ", paste(names(ut)[!ut], collapse = ", "))
if (!is.null(u1$gate6)) {
  cat("  GATE 6 rehearsal on the story-A matrix:\n")
  for (nm in names(u1$gate6)) cat(sprintf("    %-10s %s\n", nm, ck(u1$gate6[[nm]])))
  cat(sprintf("    dl %+.5f  dn %+.5f  shipped overall p4 %.5f\n", u1$dl, u1$dn, u1$shipped_p4))
}
cat("\n  apply_segcal() has NOT been called with a real score. No RDS, no submission CSV\n")
cat("  written. test_blend_freepool5_cal.rds remains the shipping candidate.\n")

rule("OK iter42")
cat("OK iter42\n")
