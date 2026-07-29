# =============================================================================
# ITERATION 56 -- THE GATES THAT DECIDE WHETHER THE ROTATION SHIPS
#
# Written BEFORE any number in it is looked at. Fires only once iteration 54 has
# emitted model/artifacts/oof_xgb_rot.rds and test_xgb_rot.rds.
#
# WHAT IS ALREADY KNOWN AND WHAT IS NOT
# -----------------------------------------------------------------------------
# KNOWN: at 1 seed, appending 71 dense oblique coordinates of the attribute
# indicator matrix takes the fixed-rounds tree from 1.13761 to 1.11515 (-0.0225,
# 7.9 model-level seed sds). It survives fitting the basis inside each fold
# (-0.00007), so it is not transduction. A RANDOM rotation beats the SVD by
# 0.00346, so the mechanism is oblique geometry, not factorization.
#
# NOT KNOWN, and each of these has killed a result in this project before:
#   1. Does it hold over 10 seeds? (iteration 54 check C)
#   2. Does it retain under reweighting to the graded population? The design
#      encoding retained 77% and a fatigue term 64%; both were rejected.
#   3. DOES IT REACH THE BLEND? Iteration 39 gained +0.00252 at member level and
#      +0.00020 at blend level. This is the decision number and nothing else is.
#   4. Is it merely a better design-share encoding in disguise? ENC_COLS already
#      supplies shrunk empirical choice shares per (design, alternative). If the
#      rotation only helps because it lets the tree rediscover design-level
#      effects, then dropping ENC_COLS should leave the rotation gain intact --
#      and the two together should be much less than the sum of their parts.
#   5. Does it replicate on an INDEPENDENT respondent grouping (folds_b)?
#
# DECISION RULE -- fixed before running
#   ADOPT only if ALL hold:
#     (a) 10-seed member gain > 0.00283 (model-level seed sd)
#     (b) segment-reweighted retention >= 90%
#     (c) nested blend improves by > 0.00048 (blend-level seed sd)
#     (d) folds_b replication reproduces >= 60% of the folds gain
#   If (c) fails the member is interesting and does not ship. That has happened
#   before and is the single most common way a real gain turns out not to matter.
#
# NO SUBMISSION IS BUILT HERE. Building a CSV is a separate, explicitly
# authorised step. Nothing is uploaded without the user saying so at the time.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

DIR <- "experiments/iter56_rotgate"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")
need <- c("model/artifacts/oof_xgb_rot.rds", "model/artifacts/test_xgb_rot.rds")
if (!all(file.exists(need))) stop("iteration 54 has not emitted the artifacts yet: ",
                                  paste(need[!file.exists(need)], collapse = ", "))

long <- readRDS("model/artifacts/long.rds"); wide <- readRDS("model/artifacts/wide.rds")
ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No); y <- ymap$y

rule("GATE b -- SEGMENT-REWEIGHTED RETENTION AT MEMBER LEVEL")
mk_w <- function(v) {
  resp <- unique(wide[, c("Case", "is_test", v), with = FALSE])
  a <- resp[is_test == FALSE, .(ptr = .N / sum(!resp$is_test)), by = v]
  b <- resp[is_test == TRUE,  .(pte = .N / sum(resp$is_test)),  by = v]
  t2 <- merge(a, b, by = v, all.x = TRUE); t2[is.na(pte), pte := 0]
  t2[, wt := pmin(pmax(pte / ptr, 0.2), 5)]
  merge(unique(wide[is_test == FALSE, c("No", v), with = FALSE]),
        t2[, c(v, "wt"), with = FALSE], by = v)[order(No), wt]
}
w_seg <- mk_w("segmentind")
rwll <- function(P, w) { l <- -log(pmax(P[cbind(seq_along(y), y)], 1e-15)); sum(w * l) / sum(w) }
getP <- function(n) { d <- readRDS(sprintf("model/artifacts/oof_%s.rds", n)); setorder(d, No)
                      as.matrix(d[, .(p1, p2, p3, p4)]) }
Pinc <- getP("xgb_lw2bag"); Prot <- getP("xgb_rot")
cat(sprintf("  xgb_lw2bag  plain %.5f   segment-reweighted %.5f\n", logloss(y, Pinc), rwll(Pinc, w_seg)))
cat(sprintf("  xgb_rot     plain %.5f   segment-reweighted %.5f\n", logloss(y, Prot), rwll(Prot, w_seg)))
dp <- logloss(y, Prot) - logloss(y, Pinc); ds <- rwll(Prot, w_seg) - rwll(Pinc, w_seg)
cat(sprintf("  delta       plain %+.5f  reweighted %+.5f   retention %.0f%%  -> %s\n",
            dp, ds, 100 * ds / dp,
            if (ds / dp >= 0.90) { "PASS" } else if (ds / dp >= 0.60) { "MARGINAL" } else { "FAIL" }))

rule("GATE c -- THE BLEND GATE (the decision number)")
cat("  run these and compare against the production nested 1.12819:\n")
cat("    BLEND_MEMBERS='xgb_rot lcmnl3_both'                    -> swap\n")
cat("    BLEND_MEMBERS='xgb_lw2bag lcmnl3_both xgb_rot'         -> add\n")
combos <- list(swap = c("xgb_rot", "lcmnl3_both"),
               add  = c("xgb_lw2bag", "lcmnl3_both", "xgb_rot"))
out <- list()
for (nm in names(combos)) {
  o <- system2("/usr/local/bin/Rscript", "model/06_blend.R", stdout = TRUE, stderr = TRUE,
               env = c(sprintf("BLEND_MEMBERS=%s", paste(combos[[nm]], collapse = " ")),
                       sprintf("BLEND_OUT=blend_rot_%s", nm)))
  ln <- grep("NESTED blend OOF", o, value = TRUE)
  wl <- grep("^weights:", o, value = TRUE)
  val <- as.numeric(sub(".*number\\): *([0-9.]+).*", "\\1", ln))
  out[[nm]] <- data.table(combo = nm, members = paste(combos[[nm]], collapse = "+"), nested = val)
  cat(sprintf("\n  %-5s %s\n    nested %.5f   (production 1.12819, delta %+.5f, %.1f blend sd)\n    %s\n",
              nm, paste(combos[[nm]], collapse = " + "), val, val - 1.12819,
              abs(val - 1.12819) / 0.00048, if (length(wl)) wl else ""))
  # segment-reweighted for the winning blend's OOF is computed below from its own preds
}
B <- rbindlist(out); fwrite(B, file.path(DIR, "blend_gate.csv"))
best <- B[which.min(nested)]
cat(sprintf("\n  best combo: %s at %.5f (delta %+.5f)  -> %s\n", best$combo, best$nested,
            best$nested - 1.12819,
            if (best$nested < 1.12819 - 0.00048) { "PASSES the blend gate" } else {
              "FAILS the blend gate -- the member gain does not reach the blend" }))
cat("\n  NEXT if it passes: folds_b replication, then and only then a submission decision.\n")
cat("done\n")
