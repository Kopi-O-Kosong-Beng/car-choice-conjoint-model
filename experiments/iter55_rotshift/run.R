# =============================================================================
# ITERATION 55 -- DOES THE ROTATION GAIN SURVIVE REWEIGHTING TO THE GRADED
#                 POPULATION? (read-only; consumes iteration 49's saved preds)
#
# Written BEFORE looking at any number here.
#
# WHY THIS IS THE DECISIVE TEST AND NOT AN AFTERTHOUGHT
# -----------------------------------------------------------------------------
# The iteration-48 workstream just established that our plain nested OOF is a
# poor predictor of the board and WHY: 98.7% of the +0.069 local-to-public
# offset is train/test SEGMENT COMPOSITION, not overfitting. Plain 1.12819 ->
# segment-reweighted 1.19610 -> public 1.197. The reweighted metric lands within
# 0.001 of the actual score; the plain one is off by 0.069.
#
# Iteration 51 measured the SVD-rotation gain at -0.019 on the PLAIN metric.
# Plain gains are exactly what this project has been fooled by before: the
# design encoding retained only 77% under reweighting and a fatigue term 64%,
# and both were rejected for it. Test is 68.8% luxury ROWS against 9.4% of
# training, so a gain concentrated in the 90% of training that is 31% of the
# graded population is nearly worthless, and a plain-OOF number cannot tell the
# difference.
#
# HYPOTHESIS: the rotation gain is structural (it is a geometry effect on the
# ATTRIBUTE design, which is randomised and therefore population-independent),
# so it should retain ~100% under reweighting -- unlike the design encoding,
# which is built from choice shares and is therefore population-specific.
#
# DECISION RULE -- fixed before running
#   retention = (reweighted gain) / (plain gain).
#     >= 90%  structural. Proceed to the blend gate with confidence.
#     60-90%  partly population-specific. Proceed but discount accordingly.
#     < 60%   same failure mode as the design encoding. Do NOT ship on the
#             strength of the plain number.
#   Reported under BOTH the segment reweighting (the one that matched the board)
#   and the income reweighting (06_blend.R's existing diagnostic), because a
#   result that depends on which reweighting is used is not a result.
#
# NOTHING IS FITTED HERE. No artifacts written. Read-only.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

DIR <- "experiments/iter55_rotshift"; dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long <- readRDS("model/artifacts/long.rds")
wide <- readRDS("model/artifacts/wide.rds")
ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No); y <- ymap$y

S <- readRDS("experiments/iter49_matfact/screen_full.rds")
Ps <- S$preds
cat("  configs available from iteration 49:", paste(names(Ps), collapse = ", "), "\n")

# ---- build both reweightings ------------------------------------------------
mk_w <- function(varname) {
  resp <- unique(wide[, c("Case", "is_test", varname), with = FALSE])
  ntr <- sum(!resp$is_test); nte <- sum(resp$is_test)
  a <- resp[is_test == FALSE, .(ptr = .N / ntr), by = varname]
  b <- resp[is_test == TRUE,  .(pte = .N / nte), by = varname]
  t2 <- merge(a, b, by = varname, all.x = TRUE)
  t2[is.na(pte), pte := 0]; t2[, wt := pmin(pmax(pte / ptr, 0.2), 5)]
  merge(unique(wide[is_test == FALSE, c("No", varname), with = FALSE]),
        t2[, c(varname, "wt"), with = FALSE], by = varname)[order(No), wt]
}
w_seg <- mk_w("segmentind")
w_inc <- mk_w("incomeind")
ess <- function(w) sum(w)^2 / sum(w^2)
cat(sprintf("  segment reweighting: ESS %.0f of %d respondents-tasks\n", ess(w_seg), length(y)))
cat(sprintf("  income  reweighting: ESS %.0f\n", ess(w_inc)))

rwll <- function(P, w) { l <- -log(pmax(P[cbind(seq_along(y), y)], 1e-15)); sum(w * l) / sum(w) }

rule("PLAIN vs REWEIGHTED, every iteration-49 config")
out <- rbindlist(lapply(names(Ps), function(n) {
  P <- Ps[[n]]
  data.table(cfg = n, plain = logloss(y, P), seg = rwll(P, w_seg), inc = rwll(P, w_inc))
}))
b <- out[cfg == "base"]
out[, `:=`(d_plain = plain - b$plain, d_seg = seg - b$seg, d_inc = inc - b$inc)]
out[, `:=`(ret_seg = ifelse(abs(d_plain) < 1e-9, NA_real_, d_seg / d_plain),
           ret_inc = ifelse(abs(d_plain) < 1e-9, NA_real_, d_inc / d_plain))]
setorder(out, plain)
print(out[, .(cfg, plain, seg, d_plain, d_seg,
              ret_seg = sprintf("%.0f%%", 100 * ret_seg),
              ret_inc = sprintf("%.0f%%", 100 * ret_inc))])
fwrite(out, file.path(DIR, "retention.csv"))

rule("VERDICT")
for (n in c("dsvd8", "dsvd16", "dsvd32")) {
  r <- out[cfg == n]
  v <- if (is.na(r$ret_seg)) { "n/a" } else if (r$ret_seg >= 0.90) { "STRUCTURAL" } else
       if (r$ret_seg >= 0.60) { "PARTLY POPULATION-SPECIFIC -- discount" } else {
         "POPULATION-SPECIFIC -- same failure mode as the design encoding" }
  cat(sprintf("  %-7s plain %+.5f  segment-reweighted %+.5f  retention %.0f%%  -> %s\n",
              n, r$d_plain, r$d_seg, 100 * r$ret_seg, v))
}
cat(sprintf("\n  For scale: the design encoding retained 77%%, a fatigue term 64%%; both rejected.\n"))
cat(sprintf("  The segment-reweighted metric predicted the live board to 0.001 (1.19610 vs 1.197),\n"))
cat(sprintf("  while plain OOF was off by 0.069. Believe the reweighted column.\n"))
cat("\ndone\n")
