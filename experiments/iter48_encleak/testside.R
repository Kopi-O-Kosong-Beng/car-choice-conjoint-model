# =============================================================================
# ITERATION 48 -- the one EXTERNAL check we own.
#
# The constant-prediction probe (submissions/log.md, 27 Jul, probe_alt4.csv -> 1.499)
# fixes the TEST none-rate exactly:  r = (1.7918 - score)/1.0986 = 0.26651.
# That is a measured property of the 4,997 test rows, not a local surrogate, and it
# is the only thing about the test set we can check a model against without spending
# a submission.
#
# A model whose design-share encoding is doing real work should predict the test
# none-rate at least as well as one without it.  A model that has learned a rule off
# the leave-own-fold-out complement identity is applying that rule to test rows whose
# encoding has NO such structure (test respondents contribute nothing to any cell,
# so their share is the plain full-sample share and their design_n is ~1.37x the
# value the model was trained on).  If that mis-fires, it should show up here.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
R_TEST <- (1.7918 - 1.499) / 1.0986
cat(sprintf("probe-measured TEST none-rate  r = %.5f\n", R_TEST))
cat(sprintf("training none-rate                 %.5f\n\n",
            mean(unique(long[is_test == FALSE, .(No, y)])$y == 4)))
fs <- sort(list.files("experiments/iter48_encleak", pattern = "^el48_test_.*\\.rds$", full.names = TRUE))
fs <- c(fs, "model/artifacts/test_xgb_lw2bag.rds", "model/artifacts/test_lcmnl3_both.rds",
        "model/artifacts/test_blend.rds")
cat(sprintf("%-46s %9s %9s\n", "artifact", "mean p4", "err vs r"))
for (f in fs) {
  if (!file.exists(f)) next
  d <- readRDS(f)
  P <- if (is.data.table(d) && "p4" %in% names(d)) { as.matrix(d[, .(p1,p2,p3,p4)]) } else { as.matrix(d) }
  m <- mean(P[, 4] / rowSums(P))
  cat(sprintf("%-46s %9.5f %+9.5f\n", basename(f), m, m - R_TEST))
}
cat("\nTESTSIDE_DONE\n")
