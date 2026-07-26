# =============================================================================
# ITERATION 21 — fold-structure robustness. Step 0: build TWO NEW respondent-grouped
# splits. model/artifacts/folds.rds (seed 42) is NEVER touched.
#
# Same construction as model/01_folds.R, different seeds only.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
tasks <- unique(long[is_test == FALSE, .(No, Case)])
setorder(tasks, No)

prod <- readRDS("model/artifacts/folds.rds"); setorder(prod, No)
stopifnot(identical(prod$No, tasks$No))

for (s in c(43L, 44L)) {
  t2 <- copy(tasks)
  t2[, fold := make_case_folds(Case, k = 5, seed = s)]
  stopifnot(nrow(t2) == 21565, all(t2[, uniqueN(fold), by = Case]$V1 == 1))
  out <- sprintf("model/artifacts/folds_%s.rds", if (s == 43L) "b" else "c")
  stopifnot(!file.exists("model/artifacts/folds.rds") == FALSE)   # sanity: prod exists
  saveRDS(t2, out)
  cat("\n=== seed", s, "->", out, "===\n")
  print(t2[, .(tasks = .N, respondents = uniqueN(Case)), keyby = fold])
  # how different is this split from production? (respondent-level agreement)
  m <- merge(unique(t2[, .(Case, fb = fold)]), unique(prod[, .(Case, fp = fold)]), by = "Case")
  cat("respondents in the same fold index as production:",
      sprintf("%.1f%% (chance 20%%)\n", 100 * mean(m$fb == m$fp)))
  # the meaningful measure: how often two respondents co-fold in both splits
  cat("adjusted Rand-ish co-membership overlap:\n")
  tb <- table(m$fp, m$fb); print(tb)
}
# production is untouched
stopifnot(identical(readRDS("model/artifacts/folds.rds"), prod))
cat("\nOK: folds_b.rds and folds_c.rds written; folds.rds unchanged\n")
