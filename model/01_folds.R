suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
tasks <- unique(long[is_test == FALSE, .(No, Case)])
tasks[, fold := make_case_folds(Case, k = 5, seed = 42)]
stopifnot(nrow(tasks) == 21565, all(tasks[, uniqueN(fold), by = Case]$V1 == 1))
print(tasks[, .(tasks = .N, respondents = uniqueN(Case)), keyby = fold])
saveRDS(tasks, "model/artifacts/folds.rds")
cat("OK: folds written\n")
