suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")

long <- readRDS("model/artifacts/long.rds")
tasks <- unique(long[, .(No, is_test)], by = "No")
expected <- list(train = sort(tasks[is_test == FALSE, No]),
                 test = sort(tasks[is_test == TRUE, No]))

import_one <- function(split) {
  out <- bundle_path(4242L, split)
  if (file.exists(out)) {
    validate_bundle(readRDS(out), 4242L, split, expected, require_test = split == "")
    return(invisible(out))
  }
  suffix <- if (nzchar(split)) "_b" else ""
  oof <- readRDS(sprintf("model/artifacts/oof_lcmnl3_both%s.rds", suffix))
  test <- if (nzchar(split)) NULL else
    readRDS("model/artifacts/test_lcmnl3_both.rds")
  bundle <- list(
    seed = 4242L,
    split = split,
    settings = settings_fingerprint(
      4242L, split, unname(tools::md5sum("experiments/iter25_taskpos/run.R")),
      3L, 25L, 1e-5),
    oof = validate_pred(oof[order(No)], expected$train, "reference oof"),
    test = if (is.null(test)) NULL else
      validate_pred(test[order(No)], expected$test, "reference test"))
  save_experiment_rds(bundle, out)
}

import_one("")
import_one("b")
cat("import_reference.R: OK\n")
