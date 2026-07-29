# Auto-resolve data file paths: prefer this year's real 2026 competition
# files if present, else fall back to last year's train2024/test2024 (used
# to build + validate the pipeline before the 2026 dataset was released).
#
# Once the real train.csv / test.csv / sample_submission.csv are dropped
# into ../data/, every script below switches automatically -- no path edits
# needed. Row counts in the 2026 brief (21565 train / 4997 test) match last
# year's files exactly, which suggests (but does NOT guarantee) it's the
# same underlying dataset -- re-validate once the real files are in before
# trusting any score.

resolve_train_path <- function() {
  if (file.exists("../data/train.csv")) "../data/train.csv" else "../data/train2024.csv"
}

resolve_test_path <- function() {
  if (file.exists("../data/test.csv")) "../data/test.csv" else "../data/test2024.csv"
}

resolve_sample_submission_path <- function() {
  if (file.exists("../data/sample_submission.csv")) "../data/sample_submission.csv" else NA
}

using_real_2026_data <- function() {
  file.exists("../data/train.csv")
}

check_data_source <- function() {
  if (using_real_2026_data()) {
    cat(">>> Using REAL 2026 competition data (data/train.csv, data/test.csv)\n")
  } else {
    cat(">>> WARNING: data/train.csv not found yet -- using last year's train2024.csv/",
        "test2024.csv as a placeholder. Scores below are NOT this year's real numbers.\n", sep = "")
  }
}
