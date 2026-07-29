# Shared submission writer + format check against sample_submission.csv
# (when the real file is available). Catches column-name / row-count
# mismatches immediately instead of silently producing a bad Kaggle upload.

write_submission <- function(test_df, probs, out_path) {
  out <- data.frame(
    No  = test_df$No,
    Ch1 = probs[, 1],
    Ch2 = probs[, 2],
    Ch3 = probs[, 3],
    Ch4 = probs[, 4]
  )
  write.csv(out, out_path, row.names = FALSE, quote = FALSE)
  cat(">>> Wrote", out_path, " rows=", nrow(out), "\n")

  sample_path <- resolve_sample_submission_path()
  if (!is.na(sample_path)) {
    sample <- read.csv(sample_path, nrows = 1)
    if (!identical(names(sample), names(out))) {
      cat("!!! WARNING: column names differ from sample_submission.csv\n")
      cat("    mine  :", paste(names(out), collapse = ", "), "\n")
      cat("    sample:", paste(names(sample), collapse = ", "), "\n")
    }
    sample_full <- read.csv(sample_path)
    if (nrow(sample_full) != nrow(out)) {
      cat("!!! WARNING: row count differs from sample_submission.csv (",
          nrow(out), " vs ", nrow(sample_full), ")\n", sep = "")
    }
  } else {
    cat("    (no sample_submission.csv found yet -- format not cross-checked)\n")
  }
  invisible(out)
}
