suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")
source("experiments/iter36_emstart/runner_config.R")

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% 1:2) stop("usage: run_seed.R <seed> [b]")
seed <- as.integer(args[1])
split <- if (length(args) == 2) args[2] else ""
if (!seed %in% EM_SEEDS) stop("seed is not pre-registered")
if (!split %in% c("", "b")) stop("split must be b")

long <- readRDS("model/artifacts/long.rds")
tasks <- unique(long[, .(No, is_test)], by = "No")
expected <- list(train = sort(tasks[is_test == FALSE, No]),
                 test = sort(tasks[is_test == TRUE, No]))
out <- bundle_path(seed, split)
expected_settings <- settings_fingerprint(
  seed, split, unname(tools::md5sum("experiments/iter25_taskpos/run.R")),
  3L, 25L, 1e-5)
if (file.exists(out)) {
  validate_bundle(readRDS(out), seed, split, expected, require_test = split == "",
                  expected_settings = expected_settings)
  cat("valid bundle already exists; skipping:", out, "\n")
  quit(save = "no")
}

runner <- file.path(R.home("bin"),
                    if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
runner_args <- c("experiments/iter25_taskpos/run.R", "3", "both",
                 if (nzchar(split)) split else character())
child_config <- lc_experiment_config(
  split,
  random_seed_raw = as.character(seed),
  output_raw = normalizePath(out, winslash = "/", mustWork = FALSE),
  skip_full_test_raw = if (split == "b") "1" else "0")
install_lc_child_env(child_config)
logfile <- file.path(ITER36_DIR, sprintf("seed_%04d%s.log", seed,
                                        if (nzchar(split)) "_b" else ""))
status <- system2(runner, runner_args, stdout = logfile, stderr = logfile)
if (status != 0L) stop("latent-class runner failed; inspect ", logfile)
validate_bundle(readRDS(out), seed, split, expected, require_test = split == "",
                expected_settings = expected_settings)
cat("completed and validated:", out, "\n")
