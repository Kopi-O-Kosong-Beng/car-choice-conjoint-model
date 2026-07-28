lc_experiment_config <- function(
    split,
    random_seed_raw = Sys.getenv("LC_RANDOM_START_SEED", "4242"),
    output_raw = Sys.getenv("LC_EXPERIMENT_OUT", ""),
    skip_full_test_raw = Sys.getenv("LC_SKIP_FULL_TEST", "0")) {
  seed <- suppressWarnings(as.integer(random_seed_raw))
  output <- trimws(output_raw)
  skip <- identical(skip_full_test_raw, "1")
  if (length(seed) != 1L || is.na(seed))
    stop("LC_RANDOM_START_SEED must be an integer")
  if (skip && !nzchar(output))
    stop("LC_SKIP_FULL_TEST requires LC_EXPERIMENT_OUT")
  if (skip && split != "b")
    stop("LC_SKIP_FULL_TEST is allowed only for split b")
  list(random_seed = seed, output = output, enabled = nzchar(output),
       skip_full_test = skip)
}

write_lc_outputs <- function(oof, test, config, settings, split,
                             historical_oof, historical_test) {
  if (config$enabled) {
    bundle <- list(seed = config$random_seed, split = split, settings = settings,
                   oof = oof, test = test)
    save_experiment_rds(bundle, config$output)
    return(invisible(config$output))
  }
  saveRDS(oof, historical_oof)
  saveRDS(test, historical_test)
  invisible(c(historical_oof, historical_test))
}

install_lc_child_env <- function(config) {
  values <- list(
    LC_RANDOM_START_SEED = as.character(config$random_seed),
    LC_EXPERIMENT_OUT = config$output,
    LC_SKIP_FULL_TEST = if (config$skip_full_test) "1" else "0")
  invisible(do.call(Sys.setenv, values))
}
