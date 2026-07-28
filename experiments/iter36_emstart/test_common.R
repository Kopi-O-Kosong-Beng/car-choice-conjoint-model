suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")
source("experiments/iter36_emstart/runner_config.R")

expect_error <- function(expr) {
  ok <- FALSE
  tryCatch(force(expr), error = function(e) ok <<- TRUE)
  stopifnot(ok)
}

stopifnot(identical(EM_SEEDS, c(4242L, 1103L, 2207L, 3301L, 4409L)))
stopifnot(identical(bundle_path(1103L, ""), file.path(
  "experiments", "iter36_emstart", "artifacts", "seed_1103.rds")))
stopifnot(identical(bundle_path(1103L, "b"), file.path(
  "experiments", "iter36_emstart", "artifacts", "seed_1103_b.rds")))

nos <- 1:3
p <- data.table(No = nos, p1 = c(.1, .2, .3), p2 = .2, p3 = .3,
                p4 = c(.4, .3, .2))
p[, p1 := 1 - p2 - p3 - p4]
v <- validate_pred(p, nos, "synthetic")
stopifnot(identical(v$No, nos), max(abs(rowSums(v[, .(p1,p2,p3,p4)]) - 1)) < 1e-12)

bad_order <- copy(p)[3:1]
expect_error(validate_pred(bad_order, nos, "bad order"))
bad_sum <- copy(p); bad_sum[1, p1 := p1 + .01]
expect_error(validate_pred(bad_sum, nos, "bad sum"))
bad_zero <- copy(p); bad_zero[1, `:=`(p1 = 0, p4 = p4 + p1)]
expect_error(validate_pred(bad_zero, nos, "bad zero"))

m <- mean_pred(list(p, p), nos, "mean")
stopifnot(max(abs(as.matrix(m[, .(p1,p2,p3,p4)]) -
                  as.matrix(p[, .(p1,p2,p3,p4)]))) < 1e-12)

b <- list(seed = 1103L, split = "",
          settings = settings_fingerprint(1103L, "", "test-md5", 3L, 25L, 1e-5),
          oof = p, test = p)
vb <- validate_bundle(b, 1103L, "", list(train = nos, test = nos), TRUE)
stopifnot(vb$seed == 1103L, nrow(vb$oof) == 3L, nrow(vb$test) == 3L)
expect_error(validate_bundle(b, 2207L, "", list(train = nos, test = nos), TRUE))

default_cfg <- lc_experiment_config("", "4242", "", "0")
stopifnot(default_cfg$random_seed == 4242L, !default_cfg$enabled,
          !default_cfg$skip_full_test)
fold_b_cfg <- lc_experiment_config("b", "1103", "bundle.rds", "1")
stopifnot(fold_b_cfg$random_seed == 1103L, fold_b_cfg$enabled,
          fold_b_cfg$skip_full_test)
expect_error(lc_experiment_config("", "1103", "bundle.rds", "1"))
expect_error(lc_experiment_config("b", "1103", "", "1"))
expect_error(lc_experiment_config("b", "abc", "bundle.rds", "1"))

td <- tempfile("iter36-output-")
dir.create(td)
exp_path <- file.path(td, "bundle.rds")
hist_oof <- file.path(td, "historical-oof.rds")
hist_test <- file.path(td, "historical-test.rds")
cfg <- lc_experiment_config("", "1103", exp_path, "0")
write_lc_outputs(
  oof = p, test = p, config = cfg,
  settings = settings_fingerprint(1103L, "", "test-md5", 3L, 25L, 1e-5),
  split = "", historical_oof = hist_oof, historical_test = hist_test)
stopifnot(file.exists(exp_path), !file.exists(hist_oof), !file.exists(hist_test))
written <- readRDS(exp_path)
stopifnot(written$seed == 1103L, written$split == "", nrow(written$oof) == 3L)
expect_error(write_lc_outputs(
  oof = p, test = p, config = cfg, settings = written$settings, split = "",
  historical_oof = hist_oof, historical_test = hist_test))

cat("test_common.R: OK\n")
