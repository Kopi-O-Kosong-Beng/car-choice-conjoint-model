suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")

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

cat("test_common.R: OK\n")
