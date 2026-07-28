test_env <- new.env(parent = globalenv())
source("experiments/iter36_emstart/finalize.R", local = test_env)
tilt_p4 <- test_env$tilt_p4
select_base <- test_env$select_base

P <- rbind(c(.10,.20,.30,.40), c(.40,.30,.20,.10), c(.25,.25,.25,.25))
target <- .30
Q <- tilt_p4(P, target)
stopifnot(abs(mean(Q[,4]) - target) < 1e-12)
stopifnot(max(abs(rowSums(Q) - 1)) < 1e-12, all(Q > 0))
stopifnot(max(abs((Q[,1] / Q[,2]) - (P[,1] / P[,2]))) < 1e-12)
stopifnot(identical(order(Q[,4]), order(P[,4])))
stopifnot(identical(select_base(FALSE),
                    "model/artifacts/test_blend.rds"))
stopifnot(identical(select_base(TRUE),
                    "model/artifacts/test_blend_emstart.rds"))

cat("test_finalize.R: OK\n")
