test_env <- new.env(parent = globalenv())
source("experiments/iter36_emstart/evaluate.R", local = test_env)
gate_decision <- test_env$gate_decision
nested_blend <- test_env$nested_blend

good <- list(start_sd = .0012, primary_gain = .0004, weighted_gain = .0002,
             b_gain = .0003, replication = .75,
             fold_gain = c(rep(.0002, 8), -.0002, -.0003),
             clustered_gain = .0003)
d <- gate_decision(good)
stopifnot(d$pass, all(unlist(d$gates)))

x <- good; x$start_sd <- .0009
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$material_variance)
x <- good; x$b_gain <- -.0001
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$independent_positive)
x <- good; x$replication <- .49
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$replication_half)
x <- good; x$fold_gain <- c(rep(.0002, 6), rep(-.0002, 4))
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$seven_folds)
x <- good; x$fold_gain[10] <- -.00101
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$no_large_fold_loss)

y <- rep(1:4, 5)
fold <- rep(1:5, each = 4)
uniform <- matrix(.25, nrow = length(y), ncol = 4)
nb <- nested_blend(uniform, uniform, y, fold)
stopifnot(abs(nb$score - log(4)) < 1e-12)
stopifnot(length(nb$fold_loss) == 5L, max(abs(nb$pred - .25)) < 1e-12)

cat("test_evaluate.R: OK\n")
