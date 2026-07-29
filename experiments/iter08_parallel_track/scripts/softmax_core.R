# Exact conditional-logit (softmax) objective for the shared-utility long format.
# Extracted so exp_softmax_objective.R and exp_softmax_followup.R share ONE
# implementation - a second copy would be a place for the two to silently diverge.
#
# Long rows are ordered task1-alt1..4, task2-alt1..4, ..., so reshaping the
# prediction vector with byrow=TRUE recovers the task x alternative matrix.
#   p_ij = exp(f_ij) / sum_k exp(f_ik);  grad = p - y;  hess = p(1-p)

softmax_rows <- function(F) {
  F <- F - apply(F, 1, max)
  E <- exp(F); E / rowSums(E)
}

make_softmax_obj <- function(n_tasks) {
  function(preds, dtrain) {
    y <- getinfo(dtrain, "label")
    P <- softmax_rows(matrix(preds, nrow = n_tasks, ncol = 4, byrow = TRUE))
    Y <- matrix(y,     nrow = n_tasks, ncol = 4, byrow = TRUE)
    list(grad = as.vector(t(P - Y)),
         hess = as.vector(t(pmax(P * (1 - P), 1e-6))))   # floor keeps splits stable
  }
}

make_softmax_eval <- function(n_tasks) {
  function(preds, dtrain) {
    y <- getinfo(dtrain, "label")
    P <- softmax_rows(matrix(preds, nrow = n_tasks, ncol = 4, byrow = TRUE))
    Y <- matrix(y,     nrow = n_tasks, ncol = 4, byrow = TRUE)
    list(metric = "task_logloss", value = -mean(log(pmax(rowSums(P * Y), 1e-15))))
  }
}

su_fit_softmax <- function(fit_df, pred_df, params, nrounds, seeds = MATE_SEEDS) {
  dm <- make_demo(fit_df, pred_df)
  ff <- build_choice_feats(fit_df, dm$train); pf <- build_choice_feats(pred_df, dm$test)
  y_long <- as.vector(sapply(fit_df$Choice, function(c) as.integer(1:4 == c)))
  nf <- nrow(fit_df); np <- nrow(pred_df)
  dtr <- xgb.DMatrix(ff$long, label = y_long); dpr <- xgb.DMatrix(pf$long)
  acc <- matrix(0, np, 4)
  for (s in seeds) {
    m <- xgb.train(c(params, list(seed = s, base_score = 0)), dtr, nrounds = nrounds,
                   obj = make_softmax_obj(nf), custom_metric = make_softmax_eval(nf),
                   verbose = 0)
    acc <- acc + softmax_rows(matrix(predict(m, dpr), nrow = np, ncol = 4, byrow = TRUE))
  }
  acc / length(seeds)
}

# ---- Gradient unit test (row-order bugs have cost this project real time twice) ----
# On a 3-task toy problem the analytic gradient must match a numerical one.
if (identical(Sys.getenv("SOFTMAX_SELFTEST"), "1")) {
  set.seed(1); nt <- 3
  y <- as.vector(sapply(c(1, 4, 2), function(c) as.integer(1:4 == c)))
  f <- rnorm(nt * 4)
  d <- xgb.DMatrix(matrix(rnorm(nt * 4 * 2), nrow = nt * 4), label = y)
  g <- make_softmax_obj(nt)(f, d)$grad
  nll <- function(v) {
    P <- softmax_rows(matrix(v, nrow = nt, ncol = 4, byrow = TRUE))
    Y <- matrix(y, nrow = nt, ncol = 4, byrow = TRUE)
    -sum(log(pmax(rowSums(P * Y), 1e-15)))
  }
  num <- sapply(seq_along(f), function(i) {
    e <- rep(0, length(f)); e[i] <- 1e-6; (nll(f + e) - nll(f - e)) / 2e-6 })
  cat(sprintf("softmax gradient self-test: max |analytic - numeric| = %.3e  %s\n",
              max(abs(g - num)), if (max(abs(g - num)) < 1e-5) "PASS" else "FAIL"))
}
