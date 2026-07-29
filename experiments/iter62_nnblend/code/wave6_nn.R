# WAVE 6 -- listwise neural network (torch), the last untried model family.
#
# Same structure as the winning tree model: ONE shared utility function
# scores every alternative; softmax over each task's 4 utilities; weighted
# cross-entropy with the segment importance weights. The function class is
# an MLP instead of boosted trees:
#   - the attr-level one-hot block acts as learned level EMBEDDINGS
#     (first linear layer over one-hots == embedding lookup),
#   - smooth interpolation where trees make staircases -- the property that
#     matters most for the 107-luxury-respondent extrapolation problem.
#
# Uses the SAME long design matrix as the validated tree pipeline (built by
# the caller), so features/weights/folds are identical between families and
# any difference is attributable to the function class alone.

library(torch)

nn_utility_net <- nn_module(
  initialize = function(n_in, width = 64, dropout = 0.2) {
    self$net <- nn_sequential(
      nn_linear(n_in, width), nn_relu(), nn_dropout(dropout),
      nn_linear(width, 32), nn_relu(), nn_dropout(dropout),
      nn_linear(32, 1)
    )
  },
  forward = function(x) self$net(x)$squeeze(2)
)

# Fit on long-format fit rows; return probability matrix (n_pred_tasks x 4)
# for the prediction long matrix. Standardization uses fit-row stats only.
nn_fit_predict <- function(X_fit, y_fit_long, w_fit_task, X_pred,
                           width = 64, dropout = 0.2, epochs = 30,
                           lr = 1e-3, weight_decay = 1e-4,
                           batch_tasks = 512, seed = 1) {
  torch_manual_seed(seed); set.seed(seed)

  mu <- colMeans(X_fit); sdv <- apply(X_fit, 2, sd); sdv[sdv < 1e-8] <- 1
  Xf <- torch_tensor(sweep(sweep(X_fit, 2, mu, "-"), 2, sdv, "/"),
                     dtype = torch_float())
  Xp <- torch_tensor(sweep(sweep(X_pred, 2, mu, "-"), 2, sdv, "/"),
                     dtype = torch_float())

  n_task <- nrow(X_fit) / 4L
  stopifnot(n_task == floor(n_task))
  # chosen alternative per task (1..4) from the long 0/1 labels
  chosen <- max.col(matrix(y_fit_long, ncol = 4, byrow = TRUE))
  target <- torch_tensor(as.integer(chosen), dtype = torch_long())
  wts <- torch_tensor(as.numeric(w_fit_task), dtype = torch_float())

  model <- nn_utility_net(ncol(X_fit), width, dropout)
  opt <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)

  model$train()
  for (ep in seq_len(epochs)) {
    perm <- sample(n_task)
    for (start in seq(1, n_task, by = batch_tasks)) {
      tb <- perm[start:min(start + batch_tasks - 1, n_task)]
      rows <- as.vector(vapply(tb, function(t) (t - 1L) * 4L + 1:4, integer(4)))
      opt$zero_grad()
      u <- model(Xf[rows, ])$view(c(length(tb), 4))
      loss_vec <- nnf_cross_entropy(u, target[tb], reduction = "none")
      loss <- (loss_vec * wts[tb])$sum() / wts[tb]$sum()
      loss$backward()
      opt$step()
    }
  }

  model$eval()
  with_no_grad({
    n_pred <- nrow(X_pred) / 4L
    out <- matrix(NA_real_, n_pred, 4)
    idx <- seq(1, nrow(X_pred), by = 40000)
    # chunked forward pass to bound memory
    u_all <- numeric(nrow(X_pred))
    for (s in idx) {
      e <- min(s + 39999, nrow(X_pred))
      u_all[s:e] <- as.numeric(model(Xp[s:e, ]))
    }
    um <- matrix(u_all, ncol = 4, byrow = TRUE)
    um <- um - apply(um, 1, max)
    eu <- exp(um)
    out <- eu / rowSums(eu)
  })
  out
}
