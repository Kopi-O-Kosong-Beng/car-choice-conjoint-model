# Case-based 5x5 CV framework matching the senior team's ensemble strategy,
# made fully panel-aware (all splits, including inner CV folds, group by Case
# so no individual leaks between train/val).

ENSEMBLE_SEEDS <- c(52, 110, 47, 9, 100)
N_FOLDS <- 5

# 70/30 Case-based split. Returns list(train_idx, val_idx) (1-based R indices).
case_split_70_30 <- function(case, seed = 123) {
  set.seed(seed)
  all_cases <- unique(case)
  n_train <- floor(0.7 * length(all_cases))
  train_cases <- sample(all_cases, n_train)
  train_mask <- case %in% train_cases
  list(train_idx = which(train_mask), val_idx = which(!train_mask))
}

# Panel-aware K-fold split, grouped by Case. Returns a list of n_splits
# list(train_idx, val_idx) pairs (1-based R indices into `case`).
case_kfold <- function(case, n_splits, seed) {
  set.seed(seed)
  unique_cases <- sort(unique(case))
  shuffled <- sample(unique_cases, length(unique_cases))
  fold_of_case <- setNames(rep(1:n_splits, length.out = length(shuffled)), shuffled)
  fold_id <- fold_of_case[as.character(case)]
  lapply(1:n_splits, function(k) {
    val_mask <- fold_id == k
    list(train_idx = which(!val_mask), val_idx = which(val_mask))
  })
}

# Multi-class log-loss matching the R/Python -log(p_chosen) convention.
# y_true: 0-based integer labels (0..3). probs: n x 4 matrix, column j = P(class j-1).
multiclass_log_loss <- function(y_true, probs, eps = 1e-15) {
  n <- length(y_true)
  chosen <- probs[cbind(seq_len(n), y_true + 1L)]
  chosen <- pmin(pmax(chosen, eps), 1 - eps)
  -mean(log(chosen))
}

top1_accuracy <- function(y_true, probs) {
  pred <- apply(probs, 1, which.max) - 1L
  mean(pred == y_true)
}
