# WAVE 18 -- pooled 4-seed OOF reference for the NN BLEND (w=0.25 on the
# standard tree recipe), for the head-on-blend confirmation.

source("paths.R"); source("data.R"); source("cv.R")
source("segment_framework.R"); source("data_v10.R"); source("data_listwise.R")
source("shift_utils.R"); source("wave5_mf.R"); source("wave6_nn.R")

SEEDS <- c(52, 110, 47, 1001003); W_NN <- 0.25
ATTRIBUTES <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
                "PP","KA","SC","TS","NV","MA","LB","AF","HU")
PARAMS <- list(max_depth = 3, eta = 0.1, min_child_weight = 2,
               subsample = 0.8, colsample_bytree = 0.8, gamma = 0.2, base_score = 0)
NROUNDS <- 250L
raw <- read.csv(resolve_train_path(), stringsAsFactors = FALSE)
train <- add_task_features(add_engineered(raw))
y <- make_label(train); case_id <- train$Case; segment <- train$segment
y_long <- long_labels(y)
tw <- segment_importance_weights(segment, 0.10); w_long <- rep(tw, each = 4)
maximum_level <- sapply(ATTRIBUTES, function(a) max(as.matrix(train[, paste0(a, 1:3)])))
build_attr_dummies <- function(data) {
  n <- nrow(data); columns <- list()
  for (attribute in ATTRIBUTES) for (level in seq_len(maximum_level[[attribute]])) {
    value <- matrix(0L, nrow = 4 * n, ncol = 1)
    for (alternative in 1:4) {
      ind <- if (alternative == 4) rep(0L, n) else
        as.integer(data[[paste0(attribute, alternative)]] == level)
      value[seq(alternative, by = 4, length.out = n), 1] <- ind
    }
    columns[[paste0(attribute, "_L", level)]] <- value
  }
  out <- do.call(cbind, columns); colnames(out) <- names(columns); out
}
BLAD <- cbind(build_long(train), build_attr_dummies(train))
long_rows <- function(rows) as.vector(vapply(rows, function(r) (r - 1L) * 4L + 1:4, integer(4)))
geo <- function(a, b, w) {
  z <- exp(w * log(pmax(a, 1e-15)) + (1 - w) * log(pmax(b, 1e-15))); z / rowSums(z)
}

acc <- matrix(0, nrow(train), 4)
for (s in SEEDS) {
  folds <- case_kfold(case_id, 5, s)
  for (fi in seq_along(folds)) {
    fr <- folds[[fi]]$train_idx; vr <- folds[[fi]]$val_idx
    mf <- fit_mf_model(train[fr, , drop = FALSE], rank = MF_RANK)
    Xall <- cbind(BLAD, predict_mf_features(mf, train))
    tree <- listwise_fit_predict(
      Xall[long_rows(fr), , drop = FALSE], y_long[long_rows(fr)],
      w_long[long_rows(fr)], Xall[long_rows(vr), , drop = FALSE],
      PARAMS, NROUNDS, seed = s * 100 + fi)
    nn <- matrix(0, length(vr), 4)
    for (init in 1:3) nn <- nn + nn_fit_predict(
      Xall[long_rows(fr), , drop = FALSE], y_long[long_rows(fr)], tw[fr],
      Xall[long_rows(vr), , drop = FALSE], width = 64, dropout = 0.35,
      epochs = 20, lr = 1e-3, weight_decay = 1e-4, seed = s * 1000 + fi * 10 + init)
    acc[vr, ] <- acc[vr, ] + geo(nn / 3, tree, W_NN)
    cat("blend ref seed", s, "fold", fi, "done", format(Sys.time()), "\n"); flush.console()
  }
}
P <- acc / length(SEEDS); P <- P / rowSums(P)
saveRDS(P, "../outputs/wave18_blend_ref.rds")
cat("blend 4-seed reference testmix:",
    sprintf("%.5f", testmix_log_loss(y, P, segment)), "\n")
cat("BLENDREF_DONE\n")
