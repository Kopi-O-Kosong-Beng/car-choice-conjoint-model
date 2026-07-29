# V10 -- add SURVEY-FATIGUE features (Task position).
#
# Discovery: respondents grow steadily more likely to pick the outside option
# ("none of these") as they work through their 19 tasks:
#     tasks 1-6   -> 26.1% choose Ch4
#     tasks 7-13  -> 31.2%
#     tasks 14-19 -> 33.3%
#     corr(Task, chose_outside) = 0.065, p = 1.5e-21
#
# The choice sets are randomised, so NOTHING else in the feature set carries
# this information -- and `Task` was never included as a predictor. The model
# therefore predicts a flat ~30% outside-option rate at every position.
# `Task` is present in test.csv too, so this is legitimately usable.
#
# Features added:
#   Task            raw position 1..19
#   Task_sq         curvature (the drift flattens out later)
#   Task_late       indicator for the fatigued back half
#   Task_x_Pricavg  fatigue interacted with price level
#   Task_x_BestVPP  fatigue interacted with deal quality (V10B only)

source("data.R")

add_task_features <- function(df) {
  df$Task_sq        <- df$Task^2
  df$Task_late      <- as.integer(df$Task >= 10)
  df$Task_x_Priceavg <- df$Task * df$Price_avg
  df
}

TASK_COLS <- c("Task", "Task_sq", "Task_late", "Task_x_Priceavg")

# V8 + task
feature_cols_v10 <- function() c(feature_cols(), TASK_COLS)

build_xgb_matrix_v10 <- function(df) {
  cols <- feature_cols_v10()
  X <- df[, cols, drop = FALSE]
  for (c in cols) if (is.character(X[[c]])) X[[c]] <- as.numeric(as.factor(X[[c]]))
  as.matrix(X)
}

load_train_v10 <- function(path) {
  tr <- read.csv(path, stringsAsFactors = FALSE)
  tr <- add_task_features(add_engineered(tr))
  list(df = tr, y = make_label(tr), case = tr$Case)
}

load_test_v10 <- function(path) {
  add_task_features(add_engineered(read.csv(path, stringsAsFactors = FALSE)))
}
