# NNBLEND FIXEDCAL -- rebuild of the wave6 NN-blend base with the
# calibration SELF-REFERENCED on the blend family's own pooled OOF
# (wave18_blend_ref.rds), fixing the reference mismatch found 29 Jul:
# the original submission_wave6_nnblend.csv calibrated against the old
# r250 tree reference, while its validating confirmation
# (wave18_gam_on_blend.R: 4/5, mean -0.0031, bootstrap P 0.980) ran the
# chain against wave18_blend_ref.rds. Same a05-bug pattern, same fix.
# alpha=0.10 tree arm (submit_wave5_mf_partial.rds, 100 models) blended
# w=0.25 with the NN arm (submit_wave6_blend_partial.rds, 50 units).
# No models are refit here.

source("paths.R")
source("data.R")
source("cv.R")
source("segment_framework.R")
source("data_v10.R")
source("data_listwise.R")
source("shift_utils.R")
source("write_submission.R")

BLEND_W <- 0.25            # frozen rung-4 weight, w on the NN side
CH4_STRENGTH <- 0.85
LUXURY_TEMPERATURE <- 1.30
GLOBAL_PENALTY <- 1
LUXURY_PENALTY <- 4
LUXURY_SEGMENTS <- c(
  "Prestige Luxury Sedan",
  "Midsize Luxury Utility segements"
)

TREE_PARTIAL_PATH <- "../outputs/submit_wave5_mf_partial.rds"
NN_PARTIAL_PATH   <- "../outputs/submit_wave6_blend_partial.rds"
REFERENCE_PATH    <- "../outputs/wave18_blend_ref.rds"
OUTPUT_PATH  <- "../submissions/submission_wave6_nnblend_FIXEDCAL.csv"
COMPARE_PATH <- "../submissions/submission_wave6_nnblend.csv"

raw_train <- read.csv(resolve_train_path(), stringsAsFactors = FALSE)
raw_test <- read.csv(resolve_test_path(), stringsAsFactors = FALSE)
train <- add_task_features(add_engineered(raw_train))
test <- add_task_features(add_engineered(raw_test))
y <- make_label(train)
segment <- train$segment
test_segment <- test$segment

train_base_long <- build_long(train, ref = test)
test_base_long <- build_long(test, ref = train)

# ------------------------------------------------ RAW ARMS (no refitting) --
tree_partial <- readRDS(TREE_PARTIAL_PATH)
stopifnot(length(tree_partial$completed_seeds) * 5L == tree_partial$model_count,
          tree_partial$model_count >= 50L)
tree_prob_raw <- tree_partial$prediction_sum / tree_partial$model_count
cat(">>> tree a05 arm:", tree_partial$model_count, "models (seeds:",
    length(tree_partial$completed_seeds), ")\n")

nn_partial <- readRDS(NN_PARTIAL_PATH)
stopifnot(nn_partial$model_count == 50L)
nn_prob_raw <- nn_partial$prediction_sum / nn_partial$model_count
cat(">>> NN arm:", nn_partial$model_count, "seed-fold units (x3 inits each)\n")

geo_blend <- function(nn_p, tree_p, w) {
  z <- exp(w * log(pmax(nn_p, 1e-15)) + (1 - w) * log(pmax(tree_p, 1e-15)))
  z / rowSums(z)
}
probability <- geo_blend(nn_prob_raw, tree_prob_raw, BLEND_W)

# ------------- CALIBRATION STACK (verbatim submit_wave6_blend.R pattern, ----
# ------------- reference swapped to this family's own pooled OOF) -----------
reference_oof <- readRDS(REFERENCE_PATH)
stopifnot(nrow(reference_oof) == nrow(train))
test_shift <- fit_apply_shifts(
  reference_oof, y, segment,
  probability, test_segment,
  shrink = CH4_STRENGTH
)
probability <- test_shift$probs
train_probability <- fit_apply_shifts(
  reference_oof, y, segment,
  reference_oof, segment,
  shrink = CH4_STRENGTH
)$probs

apply_luxury_temperature <- function(input_probability, segment_value) {
  output <- input_probability
  local_luxury <- segment_value %in% LUXURY_SEGMENTS
  score <- log(pmax(output[local_luxury, , drop = FALSE], 1e-15)) /
    LUXURY_TEMPERATURE
  exponentiated <- exp(score - apply(score, 1, max))
  output[local_luxury, ] <- exponentiated / rowSums(exponentiated)
  output
}
probability <- apply_luxury_temperature(probability, test_segment)
train_probability <- apply_luxury_temperature(train_probability, segment)

build_residual_feature <- function(data, long_design) {
  n <- nrow(data)
  price <- as.matrix(data[, paste0("Price", 1:3)])
  mean_price <- rowMeans(price)
  relative_price <- matrix(0, nrow = 4 * n, ncol = 1)
  for (alternative in 1:3) {
    relative_price[seq(alternative, by = 4, length.out = n), 1] <-
      price[, alternative] - mean_price
  }
  cbind(
    relative_price = as.numeric(relative_price),
    outside_constant = as.numeric(long_design[, "alt_id"] == 4),
    total_vs_best = long_design[, "total_vs_bestriv"]
  )
}
train_feature <- build_residual_feature(train, train_base_long)
test_feature <- build_residual_feature(test, test_base_long)
feature_scale <- sqrt(colMeans(train_feature^2))
feature_scale[!is.finite(feature_scale) | feature_scale < 1e-8] <- 1

make_residual_design <- function(feature, segment_value) {
  base <- sweep(feature, 2, feature_scale, "/")
  cbind(base, base * rep(
    as.numeric(segment_value %in% LUXURY_SEGMENTS), each = 4
  ))
}
train_residual_design <- make_residual_design(train_feature, segment)
test_residual_design <- make_residual_design(test_feature, test_segment)
parameter_names <- c(
  paste0("global_", colnames(train_feature)),
  paste0("luxury_deviation_", colnames(train_feature))
)
colnames(train_residual_design) <- parameter_names
colnames(test_residual_design) <- parameter_names

testmix_task_weights <- function(segment_value) {
  weight <- numeric(length(segment_value))
  for (segment_name in names(TEST_SEGMENT_SHARE)) {
    rows <- which(segment_value == segment_name)
    if (length(rows) && TEST_SEGMENT_SHARE[[segment_name]] > 0) {
      weight[rows] <- TEST_SEGMENT_SHARE[[segment_name]] / length(rows)
    }
  }
  weight
}
residual_weight <- testmix_task_weights(segment)
residual_penalty <- c(rep(GLOBAL_PENALTY, 3), rep(LUXURY_PENALTY, 3))
offset <- log(pmax(train_probability, 1e-15))
cached_theta <- NULL
cached_result <- NULL
evaluate <- function(theta) {
  if (!is.null(cached_theta) && identical(theta, cached_theta)) {
    return(cached_result)
  }
  cached_theta <<- theta
  utility <- offset + matrix(
    as.vector(train_residual_design %*% theta), ncol = 4, byrow = TRUE
  )
  utility <- utility - apply(utility, 1, max)
  exponentiated <- exp(utility)
  fitted_probability <- exponentiated / rowSums(exponentiated)
  chosen_index <- cbind(seq_along(y), y + 1L)
  chosen <- fitted_probability[chosen_index]
  value <- -sum(residual_weight * log(pmax(chosen, 1e-15))) /
    sum(residual_weight) + 0.5 * sum(residual_penalty * theta^2)
  residual <- fitted_probability
  residual[chosen_index] <- residual[chosen_index] - 1
  gradient <- as.vector(crossprod(
    train_residual_design, as.vector(t(residual * residual_weight))
  )) / sum(residual_weight) + residual_penalty * theta
  cached_result <<- list(value = value, gradient = gradient)
  cached_result
}
residual_fit <- optim(
  rep(0, 6),
  fn = function(theta) evaluate(theta)$value,
  gr = function(theta) evaluate(theta)$gradient,
  method = "L-BFGS-B",
  control = list(maxit = 120, factr = 1e7, pgtol = 1e-7)
)
stopifnot(residual_fit$convergence == 0)

test_utility <- log(pmax(probability, 1e-15)) + matrix(
  as.vector(test_residual_design %*% residual_fit$par), ncol = 4, byrow = TRUE
)
test_utility <- test_utility - apply(test_utility, 1, max)
test_exponentiated <- exp(test_utility)
probability <- test_exponentiated / rowSums(test_exponentiated)
write_submission(test, probability, OUTPUT_PATH)

sample_submission <- read.csv(resolve_sample_submission_path(),
                              stringsAsFactors = FALSE)
written <- read.csv(OUTPUT_PATH, stringsAsFactors = FALSE)
written_probability <- as.matrix(written[, paste0("Ch", 1:4)])
stopifnot(
  nrow(written) == 4997L,
  identical(names(written), names(sample_submission)),
  identical(written$No, sample_submission$No),
  all(is.finite(written_probability)),
  all(written_probability > 0),
  max(abs(rowSums(written_probability) - 1)) < 1e-10
)

cat("residual coefficients:\n")
print(setNames(residual_fit$par, parameter_names), digits = 9)
cat(sprintf("\nmean Ch4 overall = %.5f\n", mean(written_probability[, 4])))
print(round(tapply(written_probability[, 4], test_segment, mean), 5))
cat(sprintf("min probability = %.3e\n", min(written_probability)))

prior <- read.csv(COMPARE_PATH, stringsAsFactors = FALSE)
stopifnot(identical(prior$No, written$No))
prior_probability <- as.matrix(prior[, paste0("Ch", 1:4)])
cat(sprintf("\nvs %s:\n  mean abs difference = %.5f\n  argmax agreement    = %.4f\n",
            COMPARE_PATH,
            mean(abs(written_probability - prior_probability)),
            mean(max.col(written_probability) == max.col(prior_probability))))
cat("\nwrote", OUTPUT_PATH, "\n")
cat("NNBLEND_FIXEDCAL_DONE\n")
