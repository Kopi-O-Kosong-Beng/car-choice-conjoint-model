# WAVE 6 -- the CONFIRMED GAM outside-option head RE-CONFIRMED on the
# MF-augmented base (the wave-5 cold-start model that supersedes the banked
# 1.195 recipe).
#
# Everything about the head is inherited FROZEN from its original adoption
# (spec stable_k4_g18, p4 share 0.25, gates): the only question here is
# whether the head still adds on the better base. Accordingly:
#   - reference OOF = pooled wave-5 TREATMENT (tree+MF) phase A+B matrices,
#     replacing the old R250 selection OOF;
#   - the incumbent chain per split (Ch4 shift .85 -> luxury T1.30 ->
#     FIXEDRESID) is unchanged, refit per split from the new reference;
#   - phase A below is informational only (gate disarmed: nothing is being
#     selected); the decision rests entirely on FIVE never-before-used
#     confirmation splits with the original gates: mean gain >=.001, wins
#     >=4/5, respondent bootstrap P>=.95, 90% interval upper endpoint < 0.

source("paths.R")
source("data.R")
source("cv.R")
source("segment_framework.R")
source("data_v10.R")
source("data_listwise.R")
source("shift_utils.R")

library(mgcv)

PHASE_A_SPLIT_SEEDS <- c(1011009, 1012017, 1013019)
CONFIRMATION_SPLIT_SEEDS <- c(
  1021001, 1022003, 1023009, 1024013, 1025019
)
# Frozen after the coarse screen because it was the only share with 3/3 wins.
# The later bounded refinement may choose smoothness/regularisation only.
P4_SHARE_GRID <- 0.250
ALPHA <- 0.10
CH4_SHRINK <- 0.85
LUXURY_FULL_T <- 1.30
GLOBAL_FIXED_PENALTY <- 1
LUXURY_FIXED_PENALTY <- 4
PHASE_A_MIN_GAIN <- -Inf     # disarmed: nothing is selected in phase A here
PHASE_A_MIN_WINS <- 0L
CONFIRMATION_MIN_GAIN <- 0.001
CONFIRMATION_MIN_WINS <- 4L
BOOTSTRAP_REPLICATES <- 20000L
MIN_BOOTSTRAP_P <- 0.95
LUXURY_SEGMENTS <- c(
  "Prestige Luxury Sedan",
  "Midsize Luxury Utility segements"
)
ATTRIBUTES <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)
OUTPUT_DIRECTORY <- "../outputs/wave18_gam_on_blend"
OUTPUT_PATH <- "../outputs/wave18_gam_on_blend.rds"
XGB_HEAD_RESULT <- "../outputs/xgb_hurdle_p4_module.rds"
dir.create(OUTPUT_DIRECTORY, recursive = TRUE, showWarnings = FALSE)

raw <- read.csv(resolve_train_path(), stringsAsFactors = FALSE)
train <- add_task_features(add_engineered(raw))
y <- make_label(train)
outside <- as.integer(y == 3L)
case_id <- train$Case
segment <- train$segment
luxury <- segment %in% LUXURY_SEGMENTS
alpha_task_weight <- segment_importance_weights(segment, ALPHA)

# Reference = pooled 4-seed NN-blend OOF (wave18_blend_ref.R).
reference_probability <- readRDS("../outputs/wave18_blend_ref.rds")
stopifnot(
  identical(dim(reference_probability), c(nrow(train), 4L)),
  all(is.finite(reference_probability)),
  all(reference_probability > 0),
  max(abs(rowSums(reference_probability) - 1)) < 1e-6
)
reference_probability <- reference_probability /
  rowSums(reference_probability)

long_rows <- function(rows) {
  as.vector(vapply(
    rows,
    function(row) (row - 1L) * 4L + 1:4,
    integer(4)
  ))
}

# Permutation-invariant task summaries. "Safety" is the total categorical
# equipment/safety level across the 19 offered attributes for each real
# alternative; only its best/mean/spread enter the GAM.
price_matrix <- as.matrix(train[, paste0("Price", 1:3)])
safety_matrix <- sapply(1:3, function(alternative) {
  rowSums(as.matrix(
    train[, paste0(ATTRIBUTES, alternative), drop = FALSE]
  ))
})

gam_data <- data.frame(
  outside = outside,
  segment = factor(train$segment),
  region = factor(train$region),
  gender = factor(train$gender),
  Urb = factor(train$Urb),
  ppark = factor(train$ppark),
  night = factor(train$night),
  educ = factor(train$educ),
  Task = as.numeric(train$Task),
  age_value = as.numeric(train$agea),
  log_income = log1p(pmax(as.numeric(train$incomea), 0)),
  log_miles = log1p(pmax(as.numeric(train$milesa), 0)),
  price_min = apply(price_matrix, 1, min),
  price_mean = rowMeans(price_matrix),
  price_spread = apply(price_matrix, 1, max) -
    apply(price_matrix, 1, min),
  safety_best = apply(safety_matrix, 1, max),
  safety_mean = rowMeans(safety_matrix),
  safety_spread = apply(safety_matrix, 1, max) -
    apply(safety_matrix, 1, min)
)
stopifnot(!anyNA(gam_data))

# Four smooths only, each with at most four effective basis dimensions before
# shrinkage. Set summaries remain linear to keep total flexibility bounded.
GAM_FORMULA_K5 <- outside ~
  segment + region + gender + Urb + ppark + night + educ +
  s(Task, k = 5, bs = "cr") +
  s(age_value, k = 5, bs = "cr") +
  s(log_income, k = 5, bs = "cr") +
  s(log_miles, k = 5, bs = "cr") +
  price_min + price_mean + price_spread +
  safety_best + safety_mean + safety_spread

GAM_FORMULA_K4 <- outside ~
  segment + region + gender + Urb + ppark + night + educ +
  s(Task, k = 4, bs = "cr") +
  s(age_value, k = 4, bs = "cr") +
  s(log_income, k = 4, bs = "cr") +
  s(log_miles, k = 4, bs = "cr") +
  price_min + price_mean + price_spread +
  safety_best + safety_mean + safety_spread

# The coarse K5/gamma1.4 fit cleared the initial gate. Refinement is restricted
# to that exact specification and one more strongly regularised/lower-rank
# neighbour before any genuinely fresh confirmation split is opened.
# Spec inherited FROZEN from the original adoption -- no re-selection here.
GAM_SPECS <- list(
  stable_k4_g18 = list(
    key = "stable_k4_g18",
    formula = GAM_FORMULA_K4,
    gamma = 1.8
  )
)

gam_cache_path <- function(stage, key, split_seed) {
  file.path(
    OUTPUT_DIRECTORY,
    paste0(stage, "_", key, "_gam_p4_seed", split_seed, ".rds")
  )
}

fit_gam_split <- function(
    stage,
    spec,
    split_seed,
    fit_rows,
    validation_rows) {
  path <- gam_cache_path(stage, spec$key, split_seed)
  if (file.exists(path)) {
    cached <- readRDS(path)
    if (identical(cached$version, "lowdf_gam_p4_refined_v1") &&
        identical(cached$key, spec$key) &&
        identical(cached$validation_rows, validation_rows) &&
        length(cached$p4) == length(validation_rows)) {
      cat("reusing", path, "\n")
      return(cached)
    }
  }
  fit_data <- gam_data[fit_rows, , drop = FALSE]
  fit_data$case_weight <- alpha_task_weight[fit_rows]
  model <- mgcv::gam(
    formula = spec$formula,
    family = binomial(link = "logit"),
    data = fit_data,
    weights = case_weight,
    method = "REML",
    select = TRUE,
    gamma = spec$gamma
  )
  p4 <- as.numeric(predict(
    model,
    newdata = gam_data[validation_rows, , drop = FALSE],
    type = "response"
  ))
  p4 <- pmin(pmax(p4, 1e-6), 1 - 1e-6)
  model_summary <- summary(model)
  smooth_edf <- if (!is.null(model_summary$s.table)) {
    setNames(
      as.numeric(model_summary$s.table[, "edf"]),
      rownames(model_summary$s.table)
    )
  } else {
    numeric(0)
  }
  result <- list(
    version = "lowdf_gam_p4_refined_v1",
    key = spec$key,
    gamma = spec$gamma,
    stage = stage,
    split_seed = split_seed,
    validation_rows = validation_rows,
    p4 = p4,
    total_edf = sum(model$edf),
    smooth_edf = smooth_edf,
    deviance_explained = model_summary$dev.expl,
    n_coefficients = length(coef(model))
  )
  saveRDS(result, path)
  rm(fit_data, model)
  invisible(gc())
  cat(sprintf(
    paste0(
      "%s %s GAM seed %d | total edf %.3f | smooth edf %.3f | ",
      "deviance explained %.4f\n"
    ),
    stage,
    spec$key,
    split_seed,
    result$total_edf,
    sum(result$smooth_edf),
    result$deviance_explained
  ))
  flush.console()
  result
}

testmix_task_weights <- function(segment_value) {
  weight <- numeric(length(segment_value))
  for (segment_name in names(TEST_SEGMENT_SHARE)) {
    rows <- which(segment_value == segment_name)
    if (length(rows) && TEST_SEGMENT_SHARE[[segment_name]] > 0) {
      weight[rows] <- TEST_SEGMENT_SHARE[[segment_name]] /
        length(rows)
    }
  }
  weight
}

# Exact banked 1.195 incumbent.
shift_pair <- function(fit_rows, validation_rows) {
  fit_shift <- fit_apply_shifts(
    reference_probability[fit_rows, , drop = FALSE],
    y[fit_rows],
    segment[fit_rows],
    reference_probability[fit_rows, , drop = FALSE],
    segment[fit_rows],
    shrink = CH4_SHRINK
  )$probs
  validation_shift <- fit_apply_shifts(
    reference_probability[fit_rows, , drop = FALSE],
    y[fit_rows],
    segment[fit_rows],
    reference_probability[validation_rows, , drop = FALSE],
    segment[validation_rows],
    shrink = CH4_SHRINK
  )$probs
  list(fit = fit_shift, validation = validation_shift)
}

apply_full_luxury_temperature <- function(
    shifted_probability,
    rows) {
  output <- shifted_probability
  local_luxury <- luxury[rows]
  if (!any(local_luxury)) return(output)
  logit <- log(
    pmax(output[local_luxury, , drop = FALSE], 1e-15)
  ) / LUXURY_FULL_T
  logit <- logit - apply(logit, 1, max)
  exponentiated <- exp(logit)
  output[local_luxury, ] <-
    exponentiated / rowSums(exponentiated)
  stopifnot(max(abs(rowSums(output) - 1)) < 1e-10)
  output
}

base_long <- build_long(train)
mean_real_price <- rowMeans(price_matrix)
relative_price <- matrix(0, nrow = 4 * nrow(train), ncol = 1)
for (alternative in 1:3) {
  relative_price[
    seq(alternative, by = 4, length.out = nrow(train)),
    1
  ] <- price_matrix[, alternative] - mean_real_price
}
raw_fixed_feature <- cbind(
  relative_price = as.numeric(relative_price),
  outside_constant = as.numeric(base_long[, "alt_id"] == 4),
  total_vs_best = base_long[, "total_vs_bestriv"]
)
fixed_penalty <- c(
  rep(GLOBAL_FIXED_PENALTY, 3),
  rep(LUXURY_FIXED_PENALTY, 3)
)

make_fixed_design <- function(rows, scale) {
  base <- sweep(
    raw_fixed_feature[long_rows(rows), , drop = FALSE],
    2,
    scale,
    "/"
  )
  cbind(
    base,
    base * rep(as.numeric(luxury[rows]), each = 4)
  )
}

fit_fixed_control <- function(base_probability, rows, fixed_scale) {
  design <- make_fixed_design(rows, fixed_scale)
  label <- y[rows]
  task_weight <- testmix_task_weights(segment[rows])
  cached_theta <- NULL
  cached <- NULL
  evaluate <- function(theta) {
    if (!is.null(cached_theta) && identical(theta, cached_theta)) {
      return(cached)
    }
    cached_theta <<- theta
    utility <- log(pmax(base_probability, 1e-15)) + matrix(
      as.vector(design %*% theta),
      ncol = 4,
      byrow = TRUE
    )
    utility <- utility - apply(utility, 1, max)
    exponentiated <- exp(utility)
    probability <- exponentiated / rowSums(exponentiated)
    chosen <- cbind(seq_along(label), label + 1L)
    value <- -sum(
      task_weight * log(pmax(probability[chosen], 1e-15))
    ) / sum(task_weight) +
      0.5 * sum(fixed_penalty * theta^2)
    residual <- probability
    residual[chosen] <- residual[chosen] - 1
    gradient <- as.vector(crossprod(
      design,
      as.vector(t(residual * task_weight))
    )) / sum(task_weight) + fixed_penalty * theta
    cached <<- list(value = value, gradient = gradient)
    cached
  }
  optimization <- optim(
    rep(0, 6),
    fn = function(theta) evaluate(theta)$value,
    gr = function(theta) evaluate(theta)$gradient,
    method = "L-BFGS-B",
    control = list(maxit = 120, factr = 1e7, pgtol = 1e-7)
  )
  stopifnot(optimization$convergence == 0)
  optimization$par
}

predict_fixed_control <- function(
    base_probability,
    rows,
    fixed_scale,
    coefficient) {
  utility <- log(pmax(base_probability, 1e-15)) + matrix(
    as.vector(
      make_fixed_design(rows, fixed_scale) %*% coefficient
    ),
    ncol = 4,
    byrow = TRUE
  )
  utility <- utility - apply(utility, 1, max)
  exponentiated <- exp(utility)
  exponentiated / rowSums(exponentiated)
}

incumbent_cache_path <- function(stage, split_seed) {
  file.path(
    OUTPUT_DIRECTORY,
    paste0(stage, "_incumbent_seed", split_seed, ".rds")
  )
}

fit_incumbent_split <- function(
    stage,
    split_seed,
    fit_rows,
    validation_rows) {
  path <- incumbent_cache_path(stage, split_seed)
  if (file.exists(path)) {
    cached <- readRDS(path)
    if (identical(
        cached$version,
        "wave18_blendbase_fullt130_fixedresid_v1"
      ) &&
        identical(cached$validation_rows, validation_rows) &&
        identical(
          dim(cached$probability),
          c(length(validation_rows), 4L)
        )) {
      return(cached$probability)
    }
  }
  shifted <- shift_pair(fit_rows, validation_rows)
  base_fit <- apply_full_luxury_temperature(
    shifted$fit, fit_rows
  )
  base_validation <- apply_full_luxury_temperature(
    shifted$validation, validation_rows
  )
  fixed_scale <- sqrt(colMeans(
    raw_fixed_feature[long_rows(fit_rows), , drop = FALSE]^2
  ))
  fixed_scale[!is.finite(fixed_scale) | fixed_scale < 1e-8] <- 1
  coefficient <- fit_fixed_control(
    base_fit, fit_rows, fixed_scale
  )
  probability <- predict_fixed_control(
    base_validation,
    validation_rows,
    fixed_scale,
    coefficient
  )
  saveRDS(
    list(
      version = "wave18_blendbase_fullt130_fixedresid_v1",
      stage = stage,
      split_seed = split_seed,
      validation_rows = validation_rows,
      probability = probability,
      coefficient = coefficient
    ),
    path
  )
  probability
}

replace_p4 <- function(incumbent_probability, candidate_p4) {
  incumbent_p4 <- incumbent_probability[, 4]
  conditional <- incumbent_probability[, 1:3, drop = FALSE] /
    pmax(1 - incumbent_p4, 1e-15)
  output <- matrix(0, nrow = nrow(incumbent_probability), ncol = 4)
  output[, 1:3] <- conditional * (1 - candidate_p4)
  output[, 4] <- candidate_p4
  stopifnot(max(abs(rowSums(output) - 1)) < 1e-10)
  output
}

score_rows <- function(rows, probability) {
  testmix_log_loss(y[rows], probability, segment[rows])
}

binary_testmix_loss <- function(rows, p4) {
  label <- outside[rows]
  loss <- -label * log(pmax(p4, 1e-15)) -
    (1 - label) * log(pmax(1 - p4, 1e-15))
  weight <- testmix_task_weights(segment[rows])
  sum(weight * loss) / sum(weight)
}

p4_diagnostics <- function(
    stage,
    split_seed,
    rows,
    incumbent_probability,
    gam_p4,
    selected_p4,
    gam_fit) {
  incumbent_p4 <- incumbent_probability[, 4]
  label <- outside[rows]
  incumbent_error <- label - incumbent_p4
  gam_error <- label - gam_p4
  selected_error <- label - selected_p4
  local_case <- case_id[rows]
  data.frame(
    stage = stage,
    split_seed = split_seed,
    incumbent_binary_loss =
      binary_testmix_loss(rows, incumbent_p4),
    gam_binary_loss = binary_testmix_loss(rows, gam_p4),
    selected_binary_loss =
      binary_testmix_loss(rows, selected_p4),
    p4_probability_correlation =
      cor(incumbent_p4, gam_p4),
    p4_error_correlation =
      cor(incumbent_error, gam_error),
    respondent_error_correlation = cor(
      tapply(incumbent_error, local_case, mean),
      tapply(gam_error, local_case, mean)
    ),
    selected_error_correlation =
      cor(incumbent_error, selected_error),
    total_edf = gam_fit$total_edf,
    smooth_edf = sum(gam_fit$smooth_edf),
    deviance_explained = gam_fit$deviance_explained
  )
}

segment_diagnostics <- function(
    stage,
    split_seed,
    rows,
    incumbent_probability,
    gam_probability,
    candidate_probability) {
  incumbent_segment <- segment_log_loss(
    y[rows], incumbent_probability, segment[rows]
  )
  gam_segment <- segment_log_loss(
    y[rows], gam_probability, segment[rows]
  )
  candidate_segment <- segment_log_loss(
    y[rows], candidate_probability, segment[rows]
  )
  data.frame(
    stage = stage,
    split_seed = split_seed,
    segment = names(incumbent_segment),
    incumbent = as.numeric(incumbent_segment),
    gam = as.numeric(gam_segment),
    gam_delta = as.numeric(gam_segment - incumbent_segment),
    candidate = as.numeric(candidate_segment),
    candidate_delta = as.numeric(
      candidate_segment - incumbent_segment
    )
  )
}

run_split <- function(stage, split_seed, specs) {
  split <- case_split_70_30(case_id, seed = split_seed)
  fit_rows <- split$train_idx
  validation_rows <- split$val_idx
  stopifnot(
    !any(case_id[fit_rows] %in% case_id[validation_rows]),
    length(unique(case_id[fit_rows])) == 794L,
    length(unique(case_id[validation_rows])) == 341L
  )
  gam_fits <- setNames(
    lapply(specs, function(spec) {
      fit_gam_split(
        stage,
        spec,
        split_seed,
        fit_rows,
        validation_rows
      )
    }),
    names(specs)
  )
  list(
    stage = stage,
    split_seed = split_seed,
    fit_rows = fit_rows,
    validation_rows = validation_rows,
    incumbent = fit_incumbent_split(
      stage, split_seed, fit_rows, validation_rows
    ),
    gam = gam_fits
  )
}

# Prior XGB outside-head diagnostics are read-only context. They were evaluated
# on different respondent splits, so they are not mixed into model selection.
xgb_head_benchmark <- NULL
if (file.exists(XGB_HEAD_RESULT)) {
  xgb_result <- readRDS(XGB_HEAD_RESULT)
  xgb_diagnostic <- rbind(
    xgb_result$phase_diagnostic,
    xgb_result$confirmation_diagnostic
  )
  xgb_head_benchmark <- list(
    selected_share = xgb_result$selected_share,
    phase_a_mean_delta = xgb_result$phase_a_mean_delta,
    confirmation_mean_delta =
      xgb_result$confirmation_mean_delta,
    confirmation_wins = xgb_result$confirmation_wins,
    p4_correlation_range = range(
      xgb_diagnostic$p4_correlation
    ),
    respondent_bootstrap =
      xgb_result$respondent_bootstrap,
    respondent_audit_red_flag =
      xgb_result$respondent_audit_red_flag
  )
}

cat(">>> LOW-DF MGCV OUTSIDE HEAD\n")
cat(
  "Phase-A splits:", paste(PHASE_A_SPLIT_SEEDS, collapse = ","),
  "| p4 shares:", paste(P4_SHARE_GRID, collapse = ","), "\n"
)
phase_splits <- lapply(
  PHASE_A_SPLIT_SEEDS,
  function(seed) run_split("phase_a", seed, GAM_SPECS)
)

phase_a_detail <- data.frame()
for (key in names(GAM_SPECS)) {
  for (share in P4_SHARE_GRID) {
    for (split_value in phase_splits) {
      incumbent_p4 <- split_value$incumbent[, 4]
      candidate_p4 <- (1 - share) * incumbent_p4 +
        share * split_value$gam[[key]]$p4
      candidate_probability <- replace_p4(
        split_value$incumbent, candidate_p4
      )
      phase_a_detail <- rbind(
        phase_a_detail,
        data.frame(
          key = key,
          p4_share = share,
          split_seed = split_value$split_seed,
          control = score_rows(
            split_value$validation_rows,
            split_value$incumbent
          ),
          candidate = score_rows(
            split_value$validation_rows,
            candidate_probability
          )
        )
      )
    }
  }
}
phase_a_detail$delta <-
  phase_a_detail$candidate - phase_a_detail$control
phase_a_summary <- aggregate(
  cbind(control, candidate, delta) ~ key + p4_share,
  phase_a_detail,
  mean
)
phase_a_wins <- aggregate(
  delta ~ key + p4_share,
  phase_a_detail,
  function(value) sum(value < 0)
)
names(phase_a_wins)[[3]] <- "wins"
phase_a_summary <- merge(
  phase_a_summary,
  phase_a_wins,
  by = c("key", "p4_share")
)
phase_a_summary <- phase_a_summary[
  order(phase_a_summary$candidate),
]
row.names(phase_a_summary) <- NULL

winner <- phase_a_summary[1, ]
selected_key <- winner$key
selected_share <- winner$p4_share
phase_a_pass <- winner$delta <= -PHASE_A_MIN_GAIN &&
  winner$wins >= PHASE_A_MIN_WINS

make_diagnostics <- function(stage, split_values) {
  p4 <- data.frame()
  segments <- data.frame()
  for (split_value in split_values) {
    gam_fit <- split_value$gam[[selected_key]]
    incumbent_p4 <- split_value$incumbent[, 4]
    selected_p4 <- (1 - selected_share) * incumbent_p4 +
      selected_share * gam_fit$p4
    gam_probability <- replace_p4(
      split_value$incumbent,
      gam_fit$p4
    )
    candidate_probability <- replace_p4(
      split_value$incumbent,
      selected_p4
    )
    p4 <- rbind(
      p4,
      p4_diagnostics(
        stage,
        split_value$split_seed,
        split_value$validation_rows,
        split_value$incumbent,
        gam_fit$p4,
        selected_p4,
        gam_fit
      )
    )
    segments <- rbind(
      segments,
      segment_diagnostics(
        stage,
        split_value$split_seed,
        split_value$validation_rows,
        split_value$incumbent,
        gam_probability,
        candidate_probability
      )
    )
  }
  list(p4 = p4, segments = segments)
}
phase_diagnostics <- make_diagnostics("phase_a", phase_splits)

cat("\n>>> PHASE-A SUMMARY\n")
print(phase_a_summary, row.names = FALSE, digits = 9)
cat("\nP4 diagnostics:\n")
print(phase_diagnostics$p4, row.names = FALSE, digits = 8)
if (!is.null(xgb_head_benchmark)) {
  cat(sprintf(
    paste0(
      "\nPrior XGB-head context (different splits): p4 correlation ",
      "%.4f-%.4f | confirmation delta %+.8f | wins %d/5 | ",
      "respondent red flag %s\n"
    ),
    xgb_head_benchmark$p4_correlation_range[[1]],
    xgb_head_benchmark$p4_correlation_range[[2]],
    xgb_head_benchmark$confirmation_mean_delta,
    xgb_head_benchmark$confirmation_wins,
    xgb_head_benchmark$respondent_audit_red_flag
  ))
}
cat(sprintf(
  paste0(
    "\nFROZEN %s | p4 share %.3f | delta %+.9f | ",
    "wins %d/3 | %s\n"
  ),
  selected_key,
  selected_share,
  winner$delta,
  winner$wins,
  if (phase_a_pass) {
    "OPEN FIVE FRESH CONFIRMATIONS"
  } else {
    "REJECT; CONFIRMATION UNOPENED"
  }
))

write.csv(
  phase_a_detail,
  file.path(OUTPUT_DIRECTORY, "phase_a_detail.csv"),
  row.names = FALSE
)
write.csv(
  phase_a_summary,
  file.path(OUTPUT_DIRECTORY, "phase_a_summary.csv"),
  row.names = FALSE
)
write.csv(
  phase_diagnostics$p4,
  file.path(OUTPUT_DIRECTORY, "phase_a_p4_diagnostics.csv"),
  row.names = FALSE
)
write.csv(
  phase_diagnostics$segments,
  file.path(OUTPUT_DIRECTORY, "phase_a_segment_effects.csv"),
  row.names = FALSE
)

output <- list(
  stage = "phase_a",
  gam_version = "lowdf_gam_p4_refined_v1",
  incumbent_version =
    "wave18_blendbase_fullt130_fixedresid_v1",
  gam_specs = GAM_SPECS,
  phase_a_split_seeds = PHASE_A_SPLIT_SEEDS,
  confirmation_split_seeds = CONFIRMATION_SPLIT_SEEDS,
  p4_share_grid = P4_SHARE_GRID,
  selected_key = selected_key,
  selected_share = selected_share,
  phase_a_detail = phase_a_detail,
  phase_a_summary = phase_a_summary,
  phase_a_p4_diagnostics = phase_diagnostics$p4,
  phase_a_segments = phase_diagnostics$segments,
  phase_a_mean_delta = winner$delta,
  phase_a_wins = winner$wins,
  phase_a_pass = phase_a_pass,
  xgb_head_benchmark = xgb_head_benchmark,
  confirmation_opened = FALSE,
  adopt = FALSE
)
saveRDS(output, OUTPUT_PATH)

if (!phase_a_pass) {
  quit(save = "no", status = 0)
}

respondent_bootstrap <- function(
    rows,
    control_probability,
    candidate_probability,
    seed,
    replicates = BOOTSTRAP_REPLICATES) {
  chosen <- cbind(seq_along(rows), y[rows] + 1L)
  task_delta <- -log(pmax(
    candidate_probability[chosen], 1e-15
  )) + log(pmax(control_probability[chosen], 1e-15))
  case_delta <- tapply(task_delta, case_id[rows], mean)
  case_segment <- tapply(
    segment[rows],
    case_id[rows],
    function(value) value[[1]]
  )
  positive_segments <- names(TEST_SEGMENT_SHARE)[
    TEST_SEGMENT_SHARE > 0
  ]
  set.seed(seed)
  draw <- numeric(replicates)
  for (replicate_index in seq_len(replicates)) {
    value <- 0
    used_weight <- 0
    for (segment_name in positive_segments) {
      cases <- names(case_segment)[case_segment == segment_name]
      if (!length(cases)) next
      sampled <- sample(cases, length(cases), replace = TRUE)
      value <- value + TEST_SEGMENT_SHARE[[segment_name]] *
        mean(case_delta[sampled])
      used_weight <- used_weight +
        TEST_SEGMENT_SHARE[[segment_name]]
    }
    draw[[replicate_index]] <- value / used_weight
  }
  list(
    replicates = replicates,
    probability_improvement = mean(draw < 0),
    interval90 = unname(quantile(draw, c(0.05, 0.95))),
    interval95 = unname(quantile(draw, c(0.025, 0.975)))
  )
}

cat("\n>>> FIVE FRESH FROZEN CONFIRMATIONS\n")
confirmation_splits <- lapply(
  CONFIRMATION_SPLIT_SEEDS,
  function(seed) run_split(
    "confirmation",
    seed,
    GAM_SPECS[selected_key]
  )
)
confirmation <- data.frame()
control_sum <- matrix(0, nrow(train), 4)
candidate_sum <- matrix(0, nrow(train), 4)
prediction_count <- integer(nrow(train))
for (split_value in confirmation_splits) {
  rows <- split_value$validation_rows
  gam_fit <- split_value$gam[[selected_key]]
  selected_p4 <- (1 - selected_share) *
    split_value$incumbent[, 4] +
    selected_share * gam_fit$p4
  candidate_probability <- replace_p4(
    split_value$incumbent, selected_p4
  )
  control_score <- score_rows(rows, split_value$incumbent)
  candidate_score <- score_rows(rows, candidate_probability)
  confirmation <- rbind(
    confirmation,
    data.frame(
      split_seed = split_value$split_seed,
      control = control_score,
      candidate = candidate_score,
      delta = candidate_score - control_score
    )
  )
  control_sum[rows, ] <- control_sum[rows, ] +
    split_value$incumbent
  candidate_sum[rows, ] <- candidate_sum[rows, ] +
    candidate_probability
  prediction_count[rows] <- prediction_count[rows] + 1L
  print(tail(confirmation, 1), row.names = FALSE, digits = 9)
  write.csv(
    confirmation,
    file.path(OUTPUT_DIRECTORY, "confirmation_partial.csv"),
    row.names = FALSE
  )
}

confirmation_diagnostics <- make_diagnostics(
  "confirmation", confirmation_splits
)
confirmation_mean_delta <- mean(confirmation$delta)
confirmation_wins <- sum(confirmation$delta < 0)
covered_rows <- which(prediction_count > 0)
covered_control <- control_sum[covered_rows, , drop = FALSE] /
  prediction_count[covered_rows]
covered_candidate <- candidate_sum[covered_rows, , drop = FALSE] /
  prediction_count[covered_rows]
covered_delta <- score_rows(
  covered_rows, covered_candidate
) - score_rows(covered_rows, covered_control)
bootstrap <- respondent_bootstrap(
  covered_rows,
  covered_control,
  covered_candidate,
  seed = 20260728,
  replicates = BOOTSTRAP_REPLICATES
)
confirmation_pass <-
  confirmation_mean_delta <= -CONFIRMATION_MIN_GAIN &&
  confirmation_wins >= CONFIRMATION_MIN_WINS &&
  bootstrap$probability_improvement >= MIN_BOOTSTRAP_P &&
  bootstrap$interval90[[2]] < 0

cat(sprintf(
  paste0(
    "\nCONFIRMATION: mean delta %+.9f | wins %d/5 | ",
    "covered delta %+.9f | bootstrap P %.4f | ",
    "90%% interval [%+.9f,%+.9f] | %s\n"
  ),
  confirmation_mean_delta,
  confirmation_wins,
  covered_delta,
  bootstrap$probability_improvement,
  bootstrap$interval90[[1]],
  bootstrap$interval90[[2]],
  if (confirmation_pass) {
    "ALL GATES PASS; PACKAGING AUTHORISED"
  } else {
    "REJECT; DO NOT PACKAGE"
  }
))
write.csv(
  confirmation,
  file.path(OUTPUT_DIRECTORY, "confirmation.csv"),
  row.names = FALSE
)
write.csv(
  confirmation_diagnostics$p4,
  file.path(OUTPUT_DIRECTORY, "confirmation_p4_diagnostics.csv"),
  row.names = FALSE
)
write.csv(
  confirmation_diagnostics$segments,
  file.path(OUTPUT_DIRECTORY, "confirmation_segment_effects.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    covered_respondents = length(unique(case_id[covered_rows])),
    total_respondents = length(unique(case_id)),
    covered_delta = covered_delta,
    bootstrap_replicates = bootstrap$replicates,
    bootstrap_p_improvement =
      bootstrap$probability_improvement,
    interval90_lower = bootstrap$interval90[[1]],
    interval90_upper = bootstrap$interval90[[2]],
    interval95_lower = bootstrap$interval95[[1]],
    interval95_upper = bootstrap$interval95[[2]]
  ),
  file.path(OUTPUT_DIRECTORY, "respondent_bootstrap.csv"),
  row.names = FALSE
)

output$stage <- "confirmation"
output$confirmation_opened <- TRUE
output$confirmation <- confirmation
output$confirmation_p4_diagnostics <-
  confirmation_diagnostics$p4
output$confirmation_segments <-
  confirmation_diagnostics$segments
output$confirmation_mean_delta <- confirmation_mean_delta
output$confirmation_wins <- confirmation_wins
output$covered_rows <- covered_rows
output$covered_delta <- covered_delta
output$respondent_bootstrap <- bootstrap
output$confirmation_pass <- confirmation_pass
output$adopt <- confirmation_pass
saveRDS(output, OUTPUT_PATH)
