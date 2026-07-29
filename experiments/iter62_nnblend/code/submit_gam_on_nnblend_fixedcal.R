# Full-data deployment of the confirmed low-DF GAM outside-option head.
#
# The banked 1.195 submission supplies the exact conditional distribution over
# Ch1:Ch3. The GAM supplies an independent p(Ch4), blended at the frozen 25%
# share. No other probability ratio is changed.

source("paths.R")
source("data.R")
source("segment_framework.R")
source("data_v10.R")
source("write_submission.R")

library(mgcv)

ALPHA <- 0.10
P4_SHARE <- 0.25
GAM_GAMMA <- 1.8
ATTRIBUTES <- c(
  "CC", "GN", "NS", "BU", "FA", "LD", "BZ", "FC", "FP", "RP",
  "PP", "KA", "SC", "TS", "NV", "MA", "LB", "AF", "HU"
)
FACTOR_COLUMNS <- c(
  "segment", "region", "gender", "Urb", "ppark", "night", "educ"
)
# NNBLEND FIXEDCAL variant. Valid: wave18_gam_on_blend.R printed
# PACKAGING AUTHORISED for the head on this base family calibrated
# against wave18_blend_ref.rds (4/5, mean -0.0031, bootstrap P 0.980);
# this rebuild makes the deployed base match that validated chain.
BASE_SUBMISSION_PATH <-
  "../submissions/submission_wave6_nnblend_FIXEDCAL.csv"
OUTPUT_PATH <-
  "../submissions/submission_NNBLEND_FIXEDCAL_GAMHEAD.csv"
FIT_PATH <- "../outputs/gam_outside_head_nnblend_fixedcal_fit.rds"

raw_train <- read.csv(resolve_train_path(), stringsAsFactors = FALSE)
raw_test <- read.csv(resolve_test_path(), stringsAsFactors = FALSE)
train <- add_task_features(add_engineered(raw_train))
test <- add_task_features(add_engineered(raw_test))
y <- make_label(train)

factor_levels <- lapply(FACTOR_COLUMNS, function(column) {
  sort(unique(as.character(train[[column]])))
})
names(factor_levels) <- FACTOR_COLUMNS
for (column in FACTOR_COLUMNS) {
  unseen <- setdiff(
    unique(as.character(test[[column]])),
    factor_levels[[column]]
  )
  if (length(unseen)) {
    stop(
      "Test contains unseen level(s) for ", column, ": ",
      paste(unseen, collapse = ", ")
    )
  }
}

build_gam_data <- function(data, response = NULL) {
  price <- as.matrix(data[, paste0("Price", 1:3)])
  safety <- sapply(1:3, function(alternative) {
    rowSums(as.matrix(
      data[, paste0(ATTRIBUTES, alternative), drop = FALSE]
    ))
  })
  output <- data.frame(
    segment = factor(
      data$segment, levels = factor_levels$segment
    ),
    region = factor(
      data$region, levels = factor_levels$region
    ),
    gender = factor(
      data$gender, levels = factor_levels$gender
    ),
    Urb = factor(data$Urb, levels = factor_levels$Urb),
    ppark = factor(
      data$ppark, levels = factor_levels$ppark
    ),
    night = factor(
      data$night, levels = factor_levels$night
    ),
    educ = factor(
      data$educ, levels = factor_levels$educ
    ),
    Task = as.numeric(data$Task),
    age_value = as.numeric(data$agea),
    log_income = log1p(pmax(as.numeric(data$incomea), 0)),
    log_miles = log1p(pmax(as.numeric(data$milesa), 0)),
    price_min = apply(price, 1, min),
    price_mean = rowMeans(price),
    price_spread = apply(price, 1, max) -
      apply(price, 1, min),
    safety_best = apply(safety, 1, max),
    safety_mean = rowMeans(safety),
    safety_spread = apply(safety, 1, max) -
      apply(safety, 1, min)
  )
  if (!is.null(response)) output$outside <- response
  stopifnot(!anyNA(output))
  output
}

train_gam <- build_gam_data(
  train, response = as.integer(y == 3L)
)
test_gam <- build_gam_data(test)
train_gam$case_weight <- segment_importance_weights(
  train$segment, ALPHA
)

GAM_FORMULA <- outside ~
  segment + region + gender + Urb + ppark + night + educ +
  s(Task, k = 4, bs = "cr") +
  s(age_value, k = 4, bs = "cr") +
  s(log_income, k = 4, bs = "cr") +
  s(log_miles, k = 4, bs = "cr") +
  price_min + price_mean + price_spread +
  safety_best + safety_mean + safety_spread

fit <- mgcv::gam(
  formula = GAM_FORMULA,
  family = binomial(link = "logit"),
  data = train_gam,
  weights = case_weight,
  method = "REML",
  select = TRUE,
  gamma = GAM_GAMMA
)
gam_p4 <- as.numeric(predict(
  fit,
  newdata = test_gam,
  type = "response"
))
gam_p4 <- pmin(pmax(gam_p4, 1e-6), 1 - 1e-6)

base_submission <- read.csv(
  BASE_SUBMISSION_PATH,
  stringsAsFactors = FALSE
)
sample_submission <- read.csv(
  resolve_sample_submission_path(),
  stringsAsFactors = FALSE
)
stopifnot(
  identical(base_submission$No, sample_submission$No),
  nrow(base_submission) == nrow(test)
)
base_probability <- as.matrix(
  base_submission[, paste0("Ch", 1:4)]
)
base_p4 <- base_probability[, 4]
conditional <- base_probability[, 1:3, drop = FALSE] /
  pmax(1 - base_p4, 1e-15)
candidate_p4 <- (1 - P4_SHARE) * base_p4 +
  P4_SHARE * gam_p4
candidate_probability <- matrix(0, nrow(test), 4)
candidate_probability[, 1:3] <-
  conditional * (1 - candidate_p4)
candidate_probability[, 4] <- candidate_p4
stopifnot(
  all(is.finite(candidate_probability)),
  all(candidate_probability > 0),
  max(abs(rowSums(candidate_probability) - 1)) < 1e-10
)

write_submission(test, candidate_probability, OUTPUT_PATH)
written <- read.csv(OUTPUT_PATH, stringsAsFactors = FALSE)
written_probability <- as.matrix(
  written[, paste0("Ch", 1:4)]
)
stopifnot(
  nrow(written) == 4997L,
  identical(names(written), names(sample_submission)),
  identical(written$No, sample_submission$No),
  all(is.finite(written_probability)),
  all(written_probability > 0),
  max(abs(rowSums(written_probability) - 1)) < 1e-10,
  max(abs(written_probability[, 4] - candidate_p4)) < 1e-12
)

fit_summary <- summary(fit)
smooth_edf <- setNames(
  as.numeric(fit_summary$s.table[, "edf"]),
  rownames(fit_summary$s.table)
)
saveRDS(
  list(
    model = fit,
    formula = GAM_FORMULA,
    alpha = ALPHA,
    p4_share = P4_SHARE,
    gamma = GAM_GAMMA,
    total_edf = sum(fit$edf),
    smooth_edf = smooth_edf,
    deviance_explained = fit_summary$dev.expl,
    base_submission_path = BASE_SUBMISSION_PATH,
    output_path = OUTPUT_PATH
  ),
  FIT_PATH
)

cat(sprintf(
  paste0(
    "full GAM total edf %.3f | smooth edf %.3f | ",
    "deviance explained %.4f\n"
  ),
  sum(fit$edf),
  sum(smooth_edf),
  fit_summary$dev.expl
))
cat(sprintf(
  paste0(
    "p4 movement vs banked: MAE %.9f | max %.9f | ",
    "full-probability MAE %.9f\n"
  ),
  mean(abs(candidate_p4 - base_p4)),
  max(abs(candidate_p4 - base_p4)),
  mean(abs(candidate_probability - base_probability))
))
cat("wrote", OUTPUT_PATH, "\n")
