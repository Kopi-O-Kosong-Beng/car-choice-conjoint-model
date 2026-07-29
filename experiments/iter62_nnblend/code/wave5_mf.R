# WAVE 5 -- Idea 1: SVD/PCA latent-taste matrix-factorization cold-start
# bridge, as extra features on top of the current best confirmed recipe's
# listwise representation (build_long() + attribute-level dummies).
#
# Pipeline (fit on TRAINING respondents in a given fold's fit_rows only):
#   1. build_person_signal(): one row per respondent (Case) summarizing their
#      revealed preference across their (up to 19) tasks -- for each of the
#      19 safety/comfort attributes, mean(chosen level - offered-menu mean
#      level) over tasks where they picked a REAL bundle (Ch1-3); the
#      analogous price delta; and their raw outside-option ("none") rate.
#      This is the person x item matrix R, built ONLY from fit_rows persons.
#   2. fit_mf_model(): center R, truncated SVD to `rank` axes -> item loadings
#      V (items x rank) and each fit-person's latent score vector (persons x
#      rank). Regress each latent-score column, and separately the raw
#      outside-rate, on that person's demographics alone (lm) -- this is the
#      cold-start bridge: it never sees a held-out/test respondent's own
#      choices, only their demographics.
#   3. predict_mf_features(): for ANY respondent (fit or held-out, train or
#      test), predict their latent score vector from demographics alone,
#      reconstruct a predicted attribute-preference vector via V, and turn it
#      into two extra long-format columns:
#        mf_fit      = dot(predicted preference vector, this alternative's
#                      own [19 attrs, price]) -- 0 for the outside row by
#                      construction (all-zero attributes), so it reads as
#                      "how well does this bundle match my predicted taste".
#        mf_outside  = predicted outside-option propensity from demographics
#                      alone, repeated across the 4 rows of a task (like the
#                      existing demographic columns in build_long()).
#      Both are demographics-only predictions for EVERY row (fit and held-out
#      alike), so train/validation/test feature distributions match exactly
#      -- this is deliberate: it is what makes the cold-start test honest.

MF_ATTRS <- ATTR_NOPRICE                 # from data_listwise.R (19 letters)
MF_ITEMS <- c(MF_ATTRS, "price_delta")
MF_DEMO_NUMERIC <- c(
  "segmentind", "yearind", "milesind", "milesa", "nightind", "nighta",
  "pparkind", "genderind", "ageind", "agea", "educind", "regionind",
  "Urbind", "incomeind", "incomea"
)
MF_RANK <- 3   # frozen a priori (middle of the suggested rank 2-5 range);
               # not tuned on any fold, so it costs no extra selection d.o.f.

build_person_signal <- function(df) {
  n <- nrow(df)
  chosen_alt <- with(df, ifelse(Ch1 == 1, 1L, ifelse(Ch2 == 1, 2L, ifelse(Ch3 == 1, 3L, 4L))))
  real <- chosen_alt %in% 1:3
  chosen_idx <- ifelse(real, chosen_alt, 1L)   # placeholder for outside rows; NA'd out below

  attr_delta <- matrix(NA_real_, n, length(MF_ATTRS))
  for (j in seq_along(MF_ATTRS)) {
    a <- MF_ATTRS[j]
    vals <- as.matrix(df[, paste0(a, 1:3)])
    offer_mean <- rowMeans(vals)
    chosen_val <- vals[cbind(seq_len(n), chosen_idx)]
    d <- chosen_val - offer_mean
    d[!real] <- NA
    attr_delta[, j] <- d
  }
  price_mat <- as.matrix(df[, paste0("Price", 1:3)])
  offer_pmean <- rowMeans(price_mat)
  chosen_price <- price_mat[cbind(seq_len(n), chosen_idx)]
  price_delta <- chosen_price - offer_pmean
  price_delta[!real] <- NA

  task_level <- data.frame(Case = df$Case, attr_delta)
  colnames(task_level)[2:(1 + length(MF_ATTRS))] <- MF_ATTRS
  task_level$price_delta <- price_delta
  task_level$outside <- as.integer(!real)

  agg_mean <- function(x) {
    m <- mean(x, na.rm = TRUE)
    if (!is.finite(m)) 0 else m
  }
  person <- aggregate(task_level[, c(MF_ATTRS, "price_delta")],
                       by = list(Case = task_level$Case), FUN = agg_mean)
  outside_rate <- aggregate(outside ~ Case, task_level, mean)
  person <- merge(person, outside_rate, by = "Case", sort = FALSE)
  person
}

# Fit the SVD + demographic-regression model on fit_df ONLY (a training
# fold's fit_rows, as a data.frame of task-level rows for those persons).
fit_mf_model <- function(fit_df, rank = MF_RANK) {
  person <- build_person_signal(fit_df)
  Rmat <- as.matrix(person[, MF_ITEMS])
  center <- colMeans(Rmat)
  Rc <- sweep(Rmat, 2, center, "-")
  sv <- svd(Rc, nu = rank, nv = rank)
  V <- sv$v                                     # items x rank
  scores <- Rc %*% V                            # persons x rank

  demo_first <- fit_df[match(person$Case, fit_df$Case), MF_DEMO_NUMERIC, drop = FALSE]
  demo_first[] <- lapply(demo_first, function(col) {
    col <- as.numeric(col)
    if (any(is.na(col))) col[is.na(col)] <- median(col, na.rm = TRUE)
    col
  })

  reg_df <- demo_first
  score_models <- vector("list", rank)
  for (k in seq_len(rank)) {
    reg_df$.target <- scores[, k]
    score_models[[k]] <- lm(.target ~ ., data = reg_df)
  }
  reg_df$.target <- person$outside
  outside_model <- lm(.target ~ ., data = reg_df)

  list(center = center, V = V, score_models = score_models,
       outside_model = outside_model, demo_cols = MF_DEMO_NUMERIC, rank = rank)
}

# Predict the 2 extra long-format columns (mf_fit, mf_outside) for ANY
# task-level data.frame `df` (fit persons, held-out persons, or test), using
# ONLY demographics -- never that person's own revealed choices, even if they
# happen to be in the model's own fit set. Row order matches build_long().
predict_mf_features <- function(model, df) {
  n <- nrow(df)
  demo <- df[, model$demo_cols, drop = FALSE]
  demo[] <- lapply(demo, function(col) {
    col <- as.numeric(col)
    if (any(is.na(col))) col[is.na(col)] <- 0
    col
  })
  pred_scores <- sapply(model$score_models, function(m) as.numeric(predict(m, newdata = demo)))
  if (is.null(dim(pred_scores))) pred_scores <- matrix(pred_scores, ncol = length(model$score_models))
  pred_outside <- as.numeric(predict(model$outside_model, newdata = demo))

  pred_vec <- pred_scores %*% t(model$V)
  pred_vec <- sweep(pred_vec, 2, model$center, "+")
  colnames(pred_vec) <- MF_ITEMS

  mf_fit <- numeric(4 * n)
  for (a in 1:4) {
    if (a == 4) {
      dotv <- rep(0, n)
    } else {
      own_attr <- as.matrix(df[, paste0(MF_ATTRS, a)])
      own_price <- df[[paste0("Price", a)]]
      dotv <- rowSums(own_attr * pred_vec[, MF_ATTRS, drop = FALSE]) +
        own_price * pred_vec[, "price_delta"]
    }
    mf_fit[seq(a, by = 4, length.out = n)] <- dotv
  }
  mf_outside_long <- rep(pred_outside, each = 4)
  cbind(mf_fit = mf_fit, mf_outside = mf_outside_long)
}
