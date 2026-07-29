# LISTWISE / symmetric-utility data representation.
#
# Every model so far is WIDE: one row per task, 80 alternative-specific
# columns, 4-class softmax. This file builds the LONG form: one row per
# (task, alternative) with a single shared feature vocabulary --
# "my attributes, my price, how I compare to my rivals, who is choosing" --
# so one utility function is learned for all four slots (4x the data per
# split decision). The outside option is a real row (all-zero attributes,
# is_outside = 1). Group structure: 4 consecutive rows per task, wide-row
# order preserved -- prob matrix reshapes back with matrix(ncol=4,byrow=TRUE).
#
# This is the tree-model version of the structure that makes MNL/MXL natural
# for choice data; the senior team only ever combined it with linearity.

ATTR_NOPRICE <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
                  "PP","KA","SC","TS","NV","MA","LB","AF","HU")

# df must already have add_engineered() + add_task_features() applied.
# ref: optional second data.frame whose string-demographic LEVELS are pooled
# with df's before encoding, so train and test encode identically even if one
# side lacks a level (test has no Small Car). Not needed for train-only OOF.
build_long <- function(df, ref = NULL, lux_ix = FALSE, attr_rel = FALSE,
                       task_ix = FALSE, fatigue_ix = FALSE, ctx_ix = FALSE) {
  n <- nrow(df)
  demo <- df[, DEMO_BLOCK, drop = FALSE]
  for (c in names(demo)) if (is.character(demo[[c]])) {
    lv <- sort(unique(c(demo[[c]], if (!is.null(ref)) ref[[c]])))
    demo[[c]] <- as.numeric(factor(demo[[c]], levels = lv))
  }

  price_mat <- as.matrix(df[, paste0("Price", 1:4)])   # Price4 is 0
  tot_mat <- sapply(1:4, function(a)
    rowSums(as.matrix(df[, paste0(ATTR_NOPRICE, a)])))

  # Respondent-level OFFER CONTEXT across their 19 tasks (pure X, no labels,
  # computable identically for test): each person's own price/value
  # distribution is the natural reference frame for "is this a good deal".
  vpp_mat <- tot_mat[, 1:3] / pmax(price_mat[, 1:3], 1e-6)
  task_bestvpp <- do.call(pmax, as.data.frame(vpp_mat))
  cs3 <- rep(df$Case, 3)
  resp_pmean   <- ave(as.vector(price_mat[, 1:3]), cs3)[seq_len(n)]
  resp_psd     <- ave(as.vector(price_mat[, 1:3]), cs3, FUN = sd)[seq_len(n)]
  resp_tmean   <- ave(as.vector(tot_mat[, 1:3]), cs3)[seq_len(n)]
  resp_bestvpp <- ave(as.vector(vpp_mat), cs3, FUN = max)[seq_len(n)]

  is_lux_r <- as.integer(df$segment %in%
    c("Prestige Luxury Sedan", "Midsize Luxury Utility segements"))

  blocks <- vector("list", 4)
  for (a in 1:4) {
    own_attr <- as.matrix(df[, paste0(c(ATTR_NOPRICE, "Price"), a)])
    colnames(own_attr) <- c(ATTR_NOPRICE, "Price")
    others <- setdiff(1:3, a)                 # real rivals only (1..3)
    rival_price_min  <- if (a %in% 1:3) do.call(pmin, as.data.frame(price_mat[, others, drop = FALSE]))
                        else apply(price_mat[, 1:3], 1, min)
    rival_price_mean <- if (a %in% 1:3) rowMeans(price_mat[, others, drop = FALSE])
                        else rowMeans(price_mat[, 1:3])
    rival_tot_best   <- if (a %in% 1:3) do.call(pmax, as.data.frame(tot_mat[, others, drop = FALSE]))
                        else apply(tot_mat[, 1:3], 1, max)
    own_price <- price_mat[, a]; own_tot <- tot_mat[, a]
    price_rank <- rowSums(price_mat[, 1:3] < own_price) + 1  # among real bundles
    blk <- cbind(
      own_attr,
      alt_id          = a,
      is_outside      = as.integer(a == 4),
      own_total       = own_tot,
      price_vs_minriv = own_price - rival_price_min,
      price_vs_mnriv  = own_price - rival_price_mean,
      total_vs_bestriv= own_tot - rival_tot_best,
      price_rank      = ifelse(a == 4, 0, price_rank),
      min_rival_price = rival_price_min,
      price_x_age     = own_price * df$ageind,
      price_x_income  = own_price * df$incomeind,
      Price_avg       = df$Price_avg,
      Task            = df$Task,
      Task_late       = df$Task_late,
      as.matrix(demo)
    )
    if (lux_ix) blk <- cbind(blk,
      price_x_lux     = own_price * is_lux_r,
      pvminriv_x_lux  = (own_price - rival_price_min) * is_lux_r,
      total_x_lux     = own_tot * is_lux_r,
      outside_x_lux   = as.integer(a == 4) * is_lux_r)
    if (task_ix) {
      task_c <- (df$Task - 10) / 9
      blk <- cbind(blk,
        outside_x_task    = as.integer(a == 4) * task_c,
        outside_x_tasksq  = as.integer(a == 4) * task_c^2,
        outside_x_late    = as.integer(a == 4) * df$Task_late)
    }
    if (fatigue_ix) {
      task_c <- (df$Task - 10) / 9
      blk <- cbind(blk,
        price_x_task       = own_price * task_c,
        pvminriv_x_task    = (own_price - rival_price_min) * task_c,
        price_rank_x_task  = ifelse(a == 4, 0, price_rank) * task_c,
        total_x_task       = own_tot * task_c)
    }
    if (ctx_ix) {
      own_vpp <- if (a == 4) rep(0, n) else vpp_mat[, a]
      blk <- cbind(blk,
        ctx_p_dev    = own_price - resp_pmean,
        ctx_p_z      = (own_price - resp_pmean) / pmax(resp_psd, 1e-6),
        ctx_t_dev    = own_tot - resp_tmean,
        ctx_vpp_gap  = own_vpp - resp_bestvpp,
        ctx_task_gap = task_bestvpp - resp_bestvpp)
    }
    if (attr_rel) {
      rel <- sapply(ATTR_NOPRICE, function(t) {
        vals <- as.matrix(df[, paste0(t, 1:3)])
        own <- if (a == 4) rep(0, n) else vals[, a]
        riv <- if (a == 4) apply(vals, 1, max)
               else do.call(pmax, as.data.frame(vals[, setdiff(1:3, a), drop = FALSE]))
        own - riv
      })
      colnames(rel) <- paste0(ATTR_NOPRICE, "_rel")
      blk <- cbind(blk, rel)
    }
    blocks[[a]] <- blk
  }
  # interleave: rows (task1,alt1..4), (task2,alt1..4), ...
  X <- matrix(NA_real_, nrow = 4 * n, ncol = ncol(blocks[[1]]))
  colnames(X) <- colnames(blocks[[1]])
  for (a in 1:4) X[seq(a, by = 4, length.out = n), ] <- blocks[[a]]
  X
}

long_labels <- function(y_wide) {          # 0..3 -> binary per long row
  lab <- matrix(0L, nrow = length(y_wide), ncol = 4)
  lab[cbind(seq_along(y_wide), y_wide + 1L)] <- 1L
  as.vector(t(lab))
}

# Group softmax objective: consecutive groups of 4 long rows = one task.
# grad/hess are the standard softmax diagonal approximation, scaled by the
# per-row weight (same within a group).
softmax_group_obj <- function(preds, dtrain) {
  y <- xgboost::getinfo(dtrain, "label")
  w <- xgboost::getinfo(dtrain, "weight")
  if (is.null(w) || !length(w)) w <- rep(1, length(y))
  m <- matrix(preds, ncol = 4, byrow = TRUE)
  m <- m - apply(m, 1, max)
  e <- exp(m); p <- e / rowSums(e)
  pv <- as.vector(t(p))
  list(grad = (pv - y) * w, hess = pmax(pv * (1 - pv), 1e-6) * w)
}

# Margins -> n x 4 probability matrix.
margins_to_probs <- function(margins) {
  m <- matrix(margins, ncol = 4, byrow = TRUE)
  m <- m - apply(m, 1, max)
  e <- exp(m); e / rowSums(e)
}

# Train one listwise model, return prob matrix for X_long_pred.
# Custom objective passed via the `obj` argument (long-standing R API); if a
# given xgboost version wants it inside params as `objective`, the tryCatch
# falls back.
listwise_fit_predict <- function(X_long_tr, y_long_tr, w_long_tr,
                                 X_long_pred, params, nrounds, seed = 1) {
  set.seed(seed)
  dtr <- xgboost::xgb.DMatrix(X_long_tr, label = y_long_tr, weight = w_long_tr)
  m <- tryCatch(
    xgboost::xgb.train(params = params, data = dtr, nrounds = nrounds,
                       obj = softmax_group_obj, verbose = 0),
    error = function(e)
      xgboost::xgb.train(params = c(params, list(objective = softmax_group_obj)),
                         data = dtr, nrounds = nrounds, verbose = 0)
  )
  margins_to_probs(predict(m, xgboost::xgb.DMatrix(X_long_pred), outputmargin = TRUE))
}
