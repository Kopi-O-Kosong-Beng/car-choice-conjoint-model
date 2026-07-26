# =============================================================================
# Bundle-level encoder for iteration 16.
#
# WHY THE KEY IS A PRESENCE PATTERN AND NOT THE 20 ATTRIBUTE VALUES.
# The brief asked to key each alternative by its own 20 attribute values, on the
# premise that a bundle recurs in different choice sets. It does not. There are
# 5,681 designs and exactly 5,681 x 3 = 17,043 distinct real bundles, and the map
# bundle -> (design, position) is a bijection (verified in diagnostics.R). Every
# bundle occurs in exactly one choice set at exactly one position, so a
# bundle-keyed win rate is the SAME STATISTIC, value for value, as the existing
# (design, alt) share encoding in model/encode_design.R -- 3.83 observations each,
# not a better-supported one.
#
# What does recur is the bundle's PRESENCE PATTERN: every real bundle carries
# exactly 9 of the 19 non-price features, and the design reused 5,511 such subsets
# across ~3 bundles each, at different tier levels, different prices and against
# different rivals. That gives 14.46 observations per key -- 3.8x the design key --
# and it is a genuinely different statistic: "this COMBINATION OF FEATURES is
# attractive", pooled over the prices and rivals it happened to be paired with.
#
# Because the pooling is over instances with very different prices, the raw rate is
# confounded by them, so every key also carries the context it was measured in
# (mean own price, mean own level-sum, mean rival price, mean rival level-sum) and
# the current row's deviation from it. That is the "discount easy wins" term the
# brief asked for. Note rival RICHNESS cannot serve: richness is 9 on every real
# bundle and 0 on the none option, so rich_task_mean is the constant 6.75.
#
# LEAKAGE. No model prediction of any kind enters this encoding -- only raw
# `chosen` indicators from reference respondents. The iteration-12 failure mode
# (the reference respondents' baseline predictions were themselves fitted on the
# scored fold) is therefore closed BY CONSTRUCTION, not by argument. Respondents
# live entirely within one fold; a fold-k row is encoded from folds != k only, so
# no person contributes to their own feature. Test rows use all 1,135 training
# respondents.
# =============================================================================

BND_ALPHAS  <- c(1, 5, 20, 50)
BND_SHARE   <- paste0("b_share_a", BND_ALPHAS)
BND_SHARE_C <- paste0(BND_SHARE, "_c")
BND_RAW     <- c(BND_SHARE, "b_n",
                 "b_price_hist", "b_price_delta", "b_lvl_hist", "b_lvl_delta",
                 "b_rivprice_hist", "b_rivprice_delta",
                 "b_rivlvl_hist", "b_rivlvl_delta")
BND_ENC_COLS <- c(BND_RAW, BND_SHARE_C)
BND_COLS     <- c(BND_ENC_COLS, "riv_price_now", "riv_lvl_now")

# 19-bit presence pattern over the non-price attributes. Price is >0 on every real
# bundle, so including it would add nothing. The none option is the all-zero key.
make_pkey <- function(dt, attrs) {
  a19 <- setdiff(attrs, "Price")
  do.call(paste0, lapply(a19, function(a) as.integer(dt[[a]] > 0)))
}

# Rows must be sorted (No, alt) with exactly 4 contiguous rows per task.
check_layout <- function(dt) {
  stopifnot(nrow(dt) %% 4 == 0)
  M <- matrix(dt$alt, ncol = 4, byrow = TRUE)
  stopifnot(all(M[, 1] == 1L), all(M[, 2] == 2L), all(M[, 3] == 3L), all(M[, 4] == 4L))
  N <- matrix(dt$No, ncol = 4, byrow = TRUE)
  stopifnot(all(N[, 1] == N[, 2]), all(N[, 1] == N[, 3]), all(N[, 1] == N[, 4]))
  invisible(TRUE)
}

# Adds pkey and the CURRENT task's rival strength. Rival = the other REAL
# alternatives (alt 4 is all-zero and is not a rival to anyone). Modifies in place.
add_bundle_context <- function(dt) {
  check_layout(dt)
  dt[, pkey := make_pkey(dt, ATTRS)]
  spread <- function(v) {
    M <- matrix(v, ncol = 4, byrow = TRUE)
    s3 <- M[, 1] + M[, 2] + M[, 3]
    as.vector(t(cbind((s3 - M[, 1]) / 2, (s3 - M[, 2]) / 2, (s3 - M[, 3]) / 2, s3 / 3)))
  }
  dt[, riv_price_now := spread(as.numeric(Price))]
  dt[, riv_lvl_now   := spread(as.numeric(lvlsum))]
  invisible(NULL)
}

# Per-row raw encoding of `target` using only the respondents in `ref`.
encode_bundle_aligned <- function(ref, target) {
  refr <- ref[alt <= 3]
  pr   <- sum(refr$chosen) / nrow(refr)          # pooled real-bundle base rate
  g    <- refr[, .(mp = mean(Price), ml = mean(lvlsum),
                   mrp = mean(riv_price_now), mrl = mean(riv_lvl_now))]
  agg <- refr[, .(n = .N, n_ch = sum(chosen),
                  mp = mean(Price), ml = mean(lvlsum),
                  mrp = mean(riv_price_now), mrl = mean(riv_lvl_now)), by = pkey]

  t2 <- copy(target)[, .rid := .I][, .(.rid, pkey, alt, Price, lvlsum,
                                       riv_price_now, riv_lvl_now)]
  t2 <- merge(t2, agg, by = "pkey", all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, n_ch = 0, mp = g$mp, ml = g$ml, mrp = g$mrp, mrl = g$mrl)]
  for (a in BND_ALPHAS) t2[, paste0("b_share_a", a) := (n_ch + a * pr) / (n + a)]
  t2[, `:=`(b_n = n,
            b_price_hist    = mp,  b_price_delta    = as.numeric(Price) - mp,
            b_lvl_hist      = ml,  b_lvl_delta      = as.numeric(lvlsum) - ml,
            b_rivprice_hist = mrp, b_rivprice_delta = riv_price_now - mrp,
            b_rivlvl_hist   = mrl, b_rivlvl_delta   = riv_lvl_now - mrl)]
  data.table::setorder(t2, .rid)
  t2[, .SD, .SDcols = BND_RAW]
}

# The none option has no bundle of its own; it is defined entirely by the set it
# was offered against. Give it the set's mean bundle strength, and zero out the
# own-bundle context terms. Then task-centre the shares -- the metric is a softmax
# within a task, so only within-task contrasts carry utility.
finalize_bundle <- function(dt) {
  check_layout(dt)
  for (cc in c(BND_SHARE, "b_n")) {
    M <- matrix(dt[[cc]], ncol = 4, byrow = TRUE)
    M[, 4] <- (M[, 1] + M[, 2] + M[, 3]) / 3
    data.table::set(dt, NULL, cc, as.vector(t(M)))
  }
  for (cc in c("b_price_hist", "b_price_delta", "b_lvl_hist", "b_lvl_delta")) {
    v <- dt[[cc]]; v[dt$alt == 4L] <- 0
    data.table::set(dt, NULL, cc, v)
  }
  for (a in BND_ALPHAS) {
    cc <- paste0("b_share_a", a)
    M <- matrix(dt[[cc]], ncol = 4, byrow = TRUE)
    data.table::set(dt, NULL, paste0(cc, "_c"), as.vector(t(M - rowMeans(M))))
  }
  invisible(NULL)
}

# Fold-aware application. Mirrors apply_design_encoding in model/encode_design.R.
apply_bundle_encoding <- function(trl, tel) {
  for (cc in BND_RAW) { trl[, (cc) := NA_real_]; tel[, (cc) := NA_real_] }
  for (k in sort(unique(trl$fold))) {
    idx <- which(trl$fold == k)
    e <- encode_bundle_aligned(trl[fold != k], trl[fold == k])
    for (cc in BND_RAW) data.table::set(trl, idx, cc, e[[cc]])
  }
  e <- encode_bundle_aligned(trl, tel)
  for (cc in BND_RAW) data.table::set(tel, seq_len(nrow(tel)), cc, e[[cc]])
  finalize_bundle(trl); finalize_bundle(tel)
  stopifnot(!anyNA(trl[, ..BND_ENC_COLS]), !anyNA(tel[, ..BND_ENC_COLS]))
  invisible(NULL)
}
