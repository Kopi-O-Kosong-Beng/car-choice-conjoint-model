# =============================================================================
# Bundle-level encoding at the highest-order key that actually recurs.
#
# Established in diagnostics.R / diagnostics2.R:
#   * the 20-attribute bundle key is a bijection with (design, position), so a
#     bundle-keyed win rate IS the existing design-share encoding, bit for bit;
#   * the 19-bit presence pattern is constant within a choice task (all three real
#     bundles are built from the same 9 features), so it is a choice-set property
#     and its within-task contrast is identically zero.
# Backing off further, the highest-order key that BOTH recurs across choice sets
# AND varies within a task is the marginal (attribute, level) cell: 91 cells with
# ~17,500 observations each, versus 3.83 for the design key.
#
# A bundle is then encoded by COMPOSING its own cells -- the empirical log-odds of
# being chosen, summed over the bundle's 20 attribute levels. This is a
# bundle-level statistic in the sense the brief intended (a property of the
# alternative itself, pooled over every context it appeared in), it varies within a
# task, and it is well supported. It is a naive-Bayes style score: it ignores
# interactions between attributes, which is exactly what buys it the support.
#
# The brief's "discount easy wins" term becomes the strength of the rivals actually
# faced: the max and mean composite score of the other real alternatives, plus
# their price and level-sum. (Rival RICHNESS cannot serve -- richness is 9 on every
# real bundle, so rich_task_mean is the constant 6.75.)
#
# LEAKAGE. Only raw `chosen` indicators from reference respondents enter; no model
# prediction of any kind. The iteration-12 failure mode -- reference respondents'
# baseline predictions having themselves been fitted on the scored fold -- is closed
# BY CONSTRUCTION rather than by argument. Respondents live entirely within one
# fold, so a fold-k row is encoded from folds != k only and nobody contributes to
# their own feature. Test rows use all 1,135 training respondents.
# =============================================================================

MRG_ALPHAS  <- c(20, 100, 500)
MRG_SCORE   <- paste0("m_score_a", MRG_ALPHAS)
MRG_SCORE_C <- paste0(MRG_SCORE, "_c")
MRG_RAW     <- MRG_SCORE
MRG_COLS    <- c(MRG_SCORE, MRG_SCORE_C,
                 "m_score_rivmax", "m_score_rivmean", "m_score_edge",
                 "riv_price_now", "riv_lvl_now")

marg_check_layout <- function(dt) {
  stopifnot(nrow(dt) %% 4 == 0)
  M <- matrix(dt$alt, ncol = 4, byrow = TRUE)
  stopifnot(all(M[, 1] == 1L), all(M[, 2] == 2L), all(M[, 3] == 3L), all(M[, 4] == 4L))
  N <- matrix(dt$No, ncol = 4, byrow = TRUE)
  stopifnot(all(N[, 1] == N[, 2]), all(N[, 1] == N[, 3]), all(N[, 1] == N[, 4]))
  invisible(TRUE)
}

# Strength of the rivals the alternative actually faces. Rivals are the other REAL
# alternatives; the all-zero none option is nobody's rival.
marg_spread <- function(v) {
  M <- matrix(v, ncol = 4, byrow = TRUE)
  s3 <- M[, 1] + M[, 2] + M[, 3]
  as.vector(t(cbind((s3 - M[, 1]) / 2, (s3 - M[, 2]) / 2, (s3 - M[, 3]) / 2, s3 / 3)))
}
marg_spread_max <- function(v) {
  M <- matrix(v, ncol = 4, byrow = TRUE)
  as.vector(t(cbind(pmax(M[, 2], M[, 3]), pmax(M[, 1], M[, 3]), pmax(M[, 1], M[, 2]),
                    pmax(M[, 1], pmax(M[, 2], M[, 3])))))
}

add_marg_context <- function(dt) {
  marg_check_layout(dt)
  dt[, riv_price_now := marg_spread(as.numeric(Price))]
  dt[, riv_lvl_now   := marg_spread(as.numeric(lvlsum))]
  invisible(NULL)
}

# Composite score of each target row from the reference respondents' cell rates.
encode_marginal_aligned <- function(ref, target) {
  refr <- ref[alt <= 3]
  pr   <- sum(refr$chosen) / nrow(refr)
  lo0  <- log(pr / (1 - pr))
  cells <- data.table::rbindlist(lapply(ATTRS, function(a)
    refr[, .(attr = a, lvl = as.integer(get(a)), ch = as.integer(chosen))][
      , .(n = .N, nch = sum(ch)), by = .(attr, lvl)]))
  cells[, ckey := paste0(attr, "=", lvl)]

  out <- vector("list", length(MRG_ALPHAS))
  for (i in seq_along(MRG_ALPHAS)) {
    a <- MRG_ALPHAS[i]
    r <- (cells$nch + a * pr) / (cells$n + a)
    contrib <- setNames(log(r / (1 - r)) - lo0, cells$ckey)
    s <- numeric(nrow(target))
    for (att in ATTRS) {
      v <- contrib[paste0(att, "=", as.integer(target[[att]]))]
      v[is.na(v)] <- 0                      # level never seen on a real bundle
      s <- s + v
    }
    out[[i]] <- unname(s)
  }
  names(out) <- MRG_SCORE
  data.table::as.data.table(out)
}

# Task-centre (only within-task contrasts carry utility under a softmax) and derive
# the rival-strength terms.
finalize_marginal <- function(dt) {
  marg_check_layout(dt)
  for (cc in MRG_SCORE) {
    M <- matrix(dt[[cc]], ncol = 4, byrow = TRUE)
    data.table::set(dt, NULL, paste0(cc, "_c"), as.vector(t(M - rowMeans(M))))
  }
  mid <- MRG_SCORE[2]
  data.table::set(dt, NULL, "m_score_rivmean", marg_spread(dt[[mid]]))
  data.table::set(dt, NULL, "m_score_rivmax",  marg_spread_max(dt[[mid]]))
  data.table::set(dt, NULL, "m_score_edge",    dt[[mid]] - dt[["m_score_rivmax"]])
  invisible(NULL)
}

apply_marginal_encoding <- function(trl, tel) {
  for (cc in MRG_RAW) { trl[, (cc) := NA_real_]; tel[, (cc) := NA_real_] }
  for (k in sort(unique(trl$fold))) {
    idx <- which(trl$fold == k)
    e <- encode_marginal_aligned(trl[fold != k], trl[fold == k])
    for (cc in MRG_RAW) data.table::set(trl, idx, cc, e[[cc]])
  }
  e <- encode_marginal_aligned(trl, tel)
  for (cc in MRG_RAW) data.table::set(tel, seq_len(nrow(tel)), cc, e[[cc]])
  finalize_marginal(trl); finalize_marginal(tel)
  stopifnot(!anyNA(trl[, ..MRG_COLS]), !anyNA(tel[, ..MRG_COLS]))
  invisible(NULL)
}
