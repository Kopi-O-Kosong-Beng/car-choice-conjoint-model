# =============================================================================
# Within-task DOMINANCE / CROSS-ATTRIBUTE CONTRAST feature block.
#
# LEAKAGE STATEMENT: every quantity below is a deterministic function of the
# ATTRIBUTE MATRIX of a choice task only. No outcome, no residual, no fitted
# value, no respondent identity enters. There is nothing to cross-fit, because
# nothing is estimated from labels. This block is computed identically for
# train and test rows, in one pass, with no reference to folds. Zero leakage
# risk by construction.
#
# Rows of `long` must be sorted by (No, alt) with exactly 4 contiguous rows per
# task and alt in 1..4. Alternative 4 is the all-zero "none" option.
# =============================================================================

# non-price attributes: higher tier = more/better feature. 0 = attribute not
# shown in this partial profile. Price: higher = more expensive = worse.
build_dom_feats <- function(long, attrs) {
  NP  <- setdiff(attrs, "Price")
  n   <- nrow(long)
  stopifnot(n %% 4 == 0, all(long$alt == rep(1:4, n / 4)))
  nt  <- n / 4
  MAXL <- vapply(NP, function(a) max(long[[a]]), numeric(1))

  # per-attribute (ntask x 3) matrices over the three REAL bundles
  R <- lapply(NP, function(a) matrix(long[[a]], ncol = 4, byrow = TRUE)[, 1:3, drop = FALSE])
  names(R) <- NP
  Pm <- matrix(long$Price, ncol = 4, byrow = TRUE)[, 1:3, drop = FALSE]

  z3 <- function(x) matrix(x, nrow = nt, ncol = 3)

  nbest <- matrix(0, nt, 3); nworst <- matrix(0, nt, 3)
  nuniqbest <- matrix(0, nt, 3); nuniqworst <- matrix(0, nt, 3)
  nactive <- numeric(nt)
  nsum <- matrix(0, nt, 3)            # normalised level sum (bundle "richness")
  maxadv <- matrix(-Inf, nt, 3)       # biggest normalised edge over the best rival
  maxdis <- matrix( Inf, nt, 3)       # biggest normalised deficit vs the best rival

  # pairwise Pareto counters over the 19 non-price attributes
  ge <- array(0L, c(nt, 3, 3))        # ge[,i,k] = # attrs where i >= k
  gt <- array(0L, c(nt, 3, 3))        # gt[,i,k] = # attrs where i >  k

  for (a in NP) {
    A  <- R[[a]]
    mx <- pmax(A[, 1], A[, 2], A[, 3])
    mn <- pmin(A[, 1], A[, 2], A[, 3])
    act <- mx > mn                    # attribute actually varies in this task
    nactive <- nactive + act
    isb <- (A == mx) & act; isw <- (A == mn) & act
    nbest  <- nbest  + isb
    nworst <- nworst + isw
    cb <- rowSums(isb); cw <- rowSums(isw)
    nuniqbest  <- nuniqbest  + isb * (cb == 1)
    nuniqworst <- nuniqworst + isw * (cw == 1)
    nsum <- nsum + A / MAXL[[a]]
    # edge vs the best rival on this attribute, normalised to the attribute range
    for (i in 1:3) {
      riv <- setdiff(1:3, i)
      d <- (A[, i] - pmax(A[, riv[1]], A[, riv[2]])) / MAXL[[a]]
      d[!act] <- NA_real_
      maxadv[, i] <- pmax(maxadv[, i], d, na.rm = TRUE)
      maxdis[, i] <- pmin(maxdis[, i], d, na.rm = TRUE)
    }
    for (i in 1:3) for (k in 1:3) if (i != k) {
      ge[, i, k] <- ge[, i, k] + (A[, i] >= A[, k])
      gt[, i, k] <- gt[, i, k] + (A[, i] >  A[, k])
    }
  }
  maxadv[!is.finite(maxadv)] <- 0; maxdis[!is.finite(maxdis)] <- 0
  nA <- length(NP)

  pmin3 <- pmin(Pm[, 1], Pm[, 2], Pm[, 3])
  pmax3 <- pmax(Pm[, 1], Pm[, 2], Pm[, 3])

  ndom_f   <- matrix(0, nt, 3); ndomby_f  <- matrix(0, nt, 3)
  ndom_fp  <- matrix(0, nt, 3); ndomby_fp <- matrix(0, nt, 3)
  for (i in 1:3) for (k in 1:3) if (i != k) {
    # features only: i weakly >= k everywhere, strictly > somewhere
    dfik <- (ge[, i, k] == nA) & (gt[, i, k] > 0)
    # features AND price: also no more expensive; strict on some dimension
    dpik <- (ge[, i, k] == nA) & (Pm[, i] <= Pm[, k]) &
            ((gt[, i, k] > 0) | (Pm[, i] < Pm[, k]))
    ndom_f[,  i] <- ndom_f[,  i] + dfik; ndomby_f[,  k] <- ndomby_f[,  k] + dfik
    ndom_fp[, i] <- ndom_fp[, i] + dpik; ndomby_fp[, k] <- ndomby_fp[, k] + dpik
  }

  nact3   <- z3(nactive)
  nsum_mu <- z3(rowMeans(nsum))
  nsum_mx <- z3(pmax(nsum[, 1], nsum[, 2], nsum[, 3]))
  pnorm3  <- Pm / 12
  pn_mu   <- z3(rowMeans(pnorm3))

  cheapest <- (Pm == z3(pmin3)); dearest <- (Pm == z3(pmax3))
  richest  <- (nsum == nsum_mx); poorest <- (nsum == z3(pmin(nsum[,1],nsum[,2],nsum[,3])))

  F3 <- list(
    dom_nbest      = nbest,
    dom_nworst     = nworst,
    dom_nuniqbest  = nuniqbest,
    dom_nuniqworst = nuniqworst,
    dom_nactive    = nact3,
    dom_fbest      = nbest  / pmax(nact3, 1),
    dom_fworst     = nworst / pmax(nact3, 1),
    dom_fnet       = (nbest - nworst) / pmax(nact3, 1),
    dom_ndom_f     = ndom_f,
    dom_ndomby_f   = ndomby_f,
    dom_ndom_fp    = ndom_fp,
    dom_ndomby_fp  = ndomby_fp,
    dom_undominated= (ndomby_fp == 0) * 1.0,
    dom_dominates  = (ndom_fp   >  0) * 1.0,
    dom_nsum       = nsum,
    dom_nsum_c     = nsum - nsum_mu,
    dom_nsum_gap   = nsum - nsum_mx,
    dom_price_gap  = (Pm - z3(pmin3)) / 12,
    dom_value_c    = (nsum - nsum_mu) - (pnorm3 - pn_mu),
    dom_maxadv     = maxadv,
    dom_maxdis     = maxdis,
    dom_cheapest   = cheapest * 1.0,
    dom_richest    = richest  * 1.0,
    dom_cheap_rich = (cheapest & richest) * 1.0,
    dom_dear_poor  = (dearest & poorest)  * 1.0
  )

  # Expand to 4 rows per task. Alternative 4 (the all-zero none option) gets 0 for
  # every bundle-level feature -- it is not a bundle. It instead receives two
  # task-level "how good is the field?" summaries, which are what can legitimately
  # move the none share under a softmax over four alternatives.
  out <- vector("list", length(F3) + 2)
  nm  <- character(length(F3) + 2)
  for (j in seq_along(F3)) {
    M <- cbind(F3[[j]], 0)
    out[[j]] <- as.vector(t(M)); nm[j] <- names(F3)[j]
  }
  j <- length(F3)
  best_val <- nsum_mx[, 1] - pmin3 / 12          # value of the best available bundle
  out[[j + 1]] <- as.vector(t(cbind(matrix(0, nt, 3), best_val)))
  nm[j + 1] <- "dom_none_bestval"
  out[[j + 2]] <- as.vector(t(cbind(matrix(0, nt, 3), nsum_mx[, 1])))
  nm[j + 2] <- "dom_none_bestrich"

  D <- data.table::as.data.table(stats::setNames(out, nm))
  stopifnot(nrow(D) == n, !anyNA(D))
  D
}

DOM_COLS <- c("dom_nbest","dom_nworst","dom_nuniqbest","dom_nuniqworst","dom_nactive",
              "dom_fbest","dom_fworst","dom_fnet","dom_ndom_f","dom_ndomby_f",
              "dom_ndom_fp","dom_ndomby_fp","dom_undominated","dom_dominates",
              "dom_nsum","dom_nsum_c","dom_nsum_gap","dom_price_gap","dom_value_c",
              "dom_maxadv","dom_maxdis","dom_cheapest","dom_richest","dom_cheap_rich",
              "dom_dear_poor","dom_none_bestval","dom_none_bestrich")
