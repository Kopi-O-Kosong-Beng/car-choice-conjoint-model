# =============================================================================
# PRE-FLIGHT PROBE for iteration 24 (within-task dominance / contrast features).
#
# d10_rankfeats.R already ran a LINEAR probe of two of these features (fbest,
# fworst) against xgb_lw2's residual and found R2 = 0.000365. Before spending a
# 30-minute xgboost run I strengthen that test in the three ways it was weak:
#   (a) probe against xgb_mono, the model I would actually be modifying, not xgb_lw2
#   (b) probe NONLINEARLY -- a cross-fitted xgboost on the residual, since the
#       whole hypothesis is that these features matter in a way trees find hard
#       to assemble but easy to use once handed over. A linear probe cannot
#       refute a nonlinear claim.
#   (c) probe the features d10 never had: pairwise Pareto dominance counts,
#       undominated flags, and the cheapest-AND-richest conjunction.
#
# Cross-fitting uses the FIXED respondent-grouped folds.rds. A fold-k residual is
# predicted only by a probe trained on other folds' respondents, so the OOF R2
# below is an honest number and cannot be inflated by memorising residuals.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("experiments/iter24_contrasts/dom_feats.R")

long  <- readRDS("model/artifacts/long.rds"); setorder(long, No, alt)
folds <- readRDS("model/artifacts/folds.rds")

t0 <- proc.time()[3]
DOM <- build_dom_feats(long, ATTRS)
cat(sprintf("built %d dominance features in %.1fs\n", ncol(DOM), proc.time()[3] - t0))
long <- cbind(long, DOM)

tr <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No"); setorder(tr, No, alt)

cat("\n== feature sanity (real bundles only) ==\n")
for (v in c("dom_nbest","dom_nactive","dom_ndom_fp","dom_ndomby_fp","dom_undominated",
            "dom_cheap_rich","dom_dear_poor","dom_nsum_gap")) {
  x <- tr[alt <= 3][[v]]
  cat(sprintf("  %-16s mean %7.4f  sd %7.4f  range [%g, %g]\n", v, mean(x), sd(x), min(x), max(x)))
}
cat(sprintf("  tasks with >=1 Pareto-dominated bundle: %.1f%%\n",
            100 * tr[alt <= 3, .(any = max(dom_ndomby_fp) > 0), by = No][, mean(any)]))

for (BASE in c("xgb_mono", "xgb_lw2")) {
  oof <- readRDS(sprintf("model/artifacts/oof_%s.rds", BASE)); setorder(oof, No)
  P <- as.matrix(oof[, .(p1,p2,p3,p4)])
  tr[, phat := as.vector(t(P))]
  tr[, r := as.numeric(chosen) - phat]

  cat(sprintf("\n===== baseline %s (OOF %.5f) =====\n", BASE,
              logloss(unique(long[is_test==FALSE, .(No,y)])[order(No), y], P)))

  rl <- tr[alt <= 3]
  cat("-- LINEAR probes of residual (real bundles) --\n")
  sets <- list(
    "d10 replication: fbest+fworst"            = c("dom_fbest","dom_fworst"),
    "NEW: pareto counts"                       = c("dom_ndom_fp","dom_ndomby_fp","dom_undominated"),
    "NEW: conjunctions"                        = c("dom_cheap_rich","dom_dear_poor","dom_cheapest","dom_richest"),
    "NEW: gaps/edges"                          = c("dom_nsum_gap","dom_price_gap","dom_maxadv","dom_maxdis","dom_value_c"),
    "FULL new block"                           = setdiff(DOM_COLS, c("dom_none_bestval","dom_none_bestrich"))
  )
  for (nmk in names(sets)) {
    fs <- sets[[nmk]]
    fs <- fs[vapply(fs, function(f) sd(rl[[f]]) > 0, logical(1))]
    s <- summary(lm(reformulate(fs, "r"), data = rl))
    cat(sprintf("  %-38s k=%2d  R2 %.6f\n", nmk, length(fs), s$r.squared))
  }

  # ---- nonlinear cross-fitted probe: can ANY function of the new block predict
  #      the incumbent's out-of-fold residual on held-out respondents?
  fs <- DOM_COLS[vapply(DOM_COLS, function(f) sd(tr[[f]]) > 0, logical(1))]
  X <- as.matrix(tr[, ..fs]); rr <- tr$r
  pr <- rep(NA_real_, nrow(tr))
  for (k in 1:5) {
    i_tr <- which(tr$fold != k); i_te <- which(tr$fold == k)
    m <- xgb.train(list(objective = "reg:squarederror", eta = 0.05, max_depth = 6,
                        min_child_weight = 20, subsample = 0.8, colsample_bytree = 0.8,
                        nthread = 3),
                   xgb.DMatrix(X[i_tr, ], label = rr[i_tr]), nrounds = 400, verbose = 0)
    pr[i_te] <- predict(m, xgb.DMatrix(X[i_te, ]))
  }
  sse <- sum((rr - pr)^2); sst <- sum((rr - mean(rr))^2)
  cat(sprintf("-- NONLINEAR cross-fitted probe (all 4 alts, %d feats) --\n", length(fs)))
  cat(sprintf("  out-of-fold R2 on residual: %+.6f   (<=0 means nothing left to find)\n", 1 - sse/sst))
  ii <- which(tr$alt <= 3)
  cat(sprintf("  restricted to real bundles: %+.6f\n",
              1 - sum((rr[ii]-pr[ii])^2)/sum((rr[ii]-mean(rr[ii]))^2)))
  # generous upper bound on the logloss this could ever buy (optimal rescale of the
  # cross-fitted correction, chosen IN-sample -> optimistic on purpose)
  y <- unique(long[is_test==FALSE, .(No,y)])[order(No), y]
  Lg <- log(pmax(P, 1e-12)); Cm <- matrix(pr, ncol = 4, byrow = TRUE)
  f <- function(c) { LL <- Lg + c*Cm; Q <- exp(LL - apply(LL,1,max)); logloss(y, Q/rowSums(Q)) }
  o <- optimize(f, c(-30, 30))
  cat(sprintf("  optimistic logloss with best in-sample rescale: %.5f (gain %+.5f at c=%.2f)\n",
              o$objective, logloss(y,P) - o$objective, o$minimum))
}
cat(sprintf("\ntotal %.1fs\n", proc.time()[3] - t0))
