# =============================================================================
# ITERATION 29 — WHERE DOES THE +0.069 GO?  (read-only diagnostic, emits nothing)
#
# THE FACT NOBODY HAS EXPLAINED. Our nested OOF is 1.12819 and our public is
# 1.197. The offset is +0.069. Every other team's offset is SMALLER:
#
#     Sheil v1     1.17683 -> 1.2230   +0.046
#     rival somesh 1.161   -> 1.210    +0.049
#     us (4 mem)   1.13044 -> 1.199    +0.069
#     us (2 mem)   1.12819 -> 1.197    +0.069
#     freepool5    1.12341 -> 1.211    +0.088   <- more fitting, bigger offset
#
# The offset GROWS with how hard we fit. The alt-4 probe accounted for 0.001 of
# it. The other 0.068 is unexplained, and it is 6x the entire gap to first place.
#
# THE HYPOTHESIS. Our OOF is respondent-grouped, so it is honest about NEW
# RESPONDENTS FROM THE SAME POOL. It is not honest about a new POPULATION. Two
# distinct things could be leaking:
#   (i)  the blend PARAMETERS (weights, temperature, eps) are fitted on 1,135
#        specific respondents and may not transfer;
#   (ii) the members are OVERCONFIDENT out of population, in which case the fix
#        is a flatter temperature, which is one parameter and costs nothing.
#
# THE DESIGN. Simulate the actual test situation using only data we have. Split
# the 1,135 training respondents into two DISJOINT halves. Fit the combiner on
# half A. Evaluate on half B. Compare against the oracle that fitted on B. The
# gap is exactly "cost of applying our fitted combiner to people it never saw".
# Two splits:
#   RANDOM  — new respondents, same population. Isolates parameter overfitting.
#   INCOME  — fit on the poorer half, evaluate on the richer half. This mimics
#             the real shift, since the test respondents are ~2x wealthier.
#
# THEN, on the held-out half, ask three questions that map to three cheap moves:
#   Q1  Does an extra TEMPERATURE (flattening) help out of population?  -> tilt-
#       style one-parameter fix, appliable to the test file with no refitting.
#   Q2  Does EQUAL WEIGHTING, with no fitted temperature at all, beat the fitted
#       combiner out of population?  -> less fitting, not more.
#   Q3  Is lcmnl3_both's fitted weight too HIGH out of population? Its test
#       marginal was measurably wrong (0.224 vs a measured 0.26648), which is
#       direct evidence its demographic channel misbehaves on new people.
#
# DIAGNOSTIC ONLY. Emits nothing, changes nothing, needs no submission slot.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
tk    <- unique(long[, .(No, Case, is_test, y)], by = "No"); setorder(tk, No)
tr    <- tk[is_test == FALSE]; y <- tr$y; cases <- tr$Case

memb <- readRDS("model/artifacts/blend.rds")$members
OOF  <- lapply(memb, function(m) {
  x <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(x, No)
  as.matrix(x[, .(p1, p2, p3, p4)])
})
names(OOF) <- memb
cat("members:", paste(memb, collapse = " + "), "\n")
for (m in memb) cat(sprintf("  %-14s single OOF %.5f\n", m, logloss(y, OOF[[m]])))

# --- the production combiner, exactly as 06_blend.R defines it ---------------
pool <- function(th, rows, W = NULL) {
  w <- if (is.null(W)) { e <- exp(th[1:2]); e / sum(e) } else W
  Tt <- exp(th[3]); eA <- plogis(th[4]) * 0.10
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], 1e-12)), w, OOF))
  L <- L / Tt; Q <- exp(L - apply(L, 1, max)); Q <- Q / rowSums(Q)
  (1 - eA) * Q + eA * 0.25
}
fit <- function(rows) optim(c(0, 0, 0, -3), function(th) logloss(y[rows], pool(th, rows)),
                            method = "Nelder-Mead", control = list(maxit = 4000))$par

# extra temperature applied AFTER the combiner: one parameter, no refitting
retemp <- function(P, Tx) { Q <- exp(log(pmax(P, 1e-12)) / Tx); Q / rowSums(Q) }

report <- function(label, Atr, Bte) {
  thA <- fit(Atr); thB <- fit(Bte)
  PB_transfer <- pool(thA, Bte)          # our params, new people
  PB_oracle   <- pool(thB, Bte)          # best possible params for those people
  l_tr <- logloss(y[Bte], PB_transfer); l_or <- logloss(y[Bte], PB_oracle)

  wA <- { e <- exp(thA[1:2]); e / sum(e) }
  cat(sprintf("\n=== %s ===\n", label))
  cat(sprintf("  weights fitted on A: %s\n",
              paste(sprintf("%s %.3f", memb, wA), collapse = "  ")))
  cat(sprintf("  transfer %.5f | oracle %.5f | PARAMETER OVERFIT COST %+.5f\n",
              l_tr, l_or, l_tr - l_or))

  # Q1 -- extra temperature on the held-out half
  o <- optimize(function(t) logloss(y[Bte], retemp(PB_transfer, t)), c(0.5, 3))
  cat(sprintf("  Q1 optimal EXTRA temperature %.4f  -> gain %+.5f  %s\n",
              o$minimum, l_tr - o$objective,
              if (o$minimum > 1.01) "(model is OVERCONFIDENT out of population)"
              else if (o$minimum < 0.99) "(underconfident)" else "(already calibrated)"))

  # Q2 -- equal weights, no temperature, no eps
  Peq <- pool(c(0, 0, 0, -50), Bte, W = c(0.5, 0.5))
  cat(sprintf("  Q2 equal-weight, no temp: %.5f  vs fitted %.5f  -> %+.5f\n",
              logloss(y[Bte], Peq), l_tr, l_tr - logloss(y[Bte], Peq)))

  # Q3 -- what weight would the held-out people have wanted?
  wB <- { e <- exp(thB[1:2]); e / sum(e) }
  cat(sprintf("  Q3 held-out-optimal weights: %s   (shift %+.3f toward %s)\n",
              paste(sprintf("%s %.3f", memb, wB), collapse = "  "),
              wB[1] - wA[1], memb[1]))
  invisible(c(overfit = l_tr - l_or, temp = o$minimum, tempgain = l_tr - o$objective))
}

set.seed(42)
u  <- unique(cases); half <- sample(u, length(u) %/% 2)
report("RANDOM respondent split (new people, same population)",
       which(cases %in% half), which(!cases %in% half))

dem <- unique(long[, .(Case, incomea)])
inc <- dem[match(u, dem$Case), incomea]
poor <- u[inc <= median(inc, na.rm = TRUE)]
report("INCOME split: fit on POORER half, evaluate on RICHER half",
       which(cases %in% poor), which(!cases %in% poor))

cat("\n--- what the test set actually needs ---\n")
cat("If Q1 shows a temperature > 1 that transfers across BOTH splits, the same\n")
cat("one-parameter flattening applies to the test file with nothing refitted,\n")
cat("and it is the only lever here big enough to matter at 0.001-per-rank.\n")
