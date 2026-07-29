# =============================================================================
# ITERATION 29b — DOES A WEALTHIER POPULATION WANT A DIFFERENT BLEND WEIGHT?
#
# WHAT 29a FOUND. Fitting the combiner on the poorer half of respondents and
# evaluating on the richer half, the richer half's own optimum sits at
# xgb_lw2bag 0.610 / lcmnl3_both 0.390. Fitting on the poor gives 0.455/0.545.
# Production ships 0.528/0.472. The test respondents are ~2x wealthier.
#
# THIS AGREES WITH THE PROBE, INDEPENDENTLY. The alt-4 probe measured the test
# none-rate at 0.26648. lcmnl3_both predicted 0.224 (worst of any model); the
# tree family predicted 0.273 (nearly exact). Two unrelated lines of evidence say
# the SAME thing: lcmnl3_both's demographic channel degrades on new, wealthier
# people, and it is carrying too much weight for the test population.
#
# THE TRAP THIS MUST AVOID. CLAUDE.md, from iteration 07: predict public from the
# income-reweighted OOF, but NEVER OPTIMISE ON IT. So this does not tune a weight
# against a reweighted score. Every number below is a genuine held-out
# evaluation: the weight is read off respondents excluded from everything used to
# choose it, and a RANDOM-split control of identical size runs alongside to show
# how much of any shift is just noise.
#
# DECISION RULE, fixed before running. Adopt a weight shift only if:
#   (1) the income holdouts move toward xgb CONSISTENTLY (>=3 of 4), and
#   (2) the shift exceeds the random-split control's spread, and
#   (3) the gain at the SHRUNK weight (halfway, not all the way) is positive on
#       every income holdout.
# Anything less and production keeps 0.528. Half the evidence for freepool5 was
# stronger than this and it cost 0.014.
#
# DIAGNOSTIC ONLY -- emits nothing.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long <- readRDS("model/artifacts/long.rds")
tk   <- unique(long[, .(No, Case, is_test, y)], by = "No"); setorder(tk, No)
tr   <- tk[is_test == FALSE]; y <- tr$y; cases <- tr$Case

memb <- readRDS("model/artifacts/blend.rds")$members
OOF  <- lapply(memb, function(m) {
  x <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(x, No)
  as.matrix(x[, .(p1, p2, p3, p4)])
})
L1 <- log(pmax(OOF[[1]], 1e-12)); L2 <- log(pmax(OOF[[2]], 1e-12))

# pure log-opinion pool at weight w on member 1 -- no temperature, no eps, so the
# only thing moving is the quantity under test
at_w <- function(w, rows) {
  L <- w * L1[rows, , drop = FALSE] + (1 - w) * L2[rows, , drop = FALSE]
  Q <- exp(L - apply(L, 1, max)); Q / rowSums(Q)
}
best_w <- function(rows) optimize(function(w) logloss(y[rows], at_w(w, rows)), c(0.05, 0.95))$minimum
loss_w <- function(w, rows) logloss(y[rows], at_w(w, rows))

PROD <- 0.528
dem  <- unique(long[, .(Case, incomea)])
u    <- unique(cases); inc <- dem[match(u, dem$Case), incomea]

cat(sprintf("production weight on %s: %.3f\n", memb[1], PROD))
cat(sprintf("optimal on ALL training respondents: %.3f\n\n", best_w(seq_along(y))))

# --- income holdouts: the richest q of respondents, never used to pick anything -
cat("=== INCOME HOLDOUTS (the richer the group, the more it should want xgb) ===\n")
cat("  held-out group        n_resp   optimal w   loss@0.528   loss@opt    gain\n")
res <- data.table()
for (q in c(0.50, 0.60, 0.70, 0.80)) {
  rich <- u[inc > quantile(inc, q, na.rm = TRUE)]
  rows <- which(cases %in% rich)
  w <- best_w(rows); g <- loss_w(PROD, rows) - loss_w(w, rows)
  res <- rbind(res, data.table(q, w, gain = g))
  cat(sprintf("  top %2.0f%% by income     %4d     %.3f       %.5f    %.5f   %+.5f\n",
              100 * (1 - q), length(rich), w, loss_w(PROD, rows), loss_w(w, rows), g))
}

# --- random control: same group sizes, no income structure --------------------
cat("\n=== RANDOM CONTROL (same sizes, income ignored) ===\n")
set.seed(7)
for (q in c(0.50, 0.60, 0.70, 0.80)) {
  ws <- replicate(40, best_w(which(cases %in% sample(u, round(length(u) * (1 - q))))))
  cat(sprintf("  n=%4d   optimal w over 40 random draws: mean %.3f  sd %.3f  range [%.3f, %.3f]\n",
              round(length(u) * (1 - q)), mean(ws), sd(ws), min(ws), max(ws)))
}

# --- the shrunk candidate, evaluated on every income holdout ------------------
W_SHRUNK <- PROD + 0.5 * (median(res$w) - PROD)
cat(sprintf("\n=== SHRUNK CANDIDATE w = %.3f (halfway from %.3f to the median holdout optimum %.3f) ===\n",
            W_SHRUNK, PROD, median(res$w)))
allpos <- TRUE
for (q in c(0.50, 0.60, 0.70, 0.80)) {
  rows <- which(cases %in% u[inc > quantile(inc, q, na.rm = TRUE)])
  g <- loss_w(PROD, rows) - loss_w(W_SHRUNK, rows)
  if (g <= 0) allpos <- FALSE
  cat(sprintf("  top %2.0f%%: %+.5f\n", 100 * (1 - q), g))
}
gfull <- loss_w(PROD, seq_along(y)) - loss_w(W_SHRUNK, seq_along(y))
cat(sprintf("  cost on the FULL training population: %+.5f  (negative = we pay here)\n", gfull))

cat("\n--- the curve near the production weight, on all respondents ---\n")
for (w in seq(0.40, 0.80, by = 0.05))
  cat(sprintf("  w=%.2f  %.5f\n", w, loss_w(w, seq_along(y))))

cat(sprintf("\nDECISION: consistent(>=3/4) %s | exceeds control %s | all holdouts positive %s\n",
            sum(res$w > PROD) >= 3, median(res$w) - PROD > 0.05, allpos))
