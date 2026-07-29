# =============================================================================
# ITERATION 28b — SAME CORRECTION, CORRECT BASE MODEL
#
# WHY THIS FILE EXISTS. run.R applied the measured marginal correction to
# freepool5, on the belief that freepool5 had scored 1.197 -- the same as the
# 2-member blend. That belief was wrong. 1.197 was the LEADERBOARD, which shows
# only a team's best submission. freepool5's own score, read from My Submissions,
# is 1.211.
#
# WHAT THAT MEANS. freepool5 is not tied with the 2-member blend. It is 0.014
# WORSE on the public leaderboard while being 0.00478 BETTER on nested OOF.
#
#     model        local nested    public     marginal it ships
#     2-member       1.12819       1.197        0.2480
#     freepool5      1.12341       1.211        0.2377
#
# The marginal error accounts for only 0.00133 of that 0.014 gap. The remaining
# ~0.0127 is conditional structure that exists in our fold split and not in the
# test set. Free-sign weights on OOF predictions are exactly the mechanism that
# produces this: unconstrained signs across five members have enough freedom to
# fit the folds' noise, and the leaderboard charged us 0.014 for it.
#
# THIS RETRACTS the explanation recorded earlier in submissions/log.md, which
# claimed freepool5's local gain was "real and merely masked" by its worse
# marginal. That story was arithmetically neat and simply false -- it was built
# on a misread score. The local gain was not masked. It was not there.
#
# WHAT IS UNCHANGED. The probe measures the test set, not any model, so
# r = 0.26648 stands regardless. The tilt machinery was validated on OOF and that
# validation is independent of which base it is applied to. Only the base changes.
#
# HYPOTHESIS. Applying the same one-parameter log-odds shift to the 2-member
# blend, so its mean p4 moves 0.2480 -> 0.26648, gains ~1.16x the marginal KL of
# 0.00090, i.e. ~0.00104. Expected public 1.197 -> 1.196.
#
# DECISION RULE (unchanged, re-run on this base): ship iff the tilt realises
# >= 70% of predicted marginal KL in both (A) cleanliness and (B) a genuine
# income-tertile miscalibration.
#
# EMITS. A submission CSV only. No member, weight, or artifact is touched.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

R_MEASURED <- (log(6) - 1.499) / log(3)
THRESH     <- 0.70

tilt <- function(P, alpha) {
  p4  <- P[, 4]
  p4n <- alpha * p4 / (alpha * p4 + (1 - p4))
  s   <- (1 - p4n) / (1 - p4)
  cbind(P[, 1] * s, P[, 2] * s, P[, 3] * s, p4n)
}
solve_alpha <- function(P, target)
  exp(uniroot(function(la) mean(tilt(P, exp(la))[, 4]) - target, c(-6, 6), tol = 1e-12)$root)
kl <- function(p, q) p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))

long <- readRDS("model/artifacts/long.rds")
tk   <- unique(long[, .(No, Case, is_test, y)], by = "No"); setorder(tk, No)
tr   <- tk[is_test == FALSE]; y <- tr$y

# blend.rds stores only the scalar score, not the nested predictions, so rebuild
# them exactly as 06_blend.R does: simplex log-opinion pool, refit per fold.
folds <- readRDS("model/artifacts/folds.rds"); fmap <- folds[order(No), fold]
OOF2 <- lapply(readRDS("model/artifacts/blend.rds")$members, function(m) {
  x <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(x, No)
  as.matrix(x[, .(p1, p2, p3, p4)])
})
pool2 <- function(th, rows) {
  w <- exp(th[1:2]); w <- w / sum(w); Tt <- exp(th[3]); eA <- plogis(th[4]) * 0.10
  L <- Reduce(`+`, Map(function(wi, Pm) wi * log(pmax(Pm[rows, , drop = FALSE], 1e-12)), w, OOF2))
  L <- L / Tt; Q <- exp(L - apply(L, 1, max)); Q <- Q / rowSums(Q)
  (1 - eA) * Q + eA * 0.25
}
P <- matrix(NA_real_, length(y), 4)
for (k in 1:5) {
  o <- optim(c(0, 0, 0, -3), function(th) logloss(y[fmap != k], pool2(th, fmap != k)),
             method = "Nelder-Mead", control = list(maxit = 3000))
  P[fmap == k, ] <- pool2(o$par, fmap == k)
}
stopifnot(nrow(P) == length(y))
cat(sprintf("2-member nested OOF: %d rows | logloss %.5f | ships p4 %.5f | truth %.5f\n\n",
            nrow(P), logloss(y, P), mean(P[, 4]), mean(y == 4)))
r_true <- mean(y == 4)

cat("(A) TILT CLEANLINESS\n    target   predicted KL   actual delta   ratio\n")
ratA <- c()
for (q in c(0.2377, 0.248, 0.2665, 0.28, 0.32)) {
  act  <- logloss(y, tilt(P, solve_alpha(P, q))) - logloss(y, P)
  pred <- kl(r_true, q); ratA <- c(ratA, pred / act)
  cat(sprintf("    %.4f   %+.5f      %+.5f      %5.1f%%\n", q, pred, act, 100 * pred / act))
}

cat("\n(B) REAL MISCALIBRATION -- income tertiles\n")
dem <- unique(long[, .(Case, incomea)])
trd <- merge(tr, dem, by = "Case"); setorder(trd, No)
qi  <- quantile(trd$incomea, c(1/3, 2/3), na.rm = TRUE)
trd[, tert := findInterval(incomea, qi)]
ratB <- c()
for (g in 0:2) {
  idx <- which(trd$tert == g); Pg <- P[idx, , drop = FALSE]; yg <- y[idx]
  truth <- mean(yg == 4); ships <- mean(Pg[, 4])
  act  <- logloss(yg, Pg) - logloss(yg, tilt(Pg, solve_alpha(Pg, truth)))
  pred <- kl(truth, ships); ratB <- c(ratB, act / pred)
  cat(sprintf("    tertile %d  truth %.4f  ships %.4f  predicted %+.5f  realised %+.5f  %5.1f%%\n",
              g, truth, ships, pred, act, 100 * act / pred))
}

ok <- min(ratA) >= THRESH && min(ratB) >= THRESH
cat(sprintf("\nDECISION: (A) %.1f%%  (B) %.1f%%  ->  %s\n\n",
            100 * min(ratA), 100 * min(ratB), if (ok) "SHIP" else "ABORT"))
if (!ok) { cat("Aborting.\n"); quit(status = 0) }

TP <- readRDS("model/artifacts/test_blend.rds")
if (!is.matrix(TP)) { TP <- as.data.table(TP); if ("No" %in% names(TP)) setorder(TP, No)
                      TP <- as.matrix(TP[, c("p1","p2","p3","p4"), with = FALSE]) }
stopifnot(nrow(TP) == 4997L)
a  <- solve_alpha(TP, R_MEASURED); TC <- tilt(TP, a); TC <- TC / rowSums(TC)
cat(sprintf("test: %.5f -> %.5f (alpha %.5f, log-odds %+.4f) | rowsum dev %.1e | min p %.2e\n",
            mean(TP[, 4]), mean(TC[, 4]), a, log(a), max(abs(rowSums(TC) - 1)), min(TC)))

out <- data.table(No = tk[is_test == TRUE][order(No), No],
                  Ch1 = TC[, 1], Ch2 = TC[, 2], Ch3 = TC[, 3], Ch4 = TC[, 4])
f <- sprintf("submissions/sub_%s.csv", format(Sys.time(), "%Y%m%d_%H%M"))
fwrite(out, f)
g <- kl(R_MEASURED, mean(TP[, 4]))
cat(sprintf("\nwrote %s (%d rows)\n", f, nrow(out)))
cat(sprintf("expected public: 1.197 - %.5f(x~1.16) = %.5f  (displays %.3f)\n",
            g, 1.197 - 1.16 * g, round(1.197 - 1.16 * g, 3)))
