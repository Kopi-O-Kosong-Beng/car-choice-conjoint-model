# =============================================================================
# READ-ONLY blend probe. Identical maths to model/06_blend.R (same objective,
# same nested protocol, same 5 fixed folds) but it SAVES NOTHING -- it does not
# touch model/artifacts/blend.rds, model/artifacts/test_blend.rds, or
# model/members.txt. Purely to answer "would swapping lcmnl3 for <x> move the
# decision number?" without editing production state.
#
# Usage:
#   ... blend_probe.R mnl_pw xgb_lw2 xgb_mono lcmnl3
#   ... blend_probe.R mnl_pw xgb_lw2 xgb_mono lcmnl4
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
memb <- commandArgs(trailingOnly = TRUE)
stopifnot(length(memb) >= 2)
cat("members:", paste(memb, collapse = ", "), "\n")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
OOF <- lapply(memb, function(m) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  as.matrix(d[, .(p1, p2, p3, p4)])
})
names(OOF) <- memb
stopifnot(all(sapply(OOF, nrow) == 21565))

ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
y <- ymap$y
fmap <- folds[order(No), fold]
M <- length(memb); eps0 <- 1e-12

blend <- function(theta, Ps, rows) {
  w <- exp(theta[1:M]); w <- w / sum(w)
  Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, Ps))
  L <- L / Tt
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
obj <- function(theta, rows) logloss(y[rows], blend(theta, OOF, rows))
fit_w <- function(rows) optim(c(rep(0, M), 0, -3), obj, rows = rows,
                              method = "Nelder-Mead", control = list(maxit = 3000))

nested <- numeric(5)
for (k in 1:5) {
  o <- fit_w(fmap != k)
  nested[k] <- obj(o$par, fmap == k)
  cat("fold", k, "held-out blend logloss:", round(nested[k], 5), "\n")
}
cat(">>> NESTED blend OOF (decision number):", round(mean(nested), 5),
    "+-", round(sd(nested), 5), "\n")
o <- fit_w(rep(TRUE, length(y)))
w <- exp(o$par[1:M]); w <- w / sum(w)
cat("full-fit weights:", paste(memb, round(w, 3), collapse = " | "), "\n")
for (m in memb) cat(sprintf("  single %-14s OOF: %.5f\n", m, logloss(y, OOF[[m]])))
cat("NOTE: nothing was written. This script is read-only by construction.\n")
