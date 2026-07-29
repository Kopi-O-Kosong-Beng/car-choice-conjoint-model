# Nested simplex log-opinion-pool blend, transcribed from model/06_blend.R.
# Reads OOF matrices from ARBITRARY PATHS so an experiment can blend files that are
# not in model/artifacts and never touches members.txt / blend.rds.
#
# Args: <path1> <path2> ...     (each an oof_*.rds with No,p1..p4, 21565 rows)
# First run with the production pair to confirm it reproduces 1.12819.
suppressMessages(library(data.table))
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
paths <- commandArgs(TRUE)
stopifnot(length(paths) >= 2)
OOF <- lapply(paths, function(p) { d <- readRDS(p); setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
stopifnot(all(sapply(OOF, nrow) == 21565))
ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No); y <- ymap$y
fmap <- folds[order(No), fold]
M <- length(OOF); eps0 <- 1e-12
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
for (k in 1:5) { o <- fit_w(fmap != k); nested[k] <- obj(o$par, fmap == k) }
o <- fit_w(rep(TRUE, length(y)))
w <- exp(o$par[1:M]); w <- w / sum(w)
for (i in seq_along(paths)) cat(sprintf("  %-58s single %.5f  w %.3f\n",
    basename(paths[i]), logloss(y, OOF[[i]]), w[i]))
cat(sprintf("  per-fold: %s\n", paste(round(nested, 5), collapse = " ")))
cat(sprintf(">>> NESTED blend OOF %.5f   (plain %.5f)\n", mean(nested), o$value))
