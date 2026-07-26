# =============================================================================
# Non-destructive blend probe. Does NOT touch model/members.txt or any production
# artifact -- it re-implements 06_blend.R's objective exactly and evaluates an
# explicit member list passed on the command line.
#
#   Rscript experiments/iter17_hb/blend_probe.R xgb_lw2,xgb_mono,lcmnl3
#   Rscript experiments/iter17_hb/blend_probe.R xgb_lw2,xgb_mono,lcmnl3,hbmnl
#
# Reports the NESTED blend OOF, which is the repository's decision number.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: blend_probe.R name1,name2,...")
memb <- trimws(strsplit(args[1], ",")[[1]])
cat("members:", paste(memb, collapse = ", "), "\n")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
OOF <- lapply(memb, function(m) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  as.matrix(d[, .(p1, p2, p3, p4)])
})
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
  o <- fit_w(fmap != k); nested[k] <- obj(o$par, fmap == k)
  cat("fold", k, "held-out blend logloss:", round(nested[k], 5), "\n")
}
cat(">>> NESTED blend OOF:", round(mean(nested), 5), "+-", round(sd(nested), 5), "\n")
o <- fit_w(rep(TRUE, length(y)))
w <- exp(o$par[1:M]); w <- w / sum(w)
cat("plain OOF:", round(o$value, 5), "\n")
cat("weights:", paste(memb, round(w, 3), collapse = " | "), "\n")
for (i in seq_along(memb))
  cat(sprintf("  single %-14s OOF: %.5f\n", memb[i], logloss(y, OOF[[i]])))
# per-fold nested losses, saved so two member sets can be paired by respondent
saveRDS(list(members = memb, nested = nested, mean = mean(nested)),
        sprintf("experiments/iter17_hb/blendprobe_%s.rds",
                paste(substr(memb, 1, 4), collapse = "-")))
cat("OK\n")
