# =============================================================================
# ITERATION 21 — nested blend probe under an ALTERNATIVE fold structure.
# Identical objective and optimiser to experiments/iter17_hb/blend_probe.R (which
# mirrors model/06_blend.R); the only changes are the fold file and that the fitted
# theta vectors are saved so weight vectors can be compared across fold structures.
#
#   Rscript experiments/iter21_foldrobust/blend_probe_split.R <42|b|c> name1,name2,...
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: blend_probe_split.R <42|b|c> name1,name2,...")
SPLIT <- args[1]
memb  <- trimws(strsplit(args[2], ",")[[1]])
FOLDFILE <- if (SPLIT == "42") {
  "model/artifacts/folds.rds"
} else {
  sprintf("model/artifacts/folds_%s.rds", SPLIT)
}
cat("split:", SPLIT, " fold file:", FOLDFILE, "\nmembers:", paste(memb, collapse = ", "), "\n")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS(FOLDFILE)
OOF <- lapply(memb, function(m) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  as.matrix(d[, .(p1, p2, p3, p4)])
})
stopifnot(all(sapply(OOF, nrow) == 21565))
ymap <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
y <- ymap$y
stopifnot(identical(folds[order(No), No], ymap$No))
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

nested <- numeric(5); thetas_k <- vector("list", 5)
loss_row <- rep(NA_real_, length(y))
for (k in 1:5) {
  o <- fit_w(fmap != k); thetas_k[[k]] <- o$par
  rows <- fmap == k
  nested[k] <- obj(o$par, rows)
  P <- blend(o$par, OOF, rows)
  loss_row[which(rows)] <- -log(pmax(P[cbind(seq_len(sum(rows)), y[rows])], 1e-15))
  cat("fold", k, "held-out blend logloss:", round(nested[k], 5), "\n")
}
cat(">>> NESTED blend OOF:", round(mean(nested), 5), "+-", round(sd(nested), 5), "\n")
cat("    (row-weighted nested OOF:", round(mean(loss_row), 5), ")\n")

o_full <- fit_w(rep(TRUE, length(y)))
w <- exp(o_full$par[1:M]); w <- w / sum(w)
cat("plain OOF:", round(o_full$value, 5), "\n")
cat("full-data weights:", paste(memb, round(w, 3), collapse = " | "), "\n")
cat("  temperature T =", round(exp(o_full$par[M + 1]), 4),
    " eps_A =", round(plogis(o_full$par[M + 2]) * 0.10, 5), "\n")
for (i in seq_along(memb))
  cat(sprintf("  single %-16s OOF: %.5f\n", memb[i], logloss(y, OOF[[i]])))

saveRDS(list(split = SPLIT, members = memb, nested = nested, mean = mean(nested),
             theta_full = o_full$par, w_full = w, thetas_k = thetas_k,
             loss_row = loss_row, No = ymap$No, Case = ymap$Case),
        sprintf("experiments/iter21_foldrobust/blend_%s_%s.rds",
                SPLIT, paste(substr(memb, 1, 6), collapse = "-")))
cat("OK\n")
