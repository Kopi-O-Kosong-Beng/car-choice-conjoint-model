# =============================================================================
# ITER 57 -- PART 0: reconstruct the NESTED OOF blend PROBABILITY MATRIX.
#
# model/06_blend.R reports the nested logloss but does not persist the held-out
# blend probabilities. Every deploy-time transform has to be scored against
# those exact rows, so rebuild them here with a byte-identical objective
# (simplex weights x temperature x uniform-eps, Nelder-Mead, same starts).
#
# Reads only. Writes ONLY into this experiment directory, with the dt57_ prefix.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

DIR <- "experiments/iter57_deploytrim"

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
y     <- ymap$y
fmap  <- folds[order(No), fold]
stopifnot(length(y) == 21565L, length(fmap) == 21565L)

build <- function(memb, tag) {
  OOF <- lapply(memb, function(m) {
    d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
    as.matrix(d[, .(p1, p2, p3, p4)])
  })
  TST <- lapply(memb, function(m) {
    d <- readRDS(sprintf("model/artifacts/test_%s.rds", m)); setorder(d, No)
    as.matrix(d[, .(p1, p2, p3, p4)])
  })
  stopifnot(all(sapply(OOF, nrow) == 21565), all(sapply(TST, nrow) == 4997))
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
  fit <- function(rows) optim(c(rep(0, M), 0, -3), obj, rows = rows,
                              method = "Nelder-Mead", control = list(maxit = 3000))

  Poof <- matrix(NA_real_, 21565, 4)
  nested <- numeric(5)
  for (k in 1:5) {
    o  <- fit(fmap != k)
    te <- which(fmap == k)
    Poof[te, ] <- blend(o$par, OOF, te)
    nested[k]  <- logloss(y[te], Poof[te, ])
    cat(sprintf("  [%s] fold %d held-out blend logloss: %.5f\n", tag, k, nested[k]))
  }
  cat(sprintf(">>> [%s] NESTED blend OOF: %.5f  (+- %.5f)\n", tag, mean(nested), sd(nested)))
  stopifnot(!anyNA(Poof))

  of <- fit(rep(TRUE, length(y)))
  w  <- exp(of$par[1:M]); w <- w / sum(w)
  cat(sprintf("    full-fit weights: %s | T %.4f | eps %.5f\n",
              paste(memb, round(w, 3), collapse = " "),
              exp(of$par[M + 1]), plogis(of$par[M + 2]) * 0.10))
  Ptest <- clip_norm(blend(of$par, TST, rows = seq_len(4997)))

  saveRDS(list(members = memb, Poof = Poof, Ptest = Ptest, y = y, fold = fmap,
               Case = ymap$Case, nested = mean(nested), nested_folds = nested,
               par_full = of$par),
          file.path(DIR, sprintf("dt57_pack_%s.rds", tag)))
  invisible(NULL)
}

cat("=== production 2-member blend ===\n")
build(c("xgb_lw2bag", "lcmnl3_both"), "prod")

cat("\n=== fr10 variant (best built-but-unsent base) ===\n")
build(c("xgb_lw2fr10", "lcmnl3_both"), "fr10")

cat("\nOK\n")
