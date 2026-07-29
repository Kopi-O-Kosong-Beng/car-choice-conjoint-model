# FOLDS_B member-level replication of the free-sign control-variate mechanism.
suppressMessages(library(data.table))
setwd("/Users/sheil/Desktop/SUTD/Y2/T5/40.016 The Analytics Edge/TAE_R-izzlers")
source("model/99_utils.R")
SCR <- "/private/tmp/claude-501/-Users-sheil-Desktop-SUTD-Y2-T5-40-016-The-Analytics-Edge-TAE-R-izzlers/87738d43-6ae7-4a89-b6d6-d01a77b3f47c/scratchpad"
long <- readRDS("model/artifacts/long.rds")
fb <- readRDS("model/artifacts/folds_b.rds"); fmap <- fb[order(No), fold]
ymap <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
y <- ymap$y; n <- length(y); case <- ymap$Case
loadO <- function(m) {
  p <- if (file.exists(sprintf("model/artifacts/oof_%s.rds", m)))
    sprintf("model/artifacts/oof_%s.rds", m) else sprintf("%s/oof_%s.rds", SCR, m)
  d <- readRDS(p); setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)])
}
smx <- function(S) { E <- exp(S - apply(S, 1, max)); E / rowSums(E) }
fit_free <- function(LP, rows, eps_grid = seq(0, 0.09, by = 0.005)) {
  M <- length(LP)
  fg <- function(beta, eps) {
    S <- Reduce(`+`, Map(function(b, l) b * l[rows, , drop=FALSE], as.list(beta), LP))
    P <- smx(S); idx <- cbind(seq_along(rows), y[rows])
    q <- (1 - eps) * P[idx] + eps * 0.25
    val <- -mean(log(pmax(q, 1e-300)))
    Ind <- matrix(0, length(rows), 4); Ind[idx] <- 1
    G <- -(1 - eps) * (P[idx] / q) * (Ind - P)
    gr <- vapply(seq_len(M), function(m) sum(G * LP[[m]][rows, , drop=FALSE]), numeric(1)) / length(rows)
    list(value = val, grad = gr)
  }
  best <- NULL; b0 <- rep(1 / M, M)
  for (eps in eps_grid) {
    o <- optim(b0, fn = function(b) fg(b, eps)$value, gr = function(b) fg(b, eps)$grad,
               method = "BFGS", control = list(maxit = 500, reltol = 1e-12))
    if (is.null(best) || o$value < best$value) best <- list(value = o$value, beta = o$par, eps = eps)
    b0 <- o$par
  }
  best
}
nested <- function(memb) {
  LP <- lapply(memb, function(m) log(pmax(loadO(m), 1e-12)))
  li <- numeric(n); pf <- numeric(5); B <- list()
  for (k in 1:5) {
    tr <- which(fmap != k); te <- which(fmap == k)
    f <- fit_free(LP, tr)
    S <- Reduce(`+`, Map(function(b, l) b * l[te, , drop=FALSE], as.list(f$beta), LP))
    P <- (1 - f$eps) * smx(S) + f$eps * 0.25
    li[te] <- -log(pmax(P[cbind(seq_along(te), y[te])], 1e-15))
    pf[k] <- mean(li[te]); B[[k]] <- round(f$beta, 3)
  }
  list(nested = mean(pf), li = li, pf = pf, B = B)
}
z <- function(a, b) { d <- a - b; g <- tapply(d, case, mean); mean(g)/(sd(g)/sqrt(length(g))) }

b2 <- nested(c("xgb_lw2bag3_b", "lcmnl3_both_b"))
cat(sprintf("folds_b 2-free base           : %.5f\n", b2$nested))
b3 <- nested(c("xgb_lw2bag3_b", "lcmnl3_both_b", "xgb_long_b"))
cat(sprintf("folds_b +xgb_long_b           : %.5f  gain %+.5f  z %+.2f\n",
    b3$nested, b2$nested - b3$nested, z(b2$li, b3$li)))
cat("   betas per fold:", paste(sapply(b3$B, function(x) x[3]), collapse=" "), "\n")
b4 <- nested(c("xgb_lw2bag3_b", "lcmnl3_both_b", "xgb_long_b", "xgb_wide_b"))
cat(sprintf("folds_b +long_b+wide_b        : %.5f  gain %+.5f  z %+.2f   (vs 3: %+.5f z %+.2f)\n",
    b4$nested, b2$nested - b4$nested, z(b2$li, b4$li), b3$nested - b4$nested, z(b3$li, b4$li)))
cat("   betas per fold (long, wide):",
    paste(sapply(b4$B, function(x) paste0(x[3], "/", x[4])), collapse="  "), "\n")
bp <- nested(c("xgb_lw2bag3_b", "lcmnl3_both_b", "xgb_mono_b"))
cat(sprintf("folds_b +xgb_mono_b (placebo) : %.5f  gain %+.5f  z %+.2f  betas %s\n",
    bp$nested, b2$nested - bp$nested, z(b2$li, bp$li),
    paste(sapply(bp$B, function(x) x[3]), collapse=" ")))
cat("OK foldsb\n")
