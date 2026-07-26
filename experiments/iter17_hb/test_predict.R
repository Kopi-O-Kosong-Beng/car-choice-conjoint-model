# =============================================================================
# Unit tests for the prediction path in run.R. The function is EXTRACTED from the
# real source file (not copied) so the test cannot drift from what actually runs.
#
#   Rscript experiments/iter17_hb/test_predict.R
# =============================================================================
suppressMessages(library(bayesm))
src <- parse("experiments/iter17_hb/run.R")
env <- new.env()
got <- character(0)
for (e in as.list(src)) {
  if (is.call(e) && identical(as.character(e[[1]]), "<-") &&
      as.character(e[[2]]) %in% c("predict_pop", "probs_fixed")) {
    eval(e, envir = env); got <- c(got, as.character(e[[2]]))
  }
}
stopifnot(setequal(got, c("predict_pop", "probs_fixed")))
env$S_SIM <- 200
environment(env$predict_pop) <- env
cat("extracted:", paste(got, collapse = ", "), "\n")

set.seed(7)
nvar <- 5; nz <- 3; ntask <- 400; n <- 4 * ntask; R_ho <- 20
X <- matrix(rnorm(n * nvar), n, nvar)
Zho <- matrix(rnorm(R_ho * nz), R_ho, nz)
resp_idx <- rep(sample.int(R_ho, ntask, replace = TRUE), each = 4L)
beta0 <- rnorm(nvar)
Delta <- matrix(rnorm(nz * nvar), nz, nvar)

mkfit <- function(D, sdscale, delta_on) {
  rooti <- diag(nvar) / sdscale                       # Sigma = sdscale^2 * I
  list(nvar = nvar, nz = nz,
       compdraw = replicate(D, list(list(mu = beta0, rooti = rooti)), simplify = FALSE),
       probdraw = matrix(1, D, 1),
       Deltadraw = matrix(rep(if (delta_on) as.vector(Delta) else rep(0, nz * nvar), D),
                          D, nz * nvar, byrow = TRUE))
}
softmax4 <- function(v) { M <- matrix(v, ncol = 4, byrow = TRUE)
  m <- pmax(M[,1],M[,2],M[,3],M[,4]); E <- exp(M - m); E / rowSums(E) }

# --- 1. degenerate mixture (Sigma -> 0), Delta = 0  => plain MNL at beta0 -----
P1 <- env$predict_pop(mkfit(3, 1e-9, FALSE), X, resp_idx, Zho, S = 5, seed = 1)
E1 <- softmax4(as.vector(X %*% beta0))
cat(sprintf("test 1  degenerate mixture, no Delta : max abs err %.3e\n", max(abs(P1 - E1))))
stopifnot(max(abs(P1 - E1)) < 1e-8)

# --- 2. degenerate mixture WITH Delta => MNL at beta0 + Delta'z_r -------------
P2 <- env$predict_pop(mkfit(3, 1e-9, TRUE), X, resp_idx, Zho, S = 5, seed = 1)
B  <- Zho %*% Delta                                   # R_ho x nvar, bayesm's Z*Delta
E2 <- softmax4(rowSums(X * B[resp_idx, , drop = FALSE]) + as.vector(X %*% beta0))
cat(sprintf("test 2  degenerate mixture, + Delta  : max abs err %.3e\n", max(abs(P2 - E2))))
stopifnot(max(abs(P2 - E2)) < 1e-8)

# --- 3. use_delta = FALSE must reproduce test 1 even when Delta is non-zero ---
P3 <- env$predict_pop(mkfit(3, 1e-9, TRUE), X, resp_idx, Zho, S = 5, seed = 1,
                      use_delta = FALSE)
cat(sprintf("test 3  ablation ignores Delta       : max abs err %.3e\n", max(abs(P3 - E1))))
stopifnot(max(abs(P3 - E1)) < 1e-8)

# --- 4. rooti really is the INVERSE Cholesky root ----------------------------
# Integrate over N(beta0, s^2 I) by brute force and compare. If rooti were treated
# as the root itself, the simulated spread would be 1/s^2 instead of s^2 and this
# would fail loudly.
s <- 0.8                                  # rooti = I/s  =>  Sigma = s^2 I
P4 <- env$predict_pop(mkfit(100, s, FALSE), X, resp_idx, Zho, S = 300, seed = 3)
set.seed(11); NS <- 30000; acc <- matrix(0, ntask, 4)
for (i in seq_len(NS)) acc <- acc + softmax4(as.vector(X %*% (beta0 + s * rnorm(nvar))))
E4 <- acc / NS
cat(sprintf("test 4  Sigma scale (s = %.1f)        : max abs err %.4f  mean %.5f\n",
            s, max(abs(P4 - E4)), mean(abs(P4 - E4))))
stopifnot(max(abs(P4 - E4)) < 0.03, mean(abs(P4 - E4)) < 0.005)
# and the wrong convention (rooti used as the root) must be detectable
P4w <- env$predict_pop(mkfit(100, 1/s, FALSE), X, resp_idx, Zho, S = 300, seed = 3)
cat(sprintf("        sanity: inverted convention would be off by %.4f (mean)\n",
            mean(abs(P4w - E4))))
stopifnot(mean(abs(P4w - E4)) > 10 * mean(abs(P4 - E4)))

# --- 5. probs_fixed agrees with a per-respondent loop -------------------------
Bmat <- matrix(rnorm(R_ho * nvar), R_ho, nvar)
P5 <- env$probs_fixed(X, resp_idx, Bmat)
E5 <- softmax4(vapply(seq_len(n), function(i) sum(X[i, ] * Bmat[resp_idx[i], ]), 0))
cat(sprintf("test 5  probs_fixed                  : max abs err %.3e\n", max(abs(P5 - E5))))
stopifnot(max(abs(P5 - E5)) < 1e-12)

# --- 6. rows are probabilities ------------------------------------------------
stopifnot(max(abs(rowSums(P4) - 1)) < 1e-12, all(P4 > 0))
cat("test 6  rows sum to 1, all positive  : OK\n")
cat("\nALL PREDICTION TESTS PASSED\n")
