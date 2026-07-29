# =============================================================================
# QUANT METHODOLOGY: stop trying to PICK a winner; identify the set we cannot
# distinguish, and ship the average of it.
#
# THE DIAGNOSIS. Our paired standard error on respondent-clustered LogLoss
# differences is ~0.0007, so candidate gains of 0.0005-0.0016 sit at t = 0.7-2.2.
# After ~30 comparisons, the EXPECTED maximum t under the pure null is already
# ~2.07 (Gumbel extreme-value result for the max of N trials). Our best candidate
# is at t~2.2. That is barely above what noise alone produces after 30 tries.
# No amount of extra cross-validation fixes this: the limit comes from having
# 1,135 independent respondents, not from fold noise.
#
# SO WE APPLY FOUR TOOLS FROM THE BACKTEST-OVERFITTING LITERATURE:
#   1. MODEL CONFIDENCE SET (Hansen, Lunde & Nason 2011) - eliminate models until
#      what remains is statistically indistinguishable from the best.
#   2. REALITY CHECK / SPA (White 2000; Hansen 2005) - does the BEST of N
#      candidates genuinely beat the incumbent once the search itself is priced in?
#      The max is taken WITHIN each bootstrap resample, so correlation between our
#      near-identical candidates is handled exactly - no "effective N" guesswork.
#   3. PBO via CSCV (Bailey, Borwein, Lopez de Prado & Zhu) - probability that the
#      config we would pick in-sample lands in the bottom half out-of-sample.
#   4. DEFLATED SHARPE RATIO - the DM t-statistic deflated for N trials.
#
# The block bootstrap those methods use for time series is replaced throughout by
# an i.i.d. RESPONDENT bootstrap: aggregating each respondent's ~19 correlated
# tasks to a single mean loss makes rows exchangeable, which is exactly the
# assumption the procedures need. (This is why MCS's default k>=3 block length
# would be wrong here - our rows have no time order.)
# =============================================================================
source("exp_harness.R")

mnl    <- readRDS(file.path(CACHE, "mnl.rds"));    ourxgb <- readRDS(file.path(CACHE, "ourxgb.rds"))
mA     <- readRDS(file.path(CACHE, "mateA.rds"));  mB     <- readRDS(file.path(CACHE, "mateB.rds"))
base   <- readRDS(file.path(CACHE, "v7_baseline.rds"))
getc <- function(n) { p <- file.path(CACHE, paste0(n, ".rds")); if (file.exists(p)) readRDS(p) else NULL }
C1 <- getc("C1_searchwin"); C2 <- getc("C2_softmax_searchwin"); S1 <- getc("sm_eta03_r1500")

ours <- geo_w(mnl, ourxgb, 0.2)
v7w  <- function(mate) geo_w(ours, mate, 0.30)
mate_of <- function(...) { L <- list(...); temper(norm_rows(Reduce(`+`, L) / length(L)), 0.9) }

# ---- The candidate set (complete probability matrices we could actually ship) ----
cand <- list("v7" = base)
if (!is.null(C1))                    cand[["A,B,C1"]]       <- v7w(mate_of(mA, mB, C1))
if (!is.null(C1) && !is.null(S1))    cand[["A,B,S1,C1"]]    <- v7w(mate_of(mA, mB, S1, C1))
if (!is.null(C1) && !is.null(C2))    cand[["A,B,C1,C2"]]    <- v7w(mate_of(mA, mB, C1, C2))
if (!is.null(C1) && !is.null(C2) && !is.null(S1))
                                     cand[["A,B,S1,C1,C2"]] <- v7w(mate_of(mA, mB, S1, C1, C2))
if (!is.null(S1))                    cand[["A,B,S1"]]       <- v7w(mate_of(mA, mB, S1))
for (nm in c("sm_eta05_r800","sm_eta10_r400","sm_eta02_r2200","monoA","augA")) {
  P <- getc(nm); if (!is.null(P)) cand[[paste0("A,B,", nm)]] <- v7w(mate_of(mA, mB, P))
}
G <- getc("greedy_ensemble"); if (!is.null(G)) cand[["greedy"]] <- G
cat(sprintf("Candidates: %d\n", length(cand)))

# ---- Shared machinery: per-respondent mean loss, one column per candidate ----
case_f <- factor(train$Case); ncase <- nlevels(case_f)
per_resp <- function(P) {
  l <- -log(pmin(pmax(P[cbind(seq_len(nrow(P)), train$Choice)], 1e-15), 1 - 1e-15))
  as.numeric(tapply(l, case_f, mean))
}
L <- sapply(cand, per_resp)            # ncase x M
M <- ncol(L)
cat(sprintf("Loss matrix: %d respondents x %d models\n\n", nrow(L), M))
cat("Overall LogLoss by candidate:\n")
print(round(sort(colMeans(L)), 5))

# One bootstrap index set, reused by every test (common random numbers).
set.seed(4242); B <- 3000
W <- matrix(0L, B, ncase)
for (b in seq_len(B)) W[b, ] <- tabulate(sample.int(ncase, ncase, replace = TRUE), ncase)
CM <- (W %*% L) / ncase                # B x M bootstrap column means

# =============================================================================
# 1. MODEL CONFIDENCE SET  (fast form: the procedure depends on the data only
#    through column means, so everything is arithmetic on CM)
# =============================================================================
mcs <- function(alpha = 0.10) {
  S <- seq_len(M); elim <- character(0); pvals <- numeric(0); running <- 0
  while (length(S) > 1) {
    m  <- colMeans(L[, S, drop = FALSE])
    d  <- m - mean(m)                                  # loss vs the field
    mb <- CM[, S, drop = FALSE]
    db <- mb - rowMeans(mb)
    v  <- colMeans(sweep(db, 2, d, "-")^2)
    v[v < 1e-18] <- 1e-18
    t_ <- d / sqrt(v)
    tb <- sweep(db, 2, d, "-") / rep(sqrt(v), each = B)
    p  <- mean(apply(tb, 1, max) > max(t_))
    running <- max(running, p)
    if (running >= alpha) break
    worst <- S[which.max(t_)]
    elim <- c(elim, colnames(L)[worst]); pvals <- c(pvals, running)
    S <- setdiff(S, worst)
  }
  list(surviving = colnames(L)[S], eliminated = elim, elim_p = pvals, p_stop = running)
}
r <- mcs(0.10)
cat("\n===== MODEL CONFIDENCE SET (alpha = 0.10) =====\n")
if (length(r$eliminated)) {
  cat("Eliminated (in order, with MCS p-value):\n")
  for (i in seq_along(r$eliminated)) cat(sprintf("  %-22s p=%.3f\n", r$eliminated[i], r$elim_p[i]))
}
cat(sprintf("\nSURVIVING SET (%d models, cannot be distinguished):\n", length(r$surviving)))
for (s in r$surviving) cat(sprintf("  %-22s %.5f\n", s, mean(L[, s])))

# =============================================================================
# 2. REALITY CHECK + SPA  - does the BEST of N beat the incumbent, after search?
# =============================================================================
bench <- "v7"; others <- setdiff(colnames(L), bench)
D  <- L[, bench] - L[, others, drop = FALSE]          # positive = candidate better
d  <- colMeans(D)
Db <- CM[, bench] - CM[, others, drop = FALSE]
w_ <- sqrt(colMeans(sweep(Db, 2, d, "-")^2)); w_[w_ < 1e-12] <- 1e-12
n  <- ncase
thr <- -w_ * sqrt(2 * log(log(n)))                    # Hansen's LIL threshold
g   <- ifelse(d >= thr, d, 0)                         # selective recentring
Tobs <- max(c(d / w_, 0))
tb_rc  <- sweep(Db, 2, d, "-") / rep(w_, each = B)
tb_spa <- sweep(Db, 2, g, "-") / rep(w_, each = B)
p_rc  <- mean(apply(tb_rc,  1, function(z) max(c(z, 0))) > Tobs)
p_spa <- mean(apply(tb_spa, 1, function(z) max(c(z, 0))) > Tobs)
cat("\n===== REALITY CHECK / SPA vs v7 =====\n")
cat(sprintf("Best candidate: %s (mean gain %.5f, t = %.2f)\n",
            others[which.max(d)], max(d), max(d / w_)))
cat(sprintf("White Reality Check p = %.3f\nHansen SPA          p = %.3f\n", p_rc, p_spa))
cat(sprintf("=> %s\n", if (p_spa < 0.05)
  "The best candidate DOES beat v7 after pricing in the search." else
  "CANNOT reject the null that v7 is as good as the best of these candidates."))

# ---- Deflated Sharpe: the DM t-statistic deflated for N trials ----
N_TRIALS <- length(others)
gam <- 0.5772156649
E_max <- (1 - gam) * qnorm(1 - 1/N_TRIALS) + gam * qnorm(1 - 1/(N_TRIALS * exp(1)))
cat(sprintf("\nExpected max t under the null for N=%d trials: %.2f\n", N_TRIALS, E_max))
cat(sprintf("Observed best t: %.2f  =>  %s\n", max(d / w_),
            if (max(d / w_) > E_max) "above the null expectation" else
              "BELOW what noise alone yields after this many tries"))

# =============================================================================
# 3. PBO via CSCV - probability the in-sample winner lands in the bottom half OOS
# =============================================================================
S_GROUPS <- 10
set.seed(99); ord <- sample(ncase)
grp <- split(ord, rep(1:S_GROUPS, length.out = ncase))
combs <- combn(S_GROUPS, S_GROUPS / 2)
lam <- numeric(ncol(combs))
for (j in seq_len(ncol(combs))) {
  is_rows  <- unlist(grp[combs[, j]]);  oos_rows <- unlist(grp[-combs[, j]])
  is_m  <- colMeans(L[is_rows, , drop = FALSE])
  oos_m <- colMeans(L[oos_rows, , drop = FALSE])
  best <- which.min(is_m)                     # lower loss is better
  rk <- rank(oos_m)[best]                     # 1 = best OOS
  om <- 1 - (rk - 1) / M                      # convert so high = good
  om <- min(max(om, 1 / (M + 1)), M / (M + 1))
  lam[j] <- log(om / (1 - om))
}
cat(sprintf("\n===== PBO (CSCV, S=%d, %d splits) =====\n", S_GROUPS, ncol(combs)))
cat(sprintf("PBO = %.3f  (probability the in-sample winner is below median out-of-sample)\n",
            mean(lam <= 0)))
cat("NOTE: this measures overfitting of the SELECTION step among these candidates.\n")
cat("It cannot see contamination inside candidates that were themselves tuned on these folds.\n")

# =============================================================================
# 4. BAGGED SIMPLEX WEIGHTS (EM) - the principled fix for weight concentration
#    L(w) = -mean log(q_i'w) is CONVEX on the simplex, so the earlier optimiser
#    was not stuck in a local optimum - concentrating was the genuine in-sample
#    optimum, i.e. pure overfitting to OOF noise. Bagging over RESPONDENT
#    resamples averages many such corner solutions into a stable interior one.
# =============================================================================
comps <- list(mnl = mnl, ourxgb = ourxgb, mateA = mA, mateB = mB)
if (!is.null(C1)) comps$C1 <- C1
if (!is.null(C2)) comps$C2 <- C2
if (!is.null(S1)) comps$S1 <- S1
Q <- sapply(comps, function(P) P[cbind(seq_len(nrow(P)), train$Choice)])   # nrow x m
em_w <- function(rows, iters = 200) {
  w <- rep(1 / ncol(Q), ncol(Q))
  Qs <- Q[rows, , drop = FALSE]
  for (i in seq_len(iters)) {
    den <- as.numeric(Qs %*% w); den[den < 1e-15] <- 1e-15
    w <- w * colMeans(Qs / den)
    w <- w / sum(w)
  }
  w
}
case_rows <- split(seq_len(nrow(train)), case_f)
set.seed(7); NB <- 40
Wbag <- matrix(0, NB, ncol(Q))
for (b in seq_len(NB)) {
  rc <- sample(levels(case_f), ncase, replace = TRUE)
  Wbag[b, ] <- em_w(unlist(case_rows[rc], use.names = FALSE))
}
w_bag <- colMeans(Wbag); names(w_bag) <- colnames(Q)
w_one <- em_w(seq_len(nrow(train))); names(w_one) <- colnames(Q)
cat("\n===== ENSEMBLE WEIGHTS =====\n")
cat("Single-fit EM weights (the concentrating, overfit solution):\n"); print(round(w_one, 4))
cat("Bagged over 40 respondent resamples:\n");                        print(round(w_bag, 4))

mix <- function(w) norm_rows(Reduce(`+`, lapply(seq_along(comps), function(i) w[i] * comps[[i]])))
report("arithmetic mix, single-fit EM weights", mix(w_one), base)
report("arithmetic mix, BAGGED EM weights",     mix(w_bag), base)
for (delta in c(0.25, 0.5, 0.75)) {
  ws <- delta * w_bag + (1 - delta) / length(w_bag)
  report(sprintf("  bagged weights shrunk to 1/M, delta=%.2f", delta), mix(ws), base)
}

# =============================================================================
# 5. SHIP THE MCS SURVIVOR SET, EQUALLY WEIGHTED
#    If several candidates are statistically indistinguishable, averaging them is
#    strictly better than picking one: it keeps the variance reduction and drops
#    the selection risk.
# =============================================================================
if (length(r$surviving) > 1) {
  Zs <- Reduce(`+`, lapply(r$surviving, function(s) log(pmax(cand[[s]], 1e-15))))
  eq <- norm_rows(exp(Zs / length(r$surviving) -
                        apply(Zs / length(r$surviving), 1, max)))
  report(sprintf("EQUAL-WEIGHT AVERAGE of the %d MCS survivors", length(r$surviving)), eq, base)
  saveRDS(eq, file.path(CACHE, "mcs_average.rds"))
} else {
  cat("\nOnly one model survived the MCS - nothing to average.\n")
}
