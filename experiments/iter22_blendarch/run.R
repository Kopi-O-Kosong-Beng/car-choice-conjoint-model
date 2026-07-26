# =============================================================================
# Iteration 22 -- a richer blend architecture
#
# HYPOTHESIS (written BEFORE running; kept verbatim even if refuted)
# -----------------------------------------------------------------
# model/06_blend.R pools members in log space with ONE global exponent per member,
# one temperature and a uniform mixing term. That single scalar per member is forced
# to be a compromise, because the members do NOT disagree uniformly across the four
# alternatives -- they disagree in a structured way. A pre-registration diagnostic
# (scratchpad/ba_diag.R, run before this script) decomposes each member's OOF logloss
# exactly into
#
#     LL_total = LL_none-binary  +  P(buy) * LL_bundle|buy          (exact identity)
#
# and finds a genuine CROSSOVER in member skill:
#
#     member     binary-none LL   cond-bundle LL     rank(none)  rank(bundle)
#     lcmnl3        0.55005          0.85124             1            3
#     xgb_lw2       0.55592          0.83933             2            2
#     xgb_mono      0.55618          0.83649             3            1
#     mnl_pw        0.55824          0.85799             4            4
#
# lcmnl3 is the BEST member at deciding whether the respondent buys at all and only
# the 3rd best at deciding which bundle; xgb_mono is the exact reverse. One global
# weight cannot express "trust the latent-class model about whether they buy, trust
# the trees about which bundle". So I predict a blend that can vary its member
# weights across the none/buy split will beat the production nested 1.13044.
#
# Note the motivation given in the task brief (lcmnl3 predicts a TEST none-rate of
# 0.223 vs the trees' ~0.275) is a statement about the test set, which no OOF
# experiment can score -- OOF none-rates are 0.302-0.308 for every member against a
# 0.3023 truth. The crossover above is the same phenomenon measured where it IS
# scorable, and is the version this experiment actually tests.
#
# VARIANTS, in increasing order of ambition (free-parameter count in brackets):
#   V0  production log-pool: softmax weights + temperature + uniform mix     [M+1 = 5]
#   V1  V0 + a single additive constant on the pooled none log-prob         [M+2 = 6]
#         -- the cheapest thing that could work: pure none-rate recalibration.
#   V2  per-alternative-type weights: one unnormalised exponent vector for
#       alts 1-3, another for alt 4, renormalised over the 4 alts           [2M+1 = 9]
#   V3  V2 + the none shift of V1 (strictly nests V0, V1 and V2)            [2M+2 = 10]
#   V4  two-stage blend: pool P(none) as a 2-class problem with its own
#       weights+shift, pool P(bundle | buy) as a 3-class problem with its
#       own weights, recombine                                              [2M+2 = 10]
#   V5  covariate-dependent weights: exponent_i = exp(a_i + b_i * income_hi),
#       income_hi = respondent income above the TRAIN median, centred        [2M+1 = 9]
#   V0p production optimiser settings exactly (single Nelder-Mead, maxit 3000),
#       included only to confirm this script reproduces the 1.13044 headline.
#
# On V4 vs iteration 09. Iteration 09 rejected a two-stage MODEL (1.17169, z = -11.04):
# it FIT one classifier for none-vs-buy and a separate one for bundle-choice, so each
# stage saw only part of the data and threw away the cross-stage regularisation a
# joint 4-way fit gives -- the bundle stage in particular was fit on ~70% of rows.
# V4 is different: every member is still fit jointly on all 4 alternatives; only the
# COMBINATION rule is factorised, and no rows are discarded. Because the logloss
# identity above is exact, V4's two stages optimise two exactly-separable pieces of
# the same objective. Whether that distinction rescues the idea is the open question;
# it is not obviously enough, and iteration 09 is recorded here as a live warning.
#
# EVALUATION -- this is where the experiment can fool itself.
# Every extra blend parameter is fitted on the very OOF predictions used to score it,
# so a plain OOF number is guaranteed to improve with parameter count and is
# worthless. All variants are scored NESTED exactly as blend_probe.R does: fit the
# blend parameters on folds != k, score fold k, average over the 5 folds. Every
# variant gets an IDENTICAL optimiser budget (same inits, same restart schedule) so
# the comparison is of architectures, not of tuning effort. Variants are then
# compared to V0 by a respondent-clustered paired test on the per-row nested losses,
# the same machinery as model/compare.R.
#
# LEAKAGE. No target-derived FEATURE is built here; the only target contact is the
# blend objective itself, and it is nested. The one honest caveat, which applies
# equally to production and so does not bias the comparison: the OOF rows used to FIT
# the blend for fold k come from base models trained on fold sets that include k.
# That is the repository's existing protocol, inherited, not introduced here.
#
# Emits NO oof_/test_ artifacts. Read-only w.r.t. model/.
#
# Usage:  Rscript experiments/iter22_blendarch/run.R            # all variants
#         Rscript experiments/iter22_blendarch/run.R V0,V4      # a subset
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

MEMB <- c("mnl_pw", "xgb_lw2", "xgb_mono", "lcmnl3")
M    <- length(MEMB)
EPS0 <- 1e-12
OUTD <- "experiments/iter22_blendarch"

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
y     <- ymap$y
N     <- length(y)
fmap  <- folds[order(No), fold]
stopifnot(identical(folds[order(No), No], ymap$No), length(fmap) == N)

OOF <- lapply(MEMB, function(m) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  stopifnot(identical(d$No, ymap$No))
  P <- as.matrix(d[, .(p1, p2, p3, p4)])
  stopifnot(!anyNA(P), all(abs(rowSums(P) - 1) < 1e-6))
  P
})
names(OOF) <- MEMB
stopifnot(all(sapply(OOF, nrow) == 21565))

lg <- function(x) log(pmax(x, EPS0))
AJ <- lapply(1:4, function(j) do.call(cbind, lapply(OOF, function(P) lg(P[, j]))))   # N x M each
Q4 <- AJ[[4]]
Q0 <- do.call(cbind, lapply(OOF, function(P) lg(rowSums(P[, 1:3]))))                 # N x M
RJ <- lapply(1:3, function(j) do.call(cbind,
        lapply(OOF, function(P) lg(P[, j] / rowSums(P[, 1:3])))))                    # N x M each

wide <- readRDS("model/artifacts/wide.rds")
inc  <- unique(wide[is_test == FALSE, .(No, incomeind)])[order(No)]
stopifnot(identical(inc$No, ymap$No))
med  <- median(unique(wide[is_test == FALSE, .(Case, incomeind)])$incomeind)
XHI  <- as.numeric(inc$incomeind > med); XHI <- XHI - mean(XHI)
cat(sprintf("income split: train median incomeind = %s, share above = %.3f\n",
            med, mean(inc$incomeind > med)))

# ---- prep: subset every design block ONCE per fit (rows constant within a fit) ----
prep <- function(rows) {
  list(n = sum(rows),
       Aj = lapply(AJ, function(Z) Z[rows, , drop = FALSE]),
       Q4 = Q4[rows, , drop = FALSE],
       Q0 = Q0[rows, , drop = FALSE],
       Rj = lapply(RJ, function(Z) Z[rows, , drop = FALSE]),
       x  = XHI[rows],
       yy = y[rows])
}
pool4 <- function(D, c4) do.call(cbind, lapply(D$Aj, function(Z) Z %*% c4))          # n x 4
sm    <- function(L) { P <- exp(L - apply(L, 1, max)); P / rowSums(P) }
mixu  <- function(P, eA, k) (1 - eA) * P + eA * (1 / k)

# ---- variants ----
V <- list()

V$V0 <- list(np = M + 1, init = c(rep(0, M), 0, -3), pred = function(th, D) {
  w <- exp(th[1:M]); w <- w / sum(w)
  L <- pool4(D, w / exp(th[M + 1]))
  mixu(sm(L), plogis(th[M + 2]) * 0.10, 4)
})

V$V0p <- V$V0                                       # same model, production optimiser

V$V1 <- list(np = M + 2, init = c(rep(0, M), 0, -3, 0), pred = function(th, D) {
  w <- exp(th[1:M]); w <- w / sum(w)
  L <- pool4(D, w / exp(th[M + 1]))
  L[, 4] <- L[, 4] + th[M + 3]
  mixu(sm(L), plogis(th[M + 2]) * 0.10, 4)
})

V$V2 <- list(np = 2 * M + 1, init = c(rep(log(1 / M), 2 * M), -3), pred = function(th, D) {
  L <- pool4(D, exp(th[1:M]))
  L[, 4] <- D$Q4 %*% exp(th[(M + 1):(2 * M)])
  mixu(sm(L), plogis(th[2 * M + 1]) * 0.10, 4)
})

V$V3 <- list(np = 2 * M + 2, init = c(rep(log(1 / M), 2 * M), -3, 0), pred = function(th, D) {
  L <- pool4(D, exp(th[1:M]))
  L[, 4] <- (D$Q4 %*% exp(th[(M + 1):(2 * M)])) + th[2 * M + 2]
  mixu(sm(L), plogis(th[2 * M + 1]) * 0.10, 4)
})

V$V4 <- list(np = 2 * M + 2, init = c(rep(log(1 / M), 2 * M), 0, -3), pred = function(th, D) {
  u <- exp(th[1:M]); v <- exp(th[(M + 1):(2 * M)])
  s1 <- (D$Q4 %*% u) + th[2 * M + 1]
  s0 <-  D$Q0 %*% u
  mx <- pmax(s1, s0); e1 <- exp(s1 - mx); e0 <- exp(s0 - mx)
  q  <- as.vector(e1 / (e1 + e0))
  Lr <- do.call(cbind, lapply(D$Rj, function(Z) Z %*% v))
  Pr <- sm(Lr)
  mixu(cbind((1 - q) * Pr, q), plogis(th[2 * M + 2]) * 0.10, 4)
})

V$V5 <- list(np = 2 * M + 1, init = c(rep(log(1 / M), M), rep(0, M), -3), pred = function(th, D) {
  Cm <- exp(outer(D$x, th[(M + 1):(2 * M)]) + rep(th[1:M], each = D$n))   # n x M
  L  <- do.call(cbind, lapply(D$Aj, function(Z) rowSums(Z * Cm)))
  mixu(sm(L), plogis(th[2 * M + 1]) * 0.10, 4)
})

# ---- fitting: identical budget for every variant (V0p excepted, by design) ----
FIT_RESTARTS <- 4L
FIT_MAXIT    <- 4000L

fit_from <- function(vn, D, init, restarts = FIT_RESTARTS, maxit = FIT_MAXIT) {
  pr  <- V[[vn]]$pred
  obj <- function(th) logloss(D$yy, pr(th, D))
  th <- init; val <- Inf
  for (r in seq_len(restarts)) {
    o <- optim(th, obj, method = "Nelder-Mead", control = list(maxit = maxit))
    improved <- o$value < val - 1e-10
    th <- o$par; val <- o$value
    if (!improved) break
  }
  list(par = th, value = val)
}

fit_best <- function(vn, D) {
  if (vn == "V0p") return(fit_from(vn, D, V[[vn]]$init, restarts = 1L, maxit = 3000L))
  b <- fit_from(vn, D, V[[vn]]$init)
  set.seed(2200 + match(vn, names(V)))
  b2 <- fit_from(vn, D, V[[vn]]$init + rnorm(length(V[[vn]]$init), 0, 0.25))
  if (b2$value < b$value) b2 else b        # selected on FITTING loss only
}

nested_run <- function(vn) {
  t0 <- Sys.time()
  Pn <- matrix(NA_real_, N, 4); per_fold <- numeric(5); pars <- vector("list", 5)
  for (k in 1:5) {
    Dtr <- prep(fmap != k); Dte <- prep(fmap == k)
    o <- fit_best(vn, Dtr)
    Pn[fmap == k, ] <- V[[vn]]$pred(o$par, Dte)
    per_fold[k] <- logloss(Dte$yy, Pn[fmap == k, , drop = FALSE])
    pars[[k]] <- o$par
  }
  stopifnot(!anyNA(Pn))
  full <- fit_best(vn, prep(rep(TRUE, N)))
  cat(sprintf("%-4s np=%2d  NESTED %.5f (sd %.5f)  folds %s  plainOOF %.5f  [%.0fs]\n",
              vn, V[[vn]]$np, mean(per_fold), sd(per_fold),
              paste(sprintf("%.4f", per_fold), collapse = " "), full$value,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  flush.console()
  list(variant = vn, np = V[[vn]]$np, Pn = Pn, per_fold = per_fold,
       nested = mean(per_fold), plain = full$value, full_par = full$par, pars = pars)
}

loss_of <- function(P) { P <- pmax(P / rowSums(P), 1e-15); -log(P[cbind(seq_len(N), y)]) }

paired <- function(la, lb, A_, B_) {
  d  <- la - lb
  cl <- data.table(Case = ymap$Case, d = d)[, .(dm = mean(d)), by = Case]
  est <- mean(cl$dm); se <- sd(cl$dm) / sqrt(nrow(cl)); z <- est / se
  cat(sprintf("  %-4s -> %-4s : nested %.5f -> %.5f | delta %+.5f  SE %.5f  z %+6.2f  win%% %4.1f  %s\n",
              A_, B_, mean(la), mean(lb), est, se, z, 100 * mean(cl$dm > 0),
              if (abs(z) < 2) "noise" else if (z > 0) "BETTER" else "WORSE"))
  invisible(c(est = est, se = se, z = z))
}

args <- commandArgs(trailingOnly = TRUE)
todo <- if (length(args)) trimws(strsplit(args[1], ",")[[1]]) else
        c("V0p", "V0", "V1", "V2", "V3", "V4", "V5")
cat("variants:", paste(todo, collapse = ", "), "\n\n")
res <- list()
for (vn in todo) res[[vn]] <- nested_run(vn)
saveRDS(res, file.path(OUTD, sprintf("res_%s.rds", paste(todo, collapse = "-"))))

cat("\n--- respondent-clustered paired tests on per-row NESTED losses ---\n")
L <- lapply(res, function(r) loss_of(r$Pn))
base <- if ("V0" %in% todo) "V0" else todo[1]
for (vn in setdiff(todo, base)) paired(L[[base]], L[[vn]], base, vn)

cat("\n--- fitted full-data parameters ---\n")
for (vn in todo) {
  p <- res[[vn]]$full_par
  cat(sprintf("%-4s: %s\n", vn, paste(sprintf("%.3f", p), collapse = " ")))
}
cat("\nmembers:", paste(MEMB, collapse = ", "), "\n")
cat("OK\n")
