# =============================================================================
# Iteration 19 -- BLEND DIVERSITY PROBE  (read-only, emits NO oof_/test_ artifacts)
#
# HYPOTHESIS (stated before running):
#   H1. xgb_lw2 and xgb_mono are near-duplicates (log-odds correlation ~0.99),
#       so the 4-member blend is effectively ~2-3 independent opinions.
#   H2. The blend's remaining loss is concentrated on the none-option (y == 4)
#       and on rows where members disagree about the none-probability.
#   H3. Because lcmnl3 and the tree models disagree systematically about the
#       none-rate (0.223 vs ~0.275 on test), a blend with a SEPARATE weight
#       vector for alternative 4 will beat the single global weight vector.
#
# WHAT IS MEASURED
#   Q1  pairwise correlation of member OOF predictions on the log-odds scale,
#       per alternative, plus error (per-row logloss) correlation and the
#       eigen-spectrum of that error-correlation matrix.
#   Q2  leave-one-member-out nested blend OOF ("diversity contribution").
#   Q3  decomposition of the nested blend OOF by true class y and by a
#       member-disagreement score.
#   Q4  nested comparison of three blend architectures, ONE change apart:
#         A  production log-opinion pool           (M weights, T, eps)
#         B  A + a free none-option intercept c    (+1 param)
#         C  B + a SEPARATE weight vector for alt 4 (+M params)
#       C-vs-B isolates the split-weight question; B-vs-A isolates the much
#       cheaper "just recalibrate the none rate" alternative. Everything is
#       NESTED (weights fitted on 4 folds, scored on the 5th) so the extra
#       parameters are honestly charged for, and compared paired with
#       respondent-clustered SEs.
# RESULT (written after the run; the hypotheses above are left as they were).
#   H1 CONFIRMED, and worse than stated. xgb_lw2/xgb_mono correlate 0.985-0.986
#      on every alternative and agree on the argmax 93.2% of the time. But the
#      logit pair is nearly as bad: mnl_pw/lcmnl3 correlate 0.958-0.960. The
#      blend has TWO opinions (tree family, logit family), not four. Dropping
#      any member except lcmnl3 changes the nested blend by <0.0007.
#   H2 REFUTED. y == 4 is the EASIEST class (mean loss 1.037 vs 1.210 for y==1)
#      and carries 27.7% of total loss against a 30.2% row share -- less than
#      its share. Loss rises with member disagreement, but only 1.054 -> 1.174
#      across quintiles, and the share of total loss is nearly flat.
#   H3 NOT SUPPORTED. The alt-4 weight vector really is different (L1 |w-v| =
#      1.12 of a possible 2), but the nested gain is +0.00108 at paired z=+1.49,
#      one of five folds moves the wrong way, and the fitted v is unstable
#      across folds. Decisive context: on OOF all four members reproduce the
#      true class shares to three decimals, so the 0.223-vs-0.275 none-rate
#      disagreement that motivated H3 DOES NOT EXIST in the data the weights
#      would be fitted on. See probe2_gating.R for the follow-up.
#
#   Q5  oracle bound: best per-respondent convex combination of the 4 members,
#       fitted and scored on the same rows (cheating -- it is an upper bound on
#       what ANY per-respondent gating scheme could buy), plus an honest
#       split-half version of the same thing.
#
# NOTE ON WHAT "SEPARATE WEIGHTS FOR ALTERNATIVE 4" CAN MEAN.
#   Every choice task contains all 4 alternatives, so there is no such thing as
#   a subset of "none rows" to fit separate weights on. Fitting weights on the
#   tasks where y == 4 would condition on the label and be unusable at test
#   time. The only deployable version is a separate weight vector on the alt-4
#   COLUMN of the pool, which is what variant C does.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
set.seed(42)

MEMB <- c("mnl_pw", "xgb_lw2", "xgb_mono", "lcmnl3")
M    <- length(MEMB)
eps0 <- 1e-12

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
y     <- ymap$y
Case  <- ymap$Case
setorder(folds, No)
stopifnot(identical(folds$No, ymap$No))
fmap <- folds$fold
N    <- length(y)
stopifnot(N == 21565)

readP <- function(nm, kind = "oof") {
  d <- readRDS(sprintf("model/artifacts/%s_%s.rds", kind, nm)); setorder(d, No)
  P <- as.matrix(d[, .(p1, p2, p3, p4)])
  stopifnot(!anyNA(P), all(P > 0), max(abs(rowSums(P) - 1)) < 1e-6)
  P
}
OOF  <- lapply(MEMB, readP)
TST  <- lapply(MEMB, readP, kind = "test")
names(OOF) <- names(TST) <- MEMB
LP   <- lapply(OOF, function(P) log(pmax(P, eps0)))   # precomputed, hot path

cat("\n########## MEMBER HEADLINE ##########\n")
for (i in 1:M)
  cat(sprintf("%-10s OOF %.5f | mean p4 oof %.4f | mean p4 test %.4f\n",
              MEMB[i], logloss(y, OOF[[i]]), mean(OOF[[i]][, 4]), mean(TST[[i]][, 4])))
cat(sprintf("%-10s ---     | true none rate  %.4f\n", "(truth)", mean(y == 4)))

# ============================================================================
# Q1  DIVERSITY: correlation on the log-odds scale, per alternative
# ============================================================================
cat("\n########## Q1. PAIRWISE CORRELATION (log-odds scale, per alternative) ##########\n")
logit <- function(p) log(p / (1 - p))
for (j in 1:4) {
  X <- sapply(OOF, function(P) logit(pmin(pmax(P[, j], 1e-9), 1 - 1e-9)))
  C <- cor(X)
  cat(sprintf("\n-- alternative %d %s --\n", j, if (j == 4) "(NONE)" else ""))
  print(round(C, 4))
}
# multinomial log-odds vs the none option: log(p_j / p_4), j = 1..3
cat("\n-- multinomial log-odds vs none, log(p_j/p_4), pooled over j=1..3 --\n")
X <- sapply(OOF, function(P) as.vector(log(P[, 1:3] / P[, 4])))
print(round(cor(X), 4))

# error correlation: this, not the level correlation, is what ensembling exploits
LL <- sapply(OOF, function(P) -log(pmax(P[cbind(1:N, y)], eps0)))
cat("\n-- per-row logloss (error) correlation --\n")
Ce <- cor(LL); print(round(Ce, 4))
ev <- eigen(Ce, symmetric = TRUE)$values
cat("eigenvalues of error-corr matrix:", paste(round(ev, 3), collapse = " "), "\n")
cat(sprintf("participation ratio (effective independent members): %.2f of %d\n",
            sum(ev)^2 / sum(ev^2), M))
# hard-argmax agreement
am <- sapply(OOF, function(P) max.col(P))
cat("\n-- argmax agreement rate between members --\n")
A <- outer(1:M, 1:M, Vectorize(function(a, b) mean(am[, a] == am[, b])))
dimnames(A) <- list(MEMB, MEMB); print(round(A, 4))

# ============================================================================
# BLEND MACHINERY -- variants A / B / C. A reproduces 06_blend.R exactly.
# ============================================================================
# theta layouts
#   A: [a_1..a_m, logT, s]                (w = softmax(a); eps = plogis(s)*0.10)
#   B: [a_1..a_m, logT, s, c]             (c added to the alt-4 logit)
#   C: [a_1..a_m, b_1..b_m, logT, s, c]   (b = softmax weights for alt 4)
mk_blend <- function(type, m) function(theta, Ls, rows) {
  a <- theta[1:m]; w <- exp(a - max(a)); w <- w / sum(w)
  if (type == "C") {
    b <- theta[(m + 1):(2 * m)]; v <- exp(b - max(b)); v <- v / sum(v)
    o <- 2 * m
  } else { v <- w; o <- m }
  Tt <- exp(theta[o + 1]); eA <- plogis(theta[o + 2]) * 0.10
  cc <- if (type == "A") 0 else theta[o + 3]
  L4 <- Reduce(`+`, Map(function(vi, Lm) vi * Lm[rows, 4], v, Ls))
  L  <- Reduce(`+`, Map(function(wi, Lm) wi * Lm[rows, , drop = FALSE], w, Ls))
  L[, 4] <- L4
  L <- L / Tt
  L[, 4] <- L[, 4] + cc
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
npar <- function(type, m) switch(type, A = m + 2, B = m + 3, C = 2 * m + 3)
init <- function(type, m) switch(type,
  A = c(rep(0, m), 0, -3), B = c(rep(0, m), 0, -3, 0),
  C = c(rep(0, m), rep(0, m), 0, -3, 0))

fit_blend <- function(type, Ls, rows, start = NULL, restarts = 5) {
  m <- length(Ls); bf <- mk_blend(type, m)
  th <- if (is.null(start)) init(type, m) else start
  obj <- function(theta) logloss(y[rows], bf(theta, Ls, rows))
  o <- list(par = th, value = obj(th))
  for (r in seq_len(restarts)) {
    oo <- optim(o$par, obj, method = "Nelder-Mead",
                control = list(maxit = 6000, reltol = 1e-12))
    if (oo$value > o$value - 1e-10) { o <- oo; break }
    o <- oo
  }
  o$blend <- bf; o$type <- type; o
}

# nested evaluation: fit on folds != k, predict fold k. Returns per-row losses.
nested_eval <- function(type, Ls, warm = NULL) {
  rowloss <- numeric(N); per <- numeric(5); pars <- vector("list", 5)
  Pn <- matrix(NA_real_, N, 4)
  for (k in 1:5) {
    tr <- fmap != k; te <- fmap == k
    st <- if (is.null(warm)) NULL else warm[[k]]
    o  <- fit_blend(type, Ls, tr, start = st)
    Pk <- o$blend(o$par, Ls, te)
    Pn[te, ] <- Pk
    rowloss[te] <- -log(pmax(Pk[cbind(1:sum(te), y[te])], 1e-15))
    per[k] <- mean(rowloss[te]); pars[[k]] <- o$par
  }
  list(loss = rowloss, per = per, mean = mean(rowloss), fold_mean = mean(per),
       sd = sd(per), pars = pars, P = Pn)
}

paired <- function(la, lb, lab) {   # positive => b better
  d  <- data.table(Case = Case, d = la - lb)[, .(dm = mean(d)), by = Case]
  est <- mean(d$dm); se <- sd(d$dm) / sqrt(nrow(d))
  cat(sprintf("%-34s %+.5f  SE %.5f  z = %+5.2f  %s\n", lab, est, se, est / se,
              if (abs(est / se) < 2) "(noise)" else if (est > 0) "(REAL gain)" else "(REAL loss)"))
  invisible(c(est = est, se = se, z = est / se))
}

cat("\n########## Q4a. REPRODUCE THE PRODUCTION NESTED BLEND ##########\n")
A <- nested_eval("A", LP)
cat("per-fold:", paste(round(A$per, 5), collapse = " "), "\n")
cat(sprintf(">>> variant A nested OOF: %.5f  (+- %.5f)   [must match 1.13044]\n",
            A$fold_mean, A$sd))

# ============================================================================
# Q2  LEAVE-ONE-MEMBER-OUT nested blend  =  diversity contribution
# ============================================================================
cat("\n########## Q2. LEAVE-ONE-MEMBER-OUT (diversity contribution) ##########\n")
loo <- list()
for (i in 1:M) {
  sub <- setdiff(seq_len(M), i)
  r <- nested_eval("A", LP[sub])
  loo[[MEMB[i]]] <- r
  cat(sprintf("without %-10s nested %.5f   contribution %+.5f  ",
              MEMB[i], r$fold_mean, r$fold_mean - A$fold_mean))
  p <- data.table(Case = Case, d = r$loss - A$loss)[, .(dm = mean(d)), by = Case]
  cat(sprintf("z = %+5.2f\n", mean(p$dm) / (sd(p$dm) / sqrt(nrow(p)))))
}
cat("\nfull-data fitted weights (variant A):\n")
oA <- fit_blend("A", LP, rep(TRUE, N))
wA <- exp(oA$par[1:M]); wA <- wA / sum(wA)
cat(paste(sprintf("%s %.3f", MEMB, wA), collapse = " | "),
    sprintf("| T %.3f eps %.4f\n", exp(oA$par[M + 1]), plogis(oA$par[M + 2]) * 0.10))

# ============================================================================
# Q3  WHERE THE LOSS LIVES
# ============================================================================
cat("\n########## Q3. LOSS DECOMPOSITION ##########\n")
# disagreement score: SD across members of the alt-4 log-odds, and mean pairwise TV
sd_l4 <- apply(sapply(OOF, function(P) logit(pmin(pmax(P[, 4], 1e-9), 1 - 1e-9))), 1, sd)
tv <- rowMeans(sapply(combn(M, 2, simplify = FALSE), function(ij)
  0.5 * rowSums(abs(OOF[[ij[1]]] - OOF[[ij[2]]]))))
dt <- data.table(y = y, Case = Case, loss = A$loss, sd_l4 = sd_l4, tv = tv,
                 p_true = A$P[cbind(1:N, y)], p4 = A$P[, 4])
cat("\n-- by true class --\n")
print(dt[, .(n = .N, share_of_rows = round(.N / N, 4),
             mean_loss = round(mean(loss), 4),
             share_of_total_loss = round(sum(loss) / sum(dt$loss), 4),
             mean_p_true = round(mean(p_true), 4)), by = y][order(y)])
cat("\n-- by disagreement quintile (mean pairwise total-variation distance) --\n")
dt[, tvq := cut(tv, quantile(tv, 0:5 / 5), include.lowest = TRUE, labels = 1:5)]
print(dt[, .(n = .N, mean_tv = round(mean(tv), 4), mean_loss = round(mean(loss), 4),
             share_of_total_loss = round(sum(loss) / sum(dt$loss), 4),
             none_rate = round(mean(y == 4), 4)), by = tvq][order(tvq)])
cat("\n-- loss by (true class x disagreement quintile) --\n")
print(dcast(dt[, .(ml = round(mean(loss), 3)), by = .(y, tvq)], y ~ tvq, value.var = "ml"))
cat("\n-- none-option calibration of the nested blend, by predicted-p4 decile --\n")
dt[, p4d := cut(p4, quantile(p4, 0:10 / 10), include.lowest = TRUE, labels = 1:10)]
print(dt[, .(n = .N, mean_pred_p4 = round(mean(p4), 4),
             actual_none = round(mean(y == 4), 4)), by = p4d][order(p4d)])
cat(sprintf("\noverall: blend mean p4 %.4f vs actual %.4f\n", mean(dt$p4), mean(y == 4)))

# ============================================================================
# Q4  DOES A SEPARATE ALT-4 WEIGHT VECTOR HELP?
# ============================================================================
cat("\n########## Q4b. BLEND ARCHITECTURES, NESTED ##########\n")
Bv <- nested_eval("B", LP, warm = lapply(A$pars, function(p) c(p, 0)))
cat(sprintf("variant B (A + none intercept)      nested %.5f (+- %.5f)\n", Bv$fold_mean, Bv$sd))
Cv <- nested_eval("C", LP, warm = lapply(Bv$pars, function(p)
  c(p[1:M], p[1:M], p[M + 1], p[M + 2], p[M + 3])))
cat(sprintf("variant C (B + split alt-4 weights) nested %.5f (+- %.5f)\n", Cv$fold_mean, Cv$sd))
cat(sprintf("variant A (production)              nested %.5f (+- %.5f)\n", A$fold_mean, A$sd))
cat("\nper-fold nested losses\n")
print(data.table(fold = 1:5, A = round(A$per, 5), B = round(Bv$per, 5), C = round(Cv$per, 5)))
cat("\npaired, respondent-clustered (positive = second model better):\n")
paired(A$loss, Bv$loss, "B over A (intercept only)")
paired(Bv$loss, Cv$loss, "C over B (the split-weight bit)")
paired(A$loss, Cv$loss, "C over A (both changes)")

cat("\n-- do the fitted weights actually differ between alt-4 and alts 1-3? --\n")
oB <- fit_blend("B", LP, rep(TRUE, N), start = c(oA$par, 0))
oC <- fit_blend("C", LP, rep(TRUE, N),
                start = c(oB$par[1:M], oB$par[1:M], oB$par[M + 1], oB$par[M + 2], oB$par[M + 3]))
sm <- function(a) { e <- exp(a - max(a)); e / sum(e) }
cat("full-data fit, variant C:\n")
cat(sprintf("  w (alts 1-3): %s\n", paste(sprintf("%s %.3f", MEMB, sm(oC$par[1:M])), collapse = " | ")))
cat(sprintf("  v (alt 4)   : %s\n", paste(sprintf("%s %.3f", MEMB, sm(oC$par[(M+1):(2*M)])), collapse = " | ")))
cat(sprintf("  T %.3f  eps %.4f  c %+.4f  in-sample %.5f (B: %.5f, A: %.5f)\n",
            exp(oC$par[2*M+1]), plogis(oC$par[2*M+2]) * 0.10, oC$par[2*M+3],
            oC$value, oB$value, oA$value))
cat(sprintf("  L1 distance |w - v| = %.3f\n", sum(abs(sm(oC$par[1:M]) - sm(oC$par[(M+1):(2*M)])))))
cat("\nper-fold variant-C weights (stability across folds):\n")
for (k in 1:5) {
  p <- Cv$pars[[k]]
  cat(sprintf("  fold %d  w: %s   v: %s   c %+.3f\n", k,
              paste(sprintf("%.3f", sm(p[1:M])), collapse = " "),
              paste(sprintf("%.3f", sm(p[(M+1):(2*M)])), collapse = " "), p[2*M+3]))
}
cat(sprintf("\nfor scale: fold-to-fold SD of variant A is %.5f; the C-over-A effect must\n", A$sd))
cat("clear the PAIRED SE above, not that SD, to be believed.\n")

# ============================================================================
# Q5  ORACLE BOUND -- best per-respondent convex combination
# ============================================================================
cat("\n########## Q5. ORACLE BOUNDS (linear/arithmetic pool) ##########\n")
Q <- sapply(OOF, function(P) P[cbind(1:N, y)])   # N x M likelihood at the truth
em_w <- function(rows, grp, iters = 400) {
  q <- Q[rows, , drop = FALSE]; g <- as.integer(factor(grp[rows])); G <- max(g)
  W <- matrix(1 / M, G, M)
  for (it in 1:iters) {
    num <- q * W[g, , drop = FALSE]
    R   <- num / rowSums(num)
    S   <- rowsum(R, g); n <- tabulate(g, nbins = G)
    W   <- S / n
    W   <- pmax(W, 1e-10); W <- W / rowSums(W)
  }
  list(W = W, g = g)
}
oracle_loss <- function(W, g, rows) -mean(log(pmax(rowSums(Q[rows, , drop = FALSE] * W[g, , drop = FALSE]), 1e-15)))
# global single linear pool (in-sample, cheating only mildly)
fg <- em_w(rep(TRUE, N), rep(1L, N))
cat(sprintf("global linear pool  (1 weight vector, in-sample) : %.5f  w = %s\n",
            oracle_loss(fg$W, fg$g, rep(TRUE, N)),
            paste(sprintf("%s %.3f", MEMB, fg$W[1, ]), collapse = " | ")))
fr <- em_w(rep(TRUE, N), Case)
cat(sprintf("per-respondent ORACLE (1135 weight vectors, cheat): %.5f\n",
            oracle_loss(fr$W, fr$g, rep(TRUE, N))))
# how degenerate are the oracle weights?
cat(sprintf("  respondents whose oracle weight is >0.99 on one member: %.1f%%\n",
            100 * mean(apply(fr$W, 1, max) > 0.99)))
cat("  mean oracle weight per member:", paste(sprintf("%s %.3f", MEMB, colMeans(fr$W)), collapse = " | "), "\n")
# honest split-half version: fit on odd tasks, score even tasks, and vice versa
tk <- long[is_test == FALSE, .(No = No[1]), by = .(Case, Task)]; setorder(tk, No)
stopifnot(identical(tk$No, ymap$No))
half <- (tk$Task %% 2L) == 1L
h1 <- em_w(half, Case); h2 <- em_w(!half, Case)
# map group ids back (factor levels are the sorted Cases, identical in both halves)
lev1 <- sort(unique(Case[half])); lev2 <- sort(unique(Case[!half]))
stopifnot(identical(lev1, lev2))
g_all <- as.integer(factor(Case, levels = lev1))
l_a <- -log(pmax(rowSums(Q[!half, ] * h1$W[g_all[!half], ]), 1e-15))
l_b <- -log(pmax(rowSums(Q[half, ]  * h2$W[g_all[half], ]), 1e-15))
cat(sprintf("per-respondent, HONEST split-half (fit 9-10 tasks, score the rest): %.5f\n",
            mean(c(l_a, l_b))))
lg_a <- -log(pmax(rowSums(Q[!half, ] * matrix(em_w(half, rep(1L, N))$W[1, ], sum(!half), M, byrow = TRUE)), 1e-15))
lg_b <- -log(pmax(rowSums(Q[half, ]  * matrix(em_w(!half, rep(1L, N))$W[1, ], sum(half), M, byrow = TRUE)), 1e-15))
cat(sprintf("global pool on the same split-half protocol (control)        : %.5f\n",
            mean(c(lg_a, lg_b))))
cat(sprintf("=> honest value of per-respondent gating: %+.5f\n",
            mean(c(lg_a, lg_b)) - mean(c(l_a, l_b))))
cat(sprintf("\nreference: production nested log-pool blend = %.5f\n", A$fold_mean))
cat("\nDONE\n")
