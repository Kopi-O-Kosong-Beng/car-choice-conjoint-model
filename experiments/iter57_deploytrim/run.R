# =============================================================================
# ITERATION 57 -- DEPLOY-TIME TRANSFORMS OF THE SHIPPED PROBABILITIES
# Everything in this header was written BEFORE any result was looked at.
#
# -----------------------------------------------------------------------------
# WHY THIS CLASS AT ALL
# -----------------------------------------------------------------------------
# The none-margin shift (iteration 36) is the only correction in this project
# that transferred at ~100%: predicted +0.00218, observed +0.002. It did so
# because its parameter was ANCHORED to a measured test quantity (the constant
# probe: r_test = (1.7918 - 1.499)/1.0986 = 0.26651), not fitted to a local
# surrogate. A competitor describes a "2 system trim". "Trim" most plausibly
# means clipping / winsorising the shipped probabilities. This iteration asks
# whether any such deploy-time transform is worth anything, and separates
# ANCHORED parameters (transfer ~100%) from LOCALLY FITTED ones (~50% or fail).
#
# -----------------------------------------------------------------------------
# THE KEY STRUCTURAL OBSERVATION -- this is what makes the search finite
# -----------------------------------------------------------------------------
# Clipping at a floor, clipping at a ceiling, winsorising, uniform mixing,
# temperature/power maps, and "rank-preserving monotone maps" are all the SAME
# object: an elementwise non-decreasing map h applied to each predicted
# probability, followed by row renormalisation.
#     floor   h(p) = max(p, f)          non-decreasing
#     ceiling h(p) = min(p, c)          non-decreasing
#     unif    h(p) = (1-a)p + a/4       non-decreasing
#     temp    h(p) = p^(1/T)            non-decreasing
# So a POOLED ISOTONIC REGRESSION of the chosen-indicator on the predicted
# probability is the LEAST UPPER BOUND of the entire class: it is the
# best-possible non-decreasing h, chosen with full hindsight on the fitting
# rows. If isotonic recalibration cannot beat the identity out of sample, then
# NO clip, no winsorisation, no temperature and no monotone map can either, and
# the class is closed in one measurement rather than by grid search.
# Per-alternative isotonic (4 separate maps) is the least upper bound of that
# class PLUS any per-alternative multiplicative adjustment.
#
# -----------------------------------------------------------------------------
# METHODOLOGY -- fixed before running
# -----------------------------------------------------------------------------
# 1. Score on the NESTED OOF blend probabilities, rebuilt byte-identically from
#    model/06_blend.R (build_oofP.R reproduces 1.12819 and 1.12741 exactly).
# 2. Every transform gets TWO numbers:
#      IN-SAMPLE  parameter fitted on all 21,565 nested-OOF rows and scored on
#                 the same rows. This is an upper bound, not evidence.
#      NESTED     parameter fitted on nested-OOF rows in folds != k, applied to
#                 fold k. This is the honest number and the only one that
#                 decides anything. Iteration 46 is the precedent: in-sample
#                 +0.00116 became nested -0.00654.
# 3. Paired, respondent-clustered bootstrap SE on the nested per-row losses.
# 4. NOTHING is fitted on the segment-reweighted metric (ESS 208, bootstrap SE
#    0.021). It is reported as a veto/diagnostic only. Iterations 07 and 46.
# 5. Separately report what each transform does to the shipped TEST none-rate
#    against the measured 0.26651.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
# A LOCALLY FITTED transform is adoptable only if its NESTED gain exceeds
# 0.0010 with z > 2 -- because it will then be halved on transfer and still be
# worth ~0.0005, at the edge of visibility. Anything below that is noise
# dressed as a result and is REJECTED regardless of how good the in-sample
# number looks.
# An ANCHORED transform (parameter = a measured test quantity) needs only a
# positive analytic gain, since it does not spend a selection event.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

DIR      <- "experiments/iter57_deploytrim"
R_PROBE  <- (1.7918 - 1.499) / 1.0986        # 0.26651 -- MEASURED on the test set
EPSF     <- 1e-9

rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

pk    <- readRDS(file.path(DIR, "dt57_pack_prod.rds"))
P0    <- pk$Poof; y <- pk$y; fold <- pk$fold; Case <- pk$Case
Ptest <- pk$Ptest
n     <- length(y)
BASE  <- pk$nested
stopifnot(abs(BASE - 1.12819) < 1e-5)

# segment weights -- DIAGNOSTIC ONLY, never a fitting objective
wide <- readRDS("model/artifacts/wide.rds")
rs   <- unique(wide[, .(Case, is_test, segmentind)])
tb   <- merge(rs[is_test == FALSE, .(ptr = .N / sum(!rs$is_test)), by = segmentind],
              rs[is_test == TRUE,  .(pte = .N / sum(rs$is_test)),  by = segmentind],
              by = "segmentind", all.x = TRUE)
tb[is.na(pte), pte := 0]; tb[, wt := pte / ptr]
long <- readRDS("model/artifacts/long.rds")
segw <- merge(unique(long[is_test == FALSE, .(No, segmentind)]), tb[, .(segmentind, wt)],
              by = "segmentind")[order(No), wt]

li   <- function(P) -log(pmax(P[cbind(seq_len(nrow(P)), y[seq_len(nrow(P))])], 1e-15))
liR  <- function(P, rows) -log(pmax(P[cbind(seq_along(rows), y[rows])], 1e-15))
nrm  <- function(P) { P <- pmax(P, EPSF); P / rowSums(P) }

pair <- function(l_base, l_new, B = 2000) {      # positive = new is better
  d <- l_base - l_new; u <- unique(Case)
  num <- tapply(d, Case, sum); den <- tapply(rep(1, n), Case, sum)
  set.seed(11)
  bs <- replicate(B, { s <- sample(u, length(u), TRUE)
                       sum(num[as.character(s)]) / sum(den[as.character(s)]) })
  est <- mean(d); c(gain = est, se = sd(bs), z = est / sd(bs))
}

# ---------------------------------------------------------------------------
# TRANSFORM FAMILIES.  Each is list(np, lo, hi, start, f(P, par) -> P)
# ---------------------------------------------------------------------------
FAM <- list()

FAM$unif <- list(  # extra uniform mixing on top of the blend's own eps
  np = 1, lo = -0.05, hi = 0.30, start = 0.0,
  f = function(P, a) nrm((1 - a) * P + a * 0.25))

FAM$floor <- list( # probability FLOOR then renormalise  ("trim the low tail")
  np = 1, lo = 1e-6, hi = 0.12, start = 0.01,
  f = function(P, fl) nrm(pmax(P, fl)))

FAM$ceil <- list(  # probability CEILING then renormalise ("winsorise the top")
  np = 1, lo = 0.40, hi = 0.999, start = 0.90,
  f = function(P, ce) nrm(pmin(P, ce)))

FAM$clip2 <- list( # two-sided clip
  np = 2, lo = c(1e-6, 0.40), hi = c(0.12, 0.999), start = c(0.01, 0.90),
  f = function(P, p) nrm(pmin(pmax(P, p[1]), p[2])))

FAM$temp <- list(  # power / temperature map
  np = 1, lo = 0.60, hi = 1.80, start = 1.0,
  f = function(P, Tt) { L <- log(pmax(P, 1e-15)) / Tt
                        E <- exp(L - pmax(L[,1],L[,2],L[,3],L[,4])); nrm(E / rowSums(E)) })

MARG_TR <- as.numeric(prop.table(table(y)))       # TRAIN marginal, for OOF scoring
FAM$marg <- list(  # shrink every row toward the marginal
  np = 1, lo = -0.05, hi = 0.40, start = 0.0,
  f = function(P, a) nrm((1 - a) * P + a * matrix(MARG_TR, nrow(P), 4, byrow = TRUE)))

FAM$alt <- list(   # per-alternative multiplicative, alt 4 the reference
  np = 3, lo = rep(-0.5, 3), hi = rep(0.5, 3), start = rep(0, 3),
  f = function(P, d) nrm(P * matrix(c(exp(d), 1), nrow(P), 4, byrow = TRUE)))

FAM$tempalt <- list(  # temperature + per-alternative multiplicative
  np = 4, lo = c(0.60, rep(-0.5, 3)), hi = c(1.80, rep(0.5, 3)), start = c(1, 0, 0, 0),
  f = function(P, p) { L <- log(pmax(P, 1e-15)) / p[1]
                       E <- exp(L - pmax(L[,1],L[,2],L[,3],L[,4])); Q <- E / rowSums(E)
                       nrm(Q * matrix(c(exp(p[2:4]), 1), nrow(P), 4, byrow = TRUE)) })

HBAR <- NULL   # set from the fitting rows each time, so it never leaks
FAM$entmix <- list(   # row-adaptive mixing keyed to predictive entropy
  np = 2, lo = c(-0.05, -0.30), hi = c(0.30, 0.30), start = c(0, 0),
  f = function(P, p) { H <- -rowSums(P * log(pmax(P, 1e-15)))
                       a <- pmin(pmax(p[1] + p[2] * (H - HBAR), 0), 0.5)
                       nrm((1 - a) * P + a * 0.25) })

# ---- nonparametric least-upper-bounds of the monotone class ---------------
iso_fit <- function(p, z) {                       # z = chosen indicator
  o  <- order(p); ir <- isoreg(p[o], z[o])
  xs <- ir$x; ys <- ir$yf
  k  <- !duplicated(xs, fromLast = TRUE)          # step function knots
  approxfun(xs[k], ys[k], rule = 2)
}
iso_pooled_fit <- function(P, rows) {
  Z <- matrix(0, length(rows), 4); Z[cbind(seq_along(rows), y[rows])] <- 1
  iso_fit(as.vector(P[rows, ]), as.vector(Z))
}
iso_pooled_apply <- function(P, g) nrm(pmax(matrix(g(as.vector(P)), nrow(P), 4), 1e-6))
iso_alt_fit <- function(P, rows) {
  Z <- matrix(0, length(rows), 4); Z[cbind(seq_along(rows), y[rows])] <- 1
  lapply(1:4, function(j) iso_fit(P[rows, j], Z[, j]))
}
iso_alt_apply <- function(P, gs) {
  Q <- sapply(1:4, function(j) gs[[j]](P[, j])); nrm(pmax(Q, 1e-6))
}

# ---------------------------------------------------------------------------
# FITTING
# ---------------------------------------------------------------------------
fit_par <- function(fam, rows) {
  HBAR <<- mean(-rowSums(P0[rows, ] * log(pmax(P0[rows, ], 1e-15))))
  obj <- function(p) {
    if (any(p < fam$lo) || any(p > fam$hi)) return(1e6)
    mean(liR(fam$f(P0[rows, , drop = FALSE], p), rows))
  }
  if (fam$np == 1) {
    g  <- seq(fam$lo, fam$hi, length.out = 121)
    v  <- vapply(g, obj, numeric(1)); b <- g[which.min(v)]
    o  <- optimize(obj, c(max(fam$lo, b - (g[2]-g[1])), min(fam$hi, b + (g[2]-g[1]))),
                   tol = 1e-8)
    list(par = o$minimum, value = o$objective)
  } else {
    o <- optim(fam$start, obj, method = "Nelder-Mead",
               control = list(maxit = 4000, reltol = 1e-12))
    o <- optim(o$par, obj, method = "Nelder-Mead",
               control = list(maxit = 4000, reltol = 1e-12))
    list(par = o$par, value = o$value)
  }
}

allrows <- seq_len(n)
res <- list()

rule("PART 1 -- PARAMETRIC FAMILIES: in-sample optimum vs honest nested")
cat(sprintf("baseline nested OOF = %.5f   (production 2-member blend)\n\n", BASE))
cat(sprintf("%-9s %8s %10s %10s %10s %8s\n", "family", "par(full)", "in-samp", "in-gain",
            "NESTED", "z"))

for (nm in names(FAM)) {
  fam <- FAM[[nm]]
  fu  <- fit_par(fam, allrows)
  HBAR <<- mean(-rowSums(P0 * log(pmax(P0, 1e-15))))
  Pin <- fam$f(P0, fu$par)
  ins <- mean(liR(Pin, allrows))

  Pn <- matrix(NA_real_, n, 4); pars <- matrix(NA_real_, 5, fam$np)
  for (k in 1:5) {
    tr <- which(fold != k); te <- which(fold == k)
    fk <- fit_par(fam, tr)
    pars[k, ] <- fk$par
    HBAR <<- mean(-rowSums(P0[tr, ] * log(pmax(P0[tr, ], 1e-15))))
    Pn[te, ] <- fam$f(P0[te, , drop = FALSE], fk$par)
  }
  ln <- li(Pn); nested <- mean(ln)
  st <- pair(li(P0), ln)
  res[[nm]] <- list(par_full = fu$par, insample = ins, nested = nested,
                    pars = pars, stat = st, li = ln,
                    seg = sum(segw * ln) / sum(segw))
  cat(sprintf("%-9s %s  %.5f  %+.5f  %.5f  %+6.2f\n", nm,
              paste(sprintf("%7.4f", fu$par), collapse = ""),
              ins, BASE - ins, nested, st["z"]))
  cat(sprintf("          per-fold par: %s\n",
              paste(apply(pars, 1, function(r) paste(sprintf("%.4f", r), collapse = "/")),
                    collapse = "  ")))
  cat(sprintf("          NESTED gain %+.5f  (se %.5f)   seg-reweighted %.5f\n\n",
              BASE - nested, st["se"], res[[nm]]$seg))
}

rule("PART 2 -- ISOTONIC: the LEAST UPPER BOUND of the whole monotone class")
cat("If these are not positive out of sample, no clip / winsorisation /\n")
cat("temperature / rank-preserving monotone map can be either.\n\n")

# pooled
gfull <- iso_pooled_fit(P0, allrows)
ins_p <- mean(liR(iso_pooled_apply(P0, gfull), allrows))
Pn <- matrix(NA_real_, n, 4)
for (k in 1:5) {
  tr <- which(fold != k); te <- which(fold == k)
  g <- iso_pooled_fit(P0, tr)
  Pn[te, ] <- iso_pooled_apply(P0[te, , drop = FALSE], g)
}
ln <- li(Pn); st <- pair(li(P0), ln)
res$iso <- list(insample = ins_p, nested = mean(ln), stat = st, li = ln,
                seg = sum(segw * ln) / sum(segw))
cat(sprintf("pooled isotonic   in-sample %.5f (gain %+.5f)   NESTED %.5f (gain %+.5f, z %+.2f)\n",
            ins_p, BASE - ins_p, mean(ln), BASE - mean(ln), st["z"]))

# per-alternative
gsf   <- iso_alt_fit(P0, allrows)
ins_a <- mean(liR(iso_alt_apply(P0, gsf), allrows))
Pn <- matrix(NA_real_, n, 4)
for (k in 1:5) {
  tr <- which(fold != k); te <- which(fold == k)
  gs <- iso_alt_fit(P0, tr)
  Pn[te, ] <- iso_alt_apply(P0[te, , drop = FALSE], gs)
}
ln <- li(Pn); st <- pair(li(P0), ln)
res$isoalt <- list(insample = ins_a, nested = mean(ln), stat = st, li = ln,
                   seg = sum(segw * ln) / sum(segw))
cat(sprintf("per-alt isotonic  in-sample %.5f (gain %+.5f)   NESTED %.5f (gain %+.5f, z %+.2f)\n",
            ins_a, BASE - ins_a, mean(ln), BASE - mean(ln), st["z"]))

rule("PART 3 -- RELIABILITY: is there anything for a monotone map to fix?")
Z <- matrix(0, n, 4); Z[cbind(seq_len(n), y)] <- 1
pv <- as.vector(P0); zv <- as.vector(Z)
br <- c(0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0)
bi <- cut(pv, br, include.lowest = TRUE)
tabl <- data.table(bin = levels(bi),
                   n    = as.integer(table(bi)),
                   pred = as.numeric(tapply(pv, bi, mean)),
                   obs  = as.numeric(tapply(zv, bi, mean)))
tabl[, resid := obs - pred]
tabl[, se := sqrt(pred * (1 - pred) / n)]
tabl[, z := resid / se]
print(tabl, digits = 4)
cat("\n(se treats the 4n cells as independent, which they are not -- rows sum to 1\n")
cat(" and respondents cluster -- so |z| here is an OPTIMISTIC upper bound.)\n")

rule("PART 4 -- WHAT EACH TRANSFORM DOES TO THE SHIPPED TEST NONE-RATE")
cat(sprintf("MEASURED test none-rate from the constant probe: %.5f\n", R_PROBE))
cat(sprintf("shipped 2-member blend none-rate                : %.5f  (miss %+.5f)\n\n",
            mean(Ptest[, 4]), mean(Ptest[, 4]) - R_PROBE))
cat(sprintf("%-9s %10s %10s %9s %9s %9s\n", "family", "par", "ship p4", "min p", "max p", "entropy"))
ent <- function(M) mean(-rowSums(M * log(pmax(M, 1e-15))))
cat(sprintf("%-9s %10s %10.5f %9.5f %9.4f %9.5f\n", "(none)", "-", mean(Ptest[, 4]),
            min(Ptest), max(Ptest), ent(Ptest)))
for (nm in names(FAM)) {
  fam <- FAM[[nm]]; p <- res[[nm]]$par_full
  HBAR <<- mean(-rowSums(P0 * log(pmax(P0, 1e-15))))
  Q <- fam$f(Ptest, p)
  cat(sprintf("%-9s %10s %10.5f %9.5f %9.4f %9.5f\n", nm,
              paste(sprintf("%.3f", p), collapse = ","), mean(Q[, 4]), min(Q), max(Q), ent(Q)))
}
Q <- iso_pooled_apply(Ptest, gfull)
cat(sprintf("%-9s %10s %10.5f %9.5f %9.4f %9.5f\n", "iso", "-", mean(Q[,4]), min(Q), max(Q), ent(Q)))
Q <- iso_alt_apply(Ptest, gsf)
cat(sprintf("%-9s %10s %10.5f %9.5f %9.4f %9.5f\n", "isoalt", "-", mean(Q[,4]), min(Q), max(Q), ent(Q)))

rule("PART 5 -- THE ANCHORED TRANSFORM, for comparison (iteration 36 method)")
apply_delta <- function(P, d) { L <- log(pmax(P, 1e-15)); L[, 4] <- L[, 4] + d
                                E <- exp(L - apply(L, 1, max)); E / rowSums(E) }
solve_delta <- function(P, tgt) uniroot(function(d) mean(apply_delta(P, d)[, 4]) - tgt,
                                        c(-5, 5), tol = 1e-12)$root
kl <- function(r, q) r * log(r / q) + (1 - r) * log((1 - r) / (1 - q))
q_old <- mean(Ptest[, 4])
for (w in c(0.60, 0.85, 1.00)) {
  tgt <- w * R_PROBE + (1 - w) * q_old
  d   <- solve_delta(Ptest, tgt); Qn <- apply_delta(Ptest, d)
  cat(sprintf("  w=%.2f  target %.5f  delta %+.5f  ships %.5f  analytic gain %+.5f\n",
              w, tgt, d, mean(Qn[, 4]), kl(R_PROBE, q_old) - kl(R_PROBE, mean(Qn[, 4]))))
}
cat("\n  This parameter is ANCHORED (measured on the graded rows). Everything in\n")
cat("  Parts 1-2 is LOCALLY FITTED and must be discounted accordingly.\n")

saveRDS(res, file.path(DIR, "dt57_results_prod.rds"))
cat("\nOK\n")
