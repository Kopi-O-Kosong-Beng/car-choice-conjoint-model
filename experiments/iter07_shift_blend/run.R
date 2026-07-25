# =============================================================================
# ITERATION 07 — Calibrate the blend for the population we are actually scored on.
#
# EVIDENCE: our one trustworthy calibration point is local 1.13878 -> public 1.201.
# The test set is measurably harder than our CV. When a model faces a harder
# population than it was calibrated on it is systematically OVERCONFIDENT, and the
# fix is to soften: raise the temperature, add a little uniform mass. Our blend,
# tuned on plain OOF, chose T = 0.968 and eps = 0 -- i.e. slight SHARPENING and no
# insurance at all. That is the right answer for the training population and
# probably the wrong one for the test population.
#
# METHOD: refit the blend parameters against an income-reweighted objective
# (respondents importance-weighted so the training income distribution matches the
# test one). Report both blends under both metrics, so the trade-off is explicit.
#
# HONEST LIMITATION: reweighting closes only ~0.005 of a 0.062 gap, so income shift
# is NOT the main driver of the gap. This buys a better-calibrated blend for the
# part of the shift we can actually measure; it is not a cure for the whole gap.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter07_shift_blend/run.R
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
wide  <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")

want <- sub("#.*$", "", readLines("model/members.txt")); want <- trimws(want)
memb <- want[nzchar(want)]
memb <- memb[file.exists(sprintf("model/artifacts/test_%s.rds", memb))]
cat("members:", paste(memb, collapse = ", "), "\n")
OOF <- lapply(memb, function(m) { d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
TST <- lapply(memb, function(m) { d <- readRDS(sprintf("model/artifacts/test_%s.rds", m)); setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
names(OOF) <- names(TST) <- memb

ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
y <- ymap$y; fmap <- folds[order(No), fold]
M <- length(memb); eps0 <- 1e-12

# importance weights matching the test income distribution
resp <- unique(wide[, .(Case, is_test, incomeind)])
ntr <- resp[is_test == FALSE, .N]; nte <- resp[is_test == TRUE, .N]
wt <- merge(resp[is_test == FALSE, .(ptr = .N/ntr), by = incomeind],
            resp[is_test == TRUE,  .(pte = .N/nte), by = incomeind], by = "incomeind", all.x = TRUE)
wt[is.na(pte), pte := 0]; wt[, w := pmin(pmax(pte/ptr, 0.2), 5)]
rw <- merge(unique(wide[is_test == FALSE, .(No, incomeind)]), wt[, .(incomeind, w)],
            by = "incomeind")[order(No), w]

blend <- function(theta, Ps, rows) {
  w <- exp(theta[1:M]); w <- w / sum(w)
  Tt <- exp(theta[M+1]); eA <- plogis(theta[M+2]) * 0.15
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, Ps))
  L <- L / Tt
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
wll <- function(P, rows, weights) {
  li <- -log(pmax(P[cbind(seq_len(nrow(P)), y[rows])], 1e-15))
  if (is.null(weights)) mean(li) else sum(weights[rows] * li) / sum(weights[rows])
}
obj <- function(theta, rows, weights) wll(blend(theta, OOF, rows), rows, weights)
fit_w <- function(rows, weights) optim(c(rep(0, M), 0, -3), obj, rows = rows, weights = weights,
                                       method = "Nelder-Mead", control = list(maxit = 4000))

nested <- function(weights) {
  out_plain <- out_rw <- numeric(5)
  for (k in 1:5) {
    o <- fit_w(fmap != k, weights)
    P <- blend(o$par, OOF, fmap == k)
    out_plain[k] <- wll(P, fmap == k, NULL)
    out_rw[k]    <- wll(P, fmap == k, rw)
  }
  c(plain = mean(out_plain), reweighted = mean(out_rw))
}

cat("\n--- nested evaluation of both tuning targets ---\n")
n_plain <- nested(NULL)
cat(sprintf("blend tuned on PLAIN OOF      : plain %.5f | reweighted %.5f\n", n_plain[1], n_plain[2]))
n_rw <- nested(rw)
cat(sprintf("blend tuned on REWEIGHTED OOF : plain %.5f | reweighted %.5f\n", n_rw[1], n_rw[2]))

for (tag in c("plain", "rw")) {
  o <- fit_w(rep(TRUE, length(y)), if (tag == "plain") NULL else rw)
  w <- exp(o$par[1:M]); w <- w/sum(w)
  cat(sprintf("\n[%s] T = %.3f  eps = %.4f\n  weights: %s\n", tag,
              exp(o$par[M+1]), plogis(o$par[M+2]) * 0.15,
              paste(sprintf("%s %.3f", memb, round(w, 3)), collapse = " | ")))
  if (tag == "rw") {
    saveRDS(list(members = memb, par = o$par, nested = n_rw[["reweighted"]],
                 nested_plain = n_rw[["plain"]], tuned_on = "reweighted"),
            "model/artifacts/blend_shift.rds")
    saveRDS(clip_norm(blend(o$par, TST, seq_len(4997))), "model/artifacts/test_blend_shift.rds")
  }
}

# How much extra softening would a HARDER test population want? Scan a multiplier
# on top of the plain-tuned blend and see where the reweighted loss bottoms out.
cat("\n--- temperature sensitivity (plain-tuned blend, scored reweighted) ---\n")
o <- fit_w(rep(TRUE, length(y)), NULL)
for (mult in c(1.00, 1.05, 1.10, 1.20, 1.30)) {
  th <- o$par; th[M+1] <- th[M+1] + log(mult)
  P <- blend(th, OOF, rep(TRUE, length(y)))
  cat(sprintf("  T x %.2f  ->  plain %.5f   reweighted %.5f\n", mult,
              wll(P, rep(TRUE, length(y)), NULL), wll(P, rep(TRUE, length(y)), rw)))
}
cat("\nOK\n")
