# Independent verification of the iteration 25 artifacts.
#
# The workflow's adversarial verifier agents died on a billing limit, so this
# does the same job: recompute the headline number from the artifact on disk
# rather than trusting the run log, check structural integrity, and look for the
# per-fold signature of a leak (one fold anomalously better than the rest).
suppressMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y, Case)])
setorder(ymap, No)

check <- function(nm, claim) {
  cat("\n===", nm, "(claimed", claim, ")\n")
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", nm)); setorder(d, No)
  cat("  rows:", nrow(d), "(expect 21565)\n")
  cat("  No matches training tasks exactly:", identical(d$No, ymap$No), "\n")
  P <- as.matrix(d[, .(p1, p2, p3, p4)])
  cat("  any NA:", anyNA(P), "| max |rowsum-1|:",
      format(max(abs(rowSums(P) - 1)), digits = 3), "\n")
  cat("  min p:", format(min(P), digits = 3),
      "| max p:", format(max(P), digits = 3), "\n")
  ll <- logloss(ymap$y, P)
  cat("  RECOMPUTED logloss:", sprintf("%.5f", ll),
      if (abs(ll - claim) < 1e-4) "  MATCHES claim\n" else "  *** DISAGREES ***\n")

  fm <- folds[order(No), fold]
  pf <- sapply(1:5, function(k) logloss(ymap$y[fm == k], P[fm == k, ]))
  cat("  per fold:", paste(sprintf("%.5f", pf), collapse = "  "), "\n")
  cat("  fold spread:", sprintf("%.5f", max(pf) - min(pf)),
      "(a leak usually shows as ONE fold far better than the rest)\n")

  t <- readRDS(sprintf("model/artifacts/test_%s.rds", nm)); setorder(t, No)
  Tm <- as.matrix(t[, .(p1, p2, p3, p4)])
  cat("  test rows:", nrow(t), "(expect 4997) | any NA:", anyNA(Tm),
      "| max |rowsum-1|:", format(max(abs(rowSums(Tm) - 1)), digits = 3), "\n")
  cat("  test none-rate:", sprintf("%.4f", mean(Tm[, 4])),
      " (train 0.302, trees ~0.275, lcmnl3 0.223)\n")
  invisible(pf)
}

base <- check("lcmnl3", 1.14396)
for (nm in c("lcmnl3_drift", "lcmnl3_tilt", "lcmnl3_both"))
  check(nm, c(lcmnl3_drift = 1.14231, lcmnl3_tilt = 1.13881, lcmnl3_both = 1.13863)[nm])

cat("\n--- leak sanity: is the gain concentrated in one fold? ---\n")
d0 <- readRDS("model/artifacts/oof_lcmnl3.rds"); setorder(d0, No)
d1 <- readRDS("model/artifacts/oof_lcmnl3_both.rds"); setorder(d1, No)
fm <- folds[order(No), fold]
P0 <- as.matrix(d0[, .(p1,p2,p3,p4)]); P1 <- as.matrix(d1[, .(p1,p2,p3,p4)])
for (k in 1:5)
  cat(sprintf("  fold %d: %.5f -> %.5f  (gain %+.5f)\n", k,
              logloss(ymap$y[fm==k], P0[fm==k,]), logloss(ymap$y[fm==k], P1[fm==k,]),
              logloss(ymap$y[fm==k], P0[fm==k,]) - logloss(ymap$y[fm==k], P1[fm==k,])))
cat("A real structural gain appears in EVERY fold. A leak concentrates.\n")
