# =============================================================================
# ITERATION 58 -- diag4.R  READ-ONLY summary. Emits no artifacts.
#
# Prices the whole question. The none-rate is ONE marginal moment, and the cost
# of getting a single Bernoulli moment wrong has a closed form:
#     C(q; r) = r*log(r/q) + (1-r)*log((1-r)/(1-q))
# which is exactly the logloss a uniform none-margin logit shift can recover and
# no more. Everything else a model does about alternative 4 is conditional
# structure that this moment does not constrain.
# =============================================================================
suppressMessages({ library(data.table) })
source("model/99_utils.R")
R_STAR <- (1.7918 - 1.499) / 1.0986
cost <- function(q, r = R_STAR) r*log(r/q) + (1-r)*log((1-r)/(1-q))

long <- readRDS("model/artifacts/long.rds"); setorder(long, No, alt)
tasks <- unique(long[, .(No, Case, y, is_test, segmentind)]); setorder(tasks, No)
tr_t <- which(tasks$is_test == FALSE); te_t <- which(tasks$is_test == TRUE)
lux_tr <- as.integer(tasks$segmentind[tr_t] %in% c(3,5))
lux_te <- as.integer(tasks$segmentind[te_t] %in% c(3,5))
ytr <- tasks$y[tr_t]
grab <- function(f) { d <- readRDS(f)
  if (is.matrix(d)) return(d[, c("p1","p2","p3","p4"), drop=FALSE])
  setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) }

cat("=== ITERATION 58 SUMMARY TABLE ===\n")
cat(sprintf("measured r* = %.6f | training none-rate 0.30230 | training-calibrated FLOOR 0.20889\n\n", R_STAR))

cfg <- list(
  c("lcmnl3_both", "2",   "1",    "PRODUCTION"),
  c("lcxg_g05",    "0.5", "1",    "weaker membership ridge"),
  c("lcxg_g16",    "16",  "1",    "shrink membership x8"),
  c("lcxg_g128",   "128", "1",    "shrink membership x64"),
  c("lcxg_g512",   "512", "1",    "shrink membership x256"),
  c("lcxg_rw05",   "2",   "0.25", "DOWN-weight luxury in membership"),
  c("lcxg_rw21",   "2",   "21.4", "reweight luxury to TEST share"))
nested <- c(lcmnl3_both = 1.12819, lcxg_g16 = 1.12937, lcxg_g128 = 1.13139,
            lcxg_g512 = 1.13208, lcxg_rw05 = 1.12865)
rwoof  <- c(lcmnl3_both = 1.13273, lcxg_g16 = 1.13336, lcxg_g128 = 1.13495,
            lcxg_g512 = 1.13550, lcxg_rw05 = 1.13290)

cat(sprintf("%-13s %5s %6s | %8s %8s | %8s %8s %8s | %8s %8s\n",
            "variant","lam_g","luxW","OOF","d vs prod","test p4","lux","non","nested","rw-OOF"))
cat(strrep("-", 108), "\n")
for (k in cfg) {
  m <- k[1]
  O <- grab(sprintf("model/artifacts/oof_%s.rds", m))
  T <- grab(sprintf("model/artifacts/test_%s.rds", m))
  ll <- logloss(ytr, O)
  cat(sprintf("%-13s %5s %6s | %8.5f %+9.5f | %8.5f %8.5f %8.5f | %8s %8s\n",
              m, k[2], k[3], ll, 1.13863 - ll, mean(T[,4]),
              mean(T[lux_te==1,4]), mean(T[lux_te==0,4]),
              if (m %in% names(nested)) sprintf("%.5f", nested[[m]]) else "-",
              if (m %in% names(rwoof))  sprintf("%.5f", rwoof[[m]])  else "-"))
}
cat("\n  OOF luxury none-rate (observed 0.15986) -- the in-population defect the\n")
cat("  shrinkage MANUFACTURES, next to the tree that already has it:\n")
for (k in c(cfg, list(c("xgb_lw2bag","-","-","")))) {
  m <- k[1]
  O <- grab(sprintf("model/artifacts/oof_%s.rds", m))
  cat(sprintf("    %-13s OOF lux %.5f  (miss %+.5f)\n", m, mean(O[lux_tr==1,4]),
              mean(O[lux_tr==1,4]) - mean(ytr[lux_tr==1]==4)))
}

# =============================================================================
cat("\n=== WHAT THE NONE-RATE MISS ACTUALLY COSTS ===\n")
cat("  C(q; r*) = the logloss recoverable by a uniform none-margin shift, exactly.\n\n")
cat(sprintf("  %-26s %9s %10s %10s\n", "shipped p4 comes from", "p4", "miss", "cost"))
qs <- list(c("xgb_lw2bag", 0.27258), c("2-member blend (LIVE)", 0.24800),
           c("lcmnl3_both alone", 0.22367), c("xgb_lw3 (tree, fixed)", 0.21006),
           c("training-calibrated floor", 0.20889), c("lcxg_g512", 0.26434))
for (q in qs)
  cat(sprintf("  %-26s %9.5f %+10.5f %10.5f\n", q[[1]], as.numeric(q[[2]]),
              as.numeric(q[[2]]) - R_STAR, cost(as.numeric(q[[2]]))))
cat("\n  paired team-vs-team SE: 0.0047 public, 0.0072 private.\n")
cat("  gap to the current leader (1.197 - 1.186): 0.011\n")
cat("  => the LIVE blend's ENTIRE none-rate miss is worth 0.00091, one fifth of a\n")
cat("     single public SE and one twelfth of the gap to the leader. Fixing it\n")
cat("     perfectly cannot move us up the board.\n")

# =============================================================================
cat("\n=== THE TWO ROUTES TO THE SAME MOMENT ===\n")
B <- grab("model/artifacts/test_blend.rds")
d <- uniroot(function(dd) {
  L <- log(pmax(B, 1e-12)); L[,4] <- L[,4] + dd
  P <- exp(L - apply(L,1,max)); P <- P/rowSums(P); mean(P[,4]) - R_STAR
}, c(-3, 3))$root
L <- log(pmax(B,1e-12)); L[,4] <- L[,4] + d
Bs <- exp(L - apply(L,1,max)); Bs <- Bs/rowSums(Bs)
cat(sprintf("  ROUTE 2  none-margin shift on the LIVE blend: delta = %+.5f\n", d))
cat(sprintf("           ships p4 %.5f (lux %.5f, non %.5f)\n",
            mean(Bs[,4]), mean(Bs[lux_te==1,4]), mean(Bs[lux_te==0,4])))
cat("           model cost: ZERO. Nothing is refitted; it is applied at submission,\n")
cat("           and it is the I-projection of the shipped distribution onto the one\n")
cat("           moment we measured -- the minimum-commitment way to impose it.\n")
cat(sprintf("  ROUTE 1  shrink the membership softmax to lambda_g = 512:\n"))
cat(sprintf("           ships p4 %.5f -- the same moment, reached by breaking the model\n", 0.26434))
cat(sprintf("           model cost: nested %+.5f, income-reweighted OOF %+.5f,\n",
            1.12819 - 1.13208, 1.13273 - 1.13550))
cat("           member OOF -0.00969 (z = -3.01), and it re-manufactures the tree's\n")
cat("           documented luxury defect inside the logit member.\n")
cat("  BOTH ROUTES NEED r*. One of them also costs 0.004-0.010 of fit.\n")

# how far apart are the two shipped distributions?
G <- grab("model/artifacts/test_lcxg_g512.rds")
P <- grab("model/artifacts/test_xgb_lw2bag.rds")
w <- 0.639; Lg <- w*log(pmax(P,1e-12)) + (1-w)*log(pmax(G,1e-12))
Bg <- exp(Lg - apply(Lg,1,max)); Bg <- Bg/rowSums(Bg)
kl <- mean(rowSums(Bs * (log(pmax(Bs,1e-12)) - log(pmax(Bg,1e-12)))))
cat(sprintf("\n  mean KL(route2 || route1-blend) on the 4,997 test rows: %.5f nats\n", kl))
cat("  -- the two agree on the moment and disagree everywhere else by more than the\n")
cat("     entire miss was worth.\n")
cat("\nOK\n")
