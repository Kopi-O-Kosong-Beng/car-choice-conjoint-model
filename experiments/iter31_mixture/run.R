# =============================================================================
# ITERATION 31 — TWO THINGS I ASSERTED WITHOUT TESTING
#
# I concluded the ceiling is ~1.194. That conclusion rests on two claims I did
# not actually verify. Both could be wrong, and both are cheap to check.
#
# CLAIM 1 (untested): "the test none-rate is 0.2665 vs training 0.3023 because
# the PEOPLE differ." But iteration 30 proved demographics carry ZERO information
# (out-of-fold R2 negative on every axis). So the people canNOT be the reason.
# The only other thing that varies is the DESIGN -- the bundles they were shown.
# If test respondents face cheaper or richer choice sets, they decline less, and
# that is mechanical, not behavioural. NOBODY HAS EVER COMPARED THE TWO DESIGNS.
# If there is a shift, and especially if any of it falls outside training
# support, that is a direct, structural explanation for the +0.069 offset --
# which would make it partly FIXABLE rather than irreducible.
#
# CLAIM 2 (untested): "our blend already integrates over unobserved
# heterogeneity." Iteration 30 measured that heterogeneity and it is enormous:
# sd(a_none)=0.96, sd(b_price)=0.93, sd(T_scale)=0.66. For an unseen respondent
# the CORRECT prediction is the mixture E_theta[softmax(u + theta)], NOT
# softmax(u + E[theta]). Those differ, and the mixture is always flatter.
#
# And our combiner is a LOG-OPINION pool -- a geometric mean, a product of
# experts. Geometric pooling is SHARPER than arithmetic. If what we actually need
# is to represent uncertainty about a latent respondent type, we have been using
# the sharpening operator where the mixing operator belongs, on every single row.
# That is not a 0.001 effect if it is real; it is a systematic bias in every
# prediction we ship.
#
# The earlier temperature probe was the crude version of this and came back
# contradictory (0.88 random / 1.04 income). A uniform temperature is the WRONG
# instrument: real mixing flattens specifically along the none and price axes,
# not isotropically. This tests the structured version.
#
# CAVEAT ON THE SDs. The per-respondent coefficients were fitted, so their spread
# contains estimation noise as well as true heterogeneity -- sd 0.96 is an upper
# bound. So sweep a shrinkage factor k from 0 (current model) upward and let the
# out-of-population holdout choose, rather than assuming the measured sd.
#
# DECISION RULE: adopt only if the optimal k > 0 on the INCOME holdout (new,
# wealthier people) AND on the random holdout. In-sample gain is not evidence.
#
# DIAGNOSTIC -- emits nothing.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R"); set.seed(42)

long <- readRDS("model/artifacts/long.rds"); setorder(long, No, alt)
tk <- unique(long[, .(No, Case, is_test, y)], by = "No"); setorder(tk, No)

# ---------------------------------------------------------------- CLAIM 1 ----
cat("=== 1. DOES THE TEST DESIGN DIFFER FROM TRAINING? ===\n")
cat("(demographics carry zero signal, so if anything explains the none-rate drop it is here)\n\n")
ATTR <- c("Price","richness","lvlsum","CC","GN","NS","BU","FA","LD","BZ","FC",
          "FP","RP","PP","KA","SC","TS","NV","MA","LB","AF","HU")
real <- long[alt != 4]                                   # alt 4 is all-zero by construction
cat(sprintf("  %-10s %8s %8s %9s %10s\n", "attribute", "train", "test", "diff", "sd units"))
for (v in ATTR) {
  a <- as.numeric(real[is_test == FALSE][[v]]); b <- as.numeric(real[is_test == TRUE][[v]])
  d <- mean(b, na.rm = TRUE) - mean(a, na.rm = TRUE); s <- sd(a, na.rm = TRUE)
  if (is.finite(d) && is.finite(s) && s > 0 && abs(d / s) > 0.02)
    cat(sprintf("  %-10s %8.3f %8.3f %+9.4f %+10.3f\n", v, mean(a, na.rm=TRUE),
                mean(b, na.rm=TRUE), d, d / s))
}
# does any test row sit outside the training design space?
pt <- real[is_test == FALSE, Price]; pe <- real[is_test == TRUE, Price]
cat(sprintf("\n  Price: train range [%g, %g], test range [%g, %g]; test rows outside: %.2f%%\n",
            min(pt), max(pt), min(pe), max(pe), 100*mean(pe < min(pt) | pe > max(pt))))
cat(sprintf("  cheapest real bundle per task -- train mean %.3f, test mean %.3f\n",
            mean(long[is_test==FALSE & alt!=4, min(Price), by=No]$V1),
            mean(long[is_test==TRUE  & alt!=4, min(Price), by=No]$V1)))

# ---------------------------------------------------------------- CLAIM 2 ----
cat("\n=== 2. SHOULD WE BE MIXING INSTEAD OF SHARPENING? ===\n")
tr <- tk[is_test == FALSE]; y <- tr$y; cases <- tr$Case
memb <- readRDS("model/artifacts/blend.rds")$members
OOF <- lapply(memb, function(m){ x <- readRDS(sprintf("model/artifacts/oof_%s.rds", m))
                                 setorder(x, No); as.matrix(x[, .(p1,p2,p3,p4)]) })
Lg <- 0.528*log(pmax(OOF[[1]],1e-12)) + 0.472*log(pmax(OOF[[2]],1e-12))
Pg <- exp(Lg - apply(Lg,1,max)); Pg <- Pg/rowSums(Pg)          # geometric (production)
Pa <- 0.528*OOF[[1]] + 0.472*OOF[[2]]                          # arithmetic (mixture)
cat(sprintf("  geometric pool (production) %.5f\n  arithmetic pool (mixture)   %.5f   diff %+.5f\n",
            logloss(y,Pg), logloss(y,Pa), logloss(y,Pg)-logloss(y,Pa)))

pr <- matrix(long[is_test==FALSE, Price], ncol=4, byrow=TRUE)
pr <- (pr - mean(pr))/sd(as.vector(pr))
none <- matrix(rep(c(0,0,0,1), nrow(Pg)), ncol=4, byrow=TRUE)
LPg <- log(pmax(Pg,1e-12))

# Monte-Carlo integrate the prediction over the measured heterogeneity
SD <- c(a = 0.958, b = 0.932, Tl = 0.656)
mixed <- function(k, rows, R = 60) {
  M <- LPg[rows, , drop=FALSE]; N <- none[rows,,drop=FALSE]; PRr <- pr[rows,,drop=FALSE]
  acc <- matrix(0, nrow(M), 4)
  for (r in 1:R) {
    th <- rnorm(3) * SD * k
    Z <- M/exp(th[3]) + th[1]*N + th[2]*PRr
    Q <- exp(Z - apply(Z,1,max)); acc <- acc + Q/rowSums(Q)
  }
  acc/R
}

holdouts <- list()
u <- unique(cases)
holdouts$random <- which(cases %in% sample(u, length(u) %/% 2))
dem <- unique(long[, .(Case, incomea)]); inc <- dem[match(u, dem$Case), incomea]
holdouts$rich <- which(cases %in% u[inc > median(inc, na.rm=TRUE)])
holdouts$all <- seq_along(y)

cat("\n  k = how much of the measured heterogeneity we integrate over (0 = today)\n")
cat(sprintf("  %-8s %s\n", "holdout", paste(sprintf("k=%.2f", seq(0,1,by=0.2)), collapse="   ")))
best <- c()
for (h in names(holdouts)) {
  rows <- holdouts[[h]]
  v <- sapply(seq(0,1,by=0.2), function(k) if (k==0) logloss(y[rows], Pg[rows,,drop=FALSE])
                                           else logloss(y[rows], mixed(k, rows)))
  best[h] <- seq(0,1,by=0.2)[which.min(v)]
  cat(sprintf("  %-8s %s   best k=%.1f  gain %+.5f\n", h,
              paste(sprintf("%.5f", v), collapse=" "), best[h], v[1]-min(v)))
}
cat(sprintf("\nDECISION: mixing helps out of population iff best k > 0 on BOTH random and rich -> %s\n",
            best["random"] > 0 && best["rich"] > 0))
