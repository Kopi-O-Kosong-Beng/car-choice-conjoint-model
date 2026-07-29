# =============================================================================
# ITERATION 30b — OF THE HETEROGENEITY THAT EXISTS, HOW MUCH IS *REACHABLE*?
#
# WHAT 30a MEASURED. Granting our blend one extra parameter per respondent,
# fitted on that person's own held-out tasks:
#     scale only   (how deterministic they are)   1.03653   gain +0.091
#     none only    (propensity to decline)        0.95522   gain +0.173
# against a population blend of 1.12778. The gap to first place is 0.011. So
# respondent heterogeneity is 15x the entire spread of the leaderboard.
#
# WHY THAT IS NOT AUTOMATICALLY GOOD NEWS. For a TEST respondent we observe zero
# choices, so a_i and T_i are not estimable for them at all. That heterogeneity
# is real, enormous, and by construction invisible. It is also the answer to
# "why do people get 0.3 logloss in other competitions" -- they work on problems
# whose labels are nearly deterministic. Ours are not. The floor is high.
#
# THE ONLY QUESTION THAT MATTERS NOW. A test respondent gives us demographics.
# If a_i or T_i is even WEAKLY predictable from demographics, that slice is
# reachable, and 5% of 0.173 is 0.009 -- the whole gap to first place. If it is
# unpredictable, the field has converged and 1.186 is noise, not skill.
#
# THE DESIGN, AND WHY IT IS HONEST. Fit each respondent's a_i and T_i on their
# own tasks. Then predict those coefficients from demographics OUT OF FOLD, using
# the SAME respondent-grouped folds as everything else, so no respondent ever
# helps predict itself. Apply the out-of-fold prediction and score. Also record
# what the ORACLE value of the same coefficient would have earned, which bounds
# the direction from above.
#
# THE CONTROL THAT KEEPS THIS HONEST. A permutation: shuffle the fitted
# coefficients across respondents and re-run the demographic regression. Any
# apparent gain that survives shuffling is machinery artefact, not signal.
#
# DIAGNOSTIC ONLY -- emits nothing, needs no submission slot.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
set.seed(42)

long <- readRDS("model/artifacts/long.rds"); setorder(long, No, alt)
tk <- unique(long[, .(No, Case, is_test, y)], by = "No"); setorder(tk, No)
tr <- tk[is_test == FALSE]; y <- tr$y; cases <- tr$Case
folds <- readRDS("model/artifacts/folds.rds"); fmap <- folds[order(No), fold]

memb <- readRDS("model/artifacts/blend.rds")$members
OOF <- lapply(memb, function(m) { x <- readRDS(sprintf("model/artifacts/oof_%s.rds", m))
                                  setorder(x, No); as.matrix(x[, .(p1,p2,p3,p4)]) })
L <- 0.528 * log(pmax(OOF[[1]], 1e-12)) + 0.472 * log(pmax(OOF[[2]], 1e-12))
P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P); LP <- log(pmax(P, 1e-12))
base <- logloss(y, P)
cat(sprintf("population blend: %.5f | uniform %.5f\n\n", base, log(4)))

pr <- matrix(long[is_test == FALSE, Price], ncol = 4, byrow = TRUE)
pr <- (pr - mean(pr)) / sd(as.vector(pr))            # GLOBAL scaling: alt4 Price is always 0
none <- matrix(rep(c(0,0,0,1), nrow(P)), ncol = 4, byrow = TRUE)

apply_par <- function(rows, a, Tl, b) {
  M <- LP[rows, , drop = FALSE] / exp(Tl) + a * none[rows, , drop = FALSE] +
       b * pr[rows, , drop = FALSE]
  Q <- exp(M - apply(M, 1, max)); Q / rowSums(Q)
}

# --- fit a_i, T_i, b_i per respondent on ALL their tasks ---------------------
u <- unique(cases)
coefs <- t(sapply(u, function(cs) {
  idx <- which(cases == cs)
  optim(c(0,0,0), function(th) logloss(y[idx], apply_par(idx, th[1], th[2], th[3])) +
                               0.02*sum(th^2), method = "BFGS",
        control = list(maxit = 80))$par
}))
colnames(coefs) <- c("a_none", "T_scale", "b_price")
cat("per-respondent coefficient spread (this is the heterogeneity):\n")
print(round(apply(coefs, 2, function(z) c(sd = sd(z), q10 = quantile(z, .1),
                                          q90 = quantile(z, .9))), 4))

# --- how much of each is predictable from demographics, OUT OF FOLD? ---------
DEMO <- c("segmentind","pparkind","genderind","educind","regionind","Urbind",
          "agea","incomea","milesa","nighta","yearind","milesind","nightind")
dem <- unique(long[, c("Case", DEMO), with = FALSE])
D <- dem[match(u, dem$Case)]
rfold <- fmap[match(u, cases)]                      # respondent -> its fold

predict_oof <- function(v) {
  out <- numeric(length(v))
  for (k in 1:5) {
    trn <- rfold != k
    m <- lm(reformulate(DEMO, "v"), data = cbind(D, v = v)[trn])
    out[!trn] <- predict(m, newdata = D[!trn])
  }
  out
}

score_with <- function(avec, Tvec, bvec) {
  Q <- matrix(NA_real_, nrow(P), 4)
  for (i in seq_along(u)) {
    idx <- which(cases == u[i]); Q[idx, ] <- apply_par(idx, avec[i], Tvec[i], bvec[i])
  }
  logloss(y, Q)
}
z <- rep(0, length(u))

cat("\n=== IS IT REACHABLE FROM DEMOGRAPHICS? (out-of-fold, grouped by respondent) ===\n")
for (nm in c("a_none", "T_scale", "b_price")) {
  v <- coefs[, nm]; ph <- predict_oof(v)
  r2 <- 1 - sum((v - ph)^2) / sum((v - mean(v))^2)
  args <- list(a = z, T = z, b = z); args[[c(a = 1, T = 2, b = 3)[substr(nm,1,1)]]] <- ph
  L_oof <- score_with(if (nm == "a_none") ph else z,
                      if (nm == "T_scale") ph else z,
                      if (nm == "b_price") ph else z)
  L_or  <- score_with(if (nm == "a_none") v else z,
                      if (nm == "T_scale") v else z,
                      if (nm == "b_price") v else z)
  # permutation control
  vp <- sample(v); php <- predict_oof(vp)
  L_perm <- score_with(if (nm == "a_none") php else z,
                       if (nm == "T_scale") php else z,
                       if (nm == "b_price") php else z)
  cat(sprintf("  %-8s  out-of-fold R2 %+.4f | demo-predicted %.5f (%+.5f) | oracle %.5f (%+.5f) | shuffled %+.5f\n",
              nm, r2, L_oof, base - L_oof, L_or, base - L_or, base - L_perm))
}

cat("\nREAD: a demo-predicted gain that beats the shuffled control is reachable signal.\n")
cat("      One that does not is the machinery flattering itself.\n")
