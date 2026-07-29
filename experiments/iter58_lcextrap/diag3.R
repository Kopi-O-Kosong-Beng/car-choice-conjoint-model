# =============================================================================
# ITERATION 58 -- diag3.R  READ-ONLY. Emits NO artifacts into model/artifacts/.
#
# THE QUESTION THIS SETTLES. Under story B (test non-luxury respondents behave
# like training non-luxury respondents) the measured r* = 0.266521 implies TEST
# luxury respondents decline at 0.2436 while TRAINING luxury respondents decline
# at 0.15986. diag2.R showed that is 4.0 respondent-clustered SEs -- a POPULATION
# difference rather than estimation error.
#
# But "population difference" has two very different versions:
#   (i)  COVARIATE SHIFT. Test luxury respondents differ from train luxury
#        respondents on OBSERVED demographics, and those demographics predict
#        none-propensity. Then a better membership model CAN close the gap and
#        it is worth building one.
#   (ii) UNOBSERVED difference. Test luxury respondents are demographically
#        indistinguishable from train luxury respondents. Then NO feature-based
#        model can close the gap, at any capacity, and the only instrument that
#        sees it is the probe.
#
# These are distinguished by two measurements that need no model fit:
#   D9   a train-vs-test propensity model on demographics, LUXURY RESPONDENTS
#        ONLY, honest 5-fold AUC. AUC ~ 0.5 => indistinguishable => (ii).
#   D10  density-ratio reweighting: reweight the 107 training luxury respondents
#        by pi/(1-pi) so their demographic distribution matches the 181 test
#        luxury respondents, then recompute their OBSERVED none-rate. If it
#        stays near 0.160 instead of moving toward 0.2436, covariate shift
#        explains none of the gap.
#   D11  the same two numbers for the FULL population, as a control -- the whole
#        train/test contrast is dominated by the deliberate luxury oversample,
#        so the full-population AUC must be high. If it is, D9 being low is a
#        real finding rather than a broken propensity model.
# =============================================================================
suppressMessages({ library(data.table); library(glmnet) })
source("model/99_utils.R")
set.seed(42)
R_STAR <- (1.7918 - 1.499) / 1.0986

long <- readRDS("model/artifacts/long.rds"); setorder(long, No, alt)
tasks <- unique(long[, .(No, Case, y, is_test, segmentind)]); setorder(tasks, No)

DEMO_CAT <- c("segmentind","pparkind","genderind","educind","regionind","Urbind")
DEMO_NUM <- c("agea","incomea","milesa","nighta","yearind","milesind","nightind")
dem <- unique(long[, c("Case","is_test",DEMO_CAT,DEMO_NUM), with = FALSE]); setorder(dem, Case)
Zl <- list()
for (v in DEMO_CAT) { lv <- sort(unique(dem[[v]])); for (l in lv[-1])
  Zl[[sprintf("%s_%s", v, l)]] <- as.numeric(dem[[v]] == l) }
tf <- function(v, x) if (v %in% c("incomea","milesa","nighta")) log1p(x) else x
for (v in DEMO_NUM) { x <- tf(v, as.numeric(dem[[v]])); Zl[[v]] <- (x - mean(x))/sd(x) }
Z <- do.call(cbind, Zl); rownames(Z) <- as.character(dem$Case)
dem[, lux := as.integer(segmentind %in% c(3, 5))]
nr <- tasks[is_test == FALSE, .(nr = mean(y == 4), ntask = .N), by = Case]
dem <- merge(dem, nr, by = "Case", all.x = TRUE)
setorder(dem, Case)
stopifnot(identical(rownames(Z), as.character(dem$Case)))

auc <- function(lab, sc) {
  r <- rank(sc); n1 <- sum(lab == 1); n0 <- sum(lab == 0)
  (sum(r[lab == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# honest 5-fold propensity AUC + out-of-fold propensities
prop_cv <- function(idx, drop_seg = TRUE) {
  Zs <- Z[idx, , drop = FALSE]
  if (drop_seg) Zs <- Zs[, !grepl("^segmentind_", colnames(Zs)), drop = FALSE]
  Zs <- Zs[, apply(Zs, 2, sd) > 0, drop = FALSE]
  lab <- as.integer(dem$is_test[idx])
  fold <- sample(rep_len(1:5, length(lab)))
  oof <- numeric(length(lab))
  for (k in 1:5) {
    tr <- fold != k
    cv <- cv.glmnet(Zs[tr, , drop = FALSE], lab[tr], family = "binomial", alpha = 0, nfolds = 5)
    oof[!tr] <- as.vector(predict(cv, Zs[!tr, , drop = FALSE], s = "lambda.min", type = "response"))
  }
  list(auc = auc(lab, oof), p = oof, lab = lab, ncol = ncol(Zs))
}

cat("=== D9: can demographics tell TEST luxury respondents from TRAIN luxury ones? ===\n")
ilux <- which(dem$lux == 1)
cat(sprintf("  luxury respondents: %d train, %d TEST\n",
            sum(dem$is_test[ilux] == FALSE), sum(dem$is_test[ilux] == TRUE)))
pl <- prop_cv(ilux)
cat(sprintf("  honest 5-fold propensity AUC (segment dummies EXCLUDED, %d covariates): %.4f\n",
            pl$ncol, pl$auc))
# permutation null
nullauc <- replicate(200, { l <- sample(pl$lab); auc(l, pl$p) })
cat(sprintf("  permutation null: mean %.4f, 95%% upper %.4f\n",
            mean(nullauc), quantile(nullauc, 0.95)))

cat("\n=== D11: control -- the SAME model on the full population ===\n")
pf <- prop_cv(seq_len(nrow(dem)), drop_seg = FALSE)
cat(sprintf("  honest 5-fold propensity AUC, all respondents, segment INCLUDED: %.4f\n", pf$auc))
pf2 <- prop_cv(seq_len(nrow(dem)), drop_seg = TRUE)
cat(sprintf("  honest 5-fold propensity AUC, all respondents, segment EXCLUDED: %.4f\n", pf2$auc))
cat("  (the deliberate luxury oversample is the whole train/test contrast; with\n")
cat("   segment removed the remaining demographic contrast is what D9 measures\n")
cat("   inside the luxury stratum.)\n")

cat("\n=== D10: density-ratio reweighting of the training luxury respondents ===\n")
ltr <- which(dem$lux == 1 & dem$is_test == FALSE)
p_tr <- pl$p[match(ltr, ilux)]
w <- p_tr / pmax(1 - p_tr, 1e-6)
w <- w / mean(w)
obs <- dem$nr[ltr]
cat(sprintf("  unweighted training luxury none-rate      %.5f\n", mean(obs)))
for (cap in c(3, 5, 10, Inf)) {
  wc <- pmin(w, cap); wc <- wc / mean(wc)
  cat(sprintf("  reweighted to TEST-luxury demographics (cap %-4s) %.5f   (shift %+.5f)\n",
              format(cap), sum(wc * obs) / sum(wc), sum(wc * obs)/sum(wc) - mean(obs)))
}
f <- mean(tasks$segmentind[tasks$is_test == TRUE] %in% c(3,5))
r_non_tr <- mean(tasks[is_test == FALSE][!(segmentind %in% c(3,5))]$y == 4)
r_lux_B  <- (R_STAR - (1 - f) * r_non_tr) / f
cat(sprintf("\n  story-B implied TEST luxury none-rate     %.5f   (needs %+.5f)\n",
            r_lux_B, r_lux_B - mean(obs)))
cat("  => covariate shift within the luxury stratum explains what fraction of the gap:\n")
wc <- pmin(w, 5); wc <- wc / mean(wc)
cat(sprintf("     %.1f%%\n", 100 * (sum(wc*obs)/sum(wc) - mean(obs)) / (r_lux_B - mean(obs))))

cat("\n=== D12: the FLOOR -- what a perfectly training-calibrated model ships ===\n")
r_lux_tr <- mean(tasks[is_test == FALSE][segmentind %in% c(3,5)]$y == 4)
floor_rate <- f * r_lux_tr + (1 - f) * r_non_tr
cat(sprintf("  test luxury ROW share f = %.5f\n", f))
cat(sprintf("  a model that reproduces the TRAINING population exactly ships\n"))
cat(sprintf("      %.5f * %.5f + %.5f * %.5f = %.5f\n",
            f, r_lux_tr, 1 - f, r_non_tr, floor_rate))
cat(sprintf("  measured r* = %.5f  =>  UNLEARNABLE GAP = %+.5f\n", R_STAR, R_STAR - floor_rate))
cat(sprintf("  xgb_lw3 (the tree WITH iteration 29's luxury fix) ships 0.21006 -- the floor.\n"))
cat(sprintf("  lcmnl3_both ships 0.22367; xgb_lw2bag (defect intact) ships 0.27258.\n"))

cat("\n=== D13: the identification fork -- what r* alone can and cannot pin ===\n")
cat("  r* is ONE moment. The admissible set is the line f*r_lux + (1-f)*r_non = r*.\n")
cat(sprintf("  %-34s %9s %9s\n", "assumption", "r_lux", "r_non"))
for (s in list(list("story B: r_non = train non-lux", r_non = r_non_tr),
               list("story A: global logit shift (it42)", r_non = 0.3924),
               list("r_non = train ALL-population rate", r_non = mean(tasks[is_test==FALSE]$y == 4)),
               list("r_lux = train luxury rate", r_non = NA))) {
  if (is.na(s$r_non)) {
    rl <- r_lux_tr; rn <- (R_STAR - f * rl) / (1 - f)
  } else { rn <- s$r_non; rl <- (R_STAR - (1 - f) * rn) / f }
  cat(sprintf("  %-34s %9.5f %9.5f\n", s[[1]], rl, rn))
}
cat("\n  under story A, lcmnl3_both's shipped lux 0.18954 misses by only -0.020 while\n")
cat("  xgb_lw2bag's 0.25887 misses by +0.049 -- the diagnosis REVERSES. The claim\n")
cat("  'the latent-class model over-extrapolates on demographics' is not identified\n")
cat("  by the measurement we own.\n")
cat("\nOK\n")
