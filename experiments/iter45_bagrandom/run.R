# =============================================================================
# ITERATION 45 -- TUNE THE ENSEMBLE RANDOMNESS, NOT THE SINGLE-MODEL RANDOMNESS
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# THE GAP, AND WHY IT IS AN ERROR RATHER THAN AN OMISSION
# -----------------------------------------------------------------------------
# Production ships xgb_lw2bag: a 10-seed bag at subsample = 0.8, colsample = 0.8.
# Those two values were tuned in ITERATION 06, for a SINGLE model, before bagging
# existed in this project. Iteration 26 then wrapped 10 seeds around them and
# changed nothing else -- its own header says "ONE CHANGE UNDER TEST: the number
# of seeds averaged."
#
# But the optimal randomness for an ENSEMBLE is not the optimal randomness for
# one model, and the direction is known:
#
#   loss(bag) ~ bias^2 + rho * var + (1 - rho) * var / S
#
# where rho is the average correlation between member predictions and S the bag
# size. Raising per-model randomness RAISES var and LOWERS rho. For S = 1 that
# trade is bad, so single-model tuning drives randomness DOWN toward 0.8/0.8. For
# S = 10 the (1-rho)var/S term is divided by ten, so the rho reduction is worth
# far more than the var increase, and the optimum moves DOWN in subsample and
# colsample -- i.e. MORE randomness than a single model wants.
#
# Production is therefore almost certainly at the wrong point on this curve, and
# it is at the wrong point for a reason that is derivable rather than empirical.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# A bagged tree with MORE per-model randomness than 0.8/0.8 will beat the
# production bag, because the correlation reduction is worth more than the
# variance increase once predictions are averaged over many seeds.
#
# Directional pre-registration, so a null cannot be re-read as a win: the
# optimum should lie at subsample <= 0.8 AND colsample <= 0.8. If the grid
# instead prefers LESS randomness, the mechanism above is wrong and the result
# should be treated as noise regardless of its size.
#
# -----------------------------------------------------------------------------
# WHY THIS IS AFFORDABLE, WHICH IS ITSELF A FINDING FROM ITERATION 39
# -----------------------------------------------------------------------------
# Early stopping calls back into R to evaluate a custom metric EVERY round, over
# a model that keeps growing, so cost is roughly quadratic in rounds: ~3 min per
# fit. Iteration 39 removed the carve and the same fit takes ~11 SECONDS -- a 20x
# speedup that makes a real grid search possible for the first time in this
# project. Iteration 39's own change did not survive its blend gate, but the
# harness it built is what makes this iteration cheap.
#
# BASELINE IS THEREFORE xgb_lw2fr (fixed rounds, 0.8/0.8, OOF 1.13604), NOT
# xgb_lw2bag (early stopping, 0.8/0.8, OOF 1.13682). Comparing a fixed-rounds
# challenger to an early-stopping baseline would confound the randomness change
# with the rounds protocol -- exactly the config-confounding error iteration 26
# caught once already.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
#   1. The best grid point must beat the 0.8/0.8 baseline at the SAME seed count.
#   2. It must sit in the pre-registered direction (subsample <= 0.8 AND
#      colsample <= 0.8). A winner outside that quadrant is rejected as noise.
#   3. Multiplicity: 12 grid points searched, so the winner needs
#      z >= qnorm(1 - 0.025/12) = 2.87 on a paired respondent-clustered test at
#      the FULL seed count, not the screening count.
#   4. It must improve the PRODUCTION 2-member nested blend by more than the
#      blend-level seed sd of 0.00048. Member gains do not automatically reach
#      the blend -- iteration 39 gained +0.00252 at member level and only
#      +0.00020 at blend level in the 5-member pool.
# ADOPT only if all four pass. Screening at S=3 then confirming the winner at
# S=10 keeps the search cheap without letting a 3-seed fluke through.
#
# ARTIFACTS: oof_xgb_bagrnd.rds / test_xgb_bagrnd.rds -- NEW names, grep-verified.
# An earlier iteration overwrote a live blend member by inheriting a name from a
# copied script; the name here is built from a literal, never from a variable.
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")

DIR <- "experiments/iter45_bagrandom"
dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
NR <- 540L            # fixed rounds, the iteration-39 protocol
S_SCREEN <- 3L        # seeds per grid point while screening
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
long <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo, ENC_COLS)

sbt <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M, 1, max)
                     E <- exp(M); E / rowSums(E) }
obj_lw <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(sbt(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]

# one bagged nested OOF at a given (subsample, colsample), S seeds
bag_oof <- function(sub, col, S) {
  acc <- matrix(0, 21565L, 4L)
  for (s in seq_len(S)) {
    prm <- list(eta = 0.03, max_depth = 8, min_child_weight = 20,
                subsample = sub, colsample_bytree = col, base_score = 0,
                nthread = 3, seed = s)
    P <- matrix(NA_real_, 21565L, 4L)
    for (k in 1:5) {
      d <- trl[fold != k]
      fit <- xgb.train(params = prm,
                       data = xgb.DMatrix(as.matrix(d[, ..feat]), label = as.numeric(d$chosen)),
                       nrounds = NR, verbose = 0, obj = obj_lw, maximize = FALSE)
      v <- trl[fold == k]
      Pk <- sbt(as.vector(predict(fit, xgb.DMatrix(as.matrix(v[, ..feat])), outputmargin = TRUE)))
      P[match(unique(v$No), sort(unique(trl$No))), ] <- Pk
    }
    acc <- acc + log(pmax(P, 1e-15))
  }
  E <- exp(acc / S - apply(acc / S, 1, max)); E / rowSums(E)
}

GRID <- CJ(sub = c(0.5, 0.6, 0.7, 0.8), col = c(0.4, 0.6, 0.8))
rule(sprintf("SCREENING GRID: %d points x %d seeds x 5 folds, fixed %d rounds", nrow(GRID), S_SCREEN, NR))
cat("  baseline for comparison: xgb_lw2fr (fixed rounds, 0.8/0.8) OOF 1.13604\n")
cat("  pre-registered direction: winner must have sub <= 0.8 AND col <= 0.8\n\n")

res <- list()
for (i in seq_len(nrow(GRID))) {
  t0 <- Sys.time()
  P <- bag_oof(GRID$sub[i], GRID$col[i], S_SCREEN)
  ll <- logloss(ytr, P)
  res[[i]] <- data.table(sub = GRID$sub[i], col = GRID$col[i], oof = ll,
                         mins = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  cat(sprintf("  sub %.1f col %.1f -> OOF %.5f   (%.1f min)\n",
              GRID$sub[i], GRID$col[i], ll, res[[i]]$mins))
  saveRDS(rbindlist(res), file.path(DIR, "grid.rds"))
}
R <- rbindlist(res); setorder(R, oof)
rule("SCREENING RESULT (S = 3)")
print(R)
fwrite(R, file.path(DIR, "grid.csv"))
cat(sprintf("\n  best: sub %.1f col %.1f at %.5f\n", R$sub[1], R$col[1], R$oof[1]))
cat(sprintf("  in the pre-registered direction? %s\n",
            if (R$sub[1] <= 0.8 && R$col[1] <= 0.8) { "YES" } else { "NO -- reject as noise" }))
cat("\n  next: confirm the winner at S = 10 and run compare.R + the blend gate.\n")
