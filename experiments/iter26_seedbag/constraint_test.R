# =============================================================================
# Does the monotone price constraint actually help? A paired-by-seed test.
#
# Iteration 08 concluded that adding monotone_constraints to the production
# listwise configuration improved OOF from 1.14152 to 1.13980 (+0.00172, z = 1.35
# on the respondent-clustered paired test -- already weak) and it has carried
# blend weight ~0.31-0.34 ever since.
#
# That comparison was ONE seed of each configuration. The seed sd measured here
# is 0.00283, larger than the entire claimed margin, so a single-seed contrast
# cannot resolve it.
#
# This runs the SAME seeds through both configurations. The pairing matters a
# great deal: the seed governs the early-stopping respondent split as well as
# xgboost's subsample/colsample draws, so a seed that produces a bad split is bad
# for BOTH configurations -- visible directly in the raw numbers, where seed 2 is
# the worst draw for each. Pairing removes that shared component and tests the
# constraint alone.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long <- readRDS("model/artifacts/long.rds")
ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
y <- ymap$y

grab <- function(pat) {
  fs <- sort(Sys.glob(file.path("experiments/iter26_seedbag", pat)))
  rbindlist(lapply(fs, function(f) {
    p <- readRDS(f)
    data.table(seed = p$seed, ll = logloss(y, p$oof))
  }))
}
mo <- grab("seed_[0-9][0-9][0-9].rds")[, .(seed, mono = ll)]
lw <- grab("seed_lw2_*.rds")[, .(seed, lw2 = ll)]
d  <- merge(mo, lw, by = "seed")
d[, diff := mono - lw2]          # positive => the CONSTRAINT is worse

cat("matched seeds:", nrow(d), "of", nrow(mo), "mono and", nrow(lw), "unconstrained\n\n")
cat("  seed   mono(constrained)   lw2(unconstrained)   diff\n")
for (i in seq_len(nrow(d)))
  cat(sprintf("  %-5d  %.5f            %.5f            %+.5f\n",
              d$seed[i], d$mono[i], d$lw2[i], d$diff[i]))

cat(sprintf("\n  mean  mono %.5f   lw2 %.5f\n", mean(d$mono), mean(d$lw2)))
cat(sprintf("  sd    mono %.5f   lw2 %.5f\n", sd(d$mono), sd(d$lw2)))
cat(sprintf("  correlation between configs across seeds: %.3f\n", cor(d$mono, d$lw2)))
cat("  (a high correlation is why pairing is worth doing -- the seed's effect on the\n")
cat("   early-stopping split is shared by both configurations)\n")

n  <- nrow(d); m <- mean(d$diff); s <- sd(d$diff); se <- s / sqrt(n)
tt <- m / se
cat(sprintf("\n--- paired test on the constraint ---\n"))
cat(sprintf("  mean(mono - lw2) = %+.5f   SE %.5f   t = %+.2f on %d df\n", m, se, tt, n - 1))
cat(sprintf("  95%% CI: [%+.5f, %+.5f]\n",
            m - qt(0.975, n - 1) * se, m + qt(0.975, n - 1) * se))
cat(sprintf("  seeds where the constraint WINS: %d of %d\n", sum(d$diff < 0), n))
cat(sprintf("\n  iteration 08 claimed the constraint was worth +0.00172 (i.e. diff = -0.00172).\n"))
cat(sprintf("  Is -0.00172 inside the CI above? %s\n",
            if (-0.00172 >= m - qt(0.975, n-1)*se && -0.00172 <= m + qt(0.975, n-1)*se)
              "yes -- the old result is not excluded, but neither is zero"
            else "NO -- the old result is excluded by this test"))

cat("\n--- what this means for the blend ---\n")
cat("  If the constraint is worth nothing, xgb_mono and xgb_lw2 are the SAME model\n")
cat("  fitted twice, and the blend has been giving two weights to one model. The\n")
cat("  bagged versions are the honest way to keep whichever is real.\n")
