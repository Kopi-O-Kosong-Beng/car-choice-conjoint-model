# =============================================================================
# Leakage audit of the iteration-16 marginal-composition encoding, run BEFORE the
# model fit. This repo has been burned twice; the encoding is checked, not argued.
#
# Four checks:
#  A. Respondents live entirely inside one fold (the premise of everything else).
#  B. SCORE-SIDE HONESTY. Permute every label in fold k, recompute the fold-k
#     encoding: it must not move by one bit. This proves no fold-k label -- and in
#     particular no scored row's own label -- enters the feature it is scored with.
#  C. TRAIN-SIDE CHANNEL. The leave-one-fold-out scheme means a training row in
#     fold j (j != k) is encoded from folds != j, which INCLUDES fold k. Quantify
#     how much fold-k labels move those training features, versus the spread of the
#     feature itself. If that ratio is not tiny, the scheme must be made doubly
#     nested (encode fold-j training rows from folds != {j,k}).
#  D. Cell support, and whether the encoding varies within a task at all.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter16_bundle_encoding/audit_leak.R
# =============================================================================
suppressMessages({ library(data.table) })
source("model/99_utils.R")
source("experiments/iter16_bundle_encoding/marginal_encoder.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
setorder(trl, No, alt)

cat("=== A. respondent containment ===\n")
chk <- trl[, .(nf = uniqueN(fold)), by = Case]
cat("respondents:", nrow(chk), " with >1 fold:", sum(chk$nf > 1), "\n")
stopifnot(all(chk$nf == 1))
print(trl[, .(rows = .N, cases = uniqueN(Case)), by = fold][order(fold)])

cat("\n=== B. score-side honesty (fold-k labels permuted) ===\n")
set.seed(999)
for (k in 1:5) {
  e_real <- encode_marginal_aligned(trl[fold != k], trl[fold == k])
  t2 <- copy(trl)
  ik <- which(t2$fold == k)
  # permute chosen WITHIN task so the design stays legal but every label moves
  t2[ik, chosen := as.integer(sample(c(1L,0L,0L,0L))), by = No]
  moved <- mean(t2$chosen[ik] != trl$chosen[ik])
  e_perm <- encode_marginal_aligned(t2[fold != k], t2[fold == k])
  md <- max(abs(as.matrix(e_real) - as.matrix(e_perm)))
  cat(sprintf("fold %d: %.1f%% of fold-k labels changed -> max |d encoding| = %.3e\n",
              k, 100 * moved, md))
  stopifnot(md == 0)
}
cat("PASS: no fold-k label reaches a fold-k feature.\n")

cat("\n=== C. train-side channel (does fold k move fold-j training features?) ===\n")
# For the fold-k model, training rows come from folds j != k and are encoded from
# folds != j. Compare that against the doubly nested folds != {j,k}.
res <- list()
for (k in 1:5) {
  for (j in setdiff(1:5, k)) {
    tgt <- trl[fold == j]
    e1 <- encode_marginal_aligned(trl[fold != j], tgt)              # as implemented
    e2 <- encode_marginal_aligned(trl[!fold %in% c(j, k)], tgt)     # doubly nested
    for (cc in MRG_SCORE) {
      v1 <- e1[[cc]]; v2 <- e2[[cc]]
      res[[length(res) + 1]] <- data.table(k = k, j = j, col = cc,
        maxabs = max(abs(v1 - v2)), sd_feat = sd(v1),
        ratio = max(abs(v1 - v2)) / sd(v1), cor = cor(v1, v2))
    }
  }
}
res <- rbindlist(res)
print(res[, .(max_shift = max(maxabs), sd_of_feature = mean(sd_feat),
              worst_shift_over_sd = max(ratio), min_cor = min(cor)), by = col])

cat("\n=== D. support and within-task variation ===\n")
refr <- trl[alt <= 3]
cells <- rbindlist(lapply(ATTRS, function(a)
  refr[, .(attr = a, lvl = as.integer(get(a)))][, .(n = .N), by = .(attr, lvl)]))
cat("marginal cells:", nrow(cells), " median n:", median(cells$n),
    " min n:", min(cells$n), "\n")
e <- encode_marginal_aligned(trl[fold != 1], trl[fold == 1])
d1 <- trl[fold == 1]
M <- matrix(e[[MRG_SCORE[2]]], ncol = 4, byrow = TRUE)
cat("within-task SD of m_score_a100 (real alts only):",
    round(mean(apply(M[, 1:3], 1, sd)), 4), "\n")
cat("SD of alt-4 (none) score:", round(sd(M[, 4]), 8),
    " <- constant by construction (all-zero bundle)\n")
cat("OK\n")
