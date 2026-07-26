# =============================================================================
# ITERATION 16 — diagnostics, run BEFORE the expensive fit.
#
# Three questions, all cheap:
#   Q1. Is a BUNDLE-level key actually distinct from the choice-set (design) key?
#   Q2. Does the repaired (presence-pattern) encoding correlate POSITIVELY with the
#       row's own held-out residual?           [leak detector (a)]
#   Q3. Does adding it directly to a baseline probability improve logloss?
#                                              [leak detector (b), iter13 style]
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter16_bundle_encoding/diagnostics.R
# Runtime: ~1 min.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
source("model/encode_design.R")
source("experiments/iter16_bundle_encoding/bundle_encoder.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
long  <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)

# ---------------------------------------------------------------------------
# Q1. The literal bundle key IS the design key.
# ---------------------------------------------------------------------------
cat("=========== Q1: bundle key vs design key ===========\n")
long[, bkey := do.call(paste, c(.SD, sep = "_")), .SDcols = ATTRS]
real <- long[alt <= 3]
mp <- unique(real[, .(bkey, dkey, alt)])
cat(sprintf("unique designs                 : %d\n", uniqueN(real$dkey)))
cat(sprintf("unique real bundles            : %d   (= 3 x designs? %s)\n",
            uniqueN(real$bkey), uniqueN(real$bkey) == 3 * uniqueN(real$dkey)))
cat(sprintf("bundles appearing in >1 design  : %d\n", nrow(mp[, .N, by = bkey][N > 1])))
cat(sprintf("bundles appearing at >1 position: %d\n",
            nrow(unique(mp[, .(bkey, alt)])[, .N, by = bkey][N > 1])))
cat(sprintf("mean appearances per bundle     : %.3f\n", nrow(real) / uniqueN(real$bkey)))
cat(sprintf("mean tasks per design           : %.3f\n",
            nrow(unique(real[, .(No, dkey)])) / uniqueN(real$dkey)))

# Numerical proof: build the literal bundle-share encoding the task asked for and
# compare it, value for value, with the existing design-share encoding.
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)                      # -> share_a1/a5/a20, design_n

lit <- copy(trl)
for (cc in ENC_COLS) lit[, (cc) := NA_real_]
encode_literal_bundle <- function(ref, target) {     # keyed on bkey instead of (dkey,alt)
  t2 <- copy(target)[, .rid := .I]
  prior <- ref[, .(pr = sum(chosen) / .N), by = alt]
  cnt   <- ref[, .(n_ch = sum(chosen), n = .N), by = .(bkey)]
  t2 <- merge(t2[, .(.rid, bkey, alt)], cnt, by = "bkey", all.x = TRUE)
  t2 <- merge(t2, prior, by = "alt", all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, n_ch = 0)]
  for (a in ENC_ALPHAS) t2[, paste0("share_a", a) := (n_ch + a * pr) / (n + a)]
  t2[, design_n := n]
  setorder(t2, .rid)
  t2[, .SD, .SDcols = ENC_COLS]
}
for (k in sort(unique(lit$fold))) {
  idx <- which(lit$fold == k)
  e <- encode_literal_bundle(lit[fold != k], lit[fold == k])
  for (cc in ENC_COLS) set(lit, idx, cc, e[[cc]])
}
cat("\nliteral bundle-share encoding vs existing design-share encoding:\n")
for (cc in ENC_COLS) {
  d <- max(abs(lit[[cc]] - trl[[cc]]))
  cat(sprintf("  %-10s max abs difference = %.3e   identical: %s\n", cc, d, d < 1e-12))
}
cat("(alt 4 differs only because the all-zero bundle is one key; alts 1-3 are the test)\n")
for (cc in ENC_COLS) {
  d <- max(abs(lit[alt <= 3][[cc]] - trl[alt <= 3][[cc]]))
  cat(sprintf("  alts1-3 %-10s max abs diff = %.3e   identical: %s\n", cc, d, d < 1e-12))
}

# ---------------------------------------------------------------------------
# Structure that forces the repair: what recurs, and what is constant?
# ---------------------------------------------------------------------------
cat("\n=========== structure ===========\n")
cat("richness on real bundles (unique):", paste(sort(unique(real$richness)), collapse = ","), "\n")
cat("rich_task_mean (unique):", paste(sort(unique(long$rich_task_mean)), collapse = ","), "\n")
cat("  -> rival RICHNESS carries no information; rival strength must use Price / lvlsum\n")
cat("lvlsum on real bundles: range", paste(range(real$lvlsum), collapse = "-"),
    " sd", round(sd(real$lvlsum), 2), "\n")
pk <- make_pkey(real, ATTRS)
cat(sprintf("presence patterns: %d unique, %.2f slots each (%.1fx the design key)\n",
            uniqueN(pk), length(pk) / uniqueN(pk),
            (length(pk) / uniqueN(pk)) / (nrow(real) / uniqueN(real$bkey))))
cat(sprintf("bundles per presence pattern: %.2f\n", uniqueN(real$bkey) / uniqueN(pk)))

# ---------------------------------------------------------------------------
# Q2 / Q3. Leak detectors on the repaired encoding.
# ---------------------------------------------------------------------------
cat("\n=========== Q2/Q3: leak detectors on the presence-pattern encoding ===========\n")
trl2 <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel2 <- long[is_test == TRUE]
setorder(trl2, No, alt); setorder(tel2, No, alt)
add_bundle_context(trl2); add_bundle_context(tel2)
apply_bundle_encoding(trl2, tel2)

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]

for (BASE in c("mnl_pw", "xgb_lw2")) {
  f <- sprintf("model/artifacts/oof_%s.rds", BASE)
  if (!file.exists(f)) { cat("skip", BASE, "(absent)\n"); next }
  b <- readRDS(f); setorder(b, No)
  pm <- melt(b, id.vars = "No", measure.vars = c("p1","p2","p3","p4"),
             variable.name = "altf", value.name = "p_model")
  pm[, alt := as.integer(sub("p", "", as.character(altf)))][, altf := NULL]
  x <- merge(trl2, pm, by = c("No","alt"), all.x = TRUE)
  setorder(x, No, alt)
  x[, resid := as.numeric(chosen) - p_model]
  base_ll <- logloss(ytr, matrix(x$p_model, ncol = 4, byrow = TRUE))
  cat(sprintf("\nbaseline %-8s OOF %.5f\n", BASE, base_ll))
  for (cc in c("b_share_a5_c", "b_share_a20_c", "b_share_a50_c")) {
    cr <- cor(x[[cc]], x$resid)
    line <- sprintf("  %-14s cor(enc, own held-out resid) = %+.4f  |", cc, cr)
    for (w in c(0.1, 0.25, 0.5, 1.0)) {
      P <- matrix(pmax(x$p_model + w * x[[cc]], 1e-9), ncol = 4, byrow = TRUE)
      P <- P / rowSums(P)
      ll <- logloss(ytr, P)
      line <- paste0(line, sprintf(" w=%.2f:%.5f%s", w, ll, if (ll < base_ll) "*" else " "))
    }
    cat(line, "\n")
  }
}
cat("\npositive correlation + a '*' (direct correction beats the baseline) = genuine signal.\n")
cat("negative correlation, or no '*' anywhere, = the iter12 failure mode.\n")
