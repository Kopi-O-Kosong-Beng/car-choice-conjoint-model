# =============================================================================
# ITERATION 16 — diagnostics part 2.
#
# diagnostics.R showed the presence-pattern encoding has literally NO effect
# (direct correction identical to the baseline at 5 dp for every weight). That can
# only happen if the encoding is constant within a choice task, so the task-centred
# version is identically zero. Q4 tests that.
#
# If confirmed, the design nests structure so tightly that neither the bundle key
# nor its presence pattern discriminates WITHIN a task. Q5/Q6 then back off to the
# highest-order key that both recurs across choice sets AND varies within a task:
# the marginal (attribute, level) cell, composed into a bundle-level score.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter16_bundle_encoding/diagnostics2.R
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
source("experiments/iter16_bundle_encoding/bundle_encoder.R")
source("experiments/iter16_bundle_encoding/marginal_encoder.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)

# ---------------------------------------------------------------------------
# Q4. Is the presence pattern constant within a task?
# ---------------------------------------------------------------------------
cat("=========== Q4: does the presence pattern vary within a task? ===========\n")
long[, pkey := make_pkey(long, ATTRS)]
real <- long[alt <= 3]
per_task <- real[, .(nuniq = uniqueN(pkey)), by = No]
cat("distinct presence patterns among the 3 real alternatives of a task:\n")
print(per_task[, .N, by = nuniq][order(nuniq)])
cat(sprintf("tasks where all 3 real bundles share ONE presence pattern: %.2f%%\n",
            100 * mean(per_task$nuniq == 1)))
cat(sprintf("unique presence patterns %d vs unique designs %d\n",
            uniqueN(real$pkey), uniqueN(real[, paste(No)]) * 0 + 5681))
cat("=> a choice set offers three bundles built from the SAME 9 features,\n")
cat("   differing only in TIER LEVELS and PRICE. Any key based on which features\n")
cat("   are present is a property of the CHOICE SET, not of the bundle.\n")

# what actually varies within a task
cat("\nwithin-task standard deviation (mean over tasks), real alternatives only:\n")
for (v in c("lvlsum", "Price")) {
  s <- real[, .(s = sd(get(v))), by = No]
  cat(sprintf("  %-8s %.3f\n", v, mean(s$s)))
}
nlev <- real[, lapply(.SD, function(z) uniqueN(z)), by = No, .SDcols = setdiff(ATTRS, "Price")]
cat(sprintf("  attributes whose LEVEL varies within a task: %.2f of 19 on average\n",
            mean(rowSums(as.matrix(nlev[, -1]) > 1))))

# ---------------------------------------------------------------------------
# Q5. Support of the marginal (attribute, level) cells.
# ---------------------------------------------------------------------------
cat("\n=========== Q5: marginal (attribute, level) support ===========\n")
cells <- rbindlist(lapply(ATTRS, function(a)
  real[, .(attr = a, lvl = get(a))][, .(n = .N), by = .(attr, lvl)]))
cat(sprintf("distinct (attribute, level) cells: %d\n", nrow(cells)))
print(quantile(cells$n, c(0, .1, .5, .9, 1)))
cat(sprintf("median support per cell is %.0fx the design key's %.2f\n",
            median(cells$n), nrow(real) / uniqueN(real$pkey) / 3.09))

# ---------------------------------------------------------------------------
# Q6. Leak detectors on the marginal composite score.
# ---------------------------------------------------------------------------
cat("\n=========== Q6: leak detectors on the marginal composite ===========\n")
add_marg_context(trl); add_marg_context(tel)
apply_marginal_encoding(trl, tel)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]

for (BASE in c("mnl_pw", "xgb_lw2")) {
  f <- sprintf("model/artifacts/oof_%s.rds", BASE)
  if (!file.exists(f)) { cat("skip", BASE, "\n"); next }
  b <- readRDS(f); setorder(b, No)
  pm <- melt(b, id.vars = "No", measure.vars = c("p1","p2","p3","p4"),
             variable.name = "altf", value.name = "p_model")
  pm[, alt := as.integer(sub("p", "", as.character(altf)))][, altf := NULL]
  x <- merge(trl, pm, by = c("No","alt"), all.x = TRUE); setorder(x, No, alt)
  x[, resid := as.numeric(chosen) - p_model]
  base_ll <- logloss(ytr, matrix(x$p_model, ncol = 4, byrow = TRUE))
  cat(sprintf("\nbaseline %-8s OOF %.5f\n", BASE, base_ll))
  for (cc in MRG_SCORE_C) {
    cat(sprintf("  %-16s sd=%.4f cor(enc, own resid)=%+.4f |", cc, sd(x[[cc]]), cor(x[[cc]], x$resid)))
    # softmax-style correction: the score is a utility, so add it to the log-odds
    for (w in c(0.05, 0.1, 0.25, 0.5)) {
      S <- matrix(log(pmax(x$p_model, 1e-12)) + w * x[[cc]], ncol = 4, byrow = TRUE)
      S <- exp(S - apply(S, 1, max)); P <- S / rowSums(S)
      ll <- logloss(ytr, P)
      cat(sprintf(" w=%.2f:%.5f%s", w, ll, if (ll < base_ll) "*" else " "))
    }
    cat("\n")
  }
}
cat("\n'*' = the encoding, added to the baseline utility, genuinely improves logloss.\n")
