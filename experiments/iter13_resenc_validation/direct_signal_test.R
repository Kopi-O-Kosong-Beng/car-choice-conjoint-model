# =============================================================================
# Does the leak-free residual encoding carry GENUINE design signal?
#
# WHY THIS TEST. iter12's tree-based result was destroyed by a subtle leak, and the
# leak-free control (baseline = mnl_pw) is clean by ARGUMENT, not by construction:
# mnl_pw's own OOF prediction for a reference respondent in fold j comes from a model
# trained on folds != j, a set that still INCLUDES the scored fold k. The claim is that
# ~150 global coefficients with no design-level features cannot memorise an individual
# choice set, so the leak path is closed in practice. That is plausible but unproven.
#
# THE TEST. Strip the tree out entirely. If the encoding carries real information about
# a design, then simply ADDING it to the baseline probability must improve logloss:
#     p_corrected = p_mnl_pw + w * encoding,  renormalised
# A leak inflates a flexible learner's score but cannot make this direct correction work,
# because a leak points the wrong way (it hands back the target's own label sign-flipped).
# iter12 established exactly that: for the xgb_lw2 baseline the best direct correction was
# 1.14515, WORSE than doing nothing. If the mnl_pw-baseline encoding is genuine, the same
# correction must go the other way and beat mnl_pw's 1.15686.
#
# This costs about a minute and is more decisive than another blend comparison.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter13_resenc_validation/direct_signal_test.R
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
source("model/encode_design.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
long <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
setorder(trl, No, alt)

BASE <- "mnl_pw"
base_oof <- readRDS(sprintf("model/artifacts/oof_%s.rds", BASE)); setorder(base_oof, No)
pm <- melt(base_oof, id.vars = "No", measure.vars = c("p1","p2","p3","p4"),
           variable.name = "altf", value.name = "p_model")
pm[, alt := as.integer(sub("p", "", as.character(altf)))][, altf := NULL]
trl <- merge(trl, pm, by = c("No","alt"), all.x = TRUE)
setorder(trl, No, alt)
trl[, resid := as.numeric(chosen) - p_model]

ALPHAS <- c(1, 5, 20, 50)
encode_resid <- function(ref, target, alpha) {
  agg <- ref[, .(rs = sum(resid), n = .N), by = .(dkey, alt)]
  t2 <- copy(target)[, .rid := .I][, .(.rid, dkey, alt)]
  t2 <- merge(t2, agg, by = c("dkey","alt"), all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, rs = 0)]
  t2[, enc := rs / (n + alpha)]
  setorder(t2, .rid)
  t2$enc
}

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
base_ll <- logloss(ytr, matrix(trl$p_model, ncol = 4, byrow = TRUE))
cat(sprintf("baseline %s OOF: %.5f\n\n", BASE, base_ll))

cat("direct correction  p_corrected = p_model + w * encoding  (renormalised)\n")
cat("a genuine signal improves on the baseline; a leak cannot\n\n")
best <- list(ll = base_ll, a = NA, w = 0)
for (a in ALPHAS) {
  enc <- numeric(nrow(trl))
  for (k in sort(unique(trl$fold))) {
    idx <- which(trl$fold == k)
    enc[idx] <- encode_resid(trl[fold != k], trl[fold == k], a)
  }
  # correlation with the row's OWN held-out residual: positive = real, negative = leak
  cr <- cor(enc, trl$resid)
  line <- sprintf("alpha=%-3d cor(enc, own resid)=%+.4f  ", a, cr)
  for (w in c(0.25, 0.5, 0.75, 1.0)) {
    P <- matrix(pmax(trl$p_model + w * enc, 1e-9), ncol = 4, byrow = TRUE)
    P <- P / rowSums(P)
    ll <- logloss(ytr, P)
    line <- paste0(line, sprintf("w=%.2f:%.5f%s ", w, ll, if (ll < base_ll) "*" else " "))
    if (ll < best$ll) best <- list(ll = ll, a = a, w = w)
  }
  cat(line, "\n")
}
cat(sprintf("\nbest direct correction: %.5f (alpha=%s, w=%s) vs baseline %.5f\n",
            best$ll, best$a, best$w, base_ll))
if (best$ll < base_ll) {
  cat("VERDICT: the encoding carries GENUINE design signal (direct correction improves).\n")
  cat("         The +0.0043 tree result is credible.\n")
} else {
  cat("VERDICT: no usable direct signal. The tree gain is NOT explained by design\n")
  cat("         information and should be treated as suspect, exactly as in iter12.\n")
}
