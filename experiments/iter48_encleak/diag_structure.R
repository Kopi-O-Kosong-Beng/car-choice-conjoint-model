# =============================================================================
# ITERATION 48 -- PHASE A.  Is the design-share encoding leaking into PRODUCTION's OOF?
#
# NO MODELS FITTED HERE.  Pure structural arithmetic on the encoder in
# model/encode_design.R.  Everything below is a claim that can be checked without
# training anything, and every number printed is reproducible from long.rds +
# folds.rds alone.
#
# THE CLAIM UNDER TEST.  encode_design.R gives a row in fold j the shrunk share
# built from folds != j.  The docstring says "a person never contributes to their
# own encoding", which is true at the ROW level.  But the numerator for a
# (dkey, alt) cell is
#         n_ch(j)  =  C  -  c_j          C = cell total over ALL folds
#                                        c_j = own fold's chosen count in the cell
# so, CONDITIONAL ON THE CELL, the feature is a strictly decreasing function of the
# row's OWN FOLD's choice count -- which contains the row's own label.  A model with
# enough capacity to isolate a cell can therefore read c_j (hence the label) off the
# encoding.  Support is thin, so c_j is often literally one label.
#
# Nothing here proves a model exploits it.  Phase B does that.  Phase A measures
# (i) whether the channel exists, (ii) how wide it is, (iii) its upper bound.
#
# Run: Rscript experiments/iter48_encleak/diag_structure.R
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
source("model/encode_design.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
long  <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)

cat("=========================================================================\n")
cat("A1.  Is the encoder fold-aware as claimed?  (verified by reading + by data)\n")
cat("=========================================================================\n")
fp <- trl[, .(nf = uniqueN(fold)), by = Case]
cat(sprintf("  respondents spanning >1 fold: %d of %d   (0 = grouping intact)\n",
            sum(fp$nf > 1), nrow(fp)))
cat("  encode_design.R line 35: encode_aligned(trl[fold != k], trl[fold == k])\n")
cat("  -> a fold-k row's share EXCLUDES every fold-k respondent.  Claim TRUE at row level.\n")
cat("  BUT apply_design_encoding() is called ONCE, before any CV loop, so a TRAINING\n")
cat("  row in fold j carries an encoding built from folds != j, a set that INCLUDES\n")
cat("  the scored fold k.  Two separate channels to check: (a) train-side, (b) the\n")
cat("  within-cell complement identity on the scored side.\n\n")

# ---------------------------------------------------------------------------
cat("=========================================================================\n")
cat("A2.  Cell support -- how thin is c_j?\n")
cat("=========================================================================\n")
cell <- trl[, .(N = .N, C = sum(chosen)), by = .(dkey, alt)]
perfold <- trl[, .(n_j = .N, c_j = sum(chosen)), by = .(dkey, alt, fold)]
cat(sprintf("  unique designs (dkey)                 : %d\n", uniqueN(trl$dkey)))
cat(sprintf("  unique (dkey,alt) cells               : %d\n", nrow(cell)))
cat(sprintf("  rows per cell            mean %.2f  median %d  (N)\n", mean(cell$N), median(cell$N)))
cat(sprintf("  folds a cell appears in  mean %.2f\n",
            mean(perfold[, .N, by = .(dkey, alt)]$N)))
cat(sprintf("  rows per (cell,fold)     mean %.2f  median %d  (n_j)\n",
            mean(perfold$n_j), median(perfold$n_j)))
cat(sprintf("  share of TRAINING ROWS whose own-fold cell block has n_j == 1 : %.1f%%\n",
            100 * merge(trl[, .(dkey, alt, fold)], perfold, by = c("dkey","alt","fold"))[, mean(n_j == 1)]))
cat("  n_j == 1  =>  c_j IS the row's own label.  The encoding is then\n")
cat("  (cell constant) - (own label) exactly: fold-level LOO == row-level LOO.\n\n")

# ---------------------------------------------------------------------------
cat("=========================================================================\n")
cat("A3.  The channel: marginal vs WITHIN-CELL correlation of the encoding\n")
cat("     with the row's own label.  Marginal +ve = genuine signal.\n")
cat("     Within-cell -ve = the leave-own-fold-out complement identity.\n")
cat("=========================================================================\n")
apply_design_encoding(trl, tel)          # production encoding, verbatim
trl <- merge(trl, perfold, by = c("dkey", "alt", "fold"), all.x = TRUE)
trl <- merge(trl, cell,    by = c("dkey", "alt"),         all.x = TRUE)
setorder(trl, No, alt)
stopifnot(!anyNA(trl$share_a1))

cid <- trl[, paste(dkey, alt)]
demean <- function(x, g) x - ave(x, g, FUN = mean)
for (v in c("share_a1", "share_a5", "share_a20", "design_n")) {
  x <- trl[[v]]; y <- as.numeric(trl$chosen)
  xw <- demean(x, cid); yw <- demean(y, cid)
  cat(sprintf("  %-10s  marginal cor %+0.4f    WITHIN-CELL cor %+0.4f\n",
              v, cor(x, y), if (sd(xw) > 0) cor(xw, yw) else NA_real_))
}
cat("\n  identity check: n_ch = C - c_j ?\n")
n_ch_rec <- trl$C - trl$c_j
n_ch_act <- trl[, share_a1 * (design_n + 1)] -
            trl[, {pr <- NULL; NA}]        # placeholder, computed properly below
# recover n_ch exactly from (share_a1, design_n) and the fold-specific prior
prior_tab <- rbindlist(lapply(1:5, function(k) {
  p <- trl[fold != k, .(pr = sum(chosen) / .N), by = alt]; p[, fold := k]; p
}))
tt <- merge(trl[, .(rid = .I, alt, fold, share_a1, design_n)], prior_tab,
            by = c("alt", "fold"), all.x = TRUE)
setorder(tt, rid)
n_ch_from_feat <- tt$share_a1 * (tt$design_n + 1) - tt$pr
cat(sprintf("  max |n_ch recovered from (share_a1, design_n, prior)  -  (C - c_j)| = %.2e\n",
            max(abs(n_ch_from_feat - n_ch_rec))))
cat("  => the feature tuple the tree sees is an EXACT, invertible encoding of\n")
cat("     (C - c_j, N - n_j).  Nothing is hidden from a model that can isolate a cell.\n\n")

# ---------------------------------------------------------------------------
cat("=========================================================================\n")
cat("A4.  Upper bound on the leak: what a PERFECT cell-identifier could score\n")
cat("=========================================================================\n")
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
mk <- function(v) { P <- matrix(pmax(v, 1e-9), ncol = 4, byrow = TRUE); P / rowSums(P) }
prior_alt <- trl[, .(pr = sum(chosen) / .N), by = alt][order(alt), pr]
cat(sprintf("  constant class prior                        : %.5f\n",
            logloss(ytr, matrix(rep(prior_alt, each = length(ytr)), ncol = 4))))
for (a in c(1, 5, 20)) {
  cat(sprintf("  HONEST  shrunk share alone (alpha=%2d)       : %.5f\n",
              a, logloss(ytr, mk(trl[[paste0("share_a", a)]]))))
}
for (a in c(1, 5, 20)) {
  pr <- tt$pr
  leakshare <- (trl$c_j + a * pr) / (trl$n_j + a)      # OWN-fold share = what a perfect
  cat(sprintf("  LEAK    own-fold share  (alpha=%2d)         : %.5f   <- recoverable as C-(C-c_j)\n",
              a, logloss(ytr, mk(leakshare))))
}
cat("\n  A model that identifies the cell AND remembers C reads c_j off the encoding.\n")
cat("  The gap between the two blocks above is the size of the prize it is chasing.\n\n")

# ---------------------------------------------------------------------------
cat("=========================================================================\n")
cat("A5.  Train/OOF vs TEST feature shift -- the encoding is NOT the same variable\n")
cat("=========================================================================\n")
cat(sprintf("  design_n   train/OOF mean %.3f   TEST mean %.3f   ratio %.3f\n",
            mean(trl$design_n), mean(tel$design_n), mean(tel$design_n) / mean(trl$design_n)))
for (a in c(1, 5, 20)) {
  v <- paste0("share_a", a)
  cat(sprintf("  %-10s train/OOF sd %.4f      TEST sd %.4f      ratio %.3f\n",
              v, sd(trl[[v]]), sd(tel[[v]]), sd(tel[[v]]) / sd(trl[[v]])))
}
cat("\n  Train/OOF rows: share is over 4/5 of respondents, with the OWN fold removed.\n")
cat("  TEST rows     : share is over 5/5 of respondents, with NOTHING removed --\n")
cat("  the test respondents are new people and contribute nothing to any cell.\n")
cat("  So any within-cell rule the tree learns off the complement identity CANNOT\n")
cat("  fire correctly on test rows: their feature is systematically the LARGEST\n")
cat("  value the cell ever takes.  Fraction of test rows whose share_a1 exceeds\n")
cat("  every training value of the same cell:\n")
mx <- trl[, .(mx = max(share_a1)), by = .(dkey, alt)]
te2 <- merge(tel[, .(dkey, alt, share_a1)], mx, by = c("dkey", "alt"), all.x = TRUE)
cat(sprintf("      %.1f%%  (of the %.1f%% of test rows whose cell was seen in training)\n",
            100 * mean(te2[!is.na(mx), share_a1 > mx - 1e-12]),
            100 * mean(!is.na(te2$mx))))
cat("\nDIAG_DONE\n")
