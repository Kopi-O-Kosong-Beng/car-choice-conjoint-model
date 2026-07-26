# =============================================================================
# Artifact + leak verification for iteration 17 (hbmnl).
#   Rscript experiments/iter17_hb/verify.R hbmnl
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
nm <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(nm)) nm <- "hbmnl"

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
o <- readRDS(sprintf("model/artifacts/oof_%s.rds",  nm))
t <- readRDS(sprintf("model/artifacts/test_%s.rds", nm))
ymap <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
refte <- sort(unique(long[is_test == TRUE, No]))

chk <- function(lbl, ok) cat(sprintf("  [%s] %s\n", if (ok) "PASS" else "FAIL", lbl))
cat("--- artifact shape ---\n")
chk("oof is data.table",        is.data.table(o))
chk("oof has 21565 rows",       nrow(o) == 21565)
chk("oof columns No,p1..p4",    identical(names(o), c("No","p1","p2","p3","p4")))
chk("oof ordered by No, matches long", identical(o$No, ymap$No))
Po <- as.matrix(o[, .(p1,p2,p3,p4)])
chk("oof rows sum to 1",        max(abs(rowSums(Po) - 1)) < 1e-9)
chk("oof strictly positive",    all(Po > 0) && !anyNA(Po))
chk("test has 4997 rows",       nrow(t) == 4997)
chk("test ordered by No, matches long", identical(t$No, refte))
Pt <- as.matrix(t[, .(p1,p2,p3,p4)])
chk("test rows sum to 1",       max(abs(rowSums(Pt) - 1)) < 1e-9)
chk("test strictly positive",   all(Pt > 0) && !anyNA(Pt))

cat("\n--- honest OOF ---\n")
cat(sprintf("  %s OOF logloss: %.5f\n", nm, logloss(ymap$y, Po)))

cat("\n--- leak detector 1: per-fold coverage ---\n")
# Every OOF row must have been produced by the fold that OWNS it. The per-fold
# prediction files record which task indices they scored; check the partition is
# exactly the fold partition and that no row was written twice.
tasks <- unique(long[, .(No, is_test)]); setorder(tasks, No)
tr_t  <- which(tasks$is_test == FALSE)
fmap  <- folds[order(No), fold]
own <- integer(0); dup <- FALSE
for (k in 1:5) {
  f <- sprintf("experiments/iter17_hb/pred_main_%d.rds", k)
  if (!file.exists(f)) { cat("  (skipped: no per-fold files)\n"); own <- NULL; break }
  p <- readRDS(f)
  idx <- match(p$task_idx, tr_t)
  if (length(intersect(own, idx))) dup <- TRUE
  own <- c(own, idx)
  chk(sprintf("fold %d scored exactly its own held-out tasks", k),
      identical(sort(idx), sort(which(fmap == k))))
}
if (!is.null(own)) {
  chk("no task predicted twice", !dup)
  chk("all 21565 tasks covered once", length(own) == 21565 && !anyDuplicated(own))
}

cat("\n--- leak detector 2: prediction is a pure function of (design, demographics) ---\n")
# This is the structural proof, not an argument. A population-averaged HB
# prediction can depend on nothing but the task design and the respondent's
# demographics. Strip the demographic channel (the stored P_nodelta, produced by
# predict_pop with z := 0) and the prediction must then depend on the DESIGN ALONE.
# So any two tasks in the SAME fold with byte-identical designs must receive
# byte-identical no-demographics predictions -- regardless of who faced them or
# what either respondent chose. If a held-out respondent's own choices had reached
# the prediction, two different people seeing the same task would differ here.
ATTRS20 <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP","KA",
             "SC","TS","NV","MA","LB","AF","HU","Price")
dk <- long[is_test == FALSE][order(No, alt)][
  , .(dk = paste(unlist(.SD), collapse = "|")), by = No, .SDcols = ATTRS20]
ok2 <- TRUE; ngrp <- 0; worst <- 0
for (k in 1:5) {
  f <- sprintf("experiments/iter17_hb/pred_main_%d.rds", k)
  if (!file.exists(f)) { cat("  (skipped: no per-fold files)\n"); ok2 <- NA; break }
  p <- readRDS(f)
  d <- data.table(No = tasks$No[p$task_idx], a = p$P_nodelta[, 1], b = p$P_nodelta[, 4])
  d <- merge(d, dk, by = "No")
  s <- d[, .(n = .N, sp = (max(a) - min(a)) + (max(b) - min(b))), by = dk][n > 1]
  ngrp <- ngrp + nrow(s); worst <- max(worst, if (nrow(s)) max(s$sp) else 0)
}
if (!is.na(ok2)) {
  cat(sprintf("  %d repeated-design groups across folds; worst within-group spread = %.2e\n",
              ngrp, worst))
  chk("identical design => identical demographics-free prediction", worst < 1e-12)
  # and the demographic channel must actually MOVE predictions, else Z is inert
  p1 <- readRDS("experiments/iter17_hb/pred_main_1.rds")
  cat(sprintf("  demographic channel moves p(none) by mean |delta| = %.4f\n",
              mean(abs(p1$P[, 4] - p1$P_nodelta[, 4]))))
}

cat("\n--- leak detector 3: none-rate calibration ---\n")
cat(sprintf("  observed train none-rate %.4f | OOF predicted %.4f | test predicted %.4f\n",
            mean(ymap$y == 4), mean(Po[, 4]), mean(Pt[, 4])))
cat("  (a model that had seen held-out labels would track the observed rate too well\n",
    "   AND would score far below the 1.13-1.16 band the honest models occupy)\n")
cat("OK\n")
