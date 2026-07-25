# =============================================================================
# ITERATION 0 — Diagnostics. No model fitting; this decides where effort goes.
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter00_diagnostics/diag.R
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
wide <- readRDS("model/artifacts/wide.rds")
long <- readRDS("model/artifacts/long.rds")

cat("\n===== 1. ATTRIBUTE LEVEL STRUCTURE =====\n")
cat("(if levels are few and ordinal, linear coding may be losing part-worth detail)\n")
for (a in ATTRS) {
  v <- unlist(wide[, paste0(a, 1:3), with = FALSE])
  u <- sort(unique(v))
  cat(sprintf("%-6s n_levels=%-4d levels: %s\n", a, length(u),
              paste(utils::head(u, 12), collapse = ",")))
}

cat("\n===== 2. CHOICE-SET DESIGN REUSE (the big one) =====\n")
acols3 <- unlist(lapply(1:3, function(j) paste0(ATTRS, j)))
key <- do.call(paste, c(wide[, ..acols3], sep = "_"))
wide[, dkey := key]
alt_str <- lapply(1:3, function(j) do.call(paste, c(wide[, paste0(ATTRS, j), with = FALSE], sep = "_")))
wide[, ikey := apply(do.call(cbind, alt_str), 1, function(x) paste(sort(x), collapse = "|"))]

tr <- wide[is_test == FALSE]; te <- wide[is_test == TRUE]
cat(sprintf("exact-order design keys : train unique %d / %d rows | test unique %d / %d rows\n",
            uniqueN(tr$dkey), nrow(tr), uniqueN(te$dkey), nrow(te)))
cat(sprintf("order-invariant keys    : train unique %d | test unique %d\n",
            uniqueN(tr$ikey), uniqueN(te$ikey)))
ov  <- sum(te$dkey %in% tr$dkey); ovi <- sum(te$ikey %in% tr$ikey)
cat(sprintf("TEST rows whose design was ALSO seen in train: exact %d (%.1f%%) | order-invariant %d (%.1f%%)\n",
            ov, 100 * ov / nrow(te), ovi, 100 * ovi / nrow(te)))
sup <- tr[, .N, by = dkey][order(-N)]
cat("train support per repeated design (top): ", paste(utils::head(sup$N, 10), collapse = ","), "\n")
cat(sprintf("designs seen >=5 times in train: %d (covering %d train rows)\n",
            sum(sup$N >= 5), sum(sup$N[sup$N >= 5])))
# single-alternative reuse: does the same BUNDLE recur across tasks?
bstr <- unlist(alt_str)
cat(sprintf("unique individual bundles across all rows/positions: %d (of %d bundle slots)\n",
            uniqueN(bstr), length(bstr)))

cat("\n===== 3. TASK-BLOCK STRUCTURE =====\n")
cat("designs per Task number (are respondents shown a shared block?)\n")
print(wide[, .(rows = .N, unique_designs = uniqueN(dkey)), keyby = Task])

cat("\n===== 4. INCOME SHIFT — full distribution, not just the mean =====\n")
resp <- unique(wide[, .(Case, is_test, incomeind, incomea, educind, milesa, agea)])
print(dcast(resp[, .N, by = .(is_test, incomeind)], incomeind ~ is_test, value.var = "N", fill = 0))
cat("incomea quantiles train:", round(quantile(resp[is_test == FALSE, incomea], c(0, .25, .5, .75, .9, 1))), "\n")
cat("incomea quantiles test :", round(quantile(resp[is_test == TRUE,  incomea], c(0, .25, .5, .75, .9, 1))), "\n")
cat("respondents per Case in test:", te[, .N, by = Case][, paste(range(N), collapse = "-")], "\n")

cat("\n===== 5. WHERE ARE WE LOSING? (loss decomposition, xgb_long OOF) =====\n")
oof <- readRDS("model/artifacts/oof_xgb_long.rds"); setorder(oof, No)
ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
P <- as.matrix(oof[, .(p1, p2, p3, p4)])
y <- ymap$y
li <- -log(pmax(P[cbind(seq_along(y), y)], 1e-15))
cat("overall:", round(mean(li), 5), "\n")
print(data.table(true_class = 1:4, n = as.vector(table(y)),
                 mean_loss = round(tapply(li, y, mean), 4),
                 mean_pred_prob = round(tapply(P[cbind(seq_along(y), y)], y, mean), 4)))
cat("\nloss by whether the NONE option was chosen:\n")
print(data.table(none_chosen = c(FALSE, TRUE), mean_loss = round(tapply(li, y == 4, mean), 4)))

cat("\n===== 6. CALIBRATION (are probabilities honest?) =====\n")
flat <- data.table(p = as.vector(P), hit = as.vector(sapply(1:4, function(j) y == j)))
flat[, bin := cut(p, breaks = c(-.01, .05, .1, .15, .2, .25, .3, .4, .5, .7, 1.01))]
print(flat[, .(n = .N, mean_pred = round(mean(p), 4), observed = round(mean(hit), 4)), keyby = bin])

cat("\n===== 7. HOW MUCH DOES MORE DATA HELP? (respondent learning curve) =====\n")
cat("mean tasks per respondent:", long[is_test == FALSE, uniqueN(No)] / long[is_test == FALSE, uniqueN(Case)], "\n")
cat("train respondents:", long[is_test == FALSE, uniqueN(Case)], "| test respondents:", long[is_test == TRUE, uniqueN(Case)], "\n")
cat("=> if per-respondent loss varies a lot, respondent-level heterogeneity is the headroom\n")
rl <- data.table(No = ymap$No, li = li)
rl <- merge(rl, unique(long[is_test == FALSE, .(No, Case)]), by = "No")
pr <- rl[, .(mean_loss = mean(li)), by = Case]
cat("per-respondent mean loss: quantiles", round(quantile(pr$mean_loss, c(0, .1, .5, .9, 1)), 3), "\n")
cat("\nDIAGNOSTICS COMPLETE\n")
