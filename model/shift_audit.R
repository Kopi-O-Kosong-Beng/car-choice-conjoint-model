# =============================================================================
# Shift audit: which models / which improvements survive the train->test
# population difference?
#
# Motivation: our local gains transfer to the public leaderboard at only ~58%
# (local -0.038 produced public -0.022). If a particular model or feature owes its
# advantage to the TRAINING population specifically, that advantage should shrink
# when we reweight respondents to look like the TEST population. Anything that
# shrinks badly here is a candidate for the missing 42%.
#
# Method: importance weights on respondents so the training income distribution
# matches the test one, then rescore every member's OOF under those weights.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/shift_audit.R
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
wide <- readRDS("model/artifacts/wide.rds")
ymap <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)

resp <- unique(wide[, .(Case, is_test, incomeind)])
ntr <- resp[is_test == FALSE, .N]; nte <- resp[is_test == TRUE, .N]
wt <- merge(resp[is_test == FALSE, .(ptr = .N / ntr), by = incomeind],
            resp[is_test == TRUE,  .(pte = .N / nte), by = incomeind],
            by = "incomeind", all.x = TRUE)
wt[is.na(pte), pte := 0]
wt[, w := pmin(pmax(pte / ptr, 0.2), 5)]
rw <- merge(unique(wide[is_test == FALSE, .(No, incomeind)]), wt[, .(incomeind, w)],
            by = "incomeind")[order(No), w]
cat("importance weights: min", round(min(rw), 2), "max", round(max(rw), 2),
    "effective sample size", round(sum(rw)^2 / sum(rw^2)), "of", length(rw), "rows\n\n")

memb <- gsub("^oof_|\\.rds$", "", list.files("model/artifacts", pattern = "^oof_.*\\.rds$"))
res <- rbindlist(lapply(memb, function(m) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No)
  P <- as.matrix(d[, .(p1, p2, p3, p4)]); P <- pmax(P / rowSums(P), 1e-15)
  li <- -log(P[cbind(seq_len(nrow(P)), ymap$y)])
  data.table(model = m, plain = mean(li), reweighted = sum(rw * li) / sum(rw))
}))
res[, penalty := reweighted - plain]
setorder(res, plain)
print(res[, .(model, plain = round(plain, 5), reweighted = round(reweighted, 5),
              penalty = round(penalty, 5))])

cat("\n--- does an improvement SURVIVE reweighting? ---\n")
pairs <- list(c("xgb_long", "xgb_de"), c("xgb_de", "xgb_lw"), c("mnl", "mnl_pw"),
              c("mixl", "mixl2"), c("xgb_long", "xgb_lw"))
for (p in pairs) {
  if (!all(p %in% res$model)) next
  a <- res[model == p[1]]; b <- res[model == p[2]]
  cat(sprintf("%-10s -> %-10s  plain %+.5f   reweighted %+.5f   retained %3.0f%%\n",
              p[1], p[2], a$plain - b$plain, a$reweighted - b$reweighted,
              100 * (a$reweighted - b$reweighted) / (a$plain - b$plain)))
}
cat("\nA gain that retains ~100% is structural; one that retains much less is\n",
    "specific to the training population and will disappoint on the leaderboard.\n")
