# =============================================================================
# A THIRD OPINION ON THE TEST NONE-RATE  (read-only diagnostic, no artifacts)
#
# THE OPEN QUESTION. The two production members disagree about the test set in a
# way they never disagree on OOF:
#
#     lcmnl3_both  predicts test none-rate 0.224
#     tree family  predicts                0.273
#     shipped blend                        0.248
#     OOF: every member reproduces the truth (0.302) to within 0.005
#
# So the disagreement lives ENTIRELY out of sample, where nothing can be fitted.
# Iteration 19 concluded correctly that a corrective alt-4 weight cannot be
# validated. It did not ask whether an INDEPENDENT estimate could adjudicate.
#
# WHY IT MATTERS. Mis-specifying the marginal costs, in KL terms, about 0.0017
# either way for a 0.025 error -- comparable to everything round 3 produced.
#
# THE ESTIMATOR. Both members reach their test none-rate through their own model
# machinery (segment membership softmax vs boosted trees). A third estimate that
# shares neither: fit P(choose none | demographics) directly on training
# respondents, then average the fitted probability over the TEST respondents'
# demographics. This is just covariate-shift reweighting of the marginal, and it
# is deliberately naive -- no choice-set information, no part-worths, no segments.
# Two variants, because the functional form is the whole question:
#   (a) logistic on the demographic design used by lcmnl3's membership model
#   (b) a fully nonparametric cell estimate on income x age tertiles, which makes
#       no linearity assumption at all
# Plus (c) the classical importance-weighted training mean, which is the
# assumption-free Horvitz-Thompson version of the same quantity.
#
# HOW TO READ IT. If all three land near 0.27, the trees extrapolate correctly and
# the blend is shipping too little none-mass. If near 0.22, the latent-class
# membership channel is right. If they straddle 0.248, the blend's implicit
# split-the-difference is already the defensible choice and the question closes.
#
# THIS IS A DIAGNOSTIC, NOT A MODEL. It emits nothing, changes nothing, and is
# allowed under the freeze. Acting on it would need the freeze re-opened.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long <- readRDS("model/artifacts/long.rds")

# one row per respondent, with their observed none-rate where known
tk <- unique(long[, .(No, Case, is_test, y)])
resp <- tk[, .(n = .N, none = mean(y == 4, na.rm = TRUE),
               is_test = is_test[1]), by = Case]

DEMO_CAT <- c("segmentind", "pparkind", "genderind", "educind", "regionind", "Urbind")
DEMO_NUM <- c("agea", "incomea", "milesa", "nighta", "yearind", "milesind", "nightind")
dem <- unique(long[, c("Case", DEMO_CAT, DEMO_NUM), with = FALSE])
resp <- merge(resp, dem, by = "Case")
tr <- resp[is_test == FALSE]; te <- resp[is_test == TRUE]
cat(sprintf("respondents: %d train / %d test\n", nrow(tr), nrow(te)))
cat(sprintf("observed TRAIN none-rate: %.4f\n\n", weighted.mean(tr$none, tr$n)))

# ---- (a) logistic on the full demographic design ----------------------------
# task-level so the fit is weighted the same way the metric is
task_tr <- tk[is_test == FALSE]
task_tr <- merge(task_tr, dem, by = "Case")
task_te <- tk[is_test == TRUE]
task_te <- merge(task_te, dem, by = "Case")
for (v in DEMO_CAT) {
  task_tr[[v]] <- factor(task_tr[[v]]); task_te[[v]] <- factor(task_te[[v]],
                                                    levels = levels(task_tr[[v]]))
}
f <- as.formula(paste("I(y == 4) ~", paste(c(DEMO_CAT, DEMO_NUM), collapse = " + ")))
m <- glm(f, data = task_tr, family = binomial)
p_te <- predict(m, newdata = task_te, type = "response")
p_tr <- predict(m, type = "response")
cat(sprintf("(a) logistic on demographics\n"))
cat(sprintf("    fitted TRAIN mean %.4f  (sanity: should equal observed)\n", mean(p_tr)))
cat(sprintf("    predicted TEST mean %.4f\n", mean(p_te)))

# ---- (b) nonparametric cells: income x age tertiles --------------------------
qi <- quantile(task_tr$incomea, c(1/3, 2/3), na.rm = TRUE)
qa <- quantile(task_tr$agea,    c(1/3, 2/3), na.rm = TRUE)
cellof <- function(d) paste0(findInterval(d$incomea, qi), "_", findInterval(d$agea, qa))
task_tr[, cell := cellof(task_tr)]; task_te[, cell := cellof(task_te)]
cellrate <- task_tr[, .(r = mean(y == 4), n = .N), by = cell]
te_cells <- task_te[, .(n_te = .N), by = cell]
cmp <- merge(cellrate, te_cells, by = "cell", all = TRUE)
cat(sprintf("\n(b) nonparametric income x age cells (no linearity assumed)\n"))
print(cmp[order(cell)])
cat(sprintf("    predicted TEST mean %.4f\n",
            weighted.mean(cmp$r, cmp$n_te, na.rm = TRUE)))

# ---- (c) importance-weighted training mean ----------------------------------
# Horvitz-Thompson: reweight training respondents to the test demographic profile
wide <- readRDS("model/artifacts/wide.rds")
rr <- unique(wide[, .(Case, is_test, incomeind)])
ntr <- rr[is_test == FALSE, .N]; nte <- rr[is_test == TRUE, .N]
wt <- merge(rr[is_test == FALSE, .(ptr = .N / ntr), by = incomeind],
            rr[is_test == TRUE,  .(pte = .N / nte), by = incomeind],
            by = "incomeind", all.x = TRUE)
wt[is.na(pte), pte := 0]
wt[, w := pmin(pmax(pte / ptr, 0.2), 5)]
tw <- merge(task_tr, unique(wide[, .(Case, incomeind)]), by = "Case")
tw <- merge(tw, wt[, .(incomeind, w)], by = "incomeind")
cat(sprintf("\n(c) importance-weighted training mean (income reweighting)\n"))
cat(sprintf("    predicted TEST mean %.4f  (ESS %.0f of %d tasks)\n",
            weighted.mean(tw$y == 4, tw$w), sum(tw$w)^2 / sum(tw$w^2), nrow(tw)))

# ---- what the models say, and what an error costs ---------------------------
say <- function(nm) {
  f <- sprintf("model/artifacts/test_%s.rds", nm)
  if (!file.exists(f)) return(NA_real_)
  x <- readRDS(f)
  if (is.matrix(x)) mean(x[, 4]) else mean(x$p4)   # test_blend.rds is a matrix
}
cat("\n--- what each model ships on the test set ---\n")
for (nm in c("lcmnl3_both", "xgb_lw2bag", "xgb_lw2", "mnl_pw", "blend"))
  cat(sprintf("    %-12s %.4f\n", nm, say(nm)))

kl <- function(p, q) p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
cat("\n--- cost of shipping q when the truth is p (logloss, KL on the margin) ---\n")
for (p in c(0.21, 0.224, 0.248, 0.273, 0.296, 0.302))
  cat(sprintf("    truth %.3f -> shipping 0.248 costs %+.5f\n", p, kl(p, 0.248)))

# ---- the assumption-free check: does income predict declining AT ALL? --------
# Every low-none-rate extrapolation rests on "the wealthier test respondents buy
# more". That is a claim about a relationship measurable directly in training.
cat("\n--- (d) marginal none-rate by income tertile, training data ---\n")
task_tr[, itert := findInterval(incomea, qi)]
print(task_tr[, .(none_rate = mean(y == 4), tasks = .N), by = itert][order(itert)])
cat("    If this is flat, no amount of income shift can move the test none-rate,\n")
cat("    and a model predicting otherwise is extrapolating on something else.\n")

cat("\n--- (e) is the test demographic profile INSIDE training support? ---\n")
for (v in c("incomea", "agea", "milesa", "nighta")) {
  qtr <- quantile(task_tr[[v]], c(0, .01, .5, .99, 1), na.rm = TRUE)
  qte <- quantile(task_te[[v]], c(0, .01, .5, .99, 1), na.rm = TRUE)
  frac_out <- mean(task_te[[v]] < qtr[1] | task_te[[v]] > qtr[5], na.rm = TRUE)
  cat(sprintf("    %-9s train med %8.1f  test med %8.1f  test rows outside train range %.2f%%\n",
              v, qtr[3], qte[3], 100 * frac_out))
}
cat("\nA DIAGNOSTIC ONLY. It emits nothing and changes nothing. It raises a question\n")
cat("rather than settling one; acting on it requires re-opening the freeze AND a\n")
cat("validation design, because there is no held-out data in the disputed regime.\n")
