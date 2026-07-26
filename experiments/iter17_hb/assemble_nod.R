# Assemble the no-demographic-channel HB variant as its own artifact pair.
#
# WHY THIS IS A SEPARATE MODEL, not just a diagnostic line:
# the `combine` step reported OOF 1.23703 with the demographic channel and
# 1.16405 without it. A 0.073 logloss penalty from ADDING demographics is the
# opposite sign to latent class, where the membership channel carries 93% of the
# gain. If the ablated version is a usable model it belongs in the blend probe on
# its own merits, and the contrast is a report finding either way.
#
# Both artifacts come from draws of (pvec, mu_c, Sigma_c) fit on folds != k with
# z := 0 at prediction time. betadraw is never touched. Same disjointness guard.
suppressPackageStartupMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
# one row per choice task, ordered by No -- the same ordering run.R's build()
# produces, which is what pred_main_*.rds$task_idx indexes into.
tasks <- unique(long[, .(No, Case, is_test, y)], by = "No")
setorder(tasks, No)

tr_t <- which(!tasks$is_test)
te_t <- which(tasks$is_test)
stopifnot(length(tr_t) == 21565L, length(te_t) == 4997L)

oof <- matrix(NA_real_, length(tr_t), 4)
for (k in 1:5) {
  p <- readRDS(sprintf("experiments/iter17_hb/pred_main_%d.rds", k))
  idx <- match(p$task_idx, tr_t)
  stopifnot(!anyNA(idx))
  oof[idx, ] <- p$P_nodelta
}
stopifnot(!anyNA(oof))

y  <- tasks$y[tr_t]
ll <- logloss(y, oof)
cat(sprintf(">>> hbmnl_nod honest OOF logloss: %.5f\n", ll))
cat("    reference: hbmnl 1.23703 | lcmnl3 1.14396 | mnl_pw 1.15686 | xgb_lw2 1.14152\n")

saveRDS(data.table(No = tasks$No[tr_t], p1 = oof[, 1], p2 = oof[, 2],
                   p3 = oof[, 3], p4 = oof[, 4]),
        "model/artifacts/oof_hbmnl_nod.rds")

pf <- readRDS("experiments/iter17_hb/pred_main_full.rds")
stopifnot(identical(pf$task_idx, te_t))
te <- clip_norm(pf$P_nodelta)
saveRDS(data.table(No = tasks$No[te_t], p1 = te[, 1], p2 = te[, 2],
                   p3 = te[, 3], p4 = te[, 4]),
        "model/artifacts/test_hbmnl_nod.rds")
cat(sprintf("    wrote test_hbmnl_nod.rds (predicted none-rate %.3f vs train %.3f)\n",
            mean(te[, 4]), mean(y == 4)))
