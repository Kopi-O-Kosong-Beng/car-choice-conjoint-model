# =============================================================================
# ITERATION 04 — Part-worth coding inside the regularized model.
#
# HYPOTHESIS: iteration 02 showed part-worth coding is worth 0.020 to the plain
# MNL (z = 11.8). glmnet should gain at least as much, and possibly more: with L1/L2
# shrinkage it can afford part-worths INTERACTED with demographics -- a heterogeneous
# part-worth model -- which the unregularized MNL cannot support without exploding.
# It also keeps the design-encoding features from iteration 01, and stays linear in
# income, which matters because the test set is twice as rich as the training set.
#
# Baseline to beat: glmnet (linear coding) 1.18118, mnl_pw 1.15686.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter04_glmnet_pw/run.R
# =============================================================================
suppressMessages({ library(data.table); library(glmnet); library(Matrix) })
source("model/99_utils.R")
source("model/encode_design.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")

long <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)

# ---- part-worth dummies, reference = lowest level on a real bundle -----------
pw_cols <- character(0)
for (a in ATTRS) {
  lv_real <- sort(unique(long[alt != 4][[a]]))
  for (l in setdiff(lv_real, lv_real[1])) {
    nm <- sprintf("%s_L%s", a, l)
    long[, (nm) := as.numeric(get(a) == l)]
    pw_cols <- c(pw_cols, nm)
  }
}
# task-centered part-worths: within-choice-set contrast, the quantity a logit uses
pwc_cols <- paste0(pw_cols, "_c")
tm <- long[, lapply(.SD, mean), by = No, .SDcols = pw_cols]
setnames(tm, pw_cols, paste0(pw_cols, "_tm"))
long <- tm[long, on = "No"]
for (p in pw_cols) set(long, j = paste0(p, "_c"), value = long[[p]] - long[[paste0(p, "_tm")]])
long[, paste0(pw_cols, "_tm") := NULL]
setorder(long, No, alt)
cat("part-worth columns:", length(pw_cols), "\n")

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)

# ---- demographics: log-income, because raw income has a $5M tail -------------
demo_num <- c("agea", "milesa", "nighta", "pparkind", "genderind",
              "educind", "Urbind", "segmentind", "regionind", "yearind")
for (d in list(trl, tel)) d[, log_inc := log(pmax(incomea, 1000))]
demo_num <- c(demo_num, "log_inc")
mu <- sapply(demo_num, function(z) mean(trl[[z]])); sdv <- sapply(demo_num, function(z) sd(trl[[z]]))
for (d in list(trl, tel)) for (z in demo_num) set(d, j = paste0(z, "_z"), value = (d[[z]] - mu[z]) / sdv[z])
demo_z <- paste0(demo_num, "_z")

build_X <- function(d) {
  main <- as.matrix(d[, c(pw_cols, pwc_cols, "richness", "lvlsum", "price_rank",
                          "price_min_rival", "asc2", "asc3", "asc4", "Task",
                          ENC_COLS, demo_z), with = FALSE])
  # heterogeneous part-worths: each task-centered part-worth x each demographic
  ip <- CJ(a = pwc_cols, dz = demo_z, sorted = FALSE)
  X_int <- do.call(cbind, lapply(seq_len(nrow(ip)), function(i) d[[ip$a[i]]] * d[[ip$dz[i]]]))
  colnames(X_int) <- paste0(ip$a, ":", ip$dz)
  # opt-out propensity by demographics
  X_a4 <- do.call(cbind, lapply(demo_z, function(z) d$asc4 * d[[z]]))
  colnames(X_a4) <- paste0("asc4:", demo_z)
  cbind(main, X_int, X_a4)
}
Xtr <- build_X(trl); Xte <- build_X(tel)
cat("design matrix:", nrow(Xtr), "x", ncol(Xtr), "\n")
ytr_bin <- as.numeric(trl$chosen)
fmap <- trl$fold

# lambda picked per fold on the TRAINING part only -- fixes the leak in 09_glmnet.R,
# where one lambda chosen on folds 2-5 was then reused to score folds 2-5.
oof_p <- numeric(nrow(Xtr)); lams <- numeric(5)
for (k in 1:5) {
  set.seed(40 + k)
  cv <- cv.glmnet(Xtr[fmap != k, ], ytr_bin[fmap != k], family = "binomial",
                  alpha = 0.5, nfolds = 4, type.measure = "deviance")
  lams[k] <- cv$lambda.min
  oof_p[fmap == k] <- as.vector(predict(cv, Xtr[fmap == k, ], s = "lambda.min", type = "response"))
  cat("fold", k, "lambda", signif(cv$lambda.min, 4), "\n")
}
oof <- dcast(data.table(No = trl$No, alt = trl$alt, p = norm_by_group(oof_p, trl$No)),
             No ~ alt, value.var = "p")
setnames(oof, c("No", "p1", "p2", "p3", "p4")); setorder(oof, No)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
cat(">>> GLMNET part-worth OOF logloss:",
    round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5),
    "  (linear-coded glmnet: 1.18118)\n")
saveRDS(oof, "model/artifacts/oof_glmnet_pw.rds")

set.seed(99)
cvf <- cv.glmnet(Xtr, ytr_bin, family = "binomial", alpha = 0.5, nfolds = 4,
                 type.measure = "deviance")
p_te <- as.vector(predict(cvf, Xte, s = "lambda.min", type = "response"))
tp <- dcast(data.table(No = tel$No, alt = tel$alt, p = norm_by_group(p_te, tel$No)),
            No ~ alt, value.var = "p")
setnames(tp, c("No", "p1", "p2", "p3", "p4")); setorder(tp, No)
saveRDS(tp, "model/artifacts/test_glmnet_pw.rds")
cf <- coef(cvf, s = "lambda.min")[, 1]
cat("nonzero coefficients:", sum(cf != 0), "of", length(cf), "\n")
cat("\nstrongest surviving interactions (taste heterogeneity):\n")
ints <- cf[grepl(":", names(cf)) & cf != 0]
print(round(c(head(sort(ints, decreasing = TRUE), 6), head(sort(ints), 6)), 4))
cat("OK\n")
