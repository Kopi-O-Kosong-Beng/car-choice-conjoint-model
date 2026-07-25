# Regularized long-format logit: same "score each alternative, softmax within task"
# shape as xgb_long, but linear + L1/L2 penalized. Carries a much wider interaction
# set than the plain MNL can support, and unlike trees it extrapolates linearly in
# income -- which matters given the train->test income shift.
suppressMessages({ library(data.table); library(glmnet); library(Matrix) })
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

demo_num <- c("agea", "incomea", "milesa", "nighta", "pparkind", "genderind",
              "educind", "Urbind", "segmentind", "regionind", "yearind")
# scale demographics so penalization treats them comparably
dsc <- long[, lapply(.SD, function(z) as.numeric(scale(z))), .SDcols = demo_num]
setnames(dsc, demo_num, paste0(demo_num, "_z"))
base <- cbind(long[, c("No", "alt", "Case", "y", "is_test", "chosen", "Task"), with = FALSE],
              long[, c(ATTRS, paste0(ATTRS, "_c"), "richness", "lvlsum", "price_rank",
                       "price_min_rival", "asc2", "asc3", "asc4"), with = FALSE],
              dsc)

attr_c <- paste0(ATTRS, "_c")
demo_z <- paste0(demo_num, "_z")
main <- c(ATTRS, attr_c, "richness", "lvlsum", "price_rank", "price_min_rival",
          "asc2", "asc3", "asc4", "Task")

X_main <- as.matrix(base[, ..main])
# interactions: every attribute (task-centered) x every demographic, plus asc4 x demographics
inter_pairs <- rbind(CJ(a = attr_c, d = demo_z, sorted = FALSE),
                     CJ(a = c("asc4", "richness"), d = demo_z, sorted = FALSE))
X_int <- do.call(cbind, lapply(seq_len(nrow(inter_pairs)), function(i)
  base[[inter_pairs$a[i]]] * base[[inter_pairs$d[i]]]))
colnames(X_int) <- paste0(inter_pairs$a, ":", inter_pairs$d)
X <- cbind(X_main, as.matrix(base[, ..demo_z]), X_int)
cat("design matrix:", nrow(X), "x", ncol(X), "\n")

is_tr <- base$is_test == FALSE
fmap <- merge(base[is_tr, .(No)], folds[, .(No, fold)], by = "No")$fold
ytr_bin <- as.numeric(base$chosen[is_tr])

# lambda chosen once on fold-1-excluded data, then reused (keeps folds honest and fast)
set.seed(42)
cvfit <- cv.glmnet(X[is_tr, ][fmap != 1, ], ytr_bin[fmap != 1], family = "binomial",
                   alpha = 0.5, nfolds = 4, type.measure = "deviance")
lam <- cvfit$lambda.min
cat("chosen lambda:", signif(lam, 4), "\n")

oof_p <- numeric(sum(is_tr))
Xtr <- X[is_tr, ]
for (k in 1:5) {
  fit <- glmnet(Xtr[fmap != k, ], ytr_bin[fmap != k], family = "binomial",
                alpha = 0.5, lambda = lam)
  oof_p[fmap == k] <- as.vector(predict(fit, Xtr[fmap == k, ], type = "response"))
  cat("fold", k, "done\n")
}
tr_rows <- base[is_tr]
oof_l <- data.table(No = tr_rows$No, alt = tr_rows$alt,
                    p = norm_by_group(oof_p, tr_rows$No))
oof <- dcast(oof_l, No ~ alt, value.var = "p")
setnames(oof, c("No", "p1", "p2", "p3", "p4")); setorder(oof, No)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
cat(">>> GLMNET OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5), "\n")
saveRDS(oof, "model/artifacts/oof_glmnet.rds")

fit_full <- glmnet(Xtr, ytr_bin, family = "binomial", alpha = 0.5, lambda = lam)
te_rows <- base[!is_tr]
p_te <- as.vector(predict(fit_full, X[!is_tr, ], type = "response"))
tp_l <- data.table(No = te_rows$No, alt = te_rows$alt, p = norm_by_group(p_te, te_rows$No))
tp <- dcast(tp_l, No ~ alt, value.var = "p")
setnames(tp, c("No", "p1", "p2", "p3", "p4")); setorder(tp, No)
saveRDS(tp, "model/artifacts/test_glmnet.rds")
cf <- coef(fit_full)[, 1]
cat("nonzero coefficients:", sum(cf != 0), "of", length(cf), "\n")
print(round(head(sort(cf[cf != 0 & names(cf) != "(Intercept)"], decreasing = TRUE), 8), 4))
cat("OK\n")
