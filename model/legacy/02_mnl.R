suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

xvars <- c(ATTRS, "Price_x_age", "Price_x_ppark", "Price_x_inc",
           grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 1"))

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]

fit_predict <- function(dtr, dva) {
  dtr_x <- dfidx(as.data.frame(dtr), idx = c("No", "alt"), choice = "chosen")
  fit <- mlogit(fml, data = dtr_x)
  dva2 <- copy(dva); dva2[, chosen := FALSE]; dva2[alt == 1, chosen := TRUE]  # dummy; predict ignores
  dva_x <- dfidx(as.data.frame(dva2), idx = c("No", "alt"), choice = "chosen")
  P <- predict(fit, newdata = dva_x)
  list(P = P, nos = sort(unique(dva$No)), fit = fit)
}

oof <- data.table(No = sort(unique(trl$No)), p1 = NA_real_, p2 = NA_real_, p3 = NA_real_, p4 = NA_real_)
for (k in 1:5) {
  r <- fit_predict(trl[fold != k], trl[fold == k])
  oof[match(r$nos, No), `:=`(p1 = r$P[, 1], p2 = r$P[, 2], p3 = r$P[, 3], p4 = r$P[, 4])]
  cat("fold", k, "done\n")
}
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
cat(">>> MNL OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5), "\n")
saveRDS(oof, "model/artifacts/oof_mnl.rds")

rf <- fit_predict(trl, tel)
test_p <- data.table(No = rf$nos, p1 = rf$P[, 1], p2 = rf$P[, 2], p3 = rf$P[, 3], p4 = rf$P[, 4])
saveRDS(test_p, "model/artifacts/test_mnl.rds")
print(summary(rf$fit)$CoefTable[c("(Intercept):4", "Price", "Price_x_age", "Price_x_ppark"), ])
cat("OK\n")
