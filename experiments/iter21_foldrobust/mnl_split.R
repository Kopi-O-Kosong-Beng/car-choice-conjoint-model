# =============================================================================
# ITERATION 21 — part-worth conditional logit refit under an ALTERNATIVE split.
# Verbatim copy of model/02_mnl_partworth.R; the ONLY differences are the fold file
# (command-line argument) and the _<split> artifact suffix.
#
#   Rscript experiments/iter21_foldrobust/mnl_split.R <b|c>
# =============================================================================
suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: mnl_split.R <b|c>")
SPLIT <- args[1]; stopifnot(SPLIT %in% c("b", "c"))
NAME <- sprintf("mnl_pw_%s", SPLIT)

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS(sprintf("model/artifacts/folds_%s.rds", SPLIT))
stopifnot(nrow(folds) == 21565, all(folds[, uniqueN(fold), by = Case]$V1 == 1))

pw_cols <- character(0)
for (a in ATTRS) {
  lv_real <- sort(unique(long[alt != 4][[a]]))
  ref <- lv_real[1]
  for (l in setdiff(lv_real, ref)) {
    nm <- sprintf("%s_L%s", a, l)
    long[, (nm) := as.numeric(get(a) == l)]
    pw_cols <- c(pw_cols, nm)
  }
}
cand <- c("asc2", "asc3", "asc4", pw_cols)
X <- as.matrix(long[, ..cand])
Xc <- X - as.matrix(long[, lapply(.SD, mean), by = No, .SDcols = cand][
  match(long$No, unique(long$No)), ..cand])
qrx <- qr(Xc, tol = 1e-7)
keep <- cand[sort(qrx$pivot[seq_len(qrx$rank)])]
dropped <- setdiff(cand, keep)
if (length(dropped)) cat("dropped as unidentified:", paste(dropped, collapse = ", "), "\n")
pw_cols <- setdiff(keep, c("asc2", "asc3", "asc4"))

xvars <- c(pw_cols, "Price_x_age", "Price_x_ppark", "Price_x_inc",
           grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 1"))

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]

fit_predict <- function(dtr, dva) {
  dtr_x <- dfidx(as.data.frame(dtr), idx = c("No", "alt"), choice = "chosen")
  fit <- mlogit(fml, data = dtr_x)
  dva2 <- copy(dva); dva2[, chosen := FALSE]; dva2[alt == 1, chosen := TRUE]
  dva_x <- dfidx(as.data.frame(dva2), idx = c("No", "alt"), choice = "chosen")
  list(P = predict(fit, newdata = dva_x), nos = sort(unique(dva$No)), fit = fit)
}

oof <- data.table(No = sort(unique(trl$No)), p1 = NA_real_, p2 = NA_real_,
                  p3 = NA_real_, p4 = NA_real_)
for (k in 1:5) {
  r <- fit_predict(trl[fold != k], trl[fold == k])
  oof[match(r$nos, No), `:=`(p1 = r$P[, 1], p2 = r$P[, 2], p3 = r$P[, 3], p4 = r$P[, 4])]
  cat("fold", k, "done\n")
}
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
stopifnot(nrow(oof) == 21565, !anyNA(oof))
cat(sprintf("\n>>> %s OOF logloss: %.5f   (production split 42: 1.15686)\n",
            NAME, logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)]))))
saveRDS(oof, sprintf("model/artifacts/oof_%s.rds", NAME))

rf <- fit_predict(trl, tel)
tp <- data.table(No = rf$nos, p1 = rf$P[,1], p2 = rf$P[,2], p3 = rf$P[,3], p4 = rf$P[,4])
setorder(tp, No)
stopifnot(nrow(tp) == 4997, !anyNA(tp))
saveRDS(tp, sprintf("model/artifacts/test_%s.rds", NAME))
cat("OK\n")
