suppressMessages(library(data.table))
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No); y <- ymap$y

getP <- function(path) {
  d <- readRDS(path)
  if (is.data.frame(d)) {
    d <- as.data.table(d)
    if ("No" %in% names(d)) setorder(d, No)
    as.matrix(d[, .(p1, p2, p3, p4)])
  } else {
    as.matrix(d)
  }
}

f <- list.files("model/artifacts", pattern = "^oof_.*[.]rds$")
n <- gsub("^oof_|[.]rds$", "", f)
n <- n[file.exists(sprintf("model/artifacts/test_%s.rds", n))]
cat(length(n), "paired artifacts\n\n")
for (m in n) {
  o <- try(getP(sprintf("model/artifacts/oof_%s.rds", m)), silent = TRUE)
  t <- try(getP(sprintf("model/artifacts/test_%s.rds", m)), silent = TRUE)
  bad <- inherits(o, "try-error") || inherits(t, "try-error")
  if (bad) { cat(sprintf("%-20s  UNREADABLE\n", m)); next }
  ok <- nrow(o) == 21565 && nrow(t) == 4997
  if (!ok) { cat(sprintf("%-20s oof=%6d test=%5d  WRONG SHAPE\n", m, nrow(o), nrow(t))); next }
  ll <- logloss(y, o)
  cat(sprintf("%-20s oof=%6d test=%5d  OOF=%.5f  test_p4=%.4f\n", m, nrow(o), nrow(t), ll, mean(t[, 4])))
}
