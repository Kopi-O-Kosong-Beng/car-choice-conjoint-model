suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")
args <- commandArgs(trailingOnly = TRUE)
K <- as.integer(if (length(args)) args[1] else stop("need fold arg 1..5 or 0 (combine+test)"))

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
xvars <- c(ATTRS, "asc2", "asc3", "asc4", "Price_x_age", "Price_x_ppark")
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 0"))
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
setorder(trl, No, alt)

fit_mixl <- function(d) {
  dx <- dfidx(as.data.frame(d), idx = list(c("No", "Case"), "alt"), choice = "chosen")
  mlogit(fml, data = dx, rpar = c(Price = "n", asc4 = "n"),
         R = 100, halton = NA, panel = TRUE)
}

# Unconditional (population-averaged) probabilities via manual simulation.
# d must be sorted by (No, alt): each task = 4 consecutive rows.
predict_uncond <- function(fit, d, R = 500, seed = 99) {
  co <- coef(fit); rp <- c("Price", "asc4"); sdn <- paste0("sd.", rp)
  stopifnot(all(c(rp, sdn) %in% names(co)))
  fixed <- co[setdiff(names(co), sdn)]
  X <- as.matrix(as.data.frame(d)[, names(fixed)])
  base_u <- as.vector(X %*% fixed)
  set.seed(seed); Z <- matrix(rnorm(R * 2), R)
  acc <- numeric(nrow(d))
  for (r in seq_len(R)) {
    u <- base_u + d$Price * (Z[r, 1] * abs(co["sd.Price"])) + d$asc4 * (Z[r, 2] * abs(co["sd.asc4"]))
    U <- matrix(u, ncol = 4, byrow = TRUE)
    m <- pmax(U[, 1], U[, 2], U[, 3], U[, 4])
    E <- exp(U - m)
    acc <- acc + as.vector(t(E / rowSums(E)))
  }
  data.table(No = d$No, alt = d$alt, p = acc / R)
}

if (K >= 1 && K <= 5) {
  t0 <- Sys.time()
  fit <- fit_mixl(trl[fold != K])
  va <- trl[fold == K]; setorder(va, No, alt)
  pl <- predict_uncond(fit, va)
  saveRDS(list(fold = K, pred = pl, coefs = coef(fit)), sprintf("model/artifacts/mixl_fold%d.rds", K))
  cat("fold", K, "done in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min;",
      "sd.Price", round(abs(coef(fit)["sd.Price"]), 4), "sd.asc4", round(abs(coef(fit)["sd.asc4"]), 4), "\n")
} else {  # K == 0: combine folds + full refit for test
  parts <- lapply(1:5, function(k) readRDS(sprintf("model/artifacts/mixl_fold%d.rds", k))$pred)
  oof_l <- rbindlist(parts)
  oof <- dcast(oof_l, No ~ alt, value.var = "p")
  setnames(oof, c("No", "p1", "p2", "p3", "p4")); setorder(oof, No)
  ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
  cat(">>> MIXL OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5), "\n")
  saveRDS(oof, "model/artifacts/oof_mixl.rds")
  fit <- fit_mixl(trl)
  tel <- long[is_test == TRUE]; tel[, chosen := FALSE]; setorder(tel, No, alt)
  tp_l <- predict_uncond(fit, tel)
  tp <- dcast(tp_l, No ~ alt, value.var = "p")
  setnames(tp, c("No", "p1", "p2", "p3", "p4")); setorder(tp, No)
  saveRDS(tp, "model/artifacts/test_mixl.rds")
  saveRDS(coef(fit), "model/artifacts/mixl_full_coefs.rds")
  print(round(coef(fit)[c("Price", "sd.Price", "asc4", "sd.asc4")], 4))
  cat("OK\n")
}
