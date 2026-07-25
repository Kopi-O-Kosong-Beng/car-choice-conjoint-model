# Mixed logit v2: log-normal price coefficient.
# v1 used rpar Price="n" (normal): mean -0.19, sd 0.46 => ~34% of the simulated
# population had a POSITIVE price coefficient (prefers expensive bundles), which is
# behaviourally wrong and cost us calibration. Here we use "ln" on negPrice = -Price,
# so beta_negPrice > 0 always => the effect of Price is negative for everyone.
suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")
args <- commandArgs(trailingOnly = TRUE)
K <- as.integer(if (length(args)) args[1] else stop("need fold arg 1..5 or 0 (combine+test)"))

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
long[, negPrice := -Price]

xvars <- c(setdiff(ATTRS, "Price"), "negPrice", "asc2", "asc3", "asc4",
           "Price_x_age", "Price_x_ppark", "Price_x_inc")
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 0"))
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
setorder(trl, No, alt)

fit_mixl <- function(d) {
  dx <- dfidx(as.data.frame(d), idx = list(c("No", "Case"), "alt"), choice = "chosen")
  mlogit(fml, data = dx, rpar = c(negPrice = "ln", asc4 = "n"),
         R = 100, halton = NA, panel = TRUE)
}

# Unconditional (population-averaged) probabilities.
# negPrice is log-normal: draw exp(mu + sigma*z); asc4 is normal: mu + sigma*z.
# mlogit reports the underlying normal's mean/sd for "ln" parameters.
predict_uncond <- function(fit, d, R = 500, seed = 99) {
  co <- coef(fit)
  stopifnot(all(c("negPrice", "sd.negPrice", "asc4", "sd.asc4") %in% names(co)))
  fixed <- co[setdiff(names(co), c("sd.negPrice", "sd.asc4", "negPrice", "asc4"))]
  X <- as.matrix(as.data.frame(d)[, names(fixed)])
  base_u <- as.vector(X %*% fixed)
  mu_np <- co["negPrice"]; sd_np <- abs(co["sd.negPrice"])
  mu_a4 <- co["asc4"];     sd_a4 <- abs(co["sd.asc4"])
  set.seed(seed); Z <- matrix(rnorm(R * 2), R)
  acc <- numeric(nrow(d))
  for (r in seq_len(R)) {
    b_np <- exp(mu_np + sd_np * Z[r, 1])          # strictly positive
    b_a4 <- mu_a4 + sd_a4 * Z[r, 2]
    u <- base_u + d$negPrice * b_np + d$asc4 * b_a4
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
  saveRDS(list(fold = K, pred = pl, coefs = coef(fit)), sprintf("model/artifacts/mixl2_fold%d.rds", K))
  cat("fold", K, "done in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min; ",
      "median price effect:", round(-exp(coef(fit)["negPrice"]), 4), "\n")
} else {
  parts <- lapply(1:5, function(k) readRDS(sprintf("model/artifacts/mixl2_fold%d.rds", k))$pred)
  oof <- dcast(rbindlist(parts), No ~ alt, value.var = "p")
  setnames(oof, c("No", "p1", "p2", "p3", "p4")); setorder(oof, No)
  ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
  cat(">>> MIXL2 OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1, p2, p3, p4)])), 5), "\n")
  saveRDS(oof, "model/artifacts/oof_mixl2.rds")
  fit <- fit_mixl(trl)
  tel <- long[is_test == TRUE]; tel[, chosen := FALSE]; setorder(tel, No, alt)
  tp <- dcast(predict_uncond(fit, tel), No ~ alt, value.var = "p")
  setnames(tp, c("No", "p1", "p2", "p3", "p4")); setorder(tp, No)
  saveRDS(tp, "model/artifacts/test_mixl2.rds")
  saveRDS(coef(fit), "model/artifacts/mixl2_full_coefs.rds")
  print(round(coef(fit)[c("negPrice", "sd.negPrice", "asc4", "sd.asc4")], 4))
  cat("OK\n")
}
