# Verifies, WITHOUT fitting anything, that the arm constructions do what the header
# of runarm.R claims: that the leak channel is open in prod/leaky and shut in honest.
#
# THE STATISTIC.  For a scored fold-k row the encoding numerator is
#     enc = (cell chosen-count over the arm's reference folds).
# A model that has isolated the cell can learn the cell's total over the arm's
# UNIVERSE, C_U, from the training rows.  Then  C_U - enc  is handed to it for free.
#     prod / leaky :  C_U - enc = c_k   -- the SCORED fold's own choice count
#     honest       :  C_U - enc = c_d   -- a decoy fold's count, no fold-k information
# We report cor(C_U - enc, own label) on the scored rows.  Large positive = the label
# is sitting one subtraction away from the feature.
suppressMessages(library(data.table))
source("model/99_utils.R"); source("model/encode_design.R")
long <- readRDS("model/artifacts/long.rds"); folds <- readRDS("model/artifacts/folds.rds")
wide <- readRDS("model/artifacts/wide.rds")
long <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No"); setorder(trl, No, alt)

pf <- trl[, .(n_j = .N, c_j = sum(chosen)), by = .(dkey, alt, fold)]
numer <- function(ref_folds, target) {   # chosen-count of the cell over ref_folds
  agg <- pf[fold %in% ref_folds, .(v = sum(c_j)), by = .(dkey, alt)]
  t2 <- copy(target)[, .rid := .I][, .(.rid, dkey, alt)]
  t2 <- merge(t2, agg, by = c("dkey","alt"), all.x = TRUE)[is.na(v), v := 0]
  setorder(t2, .rid); t2$v
}
cat("cor( C_U - enc ,  own label )  on the SCORED rows of each outer fold\n")
cat("   (this is the quantity a cell-identifying model gets for free)\n\n")
cat(sprintf("%-8s %8s %8s %8s %8s %8s   %s\n","outer k","prod","leaky","honest","","",""))
res <- list()
for (k in 1:5) {
  d <- (k %% 5L) + 1L
  sc <- trl[fold == k]; y <- as.numeric(sc$chosen)
  out <- c()
  for (arm in c("prod","leaky","honest")) {
    if (arm == "prod")   { U <- 1:5;              ref <- setdiff(1:5, k) }
    if (arm == "leaky")  { U <- setdiff(1:5, d);  ref <- setdiff(U, k)   }
    if (arm == "honest") { U <- setdiff(1:5, k);  ref <- setdiff(U, d)   }
    L <- numer(U, sc) - numer(ref, sc)
    out <- c(out, cor(L, y))
  }
  res[[k]] <- out
  cat(sprintf("%-8d %+8.4f %+8.4f %+8.4f\n", k, out[1], out[2], out[3]))
}
m <- colMeans(do.call(rbind, res))
cat(sprintf("%-8s %+8.4f %+8.4f %+8.4f\n\n", "mean", m[1], m[2], m[3]))
cat("Also, the HONEST reference never contains a fold-k respondent, by construction\n")
cat("(asserted inside runarm.R with stopifnot on every fold and every training fold).\n")
cat("VERIFY_DONE\n")
