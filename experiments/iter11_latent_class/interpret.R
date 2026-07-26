# =============================================================================
# Correct interpretation of the 3-class latent-class fit (report material).
#
# The original report block in run.R permuted `betas` for display but not the
# membership matrix `G`, which inverted the reported membership drivers. The
# OOF number was never affected -- only the prose. This script reads both
# consistently, with NO permutation, so class c in the taste table is the same
# class c in the membership table.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter11_latent_class/interpret.R
# =============================================================================
f <- readRDS("experiments/iter11_latent_class/fit_C3.rds")
B <- f$betas; G <- f$G; keep <- f$keep; zc <- f$zcols; sh <- f$shares
C <- length(B)

cat("converged log-likelihood path (last 4):", round(tail(f$path, 4), 1), "\n")
cat("class shares:", paste(sprintf("C%d %.1f%%", 1:C, 100 * sh), collapse = "  "), "\n\n")

price_idx <- grep("^Price_L", keep)
price_lvl <- as.integer(sub("Price_L", "", keep[price_idx]))
ord <- order(price_lvl)

cat("=== PRICE PART-WORTHS BY CLASS (utility relative to the cheapest level) ===\n")
cat(sprintf("%-8s %s\n", "level", paste(sprintf("%6d", price_lvl[ord]), collapse = "")))
for (c in 1:C)
  cat(sprintf("class %d  %s\n", c,
      paste(sprintf("%6.2f", B[[c]][price_idx][ord]), collapse = "")))

cat("\n=== PRICE SENSITIVITY SUMMARY ===\n")
for (c in 1:C) {
  v <- B[[c]][price_idx][ord]
  cat(sprintf("class %d (%.0f%%): total drop across the price range %6.2f | mean step %5.2f\n",
              c, 100 * sh[c], v[length(v)], mean(diff(c(0, v)))))
}

a4 <- which(keep == "asc4")
if (length(a4)) {
  cat("\n=== OUTSIDE-OPTION CONSTANT (higher = more willing to buy nothing) ===\n")
  for (c in 1:C) cat(sprintf("class %d (%.0f%%): asc4 = %6.2f\n", c, 100 * sh[c], B[[c]][a4]))
}

cat("\n=== WHAT EACH CLASS VALUES MOST (top feature part-worths) ===\n")
featmask <- grepl("_L", keep) & !grepl("^Price_L", keep)
for (c in 1:C) {
  v <- B[[c]][featmask]; names(v) <- keep[featmask]
  cat(sprintf("class %d: %s\n", c,
      paste(sprintf("%s %+.2f", names(sort(v, decreasing = TRUE))[1:6],
                    sort(v, decreasing = TRUE)[1:6]), collapse = "  ")))
}

cat("\n=== MEMBERSHIP DRIVERS (class 1 is the reference; + means more likely) ===\n")
cat("G columns correspond to classes 2..C, read against the SAME class indices above.\n")
for (j in seq_len(ncol(G))) {
  g <- G[, j]; names(g) <- zc
  g <- g[names(g) != "Intercept"]
  top <- c(head(sort(g, decreasing = TRUE), 5), tail(sort(g, decreasing = TRUE), 5))
  cat(sprintf("\nclass %d vs class 1:\n  ", j + 1))
  cat(paste(sprintf("%s %+.2f", names(top), top), collapse = "\n  "), "\n")
}
