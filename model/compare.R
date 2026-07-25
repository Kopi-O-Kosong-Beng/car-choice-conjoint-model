# =============================================================================
# Paired comparison of two OOF prediction sets, with respondent-clustered SE.
#
# Why: comparing two models by their headline OOF numbers ignores that both were
# scored on the SAME rows. A paired test is far more sensitive. But rows are not
# independent -- 19 come from each respondent -- so the SE must be clustered by
# respondent or it will be optimistically small.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/compare.R baseline challenger
#   e.g. ... model/compare.R xgb_long xgb_de
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: compare.R <baseline_name> <challenger_name>")
A <- args[1]; B <- args[2]

long <- readRDS("model/artifacts/long.rds")
ymap <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)

get_loss <- function(nm) {
  f <- sprintf("model/artifacts/oof_%s.rds", nm)
  if (!file.exists(f)) stop("no such artifact: ", f)
  d <- readRDS(f); setorder(d, No)
  stopifnot(identical(d$No, ymap$No))
  P <- as.matrix(d[, .(p1, p2, p3, p4)])
  P <- pmax(P / rowSums(P), 1e-15)
  -log(P[cbind(seq_len(nrow(P)), ymap$y)])
}
la <- get_loss(A); lb <- get_loss(B)

cat(sprintf("\n%-16s OOF: %.5f\n%-16s OOF: %.5f\n", A, mean(la), B, mean(lb)))
d <- la - lb                                   # positive => challenger is better
cl <- data.table(Case = ymap$Case, d = d)[, .(dm = mean(d)), by = Case]
est <- mean(cl$dm)
se  <- sd(cl$dm) / sqrt(nrow(cl))
z   <- est / se
cat(sprintf("\nimprovement (%s over %s): %+.5f\n", B, A, est))
cat(sprintf("respondent-clustered SE: %.5f  (n = %d respondents)\n", se, nrow(cl)))
cat(sprintf("95%% CI: [%+.5f, %+.5f]   z = %.2f\n", est - 1.96 * se, est + 1.96 * se, z))
cat(if (abs(z) < 2) "VERDICT: not distinguishable from noise\n"
    else if (z > 0) "VERDICT: challenger is genuinely better\n"
    else "VERDICT: challenger is genuinely WORSE\n")
won <- mean(cl$dm > 0)
cat(sprintf("challenger wins on %.1f%% of respondents\n", 100 * won))
