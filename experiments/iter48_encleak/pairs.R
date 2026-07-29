# Paired, respondent-clustered comparison (model/compare.R's statistics) applied to
# the iteration-48 arm bags, which live outside model/artifacts.
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
ymap <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
getP <- function(f) { d <- readRDS(f); setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) }
bag <- function(arm, depth = 8) {
  fs <- list.files("experiments/iter48_encleak",
                   pattern = sprintf("^el48_oof_%s_d%d_m20_e003_n540_s.*\\.rds$", arm, depth), full.names = TRUE)
  A <- Reduce(`+`, lapply(fs, getP)) / length(fs); A / rowSums(A)
}
loss <- function(P) { P <- pmax(P / rowSums(P), 1e-15); -log(P[cbind(seq_len(nrow(P)), ymap$y)]) }
cmp <- function(lab, Pa, Pb) {          # positive => Pb better
  d <- loss(Pa) - loss(Pb)
  cl <- data.table(Case = ymap$Case, d = d)[, .(dm = mean(d)), by = Case]
  est <- mean(cl$dm); se <- sd(cl$dm) / sqrt(nrow(cl))
  cat(sprintf("%-42s %+8.5f  SE %.5f  z %7.2f  wins %5.1f%%\n",
              lab, est, se, est/se, 100*mean(cl$dm > 0)))
}
P <- setNames(lapply(c("prod","noenc","honest","leaky"), bag), c("prod","noenc","honest","leaky"))
cat("depth 8 / mcw 20 / eta .03 / 540 rounds, seed-bagged.  positive = second is better\n\n")
cmp("noenc -> prod   (production's claim)", P$noenc, P$prod)
cmp("honest -> leaky (PURE LEAK, matched)", P$honest, P$leaky)
cmp("noenc -> honest (honest value, 3-fold)", P$noenc, P$honest)
cmp("leaky -> prod   (3->4 fold reference)", P$leaky, P$prod)
cat("\nblock test (disjoint reference block, 1 fold; train = 3 folds)\n")
bag2 <- function(arm, depth) {
  fs <- list.files("experiments/iter48_encleak",
                   pattern = sprintf("^el48_oof_%s_d%d_m20_e003_n540_s.*\\.rds$", arm, depth), full.names = TRUE)
  A <- Reduce(`+`, lapply(fs, getP)) / length(fs); A / rowSums(A)
}
for (dd in c(8, 10)) cmp(sprintf("block0 -> block (depth %d)", dd), bag2("block0", dd), bag2("block", dd))
cat("\nPAIRS_DONE\n")
