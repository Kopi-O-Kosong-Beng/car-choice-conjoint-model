# Shared utilities. All scripts run with cwd = Competition folder.
ATTRS <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP","PP",
           "KA","SC","TS","NV","MA","LB","AF","HU","Price")

logloss <- function(y, P, eps = 1e-15) {
  stopifnot(length(y) == nrow(P), ncol(P) == 4, !anyNA(y))
  P <- pmax(pmin(as.matrix(P), 1 - eps), eps)
  P <- P / rowSums(P)
  -mean(log(P[cbind(seq_along(y), y)]))
}

make_case_folds <- function(cases, k = 5, seed = 42) {
  set.seed(seed)
  u <- sort(unique(cases))
  f <- sample(rep_len(seq_len(k), length(u)))
  unname(setNames(f, u)[as.character(cases)])
}

# Normalize a positive score to probabilities within groups (e.g. per task No)
norm_by_group <- function(score, group) {
  s <- ave(score, group, FUN = sum)
  score / s
}

clip_norm <- function(P, eps = 1e-6) {
  P <- pmax(pmin(as.matrix(P), 1 - eps), eps)
  P / rowSums(P)
}
