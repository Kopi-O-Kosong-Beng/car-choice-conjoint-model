# Diagnostic 1: is the experiment BLOCKED -- does each task show only a subset of attributes?
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")

# active attribute set per task (No), using only real bundles (alt 1..3)
real <- long[alt <= 3]
act <- real[, lapply(.SD, function(x) as.integer(any(x > 0))), by = .(No, Case, Task, is_test), .SDcols = ATTRS]
act[, nact := rowSums(.SD), .SDcols = ATTRS]
cat("== number of ACTIVE attributes per choice task ==\n")
print(table(act$nact))

act[, patt := do.call(paste0, .SD), .SDcols = ATTRS]
cat("\n== number of distinct active-attribute PATTERNS (blocks) ==\n")
cat(uniqueN(act$patt), "\n")
pt <- act[, .N, by = patt][order(-N)]
print(head(pt, 20))

cat("\n== block pattern by Task number (are blocks tied to task position?) ==\n")
tb <- table(act$Task, act$patt)
print(dim(tb))
# for each task, how concentrated
for (t in 1:19) {
  r <- tb[as.character(t), ]
  r <- sort(r[r>0], decreasing = TRUE)
  cat(sprintf("Task %2d: %d distinct blocks, top share %.3f\n", t, length(r), r[1]/sum(r)))
}

cat("\n== decode the blocks: which attrs active in each ==\n")
bl <- unique(act[, c("patt", ATTRS), with = FALSE])
for (i in seq_len(min(nrow(bl), 12))) {
  a <- ATTRS[as.logical(unlist(bl[i, ATTRS, with=FALSE]))]
  cat(sprintf("block %d (n=%d): %s\n", i, pt[patt == bl$patt[i], N], paste(a, collapse=" ")))
}

cat("\n== per respondent: how many distinct blocks across their 19 tasks ==\n")
print(table(act[, uniqueN(patt), by = Case]$V1))

cat("\n== does each respondent see each block the same number of times? ==\n")
print(head(dcast(act[, .N, by=.(Case, patt)], Case ~ patt, value.var="N", fill=0), 10))
