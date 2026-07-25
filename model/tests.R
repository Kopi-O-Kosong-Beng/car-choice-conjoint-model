source("model/99_utils.R")
ok <- function(name, cond) { cat(sprintf("%-45s %s\n", name, if (cond) "PASS" else "FAIL")); stopifnot(cond) }

# logloss: uniform prediction must equal ln(4) = 1.386294
P <- matrix(0.25, 3, 4)
ok("logloss uniform = 1.386294", abs(logloss(c(1,2,4), P) - log(4)) < 1e-9)
# logloss: perfect prediction ~ 0
P2 <- matrix(1e-15, 2, 4); P2[cbind(1:2, c(2,3))] <- 1
ok("logloss perfect ~ 0", logloss(c(2,3), P2) < 1e-6)

# folds: constant within case, 5 non-empty, deterministic
cases <- rep(1:100, each = 19)
f1 <- make_case_folds(cases); f2 <- make_case_folds(cases)
ok("folds deterministic", identical(f1, f2))
ok("folds constant within case", all(tapply(f1, cases, function(x) length(unique(x))) == 1))
ok("5 non-empty folds", length(unique(f1)) == 5)

# norm_by_group sums to 1
g <- rep(1:2, each = 4); s <- runif(8)
ok("norm_by_group sums to 1", all(abs(tapply(norm_by_group(s, g), g, sum) - 1) < 1e-12))

# reshape spot-check (only once artifacts exist)
if (file.exists("model/artifacts/long.rds")) {
  suppressMessages(library(data.table))
  l <- readRDS("model/artifacts/long.rds")
  w <- fread("Raw Dump/Competition Data/train2024.csv")
  set.seed(1); i <- sample(nrow(w), 8)
  for (r in i) for (j in 1:4) {
    stopifnot(l[No == w$No[r] & alt == j, Price] == w[[paste0("Price", j)]][r],
              l[No == w$No[r] & alt == j, CC]    == w[[paste0("CC", j)]][r],
              l[No == w$No[r] & alt == j, HU]    == w[[paste0("HU", j)]][r])
  }
  ok("long reshape matches wide (24 spot checks)", TRUE)
  # chosen consistency: exactly one TRUE per train task
  ok("exactly one chosen per train task", all(l[is_test == FALSE, sum(chosen), by = No]$V1 == 1))
  # price_min_rival: manual check on one task
  x <- l[No == w$No[i[1]]][order(alt)]
  ok("price_min_rival correct", all(sapply(1:4, function(j) min(x$Price[-j])) == x$price_min_rival))
}

cat("ALL TESTS PASS\n")
