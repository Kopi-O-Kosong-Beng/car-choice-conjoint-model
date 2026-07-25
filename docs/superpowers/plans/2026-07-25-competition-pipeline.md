# Analytics Edge Competition Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **This session's choice: inline execution (executing-plans).** Harness rule forbids unprompted subagents; user pre-approved continuous execution to first OOF milestone.

**Goal:** An R pipeline that produces honestly-validated (respondent-grouped, nested) OOF logloss numbers for a blend of MNL, mixed logit, and two xgboost formulations, plus a Kaggle submission file generator.

**Architecture:** Numbered R scripts under `model/`, communicating only through RDS artifacts in `model/artifacts/` (long-format data → fixed folds → per-model OOF+test prediction matrices → blend → submission CSV). Every model consumes identical folds; the blend's nested OOF is the only number used for submission decisions.

**Tech Stack:** R 4.6.0 at `C:\Program Files\R\R-4.6.0\bin\Rscript.exe` (NOT on PATH — always invoke by full path). Packages present: data.table, xgboost, mlogit, dfidx, nnet, glmnet, Matrix. Working directory for all runs: `d:\SUTD\Term5\Analytics Edge\Competition`.

## Global Constraints

- **R only** — competition rule; no Python anywhere in the pipeline.
- Data: `Raw Dump/Competition Data/train2024.csv` (21,565 rows), `test2024.csv` (4,997 rows). Never modified in place.
- All CV grouped by `Case` (respondent); one fixed fold file used by every model; seed 42.
- Probabilities: clip to [1e-6, 1-1e-6] and renormalize rows to sum 1 before writing any submission.
- Submission CSV header exactly: `No,Ch1,Ch2,Ch3,Ch4`, 4,997 rows, `No` = 21566–26562.
- No git repo here (deliberate — course folder). "Commit" steps are replaced by artifact + log verification.
- Attribute set (order matters, reused everywhere): `CC GN NS BU FA LD BZ FC FP RP PP KA SC TS NV MA LB AF HU Price`.

---

### Task 1: Scaffold, utils, and unit tests

**Files:**
- Create: `model/99_utils.R`
- Create: `model/tests.R`
- Create: `model/artifacts/` and `submissions/` directories

**Interfaces:**
- Produces: `logloss(y, P)` (y integer 1..4, P n×4 matrix → scalar), `make_case_folds(cases, k=5, seed=42)` (returns integer fold per element of `cases`, constant within Case), `ATTRS` character vector, `RSCRIPT` doc comment.

- [ ] **Step 1: Write `model/99_utils.R`**

```r
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
```

- [ ] **Step 2: Write failing-then-passing checks in `model/tests.R`**

```r
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

cat("ALL TESTS PASS\n")
```

- [ ] **Step 3: Run tests**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/tests.R`
Expected: 5 PASS lines then `ALL TESTS PASS`.

---

### Task 2: Data load + long format + features (`model/00_load.R`)

**Files:**
- Create: `model/00_load.R`

**Interfaces:**
- Produces artifacts: `model/artifacts/long.rds` (data.table, 26,562×4 rows = one per task×alternative) and `model/artifacts/wide.rds` (train+test rows with `y`, `is_test`).
- Long columns later tasks rely on: `No, Case, Task, alt (1..4), chosen (logical, NA for test), y, is_test`, the 20 `ATTRS`, `paste0(ATTRS,"_c")` (task-centered), `richness, lvlsum, price_rank, price_min_rival, price_task_mean, rich_task_mean`, `alt1..alt4` dummies, `asc2,asc3,asc4`, `Price_x_age, Price_x_ppark, Price_x_seg2..5, Price_x_reg2..4`, demographics numeric: `segmentind,yearind,milesind,milesa,nightind,nighta,pparkind,genderind,ageind,agea,educind,regionind,Urbind,incomeind,incomea`.

- [ ] **Step 1: Write `model/00_load.R`**

```r
suppressMessages(library(data.table))
source("model/99_utils.R")

tr <- fread("Raw Dump/Competition Data/train2024.csv")
te <- fread("Raw Dump/Competition Data/test2024.csv")

# --- integrity checks (fail loudly, they encode assumptions) ---
stopifnot(nrow(tr) == 21565, nrow(te) == 4997)
stopifnot(max(tr$Case) < min(te$Case))                       # disjoint respondents
stopifnot(all(rowSums(tr[, .(Ch1, Ch2, Ch3, Ch4)]) == 1L))   # exactly one choice
a4 <- paste0(ATTRS, "4")
cat("alt-4 attribute max over all rows (expect 0):",
    max(as.matrix(rbind(tr, te, fill = TRUE)[, ..a4]), na.rm = TRUE), "\n")

tr[, y := max.col(cbind(Ch1, Ch2, Ch3, Ch4))]
te[, y := NA_integer_]
te[, c("Ch1","Ch2","Ch3","Ch4") := NULL]
all_dt <- rbind(tr, te, fill = TRUE)
all_dt[, is_test := No > 21565L]

demo_keep <- c("segment","segmentind","yearind","milesind","milesa","nightind","nighta",
               "ppark","pparkind","genderind","ageind","agea","educind",
               "region","regionind","Urbind","incomeind","incomea")
idv <- c("No","Case","Task","y","is_test", demo_keep)
mv  <- setNames(lapply(ATTRS, function(a) paste0(a, 1:4)), ATTRS)
long <- melt(all_dt, id.vars = idv, measure.vars = mv, variable.name = "alt")
long[, alt := as.integer(alt)]
long[, chosen := ifelse(is_test, NA, alt == y)]
setorder(long, No, alt)

# --- engineered features (within task group No) ---
np <- setdiff(ATTRS, "Price")
long[, richness := rowSums(.SD != 0), .SDcols = np]
long[, lvlsum   := rowSums(.SD),      .SDcols = np]
long[, price_rank := frank(Price, ties.method = "average"), by = No]
long[, `:=`(price_task_mean = mean(Price), rich_task_mean = mean(richness)), by = No]
long[, price_min_rival := vapply(seq_len(.N), function(i) min(Price[-i]), 0), by = No]
for (a in ATTRS) long[, paste0(a, "_c") := get(a) - mean(get(a)), by = No]
for (j in 1:4) long[, paste0("alt", j) := as.numeric(alt == j)]
long[, `:=`(asc2 = alt2, asc3 = alt3, asc4 = alt4)]

# --- MNL interaction columns (explicit, so glmnet/manual X reuse them) ---
long[, Price_x_age   := Price * agea / 100]
long[, Price_x_ppark := Price * pparkind]
for (s in sort(unique(long$segmentind))[-1]) long[, paste0("Price_x_seg", s) := Price * (segmentind == s)]
for (r in sort(unique(long$regionind))[-1])  long[, paste0("Price_x_reg", r) := Price * (regionind == r)]

dir.create("model/artifacts", recursive = TRUE, showWarnings = FALSE)
saveRDS(long,  "model/artifacts/long.rds")
saveRDS(all_dt, "model/artifacts/wide.rds")
cat("train tasks:", uniqueN(long[!is_test, No]), " test tasks:", uniqueN(long[is_test, No]), "\n")
cat("choice shares train:", tr[, round(colMeans(cbind(Ch1,Ch2,Ch3,Ch4)), 4)], "\n")
cat("OK: artifacts written\n")
```

- [ ] **Step 2: Run it**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/00_load.R`
Expected: alt-4 max = 0 (if not 0, the "none" assumption is wrong — features stay valid, but note it in report_notes), `train tasks: 21565  test tasks: 4997`, four choice shares, `OK`.

- [ ] **Step 3: Spot-check reshape correctness**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" -e "library(data.table); l <- readRDS('model/artifacts/long.rds'); w <- fread('Raw Dump/Competition Data/train2024.csv'); i <- sample(nrow(w), 5); for (r in i) { for (j in 1:4) stopifnot(l[No==w$No[r] & alt==j, Price] == w[[paste0('Price',j)]][r], l[No==w$No[r] & alt==j, CC] == w[[paste0('CC',j)]][r]) }; cat('reshape spot-check PASS\n')"`
Expected: `reshape spot-check PASS`.

---

### Task 3: Fixed grouped folds (`model/01_folds.R`)

**Files:**
- Create: `model/01_folds.R`

**Interfaces:**
- Produces: `model/artifacts/folds.rds` — data.table `(No, Case, fold)` for the 21,565 train tasks. Every model joins on `No`.

- [ ] **Step 1: Write `model/01_folds.R`**

```r
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
tasks <- unique(long[is_test == FALSE, .(No, Case)])
tasks[, fold := make_case_folds(Case, k = 5, seed = 42)]
stopifnot(nrow(tasks) == 21565, all(tasks[, uniqueN(fold), by = Case]$V1 == 1))
print(tasks[, .(tasks = .N, respondents = uniqueN(Case)), keyby = fold])
saveRDS(tasks, "model/artifacts/folds.rds")
cat("OK: folds written\n")
```

- [ ] **Step 2: Run and eyeball balance**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/01_folds.R`
Expected: 5 folds, each ~227 respondents / ~4,300 tasks, then `OK`.

---

### Task 4: MNL with interactions (`model/02_mnl.R`)

**Files:**
- Create: `model/02_mnl.R`

**Interfaces:**
- Consumes: `long.rds`, `folds.rds`.
- Produces: `model/artifacts/oof_mnl.rds` (data.table `No, p1, p2, p3, p4` for all train tasks) and `model/artifacts/test_mnl.rds` (same for test; model refit on full train). Prints OOF logloss. This artifact contract `(oof_<name>.rds, test_<name>.rds)` is identical for every model.

- [ ] **Step 1: Write `model/02_mnl.R`**

```r
suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

xvars <- c(ATTRS, "Price_x_age", "Price_x_ppark",
           grep("^Price_x_seg|^Price_x_reg", names(long), value = TRUE))
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 1"))

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]

fit_predict <- function(dtr, dva) {
  dtr_x <- dfidx(as.data.frame(dtr), idx = c("No", "alt"), choice = "chosen")
  fit <- mlogit(fml, data = dtr_x)
  dva2 <- copy(dva); dva2[, chosen := FALSE]; dva2[alt == 1, chosen := TRUE]  # dummy, predict ignores
  dva_x <- dfidx(as.data.frame(dva2), idx = c("No", "alt"), choice = "chosen")
  P <- predict(fit, newdata = dva_x)
  list(P = P, nos = sort(unique(dva$No)), fit = fit)
}

oof <- data.table(No = sort(unique(trl$No)), p1 = NA_real_, p2 = NA_real_, p3 = NA_real_, p4 = NA_real_)
for (k in 1:5) {
  r <- fit_predict(trl[fold != k], trl[fold == k])
  oof[match(r$nos, No), `:=`(p1 = r$P[,1], p2 = r$P[,2], p3 = r$P[,3], p4 = r$P[,4])]
  cat("fold", k, "done\n")
}
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
cat(">>> MNL OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)])), 5), "\n")
saveRDS(oof, "model/artifacts/oof_mnl.rds")

rf <- fit_predict(trl, tel)
test_p <- data.table(No = rf$nos, p1 = rf$P[,1], p2 = rf$P[,2], p3 = rf$P[,3], p4 = rf$P[,4])
saveRDS(test_p, "model/artifacts/test_mnl.rds")
print(summary(rf$fit)$CoefTable[c("Price","Price_x_age","Price_x_ppark"), ])
cat("OK\n")
```

- [ ] **Step 2: Run**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/02_mnl.R` (allow ~5–15 min)
Expected: OOF logloss ≈ 1.16–1.19 (team's comparable number: ~1.17–1.18). Hard fail if > 1.30. Price coefficient negative and significant.

---

### Task 5: Long-format xgboost (`model/04_xgb_long.R`)

**Files:**
- Create: `model/04_xgb_long.R`

**Interfaces:**
- Consumes: `long.rds`, `folds.rds`. Produces: `oof_xgb_long.rds`, `test_xgb_long.rds` (same contract as Task 4).

- [ ] **Step 1: Check installed xgboost API version**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" -e "cat(as.character(packageVersion('xgboost')))"`
If ≥ 2.0: `xgb.train(evals=...)`; if 1.x: `watchlist=`. Code below uses a version guard.

- [ ] **Step 2: Write `model/04_xgb_long.R`**

```r
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
          "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo)

trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
params <- list(objective = "binary:logistic", eta = 0.05, max_depth = 6,
               min_child_weight = 10, subsample = 0.8, colsample_bytree = 0.8,
               eval_metric = "logloss", nthread = 0)
xgb_v2 <- packageVersion("xgboost") >= "2.0.0"
train_one <- function(dtr, dva_es, nrounds = 3000) {
  m_tr <- xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(dtr$chosen))
  m_es <- xgb.DMatrix(as.matrix(dva_es[, ..feat]), label = as.numeric(dva_es$chosen))
  a <- list(params = params, data = m_tr, nrounds = nrounds,
            early_stopping_rounds = 100, verbose = 0)
  if (xgb_v2) a$evals <- list(es = m_es) else a$watchlist <- list(es = m_es)
  do.call(xgb.train, a)
}
predict_norm <- function(fit, d) {
  p <- predict(fit, xgb.DMatrix(as.matrix(d[, ..feat])))
  data.table(No = d$No, alt = d$alt, p = norm_by_group(p, d$No))
}

oof_list <- list(); best_iters <- integer(5)
for (k in 1:5) {
  dtr_all <- trl[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  best_iters[k] <- fit$best_iteration
  oof_list[[k]] <- predict_norm(fit, trl[fold == k])
  cat("fold", k, "best_iter", fit$best_iteration, "\n")
}
oof_l <- rbindlist(oof_list)
oof <- dcast(oof_l, No ~ alt, value.var = "p"); setnames(oof, c("No","p1","p2","p3","p4")); setorder(oof, No)
ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
cat(">>> XGB-long OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)])), 5), "\n")
saveRDS(oof, "model/artifacts/oof_xgb_long.rds")

set.seed(7)
es_cases <- sample(unique(trl$Case), round(0.1 * uniqueN(trl$Case)))
fit_full <- train_one(trl[!Case %in% es_cases], trl[Case %in% es_cases])
tp_l <- predict_norm(fit_full, tel)
tp <- dcast(tp_l, No ~ alt, value.var = "p"); setnames(tp, c("No","p1","p2","p3","p4")); setorder(tp, No)
saveRDS(tp, "model/artifacts/test_xgb_long.rds")
imp <- xgb.importance(model = fit_full); print(head(imp, 15))
cat("OK\n")
```

*(set.seed(123) at top before the fold loop for the es_cases sampling.)*

- [ ] **Step 3: Run**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/04_xgb_long.R` (~5–20 min)
Expected: OOF ≈ 1.12–1.17. Importance list topped by Price/price-derived and richness features.

---

### Task 6: Wide 4-class xgboost (`model/05_xgb_wide.R`)

**Files:**
- Create: `model/05_xgb_wide.R`

**Interfaces:**
- Consumes: `wide.rds`, `folds.rds`. Produces: `oof_xgb_wide.rds`, `test_xgb_wide.rds`.

- [ ] **Step 1: Write `model/05_xgb_wide.R`**

```r
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
wide  <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")

acols <- as.vector(outer(setdiff(ATTRS, "Price"), 1:3, paste0))
pcols <- paste0("Price", 1:3)
wide[, `:=`(pd12 = Price1 - Price2, pd13 = Price1 - Price3, pd23 = Price2 - Price3,
            pmin123 = pmin(Price1, Price2, Price3), pmax123 = pmax(Price1, Price2, Price3))]
for (j in 1:3) wide[, paste0("rich", j) := rowSums(.SD != 0), .SDcols = paste0(setdiff(ATTRS,"Price"), j)]
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
feat <- c(acols, pcols, "pd12","pd13","pd23","pmin123","pmax123", paste0("rich",1:3), "Task", demo)

trw <- merge(wide[is_test == FALSE], folds[, .(No, fold)], by = "No")
tew <- wide[is_test == TRUE]
params <- list(objective = "multi:softprob", num_class = 4, eta = 0.05, max_depth = 6,
               min_child_weight = 10, subsample = 0.8, colsample_bytree = 0.8,
               eval_metric = "mlogloss", nthread = 0)
xgb_v2 <- packageVersion("xgboost") >= "2.0.0"
train_one <- function(dtr, des) {
  m_tr <- xgb.DMatrix(as.matrix(dtr[, ..feat]), label = dtr$y - 1)
  m_es <- xgb.DMatrix(as.matrix(des[, ..feat]), label = des$y - 1)
  a <- list(params = params, data = m_tr, nrounds = 3000,
            early_stopping_rounds = 100, verbose = 0)
  if (xgb_v2) a$evals <- list(es = m_es) else a$watchlist <- list(es = m_es)
  do.call(xgb.train, a)
}
pred_mat <- function(fit, d) {
  p <- predict(fit, xgb.DMatrix(as.matrix(d[, ..feat])))
  matrix(p, ncol = 4, byrow = TRUE)
}

set.seed(123)
oof <- data.table(No = trw$No, p1 = NA_real_, p2 = NA_real_, p3 = NA_real_, p4 = NA_real_)
for (k in 1:5) {
  dtr_all <- trw[fold != k]
  es_cases <- sample(unique(dtr_all$Case), round(0.1 * uniqueN(dtr_all$Case)))
  fit <- train_one(dtr_all[!Case %in% es_cases], dtr_all[Case %in% es_cases])
  P <- pred_mat(fit, trw[fold == k])
  oof[match(trw[fold == k, No], No), `:=`(p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])]
  cat("fold", k, "best_iter", fit$best_iteration, "\n")
}
setorder(oof, No)
cat(">>> XGB-wide OOF logloss:", round(logloss(trw[order(No), y], as.matrix(oof[, .(p1,p2,p3,p4)])), 5), "\n")
saveRDS(oof, "model/artifacts/oof_xgb_wide.rds")

es_cases <- sample(unique(trw$Case), round(0.1 * uniqueN(trw$Case)))
fit_full <- train_one(trw[!Case %in% es_cases], trw[Case %in% es_cases])
P <- pred_mat(fit_full, tew)
saveRDS(data.table(No = tew$No, p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4])[order(No)],
        "model/artifacts/test_xgb_wide.rds")
cat("OK\n")
```

- [ ] **Step 2: Run**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/05_xgb_wide.R` (~5–15 min)
Expected: OOF ≈ 1.15–1.20 (this member exists for diversity; it may trail xgb-long).

---

### Task 7: Mixed logit (`model/03_mixl.R`) — fold-parallel

**Files:**
- Create: `model/03_mixl.R` (accepts fold argument so folds run as concurrent processes)

**Interfaces:**
- Consumes: `long.rds`, `folds.rds`. Produces per fold: `model/artifacts/mixl_fold<k>.rds`; then combined `oof_mixl.rds`, `test_mixl.rds` (fold arg `0` = full-train refit + test predict + combine).
- Spec: random normal coefficients on `Price` and `asc4` (none-option ASC), fixed coefficients otherwise, panel by `Case`, 100 Halton draws for fitting; prediction = unconditional (population-averaged) probabilities via 500 manual normal draws from estimated (mean, |sd|).

- [ ] **Step 1: Write `model/03_mixl.R`**

```r
suppressMessages({ library(data.table); library(mlogit); library(dfidx) })
source("model/99_utils.R")
args <- commandArgs(trailingOnly = TRUE)
K <- as.integer(if (length(args)) args[1] else stop("need fold arg 1..5 or 0"))

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
xvars <- c(setdiff(ATTRS, character(0)), "asc2", "asc3", "asc4",
           "Price_x_age", "Price_x_ppark")
fml <- as.formula(paste("chosen ~", paste(xvars, collapse = " + "), "| 0"))
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")

fit_mixl <- function(d) {
  dx <- dfidx(as.data.frame(d), idx = list(c("No", "Case"), "alt"), choice = "chosen")
  mlogit(fml, data = dx, rpar = c(Price = "n", asc4 = "n"),
         R = 100, halton = NA, panel = TRUE)
}
# Unconditional probabilities via manual simulation (robust vs predict() quirks)
predict_uncond <- function(fit, d, R = 500, seed = 99) {
  co <- coef(fit)
  rp <- c("Price", "asc4")
  sd_names <- paste0("sd.", rp)
  stopifnot(all(c(rp, sd_names) %in% names(co)))
  fixed <- co[setdiff(names(co), c(sd_names))]
  X <- as.matrix(as.data.frame(d)[, names(fixed)])
  set.seed(seed)
  Z <- matrix(rnorm(R * length(rp)), R)
  acc <- matrix(0, nrow(d), 4)
  base_u <- X %*% fixed
  for (r in seq_len(R)) {
    u <- base_u
    for (j in seq_along(rp)) {
      delta <- Z[r, j] * abs(co[sd_names[j]])
      u <- u + as.matrix(d[[rp[j]]]) * delta
    }
    eu <- exp(u - ave(u, d$No, FUN = max))
    p <- norm_by_group(as.vector(eu), d$No)
    acc <- acc + matrix(p, ncol = 4, byrow = FALSE)[, 1, drop = TRUE] |> (\(x) acc)()  # placeholder guard, replaced below
  }
}
```

**NOTE:** the accumulation line above is intentionally rewritten in the actual implementation as a long-vector accumulator (see Step 2) — the plan records the final version below; do not keep the placeholder line.

```r
predict_uncond <- function(fit, d, R = 500, seed = 99) {
  co <- coef(fit); rp <- c("Price", "asc4"); sdn <- paste0("sd.", rp)
  stopifnot(all(c(rp, sdn) %in% names(co)))
  fixed <- co[setdiff(names(co), sdn)]
  X <- as.matrix(as.data.frame(d)[, names(fixed)])
  base_u <- as.vector(X %*% fixed)
  set.seed(seed); Z <- matrix(rnorm(R * 2), R)
  acc <- numeric(nrow(d))
  for (r in seq_len(R)) {
    u <- base_u + d$Price * (Z[r,1] * abs(co["sd.Price"])) + d$asc4 * (Z[r,2] * abs(co["sd.asc4"]))
    m <- ave(u, d$No, FUN = max)
    eu <- exp(u - m)
    acc <- acc + norm_by_group(eu, d$No)
  }
  data.table(No = d$No, alt = d$alt, p = acc / R)
}

if (K >= 1 && K <= 5) {
  fit <- fit_mixl(trl[fold != K])
  pl <- predict_uncond(fit, trl[fold == K])
  saveRDS(list(fold = K, pred = pl, coefs = coef(fit)), sprintf("model/artifacts/mixl_fold%d.rds", K))
  cat("fold", K, "done; rpar sds:", abs(coef(fit)[c("sd.Price","sd.asc4")]), "\n")
} else {  # K == 0: combine folds + full refit for test
  parts <- lapply(1:5, function(k) readRDS(sprintf("model/artifacts/mixl_fold%d.rds", k))$pred)
  oof_l <- rbindlist(parts)
  oof <- dcast(oof_l, No ~ alt, value.var = "p"); setnames(oof, c("No","p1","p2","p3","p4")); setorder(oof, No)
  ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
  cat(">>> MIXL OOF logloss:", round(logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)])), 5), "\n")
  saveRDS(oof, "model/artifacts/oof_mixl.rds")
  fit <- fit_mixl(trl)
  tel <- long[is_test == TRUE]; tel[, chosen := FALSE]
  tp_l <- predict_uncond(fit, tel)
  tp <- dcast(tp_l, No ~ alt, value.var = "p"); setnames(tp, c("No","p1","p2","p3","p4")); setorder(tp, No)
  saveRDS(tp, "model/artifacts/test_mixl.rds")
  saveRDS(coef(fit), "model/artifacts/mixl_full_coefs.rds")
  cat("OK\n")
}
```

- [ ] **Step 2: Launch folds 1–5 as background processes (2–3 at a time), then combine**

Run (background, per fold): `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/03_mixl.R 1` … `5`, then `... model/03_mixl.R 0`
Expected: each fold prints positive rpar sds; combined OOF ≈ 1.13–1.17 (should beat plain MNL by ≥ 0.01; if not, note it — heterogeneity may be mostly in ASC).
Timebox: if a single fold exceeds ~45 min, kill, drop to `R = 50` draws and/or `rpar = c(Price="n")` only, rerun.

---

### Task 8: Blend + nested evaluation (`model/06_blend.R`)

**Files:**
- Create: `model/06_blend.R`

**Interfaces:**
- Consumes every `oof_<m>.rds` / `test_<m>.rds` present (auto-discovers members).
- Produces: `model/artifacts/blend.rds` (final weights, temperature T, uniform-mix ε, member list), `model/artifacts/test_blend.rds` (final test matrix), prints **nested OOF** (the decision number) and plain OOF.

- [ ] **Step 1: Write `model/06_blend.R`**

```r
suppressMessages(library(data.table))
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

memb <- gsub("^oof_|\\.rds$", "", list.files("model/artifacts", pattern = "^oof_.*\\.rds$"))
memb <- memb[file.exists(sprintf("model/artifacts/test_%s.rds", memb))]
cat("members:", paste(memb, collapse = ", "), "\n")
OOF <- lapply(memb, function(m) { d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
TST <- lapply(memb, function(m) { d <- readRDS(sprintf("model/artifacts/test_%s.rds", m)); setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)]) })
names(OOF) <- names(TST) <- memb

ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
y <- ymap$y
fmap <- folds[order(No), fold]
M <- length(memb); eps0 <- 1e-12

blend <- function(theta, Ps, rows = TRUE) {
  w <- exp(theta[1:M]); w <- w / sum(w)
  Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, Ps))
  L <- L / Tt
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
obj <- function(theta, rows) { P <- blend(theta, OOF, rows); logloss(y[rows], P) }
fit_w <- function(rows) optim(c(rep(0, M), 0, -3), obj, rows = rows,
                              method = "Nelder-Mead", control = list(maxit = 3000))

nested <- numeric(5)
for (k in 1:5) {
  o <- fit_w(fmap != k)
  nested[k] <- obj(o$par, fmap == k)
  cat("fold", k, "held-out blend logloss:", round(nested[k], 5), "\n")
}
cat(">>> NESTED blend OOF (decision number):", round(mean(nested), 5), "±", round(sd(nested), 5), "\n")

o <- fit_w(rep(TRUE, length(y)))
w <- exp(o$par[1:M]); w <- w / sum(w)
cat("plain OOF:", round(o$value, 5), " weights:", paste(memb, round(w, 3), collapse = " "),
    " T:", round(exp(o$par[M+1]), 3), " eps:", round(plogis(o$par[M+2]) * 0.10, 4), "\n")
for (m in memb) cat(sprintf("  single %-12s OOF: %.5f\n", m, logloss(y, OOF[[m]])))
saveRDS(list(members = memb, par = o$par, nested = mean(nested)), "model/artifacts/blend.rds")
saveRDS(clip_norm(blend(o$par, TST, rows = seq_len(nrow(TST[[1]])))), "model/artifacts/test_blend.rds")
cat("OK\n")
```

*(Note: `blend(..., TST, rows=...)` indexes test rows; all TST matrices are 4,997×4 ordered by No.)*

- [ ] **Step 2: Run after ≥2 members exist; rerun as members are added**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/06_blend.R`
Expected: nested OOF < best single member; target ≤ 1.13. Weights non-degenerate (no member at ~1.0 unless it truly dominates).

---

### Task 9: Submission writer (`model/07_submit.R`)

**Files:**
- Create: `model/07_submit.R`, `submissions/log.md`

**Interfaces:**
- Consumes: `test_blend.rds`, `long.rds`. Produces `submissions/sub_<yyyymmdd_HHMM>.csv` + appended log row.

- [ ] **Step 1: Write `model/07_submit.R`**

```r
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
P <- readRDS("model/artifacts/test_blend.rds")
b <- readRDS("model/artifacts/blend.rds")
nos <- sort(unique(long[is_test == TRUE, No]))
stopifnot(nrow(P) == 4997, length(nos) == 4997, min(nos) == 21566, max(nos) == 26562)
P <- clip_norm(P)
stopifnot(all(abs(rowSums(P) - 1) < 1e-8), all(P > 0), all(P < 1))
sub <- data.table(No = nos, Ch1 = P[,1], Ch2 = P[,2], Ch3 = P[,3], Ch4 = P[,4])
f <- sprintf("submissions/sub_%s.csv", format(Sys.time(), "%Y%m%d_%H%M"))
dir.create("submissions", showWarnings = FALSE)
fwrite(sub, f)
cat(sprintf("| %s | %s | %s | %.5f | %+0.3f offset => expect public ~%.3f | (pending) |\n",
    format(Sys.time(), "%Y-%m-%d %H:%M"), basename(f),
    paste(b$members, collapse = "+"), b$nested, 0.05, b$nested + 0.05),
    file = "submissions/log.md", append = TRUE)
cat("wrote", f, "\n")
```

- [ ] **Step 2: Create `submissions/log.md` header**

```markdown
# Submission log — record public score after every Kaggle upload
| when | file | members | nested OOF | expectation | public |
|---|---|---|---|---|---|
```

- [ ] **Step 3: Run + verify format against sample_submission2024.csv**

Run: `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/07_submit.R` then compare header + row count with `Raw Dump/Competition Data/sample_submission2024.csv`.
Expected: file with 4,998 lines (header + 4,997), header `No,Ch1,Ch2,Ch3,Ch4`.

---

### Task 10: Covariate-shift check + report notes seed

**Files:**
- Create: `model/08_shift_check.R`, `report_notes.md`

- [ ] **Step 1: Write `model/08_shift_check.R`**

```r
suppressMessages(library(data.table))
wide <- readRDS("model/artifacts/wide.rds")
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
resp <- unique(wide[, c("Case","is_test", demo), with = FALSE])
tab <- resp[, lapply(.SD, mean), by = is_test, .SDcols = demo]
print(t(tab), digits = 3)
fwrite(tab, "model/artifacts/shift_check.csv")
```

- [ ] **Step 2: Run; paste the table + interpretation into `report_notes.md`** (seed the file with sections: Data, Validation design, Models tried, Blend, Public/private fit, Insights (coefficients/WTP), Limitations.)

Expected: means broadly similar train vs test; any large gap → note as explanation for local→LB offset and consider stratification (do NOT jump to reweighting without evidence it helps nested OOF).

---

## Self-Review

- **Spec coverage:** harness+folds (T1–3), MNL (T4), xgb-long (T5), xgb-wide (T6), mixed logit (T7), nested blend (T8), submission+log (T9), shift check + report notes (T10). Sheil/Kavya members: additive — same `(oof_*, test_*)` contract, added when scripts arrive. Vault: separate effort, not in this plan (content generation, not software).
- **Placeholder scan:** one intentionally flagged placeholder block in Task 7 Step 1 is immediately superseded by the final `predict_uncond` given below it — implementer uses the second version verbatim.
- **Type consistency:** artifact contract `(No, p1..p4)` data.table ordered by `No` used identically in T4–T8; `logloss(y, P)` signature consistent; `folds.rds` columns `(No, Case, fold)` consumed by name everywhere.
