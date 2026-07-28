# Full-Test Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an R-only, procedure-level EM-start variance audit, adopt latent-class
bagging only when it replicates, and emit one measured-marginal-corrected final submission.

**Architecture:** Add opt-in experiment hooks to the existing latent-class runner while
preserving its default outputs byte-for-byte in intent. Store each long-running fit as a
self-describing bundle, evaluate expected single-start versus bagged nested-blend procedures
on two respondent-grouped fold structures, and let a deterministic finalizer select either
the accepted bag or the unchanged production blend before applying the measured alternative-4
tilt.

**Tech Stack:** R 4.6.0, `data.table`, base R `optim`, existing conditional-logit/EM code,
PowerShell only for launching R commands.

## Global Constraints

- R only; do not introduce Python.
- Never regenerate `model/artifacts/folds.rds`, `folds_b.rds`, or split rows rather than
  respondents.
- The five pre-registered seeds are exactly `4242, 1103, 2207, 3301, 4409`.
- The only model change is the random EM start; all features, penalties, iteration budgets,
  convergence settings, and task-position terms remain fixed.
- Per-seed outputs stay under `experiments/iter36_emstart/` and must not overwrite production
  artifacts.
- Arithmetic probability averaging is fixed before seeing results.
- Gate thresholds are fixed by the design specification and must not be retuned.
- Do not upload or select a model based on public leaderboard performance.
- Preserve the user's existing dirty changes. In particular, do not stage the pre-existing
  edits to `CLAUDE.md`, `EXPERIMENTS.md`, `submissions/log.md`, or
  `experiments/iter11_latent_class/fit_C3.rds`.
- Use `& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe"` from the repository root.

---

## File structure

- Modify `experiments/iter25_taskpos/run.R`: add opt-in seed/output hooks; historical defaults
  remain unchanged.
- Create `experiments/iter36_emstart/common.R`: seed constants, paths, prediction and bundle
  validators, averaging, and atomic experiment writes.
- Create `experiments/iter36_emstart/test_common.R`: base-R contract tests.
- Create `experiments/iter36_emstart/run_seed.R`: resumable launcher for one seed and one fold
  structure.
- Create `experiments/iter36_emstart/import_reference.R`: package the existing seed-4242
  primary and fold-B artifacts without refitting them.
- Create `experiments/iter36_emstart/evaluate.R`: nested procedure comparison, shift audit,
  fold concentration audit, gate decision, and promotion artifacts.
- Create `experiments/iter36_emstart/test_evaluate.R`: deterministic gate and nested-prediction
  tests.
- Create `experiments/iter36_emstart/finalize.R`: choose the permitted base, apply the measured
  alternative-4 tilt, validate, and write one submission.
- Create `experiments/iter36_emstart/test_finalize.R`: tilt invariance and fallback tests.
- Modify `EXPERIMENTS.md`: append iteration 36's pre-registered hypothesis before compute and
  its result after compute; leave the already-dirty file unstaged unless its prior owner
  changes have first been committed.

---

### Task 1: Prediction-bundle contracts

**Files:**
- Create: `experiments/iter36_emstart/common.R`
- Create: `experiments/iter36_emstart/test_common.R`

**Interfaces:**
- Consumes: R `data.table`s with `No, p1, p2, p3, p4`.
- Produces:
  - `EM_SEEDS`: integer vector of length five.
  - `bundle_path(seed, split = "") -> character(1)`.
  - `validate_pred(x, expected_no, label) -> data.table`.
  - `validate_bundle(x, seed, split, expected_no, require_test,
    expected_settings = NULL) -> list`.
  - `mean_pred(xs, expected_no, label) -> data.table`.
  - `save_experiment_rds(x, path) -> invisible(path)`.

- [ ] **Step 1: Write failing contract tests**

Create `experiments/iter36_emstart/test_common.R`:

```r
suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")

expect_error <- function(expr) {
  ok <- FALSE
  tryCatch(force(expr), error = function(e) ok <<- TRUE)
  stopifnot(ok)
}

stopifnot(identical(EM_SEEDS, c(4242L, 1103L, 2207L, 3301L, 4409L)))
stopifnot(identical(bundle_path(1103L, ""), file.path(
  "experiments", "iter36_emstart", "artifacts", "seed_1103.rds")))
stopifnot(identical(bundle_path(1103L, "b"), file.path(
  "experiments", "iter36_emstart", "artifacts", "seed_1103_b.rds")))

nos <- 1:3
p <- data.table(No = nos, p1 = c(.1, .2, .3), p2 = .2, p3 = .3,
                p4 = c(.4, .3, .2))
p[, p1 := 1 - p2 - p3 - p4]
v <- validate_pred(p, nos, "synthetic")
stopifnot(identical(v$No, nos), max(abs(rowSums(v[, .(p1,p2,p3,p4)]) - 1)) < 1e-12)

bad_order <- copy(p)[3:1]
expect_error(validate_pred(bad_order, nos, "bad order"))
bad_sum <- copy(p); bad_sum[1, p1 := p1 + .01]
expect_error(validate_pred(bad_sum, nos, "bad sum"))
bad_zero <- copy(p); bad_zero[1, `:=`(p1 = 0, p4 = p4 + p1)]
expect_error(validate_pred(bad_zero, nos, "bad zero"))

m <- mean_pred(list(p, p), nos, "mean")
stopifnot(max(abs(as.matrix(m[, .(p1,p2,p3,p4)]) -
                  as.matrix(p[, .(p1,p2,p3,p4)]))) < 1e-12)

cat("test_common.R: OK\n")
```

- [ ] **Step 2: Run the test and confirm the missing-file failure**

Run:

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/test_common.R
```

Expected: failure stating that `experiments/iter36_emstart/common.R` does not exist.

- [ ] **Step 3: Implement the contract library**

Create `experiments/iter36_emstart/common.R`:

```r
suppressMessages(library(data.table))

EM_SEEDS <- c(4242L, 1103L, 2207L, 3301L, 4409L)
P_COLS <- paste0("p", 1:4)
ITER36_DIR <- file.path("experiments", "iter36_emstart")
ARTIFACT_DIR <- file.path(ITER36_DIR, "artifacts")

bundle_path <- function(seed, split = "") {
  stopifnot(length(seed) == 1L, seed %in% EM_SEEDS, split %in% c("", "b"))
  file.path(ARTIFACT_DIR, sprintf("seed_%04d%s.rds", seed,
                                  if (nzchar(split)) paste0("_", split) else ""))
}

validate_pred <- function(x, expected_no, label) {
  if (!is.data.table(x)) x <- as.data.table(x)
  if (!identical(names(x), c("No", P_COLS)))
    stop(label, ": columns must be exactly No,p1,p2,p3,p4")
  if (!identical(x$No, expected_no)) stop(label, ": No identifiers/order differ")
  P <- as.matrix(x[, ..P_COLS])
  if (anyNA(P) || any(!is.finite(P))) stop(label, ": non-finite probability")
  if (any(P <= 0)) stop(label, ": probabilities must be strictly positive")
  if (max(abs(rowSums(P) - 1)) > 1e-9) stop(label, ": row sums differ from one")
  x[]
}

settings_fingerprint <- function(seed, split, code_md5, n_screen, n_max, tol) {
  list(seed = as.integer(seed), split = split, C = 3L, taskmode = "both",
       lambda_b = 2, lambda_g = 2, n_screen = as.integer(n_screen),
       n_max = as.integer(n_max), tol = as.numeric(tol), code_md5 = code_md5)
}

validate_bundle <- function(x, seed, split, expected_no, require_test = split == "",
                            expected_settings = NULL) {
  need <- c("seed", "split", "settings", "oof", "test")
  if (!is.list(x) || !all(need %in% names(x))) stop("invalid bundle structure")
  if (!identical(as.integer(x$seed), as.integer(seed))) stop("bundle seed mismatch")
  if (!identical(x$split, split)) stop("bundle split mismatch")
  core <- c("seed","split","C","taskmode","lambda_b","lambda_g",
            "n_screen","n_max","tol","code_md5")
  if (!all(core %in% names(x$settings))) stop("bundle settings fingerprint is incomplete")
  if (x$settings$seed != seed || x$settings$split != split ||
      x$settings$C != 3L || x$settings$taskmode != "both" ||
      x$settings$lambda_b != 2 || x$settings$lambda_g != 2 ||
      x$settings$n_screen != 3L || x$settings$n_max != 25L ||
      x$settings$tol != 1e-5)
    stop("bundle settings do not match the pre-registered procedure")
  if (!is.null(expected_settings) && !identical(x$settings, expected_settings))
    stop("bundle settings fingerprint differs from this runner")
  x$oof <- validate_pred(x$oof, expected_no$train, "bundle oof")
  if (require_test) {
    x$test <- validate_pred(x$test, expected_no$test, "bundle test")
  } else if (!is.null(x$test)) {
    x$test <- validate_pred(x$test, expected_no$test, "optional bundle test")
  }
  x
}

mean_pred <- function(xs, expected_no, label) {
  if (!length(xs)) stop(label, ": no predictions")
  xs <- lapply(seq_along(xs), function(i)
    validate_pred(xs[[i]], expected_no, sprintf("%s[%d]", label, i)))
  P <- Reduce(`+`, lapply(xs, function(x) as.matrix(x[, ..P_COLS]))) / length(xs)
  P <- P / rowSums(P)
  validate_pred(data.table(No = expected_no, p1 = P[,1], p2 = P[,2],
                           p3 = P[,3], p4 = P[,4]), expected_no, label)
}

save_experiment_rds <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(path)) stop("refusing to overwrite existing experiment artifact: ", path)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  on.exit(if (file.exists(tmp)) unlink(tmp), add = TRUE)
  saveRDS(x, tmp)
  if (!file.rename(tmp, path)) stop("atomic rename failed for ", path)
  invisible(path)
}
```

- [ ] **Step 4: Run the contract tests**

Run the command from Step 2.

Expected: `test_common.R: OK`.

- [ ] **Step 5: Commit the isolated contract**

```powershell
git add -- experiments/iter36_emstart/common.R experiments/iter36_emstart/test_common.R
git commit -m "test: define EM-start artifact contracts"
```

---

### Task 2: Opt-in EM runner and resumable seed launcher

**Files:**
- Modify: `experiments/iter25_taskpos/run.R:114-154,483-506,567-633`
- Create: `experiments/iter36_emstart/run_seed.R`
- Create: `experiments/iter36_emstart/import_reference.R`
- Test: `experiments/iter36_emstart/test_common.R`

**Interfaces:**
- Consumes:
  - `LC_RANDOM_START_SEED=<integer>`;
  - `LC_EXPERIMENT_OUT=<path>`;
  - `LC_SKIP_FULL_TEST=0|1`.
- Produces: one bundle at `bundle_path(seed, split)` containing `seed`, `split`, `settings`,
  `oof`, and `test`; fold-B bundles have `test = NULL`.

- [ ] **Step 1: Extend the test with bundle validation**

Append before the final `cat()` in `test_common.R`:

```r
b <- list(seed = 1103L, split = "",
          settings = settings_fingerprint(1103L, "", "test-md5", 3L, 25L, 1e-5),
          oof = p, test = p)
vb <- validate_bundle(b, 1103L, "", list(train = nos, test = nos), TRUE)
stopifnot(vb$seed == 1103L, nrow(vb$oof) == 3L, nrow(vb$test) == 3L)
expect_error(validate_bundle(b, 2207L, "", list(train = nos, test = nos), TRUE))
```

- [ ] **Step 2: Run tests and confirm the new checks pass against the existing validator**

Run `test_common.R`.

Expected: `test_common.R: OK`.

- [ ] **Step 3: Add opt-in settings to the latent-class runner**

Immediately after `OUTDIR <- "experiments/iter11_latent_class"` in
`experiments/iter25_taskpos/run.R`, insert:

```r
RANDOM_START_SEED <- as.integer(Sys.getenv("LC_RANDOM_START_SEED", "4242"))
EXPERIMENT_OUT <- trimws(Sys.getenv("LC_EXPERIMENT_OUT", ""))
SKIP_FULL_TEST <- identical(Sys.getenv("LC_SKIP_FULL_TEST", "0"), "1")
if (!is.finite(RANDOM_START_SEED)) stop("LC_RANDOM_START_SEED must be an integer")
if (SKIP_FULL_TEST && !nzchar(EXPERIMENT_OUT))
  stop("LC_SKIP_FULL_TEST is allowed only with LC_EXPERIMENT_OUT")
if (SKIP_FULL_TEST && SPLIT != "b")
  stop("LC_SKIP_FULL_TEST is allowed only for split b")
```

Replace the hard-coded random start:

```r
starts <- list(list(kind = "det", seed = 0),
               list(kind = "rand", seed = RANDOM_START_SEED))
```

Replace the immediate OOF and test writes with in-memory tables:

```r
oof <- clip_norm(oof)
oof_dt <- data.table(No = tasks$No[tr_t], p1 = oof[,1], p2 = oof[,2],
                     p3 = oof[,3], p4 = oof[,4])

te_dt <- NULL
full <- NULL
if (!SKIP_FULL_TEST) {
  cat("--- full refit ---\n")
  full <- fit_one(tr_t)
  te <- clip_norm(predict_lc(full, te_t))
  te_dt <- data.table(No = tasks$No[te_t], p1 = te[,1], p2 = te[,2],
                      p3 = te[,3], p4 = te[,4])
}
```

Guard the existing segment-report block with:

```r
if (C > 1 && !is.null(full)) {
```

Guard its historical fit-object write with `if (!nzchar(EXPERIMENT_OUT))`, retaining the
existing `saveRDS()` body.

Replace the final `cat("OK\n")` with:

```r
if (nzchar(EXPERIMENT_OUT)) {
  source("experiments/iter36_emstart/common.R")
  expected <- list(train = sort(tasks$No[tr_t]), test = sort(tasks$No[te_t]))
  oof_dt <- validate_pred(oof_dt[order(No)], expected$train, "runner oof")
  if (!is.null(te_dt)) te_dt <- validate_pred(te_dt[order(No)], expected$test, "runner test")
  bundle <- list(
    seed = RANDOM_START_SEED,
    split = SPLIT,
    settings = settings_fingerprint(
      RANDOM_START_SEED, SPLIT,
      unname(tools::md5sum("experiments/iter25_taskpos/run.R")),
      N_SCREEN, N_MAX, TOL),
    oof = oof_dt,
    test = te_dt
  )
  save_experiment_rds(bundle, EXPERIMENT_OUT)
  cat("wrote experiment bundle:", EXPERIMENT_OUT, "\n")
} else {
  saveRDS(oof_dt, sprintf("model/artifacts/oof_%s.rds", NAME))
  saveRDS(te_dt, sprintf("model/artifacts/test_%s.rds", NAME))
}
cat("OK\n")
```

The default environment leaves `EXPERIMENT_OUT` empty, retains seed 4242, performs the full
refit, and writes the historical paths.

- [ ] **Step 4: Create the resumable launcher**

Create `experiments/iter36_emstart/run_seed.R`:

```r
suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% 1:2) stop("usage: run_seed.R <seed> [b]")
seed <- as.integer(args[1])
split <- if (length(args) == 2) args[2] else ""
if (!seed %in% EM_SEEDS) stop("seed is not pre-registered")
if (!split %in% c("", "b")) stop("split must be b")

long <- readRDS("model/artifacts/long.rds")
tasks <- unique(long[, .(No, is_test)], by = "No")
expected <- list(train = sort(tasks[is_test == FALSE, No]),
                 test = sort(tasks[is_test == TRUE, No]))
out <- bundle_path(seed, split)
expected_settings <- settings_fingerprint(
  seed, split, unname(tools::md5sum("experiments/iter25_taskpos/run.R")),
  3L, 25L, 1e-5)
if (file.exists(out)) {
  validate_bundle(readRDS(out), seed, split, expected, require_test = split == "",
                  expected_settings = expected_settings)
  cat("valid bundle already exists; skipping:", out, "\n")
  quit(save = "no")
}

runner <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
runner_args <- c("experiments/iter25_taskpos/run.R", "3", "both",
                 if (nzchar(split)) split else character())
env <- c(sprintf("LC_RANDOM_START_SEED=%d", seed),
         sprintf("LC_EXPERIMENT_OUT=%s", normalizePath(out, winslash = "/", mustWork = FALSE)),
         sprintf("LC_SKIP_FULL_TEST=%s", if (split == "b") "1" else "0"))
logfile <- file.path(ITER36_DIR, sprintf("seed_%04d%s.log", seed,
                    if (nzchar(split)) "_b" else ""))
status <- system2(runner, runner_args, stdout = logfile, stderr = logfile, env = env)
if (status != 0L) stop("latent-class runner failed; inspect ", logfile)
validate_bundle(readRDS(out), seed, split, expected, require_test = split == "",
                expected_settings = expected_settings)
cat("completed and validated:", out, "\n")
```

- [ ] **Step 5: Create the reference importer**

Create `experiments/iter36_emstart/import_reference.R`:

```r
suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")

long <- readRDS("model/artifacts/long.rds")
tasks <- unique(long[, .(No, is_test)], by = "No")
expected <- list(train = sort(tasks[is_test == FALSE, No]),
                 test = sort(tasks[is_test == TRUE, No]))

import_one <- function(split) {
  out <- bundle_path(4242L, split)
  if (file.exists(out)) {
    validate_bundle(readRDS(out), 4242L, split, expected, require_test = split == "")
    return(invisible(out))
  }
  suf <- if (nzchar(split)) "_b" else ""
  oof <- readRDS(sprintf("model/artifacts/oof_lcmnl3_both%s.rds", suf))
  test <- if (nzchar(split)) NULL else readRDS("model/artifacts/test_lcmnl3_both.rds")
  b <- list(seed = 4242L, split = split,
            settings = settings_fingerprint(
              4242L, split, unname(tools::md5sum("experiments/iter25_taskpos/run.R")),
              3L, 25L, 1e-5),
            oof = validate_pred(oof[order(No)], expected$train, "reference oof"),
            test = if (is.null(test)) NULL else
              validate_pred(test[order(No)], expected$test, "reference test"))
  save_experiment_rds(b, out)
}

import_one("")
import_one("b")
cat("import_reference.R: OK\n")
```

- [ ] **Step 6: Parse and contract-test the runner**

Run:

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" -e "parse(file='experiments/iter25_taskpos/run.R'); parse(file='experiments/iter36_emstart/run_seed.R'); parse(file='experiments/iter36_emstart/import_reference.R')"
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/test_common.R
```

Expected: all files parse and `test_common.R: OK`.

- [ ] **Step 7: Commit only the runner code**

```powershell
git add -- experiments/iter25_taskpos/run.R experiments/iter36_emstart/run_seed.R experiments/iter36_emstart/import_reference.R experiments/iter36_emstart/test_common.R
git commit -m "feat: make latent-class EM starts reproducible"
```

---

### Task 3: Procedure-level evaluator and fixed gates

**Files:**
- Create: `experiments/iter36_emstart/evaluate.R`
- Create: `experiments/iter36_emstart/test_evaluate.R`

**Interfaces:**
- Consumes: five validated primary bundles and, after Gate 1, five validated fold-B bundles.
- Produces:
  - `nested_blend(Ptree, Plc, y, fold) -> list(score, fold_loss, pred)`;
  - `gate_decision(metrics) -> list(pass, gates, reasons)`;
  - `experiments/iter36_emstart/evaluation.rds`;
  - `experiments/iter36_emstart/per_fold.csv`;
  - standard candidate member artifacts only when all gates pass.

- [ ] **Step 1: Write deterministic gate tests**

Create `experiments/iter36_emstart/test_evaluate.R`:

```r
source("experiments/iter36_emstart/evaluate.R", local = TRUE)

good <- list(start_sd = .0012, primary_gain = .0004, weighted_gain = .0002,
             b_gain = .0003, replication = .75,
             fold_gain = c(rep(.0002, 8), -.0002, -.0003),
             clustered_gain = .0003)
d <- gate_decision(good)
stopifnot(d$pass, all(unlist(d$gates)))

x <- good; x$start_sd <- .0009
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$material_variance)
x <- good; x$b_gain <- -.0001
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$independent_positive)
x <- good; x$replication <- .49
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$replication_half)
x <- good; x$fold_gain <- c(rep(.0002, 6), rep(-.0002, 4))
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$seven_folds)
x <- good; x$fold_gain[10] <- -.00101
stopifnot(!gate_decision(x)$pass, !gate_decision(x)$gates$no_large_fold_loss)

cat("test_evaluate.R: OK\n")
```

At the top of `evaluate.R`, execution must be guarded by:

```r
RUN_MAIN <- identical(environment(), globalenv())
```

and the artifact-loading main body must run only inside `if (RUN_MAIN)`, so sourcing the file
for tests defines functions without requiring long-run artifacts.

- [ ] **Step 2: Run the test and confirm the missing-file failure**

Run:

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/test_evaluate.R
```

Expected: failure because `evaluate.R` does not exist.

- [ ] **Step 3: Implement the nested blend and gate functions**

Create `experiments/iter36_emstart/evaluate.R` with these function contracts:

```r
suppressMessages(library(data.table))
source("model/99_utils.R")
source("experiments/iter36_emstart/common.R")
RUN_MAIN <- identical(environment(), globalenv())

nested_blend <- function(Ptree, Plc, y, fold) {
  stopifnot(nrow(Ptree) == length(y), nrow(Plc) == length(y),
            length(fold) == length(y), identical(sort(unique(fold)), 1:5))
  Ps <- list(Ptree, Plc); M <- 2L
  blend <- function(theta, rows) {
    w <- exp(theta[1:M]); w <- w / sum(w)
    Tt <- exp(theta[M + 1L]); eps <- plogis(theta[M + 2L]) * .10
    L <- w[1] * log(pmax(Ps[[1]][rows,,drop=FALSE], 1e-12)) +
         w[2] * log(pmax(Ps[[2]][rows,,drop=FALSE], 1e-12))
    L <- L / Tt
    Q <- exp(L - apply(L, 1, max)); Q <- Q / rowSums(Q)
    (1 - eps) * Q + eps * .25
  }
  pred <- matrix(NA_real_, length(y), 4)
  fold_loss <- numeric(5)
  for (k in 1:5) {
    tr <- fold != k; va <- fold == k
    obj <- function(th) logloss(y[tr], blend(th, tr))
    fit <- optim(c(0,0,0,-3), obj, method = "Nelder-Mead",
                 control = list(maxit = 3000))
    pred[va,] <- blend(fit$par, va)
    fold_loss[k] <- logloss(y[va], pred[va,,drop=FALSE])
  }
  list(score = mean(-log(pmax(pred[cbind(seq_along(y), y)], 1e-15))),
       fold_loss = fold_loss, pred = pred)
}

gate_decision <- function(m) {
  gates <- list(
    material_variance = m$start_sd >= .001,
    primary_positive = m$primary_gain > 0,
    weighted_nonnegative = m$weighted_gain >= 0,
    independent_positive = m$b_gain > 0,
    replication_half = m$replication >= .50,
    seven_folds = sum(m$fold_gain > 0) >= 7,
    no_large_fold_loss = min(m$fold_gain) >= -.001,
    clustered_positive = m$clustered_gain > 0
  )
  list(pass = all(unlist(gates)), gates = gates,
       reasons = names(gates)[!unlist(gates)])
}
```

After the function definitions, add this main body:

```r
if (RUN_MAIN) {
  long <- readRDS("model/artifacts/long.rds")
  tasks <- unique(long[, .(No, Case, is_test, y)], by = "No"); setorder(tasks, No)
  tr <- tasks[is_test == FALSE]; te <- tasks[is_test == TRUE]
  expected <- list(train = tr$No, test = te$No)
  y <- tr$y

  get_bundle <- function(seed, split)
    validate_bundle(readRDS(bundle_path(seed, split)), seed, split, expected,
                    require_test = split == "")
  primary <- lapply(EM_SEEDS, get_bundle, split = "")
  names(primary) <- EM_SEEDS
  PlcA <- lapply(primary, function(z) as.matrix(z$oof[, ..P_COLS]))
  model_loss <- vapply(PlcA, function(P) logloss(y, P), numeric(1))
  start_sd <- sd(model_loss)
  eval_path <- file.path(ITER36_DIR, "evaluation.rds")

  if (start_sd < .001) {
    decision <- list(pass = FALSE, gates = list(material_variance = FALSE),
                     reasons = "material_variance")
    saveRDS(list(needs_fold_b = FALSE, model_loss = model_loss, start_sd = start_sd,
                 decision = decision), eval_path)
    cat(sprintf("Gate 1 REJECT: start SD %.6f < 0.001\n", start_sd))
    quit(save = "no")
  }

  missing_b <- EM_SEEDS[!file.exists(vapply(EM_SEEDS, bundle_path, character(1), split = "b"))]
  if (length(missing_b)) {
    saveRDS(list(needs_fold_b = TRUE, missing_b = missing_b,
                 model_loss = model_loss, start_sd = start_sd), eval_path)
    cat("Gate 1 PASS; run fold-B seeds:", paste(missing_b, collapse = ", "), "\n")
    quit(save = "no")
  }

  secondary <- lapply(EM_SEEDS, get_bundle, split = "b")
  names(secondary) <- EM_SEEDS
  PlcB <- lapply(secondary, function(z) as.matrix(z$oof[, ..P_COLS]))
  bagA_dt <- mean_pred(lapply(primary, `[[`, "oof"), expected$train, "primary bag")
  bagB_dt <- mean_pred(lapply(secondary, `[[`, "oof"), expected$train, "fold-B bag")
  test_bag_dt <- mean_pred(lapply(primary, `[[`, "test"), expected$test, "test bag")
  bagA <- as.matrix(bagA_dt[, ..P_COLS]); bagB <- as.matrix(bagB_dt[, ..P_COLS])

  get_oof <- function(name) {
    x <- readRDS(sprintf("model/artifacts/oof_%s.rds", name)); setorder(x, No)
    as.matrix(validate_pred(x, expected$train, name)[, ..P_COLS])
  }
  treeA <- get_oof("xgb_lw2bag")
  treeB <- get_oof("xgb_lw2_b")
  foldA <- readRDS("model/artifacts/folds.rds")[order(No), fold]
  foldB <- readRDS("model/artifacts/folds_b.rds")[order(No), fold]

  oneA <- lapply(PlcA, function(P) nested_blend(treeA, P, y, foldA))
  oneB <- lapply(PlcB, function(P) nested_blend(treeB, P, y, foldB))
  avgA <- nested_blend(treeA, bagA, y, foldA)
  avgB <- nested_blend(treeB, bagB, y, foldB)
  loss_vec <- function(z) -log(pmax(z$pred[cbind(seq_along(y), y)], 1e-15))
  single_loss_A <- Reduce(`+`, lapply(oneA, loss_vec)) / length(oneA)
  single_loss_B <- Reduce(`+`, lapply(oneB, loss_vec)) / length(oneB)
  bag_loss_A <- loss_vec(avgA); bag_loss_B <- loss_vec(avgB)
  effectA <- single_loss_A - bag_loss_A
  effectB <- single_loss_B - bag_loss_B
  primary_gain <- mean(effectA)
  b_gain <- mean(effectB)

  wide <- readRDS("model/artifacts/wide.rds")
  resp <- unique(wide[, .(Case, is_test, incomeind)])
  ntr <- resp[is_test == FALSE, .N]; nte <- resp[is_test == TRUE, .N]
  wt <- merge(resp[is_test == FALSE, .(ptr = .N / ntr), by = incomeind],
              resp[is_test == TRUE, .(pte = .N / nte), by = incomeind],
              by = "incomeind", all.x = TRUE)
  wt[is.na(pte), pte := 0]
  wt[, w := pmin(pmax(pte / ptr, .2), 5)]
  rw <- merge(unique(wide[is_test == FALSE, .(No, incomeind)]),
              wt[, .(incomeind, w)], by = "incomeind")[order(No), w]
  weighted_gain <- sum(rw * effectA) / sum(rw)

  gainA <- colMeans(do.call(rbind, lapply(oneA, `[[`, "fold_loss"))) - avgA$fold_loss
  gainB <- colMeans(do.call(rbind, lapply(oneB, `[[`, "fold_loss"))) - avgB$fold_loss
  fold_gain <- c(gainA, gainB)
  per_fold <- rbind(
    data.table(split = "primary", fold = 1:5,
               mean_single = colMeans(do.call(rbind, lapply(oneA, `[[`, "fold_loss"))),
               bagged = avgA$fold_loss, gain = gainA),
    data.table(split = "b", fold = 1:5,
               mean_single = colMeans(do.call(rbind, lapply(oneB, `[[`, "fold_loss"))),
               bagged = avgB$fold_loss, gain = gainB))

  by_case <- data.table(Case = tr$Case, effect = (effectA + effectB) / 2)[,
    .(effect = mean(effect)), by = Case]
  clustered_gain <- mean(by_case$effect)
  clustered_se <- sd(by_case$effect) / sqrt(nrow(by_case))
  metrics <- list(start_sd = start_sd, primary_gain = primary_gain,
                  weighted_gain = weighted_gain, b_gain = b_gain,
                  replication = b_gain / primary_gain, fold_gain = fold_gain,
                  clustered_gain = clustered_gain, clustered_se = clustered_se)
  decision <- gate_decision(metrics)
  evaluation <- list(needs_fold_b = FALSE, model_loss = model_loss,
                     primary_single_nested = vapply(oneA, `[[`, numeric(1), "score"),
                     b_single_nested = vapply(oneB, `[[`, numeric(1), "score"),
                     primary_bagged_nested = avgA$score,
                     b_bagged_nested = avgB$score,
                     metrics = metrics, decision = decision)
  saveRDS(evaluation, eval_path)
  fwrite(per_fold, file.path(ITER36_DIR, "per_fold.csv"))

  if (decision$pass) {
    saveRDS(bagA_dt, "model/artifacts/oof_lcmnl3_bothbag.rds")
    saveRDS(test_bag_dt, "model/artifacts/test_lcmnl3_bothbag.rds")
    saveRDS(bagB_dt, "model/artifacts/oof_lcmnl3_bothbag_b.rds")
    validate_pred(readRDS("model/artifacts/oof_lcmnl3_bothbag.rds"),
                  expected$train, "promoted primary bag")
    validate_pred(readRDS("model/artifacts/test_lcmnl3_bothbag.rds"),
                  expected$test, "promoted test bag")
    validate_pred(readRDS("model/artifacts/oof_lcmnl3_bothbag_b.rds"),
                  expected$train, "promoted fold-B bag")
  }
  print(unlist(decision$gates))
  cat(if (decision$pass) "PASS\n" else
      paste("REJECT:", paste(decision$reasons, collapse = ", "), "\n"))
}
```

The script writes standard member artifacts only after a full pass, reads them back, and
validates them before printing `PASS`.

- [ ] **Step 4: Run evaluator unit tests**

Run `test_evaluate.R`.

Expected: `test_evaluate.R: OK`.

- [ ] **Step 5: Parse the evaluator**

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" -e "parse(file='experiments/iter36_emstart/evaluate.R')"
```

Expected: exit code 0.

- [ ] **Step 6: Commit evaluator and tests**

```powershell
git add -- experiments/iter36_emstart/evaluate.R experiments/iter36_emstart/test_evaluate.R
git commit -m "feat: evaluate EM bagging as a procedure"
```

---

### Task 4: Deterministic finalizer and marginal correction

**Files:**
- Create: `experiments/iter36_emstart/finalize.R`
- Create: `experiments/iter36_emstart/test_finalize.R`

**Interfaces:**
- Consumes:
  - `evaluation.rds` when present;
  - `model/artifacts/test_blend_emstart.rds` only after an accepted bag;
  - otherwise `model/artifacts/test_blend.rds`.
- Produces:
  - `tilt_p4(P, target) -> matrix`;
  - exactly one `submissions/sub_final_robust_20260729.csv`;
  - `experiments/iter36_emstart/final_manifest.rds`.

- [ ] **Step 1: Write tilt and fallback tests**

Create `experiments/iter36_emstart/test_finalize.R`:

```r
source("experiments/iter36_emstart/finalize.R", local = TRUE)

P <- rbind(c(.10,.20,.30,.40), c(.40,.30,.20,.10), c(.25,.25,.25,.25))
target <- .30
Q <- tilt_p4(P, target)
stopifnot(abs(mean(Q[,4]) - target) < 1e-12)
stopifnot(max(abs(rowSums(Q) - 1)) < 1e-12, all(Q > 0))
stopifnot(max(abs((Q[,1] / Q[,2]) - (P[,1] / P[,2]))) < 1e-12)
stopifnot(identical(order(Q[,4]), order(P[,4])))
stopifnot(identical(select_base(FALSE),
                    "model/artifacts/test_blend.rds"))
stopifnot(identical(select_base(TRUE),
                    "model/artifacts/test_blend_emstart.rds"))

cat("test_finalize.R: OK\n")
```

As with `evaluate.R`, `finalize.R` must use `RUN_MAIN <- identical(environment(), globalenv())`
and guard its file-writing main body.

- [ ] **Step 2: Run the test and confirm the missing-file failure**

Run `test_finalize.R`.

Expected: failure because `finalize.R` does not exist.

- [ ] **Step 3: Implement the finalizer**

Create `experiments/iter36_emstart/finalize.R`:

```r
suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")
RUN_MAIN <- identical(environment(), globalenv())
R_MEASURED <- (log(6) - 1.499) / log(3)

tilt_p4 <- function(P, target) {
  tilt <- function(alpha) {
    p4n <- alpha * P[,4] / (alpha * P[,4] + 1 - P[,4])
    s <- (1 - p4n) / (1 - P[,4])
    cbind(P[,1] * s, P[,2] * s, P[,3] * s, p4 = p4n)
  }
  alpha <- exp(uniroot(function(la) mean(tilt(exp(la))[,4]) - target,
                       c(-6, 6), tol = 1e-12)$root)
  Q <- tilt(alpha); Q / rowSums(Q)
}

select_base <- function(bag_passed) {
  if (isTRUE(bag_passed)) "model/artifacts/test_blend_emstart.rds"
  else "model/artifacts/test_blend.rds"
}
```

Add this guarded main body:

```r
if (RUN_MAIN) {
  evaluation_path <- file.path(ITER36_DIR, "evaluation.rds")
  evaluation <- if (file.exists(evaluation_path)) readRDS(evaluation_path) else NULL
  bag_passed <- !is.null(evaluation) && isTRUE(evaluation$decision$pass)
  base_path <- select_base(bag_passed)
  if (!file.exists(base_path)) stop("selected base does not exist: ", base_path)

  long <- readRDS("model/artifacts/long.rds")
  test_no <- sort(unique(long[is_test == TRUE, No]))
  raw <- readRDS(base_path)
  if (is.data.table(raw) || is.data.frame(raw)) {
    raw <- as.data.table(raw)
    if ("No" %in% names(raw)) setorder(raw, No)
    P <- as.matrix(raw[, intersect(c("p1","p2","p3","p4"), names(raw)), with = FALSE])
  } else {
    P <- as.matrix(raw)
  }
  if (!identical(dim(P), c(4997L, 4L))) stop("base must be a 4997 x 4 matrix")
  base_dt <- validate_pred(data.table(No = test_no, p1 = P[,1], p2 = P[,2],
                                      p3 = P[,3], p4 = P[,4]),
                           test_no, "selected blend")
  P <- as.matrix(base_dt[, ..P_COLS])
  Q <- tilt_p4(P, R_MEASURED)
  if (abs(mean(Q[,4]) - R_MEASURED) > 1e-12) stop("p4 target mismatch")
  if (max(abs(rowSums(Q) - 1)) > 1e-9) stop("tilted rows do not sum to one")
  if (anyNA(Q) || any(!is.finite(Q)) || any(Q <= 0)) stop("invalid tilted probability")
  if (!identical(order(Q[,4]), order(P[,4]))) stop("p4 order changed")
  if (max(abs(Q[,1] / Q[,2] - P[,1] / P[,2])) > 1e-12 ||
      max(abs(Q[,1] / Q[,3] - P[,1] / P[,3])) > 1e-12)
    stop("relative odds among alternatives 1-3 changed")

  out <- "submissions/sub_final_robust_20260729.csv"
  if (file.exists(out)) stop("refusing to overwrite existing final submission: ", out)
  fwrite(data.table(No = test_no, Ch1 = Q[,1], Ch2 = Q[,2],
                    Ch3 = Q[,3], Ch4 = Q[,4]), out)
  check <- fread(out)
  if (!identical(names(check), c("No","Ch1","Ch2","Ch3","Ch4")) ||
      !identical(check$No, test_no) || nrow(check) != 4997L)
    stop("CSV schema or identifiers changed on read-back")
  CQ <- as.matrix(check[, .(Ch1,Ch2,Ch3,Ch4)])
  if (anyNA(CQ) || any(!is.finite(CQ)) || any(CQ <= 0) ||
      max(abs(rowSums(CQ) - 1)) > 1e-9)
    stop("CSV probabilities failed read-back validation")

  manifest <- list(base = base_path, bag_passed = bag_passed,
                   target_p4 = R_MEASURED, achieved_p4 = mean(CQ[,4]),
                   csv = out, csv_md5 = unname(tools::md5sum(out)),
                   created_at = format(Sys.time(), tz = "Asia/Singapore", usetz = TRUE))
  manifest_path <- file.path(ITER36_DIR, "final_manifest.rds")
  save_experiment_rds(manifest, manifest_path)
  cat("wrote", out, "from", base_path, "with mean p4", mean(CQ[,4]), "\n")
}
```

- [ ] **Step 4: Run finalizer tests and parse checks**

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/test_finalize.R
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" -e "parse(file='experiments/iter36_emstart/finalize.R')"
```

Expected: `test_finalize.R: OK` and exit code 0.

- [ ] **Step 5: Commit the finalizer**

```powershell
git add -- experiments/iter36_emstart/finalize.R experiments/iter36_emstart/test_finalize.R
git commit -m "feat: finalize one robust corrected submission"
```

---

### Task 5: Pre-register and run the primary variance gate

**Files:**
- Modify: `EXPERIMENTS.md`
- Create at runtime: `experiments/iter36_emstart/artifacts/seed_*.rds`
- Create at runtime: `experiments/iter36_emstart/seed_*.log`

**Interfaces:**
- Consumes: Tasks 1--4 and the primary fold structure.
- Produces: five valid primary bundles and a measured primary start SD.

- [ ] **Step 1: Append the pre-registration before fitting**

Append this heading and decision record to `EXPERIMENTS.md`:

```markdown
## Iteration 36 — EM-start variance of `lcmnl3_both`

**Hypothesis, written before compute.** The latent-class half of production screens a
deterministic and one random EM start, but its start variance has never been measured. If
random local optima materially move OOF predictions, arithmetic averaging should reduce
variance without selecting a model family, feature, or hyperparameter.

**Fixed procedures.** Random-start seeds 4242, 1103, 2207, 3301, and 4409; deterministic
start, features, penalties, task-position terms, EM budgets, and convergence tolerance are
identical in every arm.

**Decision rule.** Stop if primary single-start model SD is below 0.001. Otherwise adopt the
bag only if expected-single-start minus bagged nested loss is positive on both the original
and fold-B respondent groupings, fold-B retains at least 50%, the income-reweighted primary
gain is non-negative, at least 7/10 held-out folds improve, no fold loses more than 0.001,
and the respondent-clustered mean effect is positive. Arithmetic averaging was chosen before
results. A failed result is logged and production remains unchanged.
```

Do not stage `EXPERIMENTS.md`, because it already contains substantial uncommitted work that
predates iteration 36.

- [ ] **Step 2: Import and validate seed 4242**

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/import_reference.R
```

Expected: `import_reference.R: OK`.

- [ ] **Step 3: Launch seed 1103 in the background**

```powershell
$p1103 = Start-Process -FilePath "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" -ArgumentList "experiments/iter36_emstart/run_seed.R","1103" -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru
$p1103.Id
```

Expected: a process ID. Do not launch more than two EM jobs simultaneously.

- [ ] **Step 4: Launch seed 2207 in the background**

Use the Step 3 command with variable `$p2207` and seed `2207`.

Expected: a process ID.

- [ ] **Step 5: Poll the first pair without blocking longer than 60 seconds**

```powershell
Get-Process -Id $p1103.Id,$p2207.Id -ErrorAction SilentlyContinue
Get-Content experiments/iter36_emstart/seed_1103.log -Tail 5
Get-Content experiments/iter36_emstart/seed_2207.log -Tail 5
```

Expected: eventual validated bundle messages; report progress to the user at least once per
60 seconds while jobs run.

- [ ] **Step 6: Launch and complete seeds 3301 and 4409**

Repeat Steps 3--5 for `3301` and `4409`, still with at most two simultaneous jobs.

- [ ] **Step 7: Validate all five primary bundles and compute Gate 1**

Run:

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/evaluate.R
```

The evaluator is allowed to stop successfully after Gate 1 when fold-B bundles are absent,
provided it writes `evaluation.rds` with either:

- `needs_fold_b = TRUE` and the measured SD when SD is at least 0.001; or
- `decision$pass = FALSE`, `needs_fold_b = FALSE`, reason `material_variance` when SD is
  below 0.001.

- [ ] **Step 8: Follow the fixed Gate-1 branch**

If SD is below 0.001, skip Task 6's fold-B compute and proceed directly to its fallback
finalization steps. If SD is at least 0.001, continue to Task 6 without changing any threshold.

---

### Task 6: Replicate, decide, finalize, and verify

**Files:**
- Modify: `model/members.txt` only on a full gate pass
- Modify: `EXPERIMENTS.md`
- Create at runtime: fold-B bundles, evaluation outputs, candidate member artifacts,
  experimental blend artifacts, and one final CSV

**Interfaces:**
- Consumes: Gate-1 result and all code from Tasks 1--5.
- Produces: final pass/fail decision and `submissions/sub_final_robust_20260729.csv`.

- [ ] **Step 1: Run fold-B seeds only when Gate 1 requests them**

For each of `1103, 2207, 3301, 4409`, launch:

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/run_seed.R <seed> b
```

Use the same two-process maximum and progress polling as Task 5. Seed 4242's fold-B bundle
was imported in Task 5.

- [ ] **Step 2: Run the full evaluator**

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/evaluate.R
```

Expected: a printed gate table, `evaluation.rds`, `per_fold.csv`, and either `PASS` with three
validated candidate member artifacts or `REJECT` with production artifacts untouched.

- [ ] **Step 3: Run required diagnostics**

Only on `PASS`:

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/compare.R lcmnl3_both lcmnl3_bothbag
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/shift_audit.R
$env:BLEND_MEMBERS="xgb_lw2bag lcmnl3_bothbag"
$env:BLEND_OUT="blend_emstart"
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/06_blend.R
Remove-Item Env:BLEND_MEMBERS
Remove-Item Env:BLEND_OUT
```

Expected: the logged artifact comparison, shift table, and
`model/artifacts/test_blend_emstart.rds`.

- [ ] **Step 4: Promote membership only after the pass diagnostics**

On `PASS`, replace the second active member line in `model/members.txt` with:

```text
lcmnl3_bothbag # latent class + task position, 5 EM starts averaged
```

Then run `model/06_blend.R` without environment overrides and confirm the production nested
score reproduces the experimental blend score. On `REJECT`, do not edit `members.txt`.

- [ ] **Step 5: Build the one permitted final CSV**

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/finalize.R
```

Expected: `submissions/sub_final_robust_20260729.csv`, with mean `Ch4` equal to the measured
target and a manifest identifying either `test_blend_emstart.rds` or `test_blend.rds`.

- [ ] **Step 6: Run the complete verification suite**

```powershell
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/test_common.R
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/test_evaluate.R
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter36_emstart/test_finalize.R
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/tests.R
git diff --check
```

Expected: all three iteration tests print `OK`, model tests pass, and `git diff --check`
reports no whitespace errors.

- [ ] **Step 7: Append the measured result and honest reflection**

Append to iteration 36 in `EXPERIMENTS.md`:

- the five single-start model losses and SD;
- expected single-start and bagged nested scores for both folds;
- income-reweighted gain;
- ten fold gains;
- clustered effect;
- each gate's Boolean result;
- final `ADOPT` or `REJECT`;
- the selected final base and measured-correction target;
- a reflection stating whether the variance question mattered and why no further modelling
  follows.

Do not stage this already-dirty file.

- [ ] **Step 8: Commit only an approved membership change**

```powershell
git status --short
if ((git diff -- model/members.txt).Length -gt 0) {
  git add -- model/members.txt
  git commit -m "feat: promote variance-reduced latent class"
}
```

Tasks 1--4 already committed all implementation `.R` files explicitly. Do not stage runtime
`.rds`, `.log`, `.csv`, or any pre-existing dirty file. Do not commit `EXPERIMENTS.md` until
its earlier uncommitted ownership is resolved.

- [ ] **Step 9: Report the final decision without claiming an unmeasured score**

Report:

- whether EM bagging passed;
- the exact validation gains;
- which base produced the CSV;
- the final CSV path;
- that Kaggle upload remains a user action unless separately requested;
- that a 1.18 score was not promised.
