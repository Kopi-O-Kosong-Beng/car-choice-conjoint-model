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
  core <- c("seed", "split", "C", "taskmode", "lambda_b", "lambda_g",
            "n_screen", "n_max", "tol", "code_md5")
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
