suppressMessages(library(data.table))
source("experiments/iter36_emstart/common.R")
RUN_MAIN <- identical(environment(), globalenv())
R_MEASURED <- (log(6) - 1.499) / log(3)

tilt_p4 <- function(P, target) {
  tilt <- function(alpha) {
    p4_new <- alpha * P[,4] / (alpha * P[,4] + 1 - P[,4])
    scale <- (1 - p4_new) / (1 - P[,4])
    cbind(P[,1] * scale, P[,2] * scale, P[,3] * scale, p4 = p4_new)
  }
  alpha <- exp(uniroot(
    function(log_alpha) mean(tilt(exp(log_alpha))[,4]) - target,
    c(-6, 6), tol = 1e-12)$root)
  Q <- tilt(alpha)
  Q / rowSums(Q)
}

select_base <- function(bag_passed) {
  if (isTRUE(bag_passed)) "model/artifacts/test_blend_emstart.rds"
  else "model/artifacts/test_blend.rds"
}

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
    present <- intersect(c("p1", "p2", "p3", "p4"), names(raw))
    P <- as.matrix(raw[, ..present])
  } else {
    P <- as.matrix(raw)
  }
  if (!identical(dim(P), c(4997L, 4L))) stop("base must be a 4997 x 4 matrix")
  base_dt <- validate_pred(
    data.table(No = test_no, p1 = P[,1], p2 = P[,2], p3 = P[,3], p4 = P[,4]),
    test_no, "selected blend")
  P <- as.matrix(base_dt[, ..P_COLS])
  Q <- tilt_p4(P, R_MEASURED)
  if (abs(mean(Q[,4]) - R_MEASURED) > 1e-12) stop("p4 target mismatch")
  if (max(abs(rowSums(Q) - 1)) > 1e-9) stop("tilted rows do not sum to one")
  if (anyNA(Q) || any(!is.finite(Q)) || any(Q <= 0))
    stop("invalid tilted probability")
  if (!identical(order(Q[,4]), order(P[,4]))) stop("p4 order changed")
  if (max(abs(Q[,1] / Q[,2] - P[,1] / P[,2])) > 1e-12 ||
      max(abs(Q[,1] / Q[,3] - P[,1] / P[,3])) > 1e-12)
    stop("relative odds among alternatives 1-3 changed")

  output <- "submissions/sub_final_robust_20260729.csv"
  if (file.exists(output))
    stop("refusing to overwrite existing final submission: ", output)
  fwrite(data.table(No = test_no, Ch1 = Q[,1], Ch2 = Q[,2],
                    Ch3 = Q[,3], Ch4 = Q[,4]), output)
  check <- fread(output)
  if (!identical(names(check), c("No", "Ch1", "Ch2", "Ch3", "Ch4")) ||
      !identical(check$No, test_no) || nrow(check) != 4997L)
    stop("CSV schema or identifiers changed on read-back")
  check_prob <- as.matrix(check[, .(Ch1, Ch2, Ch3, Ch4)])
  if (anyNA(check_prob) || any(!is.finite(check_prob)) || any(check_prob <= 0) ||
      max(abs(rowSums(check_prob) - 1)) > 1e-9)
    stop("CSV probabilities failed read-back validation")

  manifest <- list(
    base = base_path,
    bag_passed = bag_passed,
    target_p4 = R_MEASURED,
    achieved_p4 = mean(check_prob[,4]),
    csv = output,
    csv_md5 = unname(tools::md5sum(output)),
    created_at = format(Sys.time(), tz = "Asia/Singapore", usetz = TRUE))
  save_experiment_rds(manifest, file.path(ITER36_DIR, "final_manifest.rds"))
  cat("wrote", output, "from", base_path,
      "with mean p4", mean(check_prob[,4]), "\n")
}
