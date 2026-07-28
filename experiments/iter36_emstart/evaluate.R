suppressMessages(library(data.table))
source("model/99_utils.R")
source("experiments/iter36_emstart/common.R")
RUN_MAIN <- identical(environment(), globalenv())

nested_blend <- function(Ptree, Plc, y, fold) {
  stopifnot(nrow(Ptree) == length(y), nrow(Plc) == length(y),
            length(fold) == length(y), identical(sort(unique(fold)), 1:5))
  predictors <- list(Ptree, Plc)
  blend <- function(theta, rows) {
    w <- exp(theta[1:2]); w <- w / sum(w)
    temperature <- exp(theta[3])
    epsilon <- plogis(theta[4]) * .10
    L <- w[1] * log(pmax(predictors[[1]][rows,,drop = FALSE], 1e-12)) +
         w[2] * log(pmax(predictors[[2]][rows,,drop = FALSE], 1e-12))
    L <- L / temperature
    Q <- exp(L - apply(L, 1, max)); Q <- Q / rowSums(Q)
    (1 - epsilon) * Q + epsilon * .25
  }
  pred <- matrix(NA_real_, length(y), 4)
  fold_loss <- numeric(5)
  for (k in 1:5) {
    train <- fold != k
    valid <- fold == k
    objective <- function(theta) logloss(y[train], blend(theta, train))
    fit <- optim(c(0, 0, 0, -3), objective, method = "Nelder-Mead",
                 control = list(maxit = 3000))
    pred[valid,] <- blend(fit$par, valid)
    fold_loss[k] <- logloss(y[valid], pred[valid,,drop = FALSE])
  }
  list(score = mean(fold_loss), fold_loss = fold_loss, pred = pred)
}

gate_decision <- function(metrics) {
  gates <- list(
    material_variance = metrics$start_sd >= .001,
    primary_positive = metrics$primary_gain > 0,
    weighted_nonnegative = metrics$weighted_gain >= 0,
    independent_positive = metrics$b_gain > 0,
    replication_half = metrics$replication >= .50,
    seven_folds = sum(metrics$fold_gain > 0) >= 7,
    no_large_fold_loss = min(metrics$fold_gain) >= -.001,
    clustered_positive = metrics$clustered_gain > 0
  )
  list(pass = all(unlist(gates)), gates = gates,
       reasons = names(gates)[!unlist(gates)])
}

if (RUN_MAIN) {
  long <- readRDS("model/artifacts/long.rds")
  tasks <- unique(long[, .(No, Case, is_test, y)], by = "No")
  setorder(tasks, No)
  train_tasks <- tasks[is_test == FALSE]
  test_tasks <- tasks[is_test == TRUE]
  expected <- list(train = train_tasks$No, test = test_tasks$No)
  y <- train_tasks$y

  get_bundle <- function(seed, split) {
    path <- bundle_path(seed, split)
    if (!file.exists(path)) stop("missing bundle: ", path)
    validate_bundle(readRDS(path), seed, split, expected,
                    require_test = split == "")
  }
  primary <- lapply(EM_SEEDS, get_bundle, split = "")
  names(primary) <- EM_SEEDS
  PlcA <- lapply(primary, function(z) as.matrix(z$oof[, ..P_COLS]))
  model_loss <- vapply(PlcA, function(P) logloss(y, P), numeric(1))
  start_sd <- sd(model_loss)
  evaluation_path <- file.path(ITER36_DIR, "evaluation.rds")

  if (start_sd < .001) {
    decision <- list(pass = FALSE, gates = list(material_variance = FALSE),
                     reasons = "material_variance")
    saveRDS(list(needs_fold_b = FALSE, model_loss = model_loss,
                 start_sd = start_sd, decision = decision), evaluation_path)
    cat(sprintf("Gate 1 REJECT: start SD %.6f < 0.001\n", start_sd))
    quit(save = "no")
  }

  fold_b_paths <- vapply(EM_SEEDS, bundle_path, character(1), split = "b")
  missing_b <- EM_SEEDS[!file.exists(fold_b_paths)]
  if (length(missing_b)) {
    saveRDS(list(needs_fold_b = TRUE, missing_b = missing_b,
                 model_loss = model_loss, start_sd = start_sd), evaluation_path)
    cat("Gate 1 PASS; run fold-B seeds:", paste(missing_b, collapse = ", "), "\n")
    quit(save = "no")
  }

  secondary <- lapply(EM_SEEDS, get_bundle, split = "b")
  names(secondary) <- EM_SEEDS
  PlcB <- lapply(secondary, function(z) as.matrix(z$oof[, ..P_COLS]))
  bagA_dt <- mean_pred(lapply(primary, `[[`, "oof"), expected$train, "primary bag")
  bagB_dt <- mean_pred(lapply(secondary, `[[`, "oof"), expected$train, "fold-B bag")
  test_bag_dt <- mean_pred(lapply(primary, `[[`, "test"), expected$test, "test bag")
  bagA <- as.matrix(bagA_dt[, ..P_COLS])
  bagB <- as.matrix(bagB_dt[, ..P_COLS])

  get_oof <- function(name) {
    x <- readRDS(sprintf("model/artifacts/oof_%s.rds", name))
    setorder(x, No)
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
  loss_vector <- function(result)
    -log(pmax(result$pred[cbind(seq_along(y), y)], 1e-15))
  single_loss_A <- Reduce(`+`, lapply(oneA, loss_vector)) / length(oneA)
  single_loss_B <- Reduce(`+`, lapply(oneB, loss_vector)) / length(oneB)
  bag_loss_A <- loss_vector(avgA)
  bag_loss_B <- loss_vector(avgB)
  effectA <- single_loss_A - bag_loss_A
  effectB <- single_loss_B - bag_loss_B
  single_score_A <- vapply(oneA, `[[`, numeric(1), "score")
  single_score_B <- vapply(oneB, `[[`, numeric(1), "score")
  primary_gain <- mean(single_score_A) - avgA$score
  b_gain <- mean(single_score_B) - avgB$score

  wide <- readRDS("model/artifacts/wide.rds")
  respondents <- unique(wide[, .(Case, is_test, incomeind)])
  n_train <- respondents[is_test == FALSE, .N]
  n_test <- respondents[is_test == TRUE, .N]
  weights <- merge(
    respondents[is_test == FALSE, .(ptr = .N / n_train), by = incomeind],
    respondents[is_test == TRUE, .(pte = .N / n_test), by = incomeind],
    by = "incomeind", all.x = TRUE)
  weights[is.na(pte), pte := 0]
  weights[, w := pmin(pmax(pte / ptr, .2), 5)]
  row_weight <- merge(
    unique(wide[is_test == FALSE, .(No, incomeind)]),
    weights[, .(incomeind, w)], by = "incomeind")[order(No), w]
  weighted_gain <- sum(row_weight * effectA) / sum(row_weight)

  fold_matrix_A <- do.call(rbind, lapply(oneA, `[[`, "fold_loss"))
  fold_matrix_B <- do.call(rbind, lapply(oneB, `[[`, "fold_loss"))
  gainA <- colMeans(fold_matrix_A) - avgA$fold_loss
  gainB <- colMeans(fold_matrix_B) - avgB$fold_loss
  fold_gain <- c(gainA, gainB)
  per_fold <- rbind(
    data.table(split = "primary", fold = 1:5,
               mean_single = colMeans(fold_matrix_A),
               bagged = avgA$fold_loss, gain = gainA),
    data.table(split = "b", fold = 1:5,
               mean_single = colMeans(fold_matrix_B),
               bagged = avgB$fold_loss, gain = gainB))

  by_case <- data.table(Case = train_tasks$Case,
                        effect = (effectA + effectB) / 2)[,
    .(effect = mean(effect)), by = Case]
  clustered_gain <- mean(by_case$effect)
  clustered_se <- sd(by_case$effect) / sqrt(nrow(by_case))
  metrics <- list(
    start_sd = start_sd,
    primary_gain = primary_gain,
    weighted_gain = weighted_gain,
    b_gain = b_gain,
    replication = b_gain / primary_gain,
    fold_gain = fold_gain,
    clustered_gain = clustered_gain,
    clustered_se = clustered_se)
  decision <- gate_decision(metrics)
  evaluation <- list(
    needs_fold_b = FALSE,
    model_loss = model_loss,
    bag_model_loss = c(primary = logloss(y, bagA), b = logloss(y, bagB)),
    primary_single_nested = single_score_A,
    b_single_nested = single_score_B,
    primary_bagged_nested = avgA$score,
    b_bagged_nested = avgB$score,
    metrics = metrics,
    decision = decision)
  saveRDS(evaluation, evaluation_path)
  fwrite(per_fold, file.path(ITER36_DIR, "per_fold.csv"))

  if (decision$pass) {
    saveRDS(bagA_dt, "model/artifacts/oof_lcmnl3_bothbag.rds")
    saveRDS(test_bag_dt, "model/artifacts/test_lcmnl3_bothbag.rds")
    saveRDS(bagB_dt, "model/artifacts/oof_lcmnl3_bothbag_b.rds")
    checkA <- validate_pred(readRDS("model/artifacts/oof_lcmnl3_bothbag.rds"),
                            expected$train, "promoted primary bag")
    checkT <- validate_pred(readRDS("model/artifacts/test_lcmnl3_bothbag.rds"),
                            expected$test, "promoted test bag")
    checkB <- validate_pred(readRDS("model/artifacts/oof_lcmnl3_bothbag_b.rds"),
                            expected$train, "promoted fold-B bag")
    stopifnot(abs(logloss(y, as.matrix(checkA[, ..P_COLS])) -
                  evaluation$bag_model_loss["primary"]) < 1e-10)
    stopifnot(abs(logloss(y, as.matrix(checkB[, ..P_COLS])) -
                  evaluation$bag_model_loss["b"]) < 1e-10)
    stopifnot(nrow(checkT) == 4997L)
  }
  print(unlist(decision$gates))
  if (decision$pass) {
    cat("PASS\n")
  } else {
    cat("REJECT:", paste(decision$reasons, collapse = ", "), "\n")
  }
}
