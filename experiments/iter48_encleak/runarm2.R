# =============================================================================
# ITERATION 48 -- PHASE B2.  "Does the design-share encoding contain ANY honest
# out-of-sample signal at all?"  The unimpeachable construction.
#
# Args: ARM DEPTH MCW ETA NROUNDS SEED       ARM in {block, block0}
#
# For scored outer fold k, let r = (k mod 5) + 1 be a RESERVED reference fold.
#   train on folds not in {k, r}      (3 folds, 681 respondents)
#   encoding for EVERY row -- training rows and scored fold-k rows alike -- is built
#   from fold r ONLY, a block of respondents that appears in neither the training set
#   nor the scored set.
#
# Consequences, all by construction rather than by argument:
#   * no row's own respondent is in its own encoding          (as in production)
#   * no row's own FOLD is in its own encoding                (NOT true in production)
#   * so the feature is NOT (cell total) - (own fold count); there is no complement
#     identity for a deep tree to invert, and no within-cell channel to the label
#   * fold-k labels appear nowhere: not in the encoding, not in the training set
#   * training rows and scored rows see the SAME reference block, exactly as a test
#     row and its training rows would if the encoder were built the honest way
#
#   block0 is the identical experiment with ENC_COLS dropped: same 3 training folds,
#   same rounds, same seed.  (block - block0) is the honest value of the encoding,
#   measured at 1-fold reference support (227 respondents, thinner than production's
#   908, so this is a LOWER bound on the honest value, not an upper one).
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")

a <- commandArgs(TRUE)
ARM <- a[1]; DEPTH <- as.integer(a[2]); MCW <- as.integer(a[3])
ETA <- as.numeric(a[4]); NR <- as.integer(a[5]); SEED <- as.integer(a[6])
stopifnot(ARM %in% c("block", "block0"))
TAG <- sprintf("%s_d%d_m%d_e%s_n%d_s%d", ARM, DEPTH, MCW, sub("\\.", "", format(ETA)), NR, SEED)
cat("=== ARM", TAG, "===\n"); flush.console()

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
long  <- add_design_key(long, wide, ATTRS); setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No"); setorder(trl, No, alt)

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
base_feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
               "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo)
feat <- if (ARM == "block0") { base_feat } else { c(base_feat, ENC_COLS) }

softmax_by_task <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M,1,max); E <- exp(M); E/rowSums(E) }
obj_listwise <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(softmax_by_task(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
params <- list(eta = ETA, max_depth = DEPTH, min_child_weight = MCW,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0,
               nthread = as.integer(Sys.getenv("XGB_NTHREAD", "8")), seed = SEED)

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
set.seed(SEED)
oof_list <- list(); t0 <- Sys.time()
for (k in 1:5) {
  r <- (k %% 5L) + 1L
  trf <- setdiff(1:5, c(k, r))
  stopifnot(length(trf) == 3L)
  dtr <- trl[fold %in% trf]; dsc <- trl[fold == k]
  Xtr <- dtr[, ..base_feat]; Xsc <- dsc[, ..base_feat]
  if (ARM == "block") {
    ref <- trl[fold == r]
    stopifnot(!(k %in% unique(ref$fold)), all(!(unique(dtr$fold) %in% r)))
    Xtr <- cbind(Xtr, encode_aligned(ref, dtr))
    Xsc <- cbind(Xsc, encode_aligned(ref, dsc))
  }
  dm <- xgb.DMatrix(as.matrix(Xtr[, ..feat]), label = as.numeric(dtr$chosen))
  fit <- xgb.train(params = params, data = dm, nrounds = NR, verbose = 0,
                   obj = obj_listwise, maximize = FALSE)
  P <- softmax_by_task(as.vector(predict(fit, xgb.DMatrix(as.matrix(Xsc[, ..feat])), outputmargin = TRUE)))
  oof_list[[k]] <- data.table(No = unique(dsc$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
  cat(sprintf("  fold %d (ref fold %d, train folds %s) done %.1f min\n",
              k, r, paste(trf, collapse=","), as.numeric(difftime(Sys.time(), t0, units="mins")))); flush.console()
}
oof <- rbindlist(oof_list); setorder(oof, No)
stopifnot(nrow(oof) == 21565L)
ll <- logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)]))
cat(sprintf(">>> %s  OOF %.5f\n", TAG, ll))
saveRDS(oof, sprintf("experiments/iter48_encleak/el48_oof_%s.rds", TAG))
cat(sprintf("ARM_DONE %s %.5f\n", TAG, ll))
