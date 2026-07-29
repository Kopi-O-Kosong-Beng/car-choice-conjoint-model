# =============================================================================
# ITERATION 48 -- PHASE B.  The controlled leak test for the design-share encoding.
#
# Args: ARM DEPTH MCW ETA NROUNDS SEED
#   ARM in {prod, noenc, honest, leaky}
#
# THE FOUR ARMS.  Let k be the scored outer fold and d = (k mod 5) + 1 a DECOY fold
# (d != k, always a training fold).
#
#   prod    production encoding: every row's share comes from "all folds except my
#           own".  4-fold support.  Built once, before the CV loop.
#
#   noenc   ENC_COLS dropped from the feature list entirely.
#
#   honest  universe V = folds != k.   training row in fold j -> ref V\{j}
#                                      scored fold-k row      -> ref V\{d}
#           NO fold-k respondent appears in ANY encoding, train or score. 3-fold support.
#
#   leaky   universe V' = folds != d.  training row in fold j != d -> ref V'\{j}
#                                      training row in fold d      -> ref V'\{k}
#                                      scored fold-k row           -> ref V'\{k}
#           3-fold support, IDENTICAL construction family to honest -- the only
#           difference is which fold the universe omits.  Here fold k is INSIDE the
#           universe, so the scored row's share is (cell total over V') - c_k, sharing
#           its constant with the training rows' (cell total over V') - c_j.  That is
#           exactly production's structure, at honest's support.
#
#   honest and leaky are isomorphic: 3 training folds with own-complement refs plus 1
#   training fold sharing the scored rows' ref, 3-fold support throughout.  So
#   (honest - leaky) isolates LEAKAGE with support, feature count, capacity, seed and
#   round count all held fixed.  (prod - leaky) isolates the support difference.
#
# FIXED ROUNDS, no early stopping: removes the ES carve as a confound and makes each
# arm a deterministic function of (features, seed).
# =============================================================================
suppressMessages({ library(data.table); library(xgboost) })
source("model/99_utils.R")
source("model/encode_design.R")

a <- commandArgs(TRUE)
ARM <- a[1]; DEPTH <- as.integer(a[2]); MCW <- as.integer(a[3])
ETA <- as.numeric(a[4]); NR <- as.integer(a[5]); SEED <- as.integer(a[6])
stopifnot(ARM %in% c("prod", "noenc", "honest", "leaky"))
TAG <- sprintf("%s_d%d_m%d_e%s_n%d_s%d", ARM, DEPTH, MCW, sub("\\.", "", format(ETA)), NR, SEED)
cat("=== ARM", TAG, "===\n"); flush.console()

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")
long  <- add_design_key(long, wide, ATTRS)
setorder(long, No, alt)
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)
apply_design_encoding(trl, tel)          # production encoding (used by arm "prod")
PROD_ENC <- as.data.table(trl[, ..ENC_COLS])

demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
base_feat <- c(ATTRS, paste0(ATTRS, "_c"), "richness","lvlsum","price_rank","price_min_rival",
               "price_task_mean","rich_task_mean","Task", paste0("alt", 1:4), demo)
feat <- if (ARM == "noenc") { base_feat } else { c(base_feat, ENC_COLS) }

softmax_by_task <- function(s) { M <- matrix(s, ncol = 4, byrow = TRUE); M <- M - apply(M,1,max); E <- exp(M); E/rowSums(E) }
obj_listwise <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label"); p <- as.vector(t(softmax_by_task(preds)))
  list(grad = p - y, hess = pmax(2 * p * (1 - p), 1e-6))
}
params <- list(eta = ETA, max_depth = DEPTH, min_child_weight = MCW,
               subsample = 0.8, colsample_bytree = 0.8, base_score = 0,
               nthread = as.integer(Sys.getenv("XGB_NTHREAD", "8")))

# ---- per-arm encoding for one outer fold ------------------------------------
# returns list(tr = data.table of ENC_COLS for trl[fold!=k], te = ... for trl[fold==k])
enc_for_fold <- function(k) {
  if (ARM == "noenc") return(NULL)
  if (ARM == "prod")  return(list(tr = PROD_ENC[trl$fold != k], te = PROD_ENC[trl$fold == k]))
  d <- (k %% 5L) + 1L
  U <- if (ARM == "honest") { setdiff(1:5, k) } else { setdiff(1:5, d) }
  piv <- if (ARM == "honest") { d } else { k }          # fold removed for the SCORED rows
  tri <- which(trl$fold != k)
  E <- data.table(matrix(NA_real_, nrow = length(tri), ncol = length(ENC_COLS)))
  setnames(E, ENC_COLS)
  for (j in sort(unique(trl$fold[tri]))) {
    ref_folds <- if (j %in% U) { setdiff(U, j) } else { setdiff(U, piv) }
    stopifnot(length(ref_folds) == 3L, !(j %in% ref_folds))
    if (ARM == "honest") stopifnot(!(k %in% ref_folds))
    rows <- which(trl$fold[tri] == j)
    e <- encode_aligned(trl[fold %in% ref_folds], trl[tri][rows])
    for (cc in ENC_COLS) set(E, rows, cc, e[[cc]])
  }
  ref_sc <- setdiff(U, piv)
  stopifnot(length(ref_sc) == 3L)
  if (ARM == "honest") stopifnot(!(k %in% ref_sc))
  Es <- encode_aligned(trl[fold %in% ref_sc], trl[fold == k])
  stopifnot(!anyNA(E))
  list(tr = E, te = as.data.table(Es))
}

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
set.seed(SEED)
oof_list <- list(); t0 <- Sys.time()
for (k in 1:5) {
  e <- enc_for_fold(k)
  dtr <- trl[fold != k, ..base_feat]; dsc <- trl[fold == k, ..base_feat]
  if (!is.null(e)) { dtr <- cbind(dtr, e$tr); dsc <- cbind(dsc, e$te) }
  dm <- xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(trl$chosen[trl$fold != k]))
  fit <- xgb.train(params = params, data = dm, nrounds = NR, verbose = 0,
                   obj = obj_listwise, maximize = FALSE)
  P <- softmax_by_task(as.vector(predict(fit, xgb.DMatrix(as.matrix(dsc[, ..feat])), outputmargin = TRUE)))
  oof_list[[k]] <- data.table(No = unique(trl$No[trl$fold == k]), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4])
  cat(sprintf("  fold %d done (%.1f min)\n", k, as.numeric(difftime(Sys.time(), t0, units="mins")))); flush.console()
}
oof <- rbindlist(oof_list); setorder(oof, No)
stopifnot(nrow(oof) == 21565L, identical(oof$No, sort(unique(trl$No))))
ll <- logloss(ytr, as.matrix(oof[, .(p1,p2,p3,p4)]))
cat(sprintf(">>> %s  OOF %.5f\n", TAG, ll)); flush.console()

# ---- test-side prediction: only for arms that have a well-defined full-data form
if (ARM %in% c("prod", "noenc")) {
  dtr <- trl[, ..base_feat]; dte <- tel[, ..base_feat]
  if (ARM == "prod") { dtr <- cbind(dtr, PROD_ENC); dte <- cbind(dte, as.data.table(tel[, ..ENC_COLS])) }
  dm <- xgb.DMatrix(as.matrix(dtr[, ..feat]), label = as.numeric(trl$chosen))
  fitf <- xgb.train(params = params, data = dm, nrounds = NR, verbose = 0,
                    obj = obj_listwise, maximize = FALSE)
  P <- softmax_by_task(as.vector(predict(fitf, xgb.DMatrix(as.matrix(dte[, ..feat])), outputmargin = TRUE)))
  tp <- data.table(No = unique(tel$No), p1=P[,1], p2=P[,2], p3=P[,3], p4=P[,4]); setorder(tp, No)
  stopifnot(nrow(tp) == 4997L)
  saveRDS(tp, sprintf("experiments/iter48_encleak/el48_test_%s.rds", TAG))
}
saveRDS(oof, sprintf("experiments/iter48_encleak/el48_oof_%s.rds", TAG))
cat(sprintf("ARM_DONE %s %.5f\n", TAG, ll))
