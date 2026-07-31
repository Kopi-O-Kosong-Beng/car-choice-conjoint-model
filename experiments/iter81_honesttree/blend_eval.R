# =============================================================================
# ITERATION 81 — THE DECISION MEASURE
#
# Computes, for a member set, the NESTED blend OOF (weights fitted on the outer
# training folds only, evaluated on the held-out fold) and reports it three ways:
#
#   plain            -- for the record only. NOT comparable across arms that
#                       differ in encoding-leak exposure.
#   segment-reweighted -- the decision measure. Training respondents are reweighted
#                       by p_test(segmentind)/p_train(segmentind), UNCLIPPED.
#                       CLAUDE.md: reads 1.19610 for the production blend against an
#                       actual public of 1.197.
#   paired vs baseline -- per-respondent differences, respondent-clustered SE, so a
#                       z is available. ESS is reported in RESPONDENTS, not tasks
#                       (the task-level count is 19x inflated because the weight is
#                       constant within a respondent).
#
# Usage:
#   BE_MEMBERS="xgbh_noenc5 lcmnl3_both" \
#   BE_BASE="xgb_lw2bag lcmnl3_both" \
#   Rscript experiments/iter81_honesttree/blend_eval.R
#
# This script NEVER writes to model/artifacts or model/members.txt.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
wide  <- readRDS("model/artifacts/wide.rds")

ymap <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
y    <- ymap$y
case <- ymap$Case
fmap <- folds[order(No), fold]
stopifnot(length(y) == 21565L, length(fmap) == 21565L)

# ---- segment reweighting, at RESPONDENT level, unclipped ---------------------
resp <- unique(wide[, .(Case, is_test, segmentind)])
ptr  <- resp[is_test == FALSE, .(ptr = .N / sum(!resp$is_test)), by = segmentind]
pte  <- resp[is_test == TRUE,  .(pte = .N / sum(resp$is_test)),  by = segmentind]
wtab <- merge(ptr, pte, by = "segmentind", all.x = TRUE)
wtab[is.na(pte), pte := 0]
wtab[, wt := pte / ptr]                                   # UNCLIPPED, by design
rmap <- merge(resp[is_test == FALSE, .(Case, segmentind)], wtab[, .(segmentind, wt)],
              by = "segmentind")
setkey(rmap, Case)
w_case <- rmap[J(sort(unique(case))), wt]                 # one weight per respondent
names(w_case) <- sort(unique(case))
w_row <- unname(w_case[as.character(case)])
ess_resp <- sum(w_case)^2 / sum(w_case^2)
cat(sprintf("segment reweighting: %d respondents, ESS %.0f (%.1f%%)\n",
            length(w_case), ess_resp, 100 * ess_resp / length(w_case)))
print(wtab[order(segmentind)])

# ---- blend machinery, identical to model/06_blend.R (simplex mode) -----------
load_set <- function(memb) {
  O <- lapply(memb, function(m) {
    f <- sprintf("model/artifacts/oof_%s.rds", m)
    if (!file.exists(f)) stop("missing artifact: ", f)
    d <- readRDS(f); setorder(d, No); stopifnot(identical(d$No, ymap$No))
    as.matrix(d[, .(p1,p2,p3,p4)])
  })
  names(O) <- memb; O
}
nested_oof <- function(memb) {
  OOF <- load_set(memb); M <- length(memb); eps0 <- 1e-12
  blend <- function(theta, rows) {
    w <- exp(theta[1:M]); w <- w / sum(w)
    Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
    L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, OOF))
    L <- L / Tt
    P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
    (1 - eA) * P + eA * 0.25
  }
  obj <- function(theta, rows) logloss(y[rows], blend(theta, rows))
  Pn <- matrix(NA_real_, length(y), 4)
  Wk <- vector("list", 5)
  for (k in 1:5) {
    o <- optim(c(rep(0, M), 0, -3), obj, rows = (fmap != k),
               method = "Nelder-Mead", control = list(maxit = 3000))
    Pn[fmap == k, ] <- blend(o$par, fmap == k)
    wk <- exp(o$par[1:M]); Wk[[k]] <- wk / sum(wk)
  }
  stopifnot(!anyNA(Pn))
  list(P = Pn, W = do.call(rbind, Wk))
}

score <- function(P) {
  ll <- -log(pmax(P[cbind(seq_along(y), y)], 1e-15))
  list(ll = ll,
       plain = mean(ll),
       segrw = sum(w_row * ll) / sum(w_row))
}

MEMB <- strsplit(trimws(Sys.getenv("BE_MEMBERS")), "[[:space:]]+")[[1]]
BASE <- strsplit(trimws(Sys.getenv("BE_BASE", "xgb_lw2bag lcmnl3_both")), "[[:space:]]+")[[1]]
MEMB <- MEMB[nzchar(MEMB)]; BASE <- BASE[nzchar(BASE)]
stopifnot(length(MEMB) >= 2L, length(BASE) >= 2L)

cat(sprintf("\nCHALLENGER: %s\nBASELINE  : %s\n\n", paste(MEMB, collapse=" + "), paste(BASE, collapse=" + ")))

nb <- nested_oof(BASE); sb <- score(nb$P)
nc <- nested_oof(MEMB); sc <- score(nc$P)

cat("per-fold weights (challenger):\n"); print(round(nc$W, 4))
cat("per-fold weights (baseline):\n");  print(round(nb$W, 4))

cat(sprintf("\n%-12s  plain %.5f   segment-reweighted %.5f\n", "BASELINE",   sb$plain, sb$segrw))
cat(sprintf("%-12s  plain %.5f   segment-reweighted %.5f\n",   "CHALLENGER", sc$plain, sc$segrw))
cat(sprintf("%-12s  plain %+.5f  segment-reweighted %+.5f   (negative = challenger better)\n",
            "DELTA", sc$plain - sb$plain, sc$segrw - sb$segrw))

# ---- paired, respondent-clustered, on the segment-reweighted metric ----------
d_row <- sc$ll - sb$ll
agg   <- data.table(case = case, d = d_row)[, .(d = mean(d)), by = case]
setkey(agg, case)
W     <- unname(w_case[as.character(agg$case)])
est   <- sum(W * agg$d) / sum(W)
vr    <- sum(W^2 * (agg$d - est)^2) / (sum(W)^2)
se    <- sqrt(vr)
cat(sprintf("\nPAIRED (segment-weighted, clustered by respondent, n=%d):\n", nrow(agg)))
cat(sprintf("  delta %+.5f   SE %.5f   z %+.2f   %s\n", est, se, est / se,
            if (est < 0) "challenger better" else "baseline better"))

# unweighted paired, for contrast
est0 <- mean(agg$d); se0 <- sd(agg$d) / sqrt(nrow(agg))
cat(sprintf("  unweighted: delta %+.5f   SE %.5f   z %+.2f\n", est0, se0, est0 / se0))

# per-fold, so a structural leak (uniform across folds) is visible
pf <- data.table(fold = fmap, d = d_row)[, .(d = mean(d)), by = fold][order(fold)]
cat("  per-fold delta:", paste(sprintf("%+.5f", pf$d), collapse = " "), "\n")
cat("done\n")
