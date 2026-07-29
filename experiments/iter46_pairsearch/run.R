# =============================================================================
# ITERATION 46 -- EXHAUSTIVE PAIR SEARCH under the PRODUCTION simplex combiner
#
# QUESTION (posed by the orchestrator, not by me):
#   A competitor reached public 1.188 with a TWO-model blend. Our production
#   two-member pool (xgb_lw2bag + lcmnl3_both) is nested 1.12819 -> public 1.197.
#   Is there a BETTER PAIR already sitting in model/artifacts/?
#
# WHAT THIS SCRIPT DOES
#   1. Enumerates every admissible pair of existing oof_*/test_* artifacts.
#   2. Scores each pair with an EXACT reimplementation of model/06_blend.R's
#      simplex log-opinion pool, nested over folds.rds (the decision number).
#   3. For every pair also records: plain OOF, fitted weights/T/eps,
#      the SHIPPED TEST none-rate mean(p4) from the 100%-data fit, and the
#      SEGMENT-reweighted OOF (train reweighted to the test segment mix).
#
# EXCLUSIONS (pre-registered, from the orchestrator's brief)
#   xgb_pt        run killed at fold 3, artifact misnamed  (iter30 run.R)
#   xgb_resenc*   100% leakage, proven iteration 15
#   *_b / *_c     validation-split artifacts; using them as members leaks the
#                 validation split that is supposed to referee the result
#   blend*        not base models
#   *_cal         calibrated derivatives
#   segcal5       ADDED BY ME: iter44's oof_segcal5 is the output of a 5-member
#                 FREE-SIGN pool, i.e. it is a blend, and free-sign is refuted
#                 (1.209 public). It fails the spirit of "blend*". Excluded.
#   NOTE lcmnl3b is NOT a validation artifact -- it is the richer-membership
#        latent class variant (EXPERIMENTS.md line 133). It is kept.
#
# WRITES ONLY: experiments/iter46_pairsearch/pairs.csv (+ top-pair detail).
#   NOTHING is written to model/artifacts/. No production file is opened.
#
# DECISION RULE (written before the run)
#   Production nested = 1.12819. Blend-level seed sd = 0.00048. A pair "beats
#   production" only if nested <= 1.12819 - 0.00048 = 1.12771. Anything closer
#   is a tie. And because this is a search over ~900 pairs, the winner's margin
#   is inflated by selection -- so a beat is a HYPOTHESIS, not a result.
# =============================================================================

suppressMessages(library(data.table))
suppressMessages(library(parallel))
source("model/99_utils.R")

OUTDIR <- "experiments/iter46_pairsearch"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
y     <- ymap$y
fmap  <- folds[order(No), fold]
NTR   <- length(y); stopifnot(NTR == 21565)

# ---- segment reweighting (train respondents reweighted to the TEST segment mix)
wide <- as.data.table(readRDS("model/artifacts/wide.rds"))
resp <- unique(wide[, .(Case, is_test, segmentind)])
ptr  <- resp[is_test == FALSE, .N, by = segmentind][, .(segmentind, ptr = N / sum(N))]
pte  <- resp[is_test == TRUE,  .N, by = segmentind][, .(segmentind, pte = N / sum(N))]
stab <- merge(ptr, pte, by = "segmentind", all.x = TRUE)
stab[is.na(pte), pte := 0]
stab[, wt := pte / ptr]
segw <- merge(unique(wide[is_test == FALSE, .(No, segmentind)]),
              stab[, .(segmentind, wt)], by = "segmentind")[order(No), wt]
stopifnot(length(segw) == NTR)
cat(sprintf("segment reweighting: effective respondents %.1f of %d\n",
            sum(segw)^2 / sum(segw^2) / 19, uniqueN(resp[is_test == FALSE, Case])))

# ---- artifact pool --------------------------------------------------------
getP <- function(path) {
  d <- readRDS(path)
  if (is.data.frame(d)) {
    d <- as.data.table(d); if ("No" %in% names(d)) setorder(d, No)
    as.matrix(d[, .(p1, p2, p3, p4)])
  } else as.matrix(d)
}
allf <- gsub("^oof_|[.]rds$", "", list.files("model/artifacts", pattern = "^oof_.*[.]rds$"))
allf <- allf[file.exists(sprintf("model/artifacts/test_%s.rds", allf))]

drop_pat <- "^blend|_b$|_c$|_cal$|^xgb_pt$|^xgb_resenc|^segcal5$"
MEMB <- sort(setdiff(allf, grep(drop_pat, allf, value = TRUE)))
cat("excluded:", paste(sort(grep(drop_pat, allf, value = TRUE)), collapse = ", "), "\n")
cat("admissible members:", length(MEMB), "\n")
cat(paste(MEMB, collapse = ", "), "\n\n")

EPS0 <- 1e-12
LO <- lapply(MEMB, function(m) log(pmax(getP(sprintf("model/artifacts/oof_%s.rds",  m)), EPS0)))
LT <- lapply(MEMB, function(m) log(pmax(getP(sprintf("model/artifacts/test_%s.rds", m)), EPS0)))
names(LO) <- names(LT) <- MEMB
stopifnot(all(sapply(LO, nrow) == 21565), all(sapply(LT, nrow) == 4997))

# duplicate detection -- identical artifacts under two names would fake a "pair"
sig <- sapply(LO, function(M) paste(round(M[1:50, ], 10), collapse = ","))
dups <- split(names(sig), sig); dups <- dups[sapply(dups, length) > 1]
if (length(dups)) for (d in dups) cat("DUPLICATE OOF (head-identical):", paste(d, collapse = " == "), "\n")

# ---- exact reimplementation of 06_blend.R's simplex pool, M = 2 ------------
# theta = (t1, t2, log T, logit eps).  w = softmax(t1,t2); Tt = exp(theta3);
# eA = plogis(theta4)*0.10;  L = (w1 logP1 + w2 logP2)/Tt; P = softmax(L);
# Pfinal = (1-eA) P + eA * 0.25.   Identical algebra, faster inner loop.
mk_obj <- function(A, B, yy, wts = NULL) {
  n <- nrow(A)
  a1 <- A[, 1]; a2 <- A[, 2]; a3 <- A[, 3]; a4 <- A[, 4]
  b1 <- B[, 1]; b2 <- B[, 2]; b3 <- B[, 3]; b4 <- B[, 4]
  ay <- A[cbind(seq_len(n), yy)]; by <- B[cbind(seq_len(n), yy)]
  function(theta) {
    e1 <- exp(theta[1]); e2 <- exp(theta[2]); s <- e1 + e2
    it <- exp(-theta[3])
    ca <- (e1 / s) * it; cb <- (e2 / s) * it
    eA <- plogis(theta[4]) * 0.10
    L1 <- ca * a1 + cb * b1; L2 <- ca * a2 + cb * b2
    L3 <- ca * a3 + cb * b3; L4 <- ca * a4 + cb * b4
    mx <- pmax(L1, L2, L3, L4)
    den <- exp(L1 - mx) + exp(L2 - mx) + exp(L3 - mx) + exp(L4 - mx)
    p <- exp((ca * ay + cb * by) - mx) / den
    p <- (1 - eA) * p + eA * 0.25
    p <- pmax(pmin(p, 1 - 1e-15), 1e-15)
    if (is.null(wts)) -mean(log(p)) else -sum(wts * log(p)) / sum(wts)
  }
}
pool_P <- function(theta, A, B) {                 # full 4-column probabilities
  e1 <- exp(theta[1]); e2 <- exp(theta[2]); s <- e1 + e2
  it <- exp(-theta[3]); ca <- (e1 / s) * it; cb <- (e2 / s) * it
  eA <- plogis(theta[4]) * 0.10
  L <- ca * A + cb * B
  mx <- do.call(pmax, lapply(1:4, function(j) L[, j]))
  E <- exp(L - mx); P <- E / rowSums(E)
  (1 - eA) * P + eA * 0.25
}
fit <- function(objf) optim(c(0, 0, 0, -3), objf, method = "Nelder-Mead",
                            control = list(maxit = 3000))

score_pair <- function(i, j) {
  A <- LO[[i]]; B <- LO[[j]]
  nested <- numeric(5)
  for (k in 1:5) {
    tr <- fmap != k; te <- fmap == k
    o <- fit(mk_obj(A[tr, , drop = FALSE], B[tr, , drop = FALSE], y[tr]))
    nested[k] <- mk_obj(A[te, , drop = FALSE], B[te, , drop = FALSE], y[te])(o$par)
  }
  of <- fit(mk_obj(A, B, y))
  th <- of$par
  e1 <- exp(th[1]); e2 <- exp(th[2]); w1 <- e1 / (e1 + e2)
  Pte <- clip_norm(pool_P(th, LT[[i]], LT[[j]]))
  Ptr <- pool_P(th, A, B)
  llr <- -sum(segw * log(pmax(Ptr[cbind(seq_len(NTR), y)], 1e-15))) / sum(segw)
  data.table(m1 = i, m2 = j,
             nested = mean(nested), nested_sd = sd(nested), plain = of$value,
             w1 = w1, w2 = 1 - w1, Temp = exp(th[3]), eps = plogis(th[4]) * 0.10,
             test_p4 = mean(Pte[, 4]), seg_oof = llr,
             f1 = nested[1], f2 = nested[2], f3 = nested[3], f4 = nested[4], f5 = nested[5])
}

# ---- sanity: production pair must reproduce 1.12819 ------------------------
cat("SANITY -- reproducing production (xgb_lw2bag + lcmnl3_both)\n")
prod_row <- score_pair("xgb_lw2bag", "lcmnl3_both")
print(prod_row[, .(nested, plain, w1, w2, Temp, eps, test_p4, seg_oof)])
cat(sprintf("  06_blend.R says 1.12819 / weights 0.528 0.472 ; here %.5f / %.3f %.3f\n\n",
            prod_row$nested, prod_row$w1, prod_row$w2))

# ---- exhaustive pair sweep -------------------------------------------------
cmb <- combn(MEMB, 2, simplify = FALSE)
# byte-identical artifacts: a "pair" of two copies of one model is not a pair
DUPSET <- unlist(dups)
cmb <- Filter(function(p) !(p[1] %in% DUPSET && p[2] %in% DUPSET &&
                            any(sapply(dups, function(d) all(p %in% d)))), cmb)
cat("pairs to evaluate:", length(cmb), "(degenerate duplicate pairs removed)\n"); flush.console()
t0 <- Sys.time()
NC <- max(1L, detectCores() - 1L)
chunks <- split(seq_along(cmb), ceiling(seq_along(cmb) / 60))
acc <- list()
for (ci in seq_along(chunks)) {
  idx <- chunks[[ci]]
  out <- mclapply(idx, function(z) {
    r <- try(score_pair(cmb[[z]][1], cmb[[z]][2]), silent = TRUE)
    if (inherits(r, "try-error")) NULL else r
  }, mc.cores = NC)
  acc[[ci]] <- rbindlist(Filter(Negate(is.null), out))
  cat(sprintf("  chunk %d/%d  (%d pairs)  elapsed %s\n", ci, length(chunks),
              sum(sapply(acc, nrow)), format(Sys.time() - t0))); flush.console()
}
res <- rbindlist(acc)
cat("elapsed:", format(Sys.time() - t0), " scored:", nrow(res), "\n")

PROD <- prod_row$nested
SEED_SD <- 0.00048
res[, delta_vs_prod := nested - PROD]
res[, p4_gap := abs(test_p4 - 0.26651)]
setorder(res, nested)
fwrite(res, file.path(OUTDIR, "pairs.csv"))

cat("\n================ TOP 25 PAIRS BY NESTED OOF ================\n")
top <- head(res, 25)
print(top[, .(m1, m2, nested = round(nested, 5), d = round(delta_vs_prod, 5),
              w1 = round(w1, 3), test_p4 = round(test_p4, 4),
              p4_gap = round(p4_gap, 4), seg_oof = round(seg_oof, 4))])

cat("\n---- production reference ----\n")
cat(sprintf("xgb_lw2bag + lcmnl3_both  nested %.5f  test_p4 %.4f  seg_oof %.4f  rank %d\n",
            PROD, prod_row$test_p4, prod_row$seg_oof,
            which(res$m1 == "xgb_lw2bag" & res$m2 == "lcmnl3_both" |
                  res$m2 == "xgb_lw2bag" & res$m1 == "lcmnl3_both")))
cat(sprintf("beats production by > %.5f : %d pairs\n", SEED_SD, sum(res$nested < PROD - SEED_SD)))

cat("\n===== TOP 40 BY NESTED, RE-RANKED BY DISTANCE TO PROBE none-rate 0.26651 =====\n")
t40 <- head(res, 40)[order(p4_gap)]
print(t40[, .(m1, m2, nested = round(nested, 5), test_p4 = round(test_p4, 4),
              p4_gap = round(p4_gap, 4), seg_oof = round(seg_oof, 4))])

cat("\n===== PAIRS WITH test_p4 IN THE PROBE BAND [0.2600, 0.2730], BEST 20 BY NESTED =====\n")
band <- res[test_p4 >= 0.26 & test_p4 <= 0.273]
print(head(band, 20)[, .(m1, m2, nested = round(nested, 5), test_p4 = round(test_p4, 4),
                         seg_oof = round(seg_oof, 4))])

cat("\nwrote", file.path(OUTDIR, "pairs.csv"), "\n")
cat("OK\n")
