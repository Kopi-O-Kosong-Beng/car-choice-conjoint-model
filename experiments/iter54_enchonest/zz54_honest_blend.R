# =============================================================================
# ITERATION 54 -- THE CONSEQUENCE OF THE ENCODING LEAK FOR PRODUCTION'S BLEND.
#
# Iteration 48 established (arms prod / noenc / honest / leaky / block) that the
# design-share encoding's +0.0218 at production's tree config is leakage, not
# signal. This script asks the two questions that follow and that iteration 48
# did not close:
#
#   Q1  Production's tree hyperparameters (depth 8) were chosen while the leak was
#       open. What is the HONEST tuning optimum, and what does the 2-member blend
#       score when its tree member is tuned WITHOUT the encoding?
#
#   Q2  Production's blend weight w_tree = 0.528 was fitted on an OOF inflated by
#       the leak. If the tree member's true test-time quality is the HONEST arm's,
#       what does 0.528 cost relative to the weight an honest OOF would have
#       chosen?
#
# NOTHING is written to model/artifacts. No file here shares a name with anything
# in model/artifacts. members.txt / blend.rds are never read or touched.
# Screen output only.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

D48   <- "experiments/iter48_encleak"
long  <- readRDS("model/artifacts/long.rds")
wide  <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No); y <- ymap$y
fmap  <- folds[order(No), fold]

# income reweighting toward the test population (the diagnostic that has tracked
# the public board -- 06_blend.R prints the same quantity)
resp <- unique(wide[, .(Case, is_test, incomeind)])
wt <- merge(resp[is_test == FALSE, .(ptr = .N / sum(!resp$is_test)), by = incomeind],
            resp[is_test == TRUE,  .(pte = .N / sum(resp$is_test)),  by = incomeind],
            by = "incomeind", all.x = TRUE)
wt[is.na(pte), pte := 0][, w := pmin(pmax(pte / ptr, 0.2), 5)]
rw <- merge(unique(wide[is_test == FALSE, .(No, incomeind)]), wt[, .(incomeind, w)],
            by = "incomeind")[order(No), w]

getP <- function(f) { d <- readRDS(f); setorder(d, No); stopifnot(nrow(d) == 21565L)
                      as.matrix(d[, .(p1, p2, p3, p4)]) }
bagof <- function(files) { A <- Reduce(`+`, lapply(files, getP)) / length(files); A / rowSums(A) }
f48 <- function(...) file.path(D48, paste0("el48_oof_", c(...), ".rds"))

LC <- getP("model/artifacts/oof_lcmnl3_both.rds")

# ---- the log-opinion pool, transcribed from model/06_blend.R -----------------
M <- 2L; eps0 <- 1e-12
poolP <- function(w, Tt, eA, Ps, rows) {
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, Ps))
  L <- L / Tt
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
unpack_full <- function(th) { w <- exp(th[1:M]); list(w = w / sum(w), Tt = exp(th[M+1]),
                                                      eA = plogis(th[M+2]) * 0.10) }
unpack_fix  <- function(th, wfix) list(w = wfix, Tt = exp(th[1]), eA = plogis(th[2]) * 0.10)

nested_blend <- function(PX, wfix = NULL) {
  Ps <- list(PX, LC)
  if (is.null(wfix)) {
    obj  <- function(th, rows) { u <- unpack_full(th); logloss(y[rows], poolP(u$w, u$Tt, u$eA, Ps, rows)) }
    st   <- c(rep(0, M), 0, -3)
  } else {
    obj  <- function(th, rows) { u <- unpack_fix(th, wfix); logloss(y[rows], poolP(u$w, u$Tt, u$eA, Ps, rows)) }
    st   <- c(0, -3)
  }
  fitw <- function(rows) optim(st, obj, rows = rows, method = "Nelder-Mead",
                               control = list(maxit = 3000))
  nested <- sapply(1:5, function(k) obj(fitw(fmap != k)$par, fmap == k))
  o  <- fitw(rep(TRUE, length(y)))
  u  <- if (is.null(wfix)) unpack_full(o$par) else unpack_fix(o$par, wfix)
  Pf <- poolP(u$w, u$Tt, u$eA, Ps, rep(TRUE, length(y)))
  li <- -log(pmax(Pf[cbind(seq_along(y), y)], 1e-15))
  list(member = logloss(y, PX), nested = mean(nested), plain = o$value,
       w = u$w[1], rw = sum(rw * li) / sum(rw))
}
say <- function(lab, r) cat(sprintf("%-40s member %.5f | nested %.5f | w_tree %.3f | income-rw %.5f\n",
                                    lab, r$member, r$nested, r$w, r$rw))

cat("=========================================================================\n")
cat("Q0.  REFERENCE -- reproduce production\n")
cat("=========================================================================\n")
say("live xgb_lw2bag (10 seeds, prod enc)", nested_blend(getP("model/artifacts/oof_xgb_lw2bag.rds")))
PRODB <- bagof(f48(paste0("prod_d8_m20_e003_n540_s", c(123, 2024, 31, 55, 7))))
say("iter48 prod bag d8 (5 seeds)", nested_blend(PRODB))

cat("\n=========================================================================\n")
cat("Q1.  THE HONEST TUNING CURVE -- tree member with ENC_COLS removed entirely\n")
cat("     (production's depth was chosen with the leak open)\n")
cat("=========================================================================\n")
CAND <- list(
  list(nm = "noenc d8 /540  (prod config, 5 seeds)",
       f = f48(paste0("noenc_d8_m20_e003_n540_s", c(123, 2024, 31, 55, 7)))),
  list(nm = "noenc d6 /540  (3 seeds)",
       f = f48(paste0("noenc_d6_m20_e003_n540_s", c(123, 2024, 7)))),
  list(nm = "noenc d6 /900  (2 seeds)",
       f = f48(paste0("noenc_d6_m20_e003_n900_s", c(123, 7)))),
  list(nm = "noenc d5 /900  (2 seeds)",
       f = f48(paste0("noenc_d5_m20_e003_n900_s", c(123, 7)))),
  list(nm = "noenc d4 /900  (1 seed)",  f = f48("noenc_d4_m20_e003_n900_s123")),
  list(nm = "noenc d4 /1400 (1 seed)",  f = f48("noenc_d4_m20_e003_n1400_s123"))
)
NOENC_BEST <- NULL; best <- Inf
for (cc in CAND) {
  if (!all(file.exists(cc$f))) { cat(sprintf("%-40s MISSING\n", cc$nm)); next }
  P <- bagof(cc$f); r <- nested_blend(P); say(cc$nm, r)
  if (r$nested < best) { best <- r$nested; NOENC_BEST <- cc$nm }
}
cat(sprintf("\n  best no-encoding blend: %s  ->  %.5f\n", NOENC_BEST, best))

cat("\n=========================================================================\n")
cat("Q2.  WHAT PRODUCTION'S LEAK-FITTED WEIGHT COSTS\n")
cat("     The HONEST arm is the prod tree config scored on features whose\n")
cat("     leave-own-fold-out complement structure is absent -- which is exactly\n")
cat("     the TEST-TIME situation (test rows are encoded from all of trl with\n")
cat("     nothing removed). Treat it as the tree member's true test-time quality.\n")
cat("=========================================================================\n")
HON <- bagof(f48(paste0("honest_d8_m20_e003_n540_s", c(123, 2024, 31, 55, 7))))
rh_free  <- nested_blend(HON)
rh_fixed <- nested_blend(HON, wfix = c(0.528, 0.472))
say("honest tree, weight refitted honestly", rh_free)
say("honest tree, weight FORCED to 0.528  ", rh_fixed)
cat(sprintf("\n  cost of production's leak-fitted weight, if the tree's true quality\n"))
cat(sprintf("  is the honest arm's:   %+.5f nats  (0.528 vs honest optimum %.3f)\n",
            rh_fixed$nested - rh_free$nested, rh_free$w))
NOE <- bagof(f48(paste0("noenc_d8_m20_e003_n540_s", c(123, 2024, 31, 55, 7))))
rn_free  <- nested_blend(NOE)
rn_fixed <- nested_blend(NOE, wfix = c(0.528, 0.472))
cat(sprintf("  same check with the noenc tree: %+.5f nats (0.528 vs optimum %.3f)\n",
            rn_fixed$nested - rn_free$nested, rn_free$w))

cat("\n=========================================================================\n")
cat("Q3.  THE ENCODING'S CONTRIBUTION, DECOMPOSED AT MEMBER LEVEL (depth 8)\n")
cat("=========================================================================\n")
LEA <- bagof(f48(paste0("leaky_d8_m20_e003_n540_s", c(123, 2024, 31, 55, 7))))
m <- c(prod = logloss(y, PRODB), noenc = logloss(y, NOE),
       honest = logloss(y, HON), leaky = logloss(y, LEA))
cat(sprintf("  noenc %.5f | prod %.5f | leaky %.5f | honest %.5f\n",
            m["noenc"], m["prod"], m["leaky"], m["honest"]))
cat(sprintf("  production's claimed encoding gain   noenc - prod   = %+.5f\n", m["noenc"] - m["prod"]))
cat(sprintf("    of which  HONEST VALUE             noenc - honest = %+.5f\n", m["noenc"] - m["honest"]))
cat(sprintf("              PURE LEAK (matched)      honest - leaky = %+.5f\n", m["honest"] - m["leaky"]))
cat(sprintf("              leak amplified by support leaky - prod  = %+.5f\n", m["leaky"] - m["prod"]))
cat("\nZZ54_DONE\n")
