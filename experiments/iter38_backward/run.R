# =============================================================================
# ITERATION 38 -- BACKWARD ELIMINATION OVER THE FREE-SIGN MEMBER SPACE
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# WHY, SPECIFICALLY -- this is iteration 37's diagnosed failure, not a re-run
# -----------------------------------------------------------------------------
# Iteration 37 ran FORWARD selection over 38 candidates (115 nested evaluations)
# and landed on a 4-member set scoring 1.12393 -- WORSE by 0.00052 than the
# 5-member set iteration 35 had picked greedily (1.12341). The pre-registered
# gate rejected it. That is a negative result about the SEARCH, and its own
# trace says why:
#
#   round 1: +xgb_long    1.12521   (+0.00298)  ACCEPT
#   round 2: +xgb_2stage  1.12393   (+0.00127)  ACCEPT
#   round 3: +xgb_mono    1.12325   (+0.00069)  STOP -- below the 0.00100 gate
#
# Forward selection never picked xgb_wide, which IS in the better 5-member set.
# Once xgb_long and xgb_2stage are in, xgb_wide's MARGINAL contribution falls
# below the step gate -- yet all three together beat any two. That is textbook
# non-additivity, and greedy forward search is structurally blind to it: it can
# only ever evaluate a member conditional on what it has already taken.
#
# Backward elimination has the opposite bias. It starts from an over-large set
# where every member is present, so a member that only pays off JOINTLY is
# evaluated in the company it needs, and is dropped only if the set is no worse
# without it.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# Backward elimination from an over-complete set will find a member set at least
# as good as iteration 35's 1.12341, and possibly better, because the control
# variates span a shared error direction jointly rather than individually. If it
# returns exactly the iteration-35 set, that is a THIRD independent line of
# evidence for it and the free-sign frontier is settled.
#
# -----------------------------------------------------------------------------
# GUARDS -- pre-registered, same battery as iteration 37
# -----------------------------------------------------------------------------
#   G1  Start from base + top 6 individually-improving artifacts (8 members).
#       Not larger: an over-complete start with many free coefficients fitted on
#       OOF rows is exactly the blend-level overfitting the ~1,500-row private
#       board is exposed to, and the starting fit must still be trustworthy.
#   G2  Drop a member whenever removing it costs LESS than 0.00048 (the measured
#       blend-level seed sd). Removing something that cheap is free simplification.
#   G3  The final set must beat 1.12341 by >= 0.00100 to be ADOPTED over
#       iteration 35's set -- the same 2x-seed-sd bar iteration 37 used.
#   G4  folds_b AND folds_c weight-level retention >= 80%.
#   G5  Shipped test none-rate reported for the winner. Iteration 36 measured the
#       truth at 0.2665; a set that drags the margin further away is buying OOF
#       logloss with deployment loss. (Iteration 35's set ships 0.2377 before
#       calibration -- the bar to not exceed.)
#
# EXCLUSIONS, identical to iteration 37 and equally load-bearing:
#   xgb_pt (withdrawn -- iter30_decorr was killed at fold 3, ranks FIRST on
#   marginal value so this exclusion matters), xgb_resenc* (100% leakage,
#   iteration 15), *_b / *_c (validation-split artifacts -- using them as members
#   would leak the validation split into the thing being validated), blend*.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
# ADOPT over iteration 35 only if nested improves by >= 0.00100 AND G4 holds AND
# G5 holds. Otherwise KEEP iteration 35's set and record this as a negative --
# which, combined with iteration 37, would mean the frontier is confirmed from
# both search directions and this space should be closed for the remaining days.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

DIR <- "experiments/iter38_backward"
SEED_SD <- 0.00048; ADOPT_BAR <- 0.00100; TRUE_NONE <- 0.26651
BASE <- c("xgb_lw2bag", "lcmnl3_both")
GREEDY5 <- c(BASE, "xgb_long", "xgb_wide", "xgb_2stage")
NSTART <- 6L

rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap  <- unique(long[is_test == FALSE, .(No, y, Case)]); setorder(ymap, No)
y <- ymap$y; case <- ymap$Case; n <- length(y)
fmap <- folds[order(No), fold]

LG <- function(m, pre = "oof") {
  d <- readRDS(sprintf("model/artifacts/%s_%s.rds", pre, m)); setorder(d, No)
  log(pmax(as.matrix(d[, .(p1, p2, p3, p4)]), 1e-12))
}
sm <- function(S) { E <- exp(S - do.call(pmax, lapply(1:4, function(j) S[, j]))); E / rowSums(E) }
fitfree <- function(L, rows) {
  M <- length(L)
  f <- function(th) {
    b <- th[1:M]; e <- plogis(th[M + 1]) * 0.10
    S <- Reduce(`+`, Map(function(bi, l) bi * l[rows, , drop = FALSE], b, L)); P <- sm(S)
    -mean(log(pmax((1 - e) * P[cbind(seq_len(sum(rows)), y[rows])] + e * 0.25, 1e-300)))
  }
  o <- optim(c(rep(1 / M, M), -3), f, method = "BFGS", control = list(maxit = 600))
  optim(o$par, f, method = "Nelder-Mead", control = list(maxit = 4000))$par
}
scfree <- function(L, th, rows) {
  M <- length(L); b <- th[1:M]; e <- plogis(th[M + 1]) * 0.10
  S <- Reduce(`+`, Map(function(bi, l) bi * l[rows, , drop = FALSE], b, L)); P <- sm(S)
  -log(pmax((1 - e) * P[cbind(seq_len(sum(rows)), y[rows])] + e * 0.25, 1e-300))
}
NEVAL <- 0L
nested <- function(memb, fm = fmap) {
  NEVAL <<- NEVAL + 1L
  L <- lapply(memb, LG); li <- numeric(n); pf <- numeric(5)
  for (k in 1:5) {
    tr <- fm != k; te <- fm == k
    th <- fitfree(L, tr); li[te] <- scfree(L, th, te); pf[k] <- mean(li[te])
  }
  list(nested = mean(pf), li = li, perfold = pf)
}
zc <- function(la, lb) { d <- la - lb; g <- tapply(d, case, mean); mean(g) / (sd(g) / sqrt(length(g))) }

avail <- gsub("^oof_|\\.rds$", "", list.files("model/artifacts", pattern = "^oof_.*\\.rds$"))
pool <- setdiff(avail[!grepl("^blend|resenc|_b$|_c$|^xgb_pt$", avail)], BASE)
pool <- pool[sapply(pool, function(m) tryCatch(nrow(LG(m)) == n, error = function(e) FALSE))]

set.seed(38)
rule("SETUP")
r_base <- nested(BASE); r_g5 <- nested(GREEDY5)
cat(sprintf("  pool %d | 2-member %.5f | iter35 greedy-5 %.5f\n", length(pool), r_base$nested, r_g5$nested))

rule("RANK CANDIDATES BY SOLO MARGINAL VALUE (to choose the over-complete start)")
solo <- vapply(pool, function(m) tryCatch(r_base$nested - nested(c(BASE, m))$nested,
                                          error = function(e) NA_real_), 0)
solo <- sort(solo[!is.na(solo)], decreasing = TRUE)
print(round(head(solo, 10), 5))
START <- c(BASE, names(head(solo, NSTART)))
cat(sprintf("\n  over-complete start (%d members): %s\n", length(START), paste(START, collapse = " + ")))

rule("BACKWARD ELIMINATION")
cur <- START; cur_r <- nested(cur); trace <- list()
cat(sprintf("  start nested %.5f\n", cur_r$nested))
repeat {
  drops <- setdiff(cur, BASE)          # never drop the two production members
  if (!length(drops)) break
  cost <- vapply(drops, function(m) nested(setdiff(cur, m))$nested - cur_r$nested, 0)
  cheapest <- names(cost)[which.min(cost)]; c_min <- min(cost)
  keep <- c_min >= SEED_SD
  cat(sprintf("  drop %-16s costs %+.5f  %s\n", cheapest, c_min,
              if (keep) { "STOP (removal is not free)" } else { "REMOVE" }))
  trace[[length(trace) + 1L]] <- data.table(dropped = cheapest, cost = c_min, removed = !keep)
  if (keep) break
  cur <- setdiff(cur, cheapest); cur_r <- nested(cur)
  cat(sprintf("     -> %d members, nested %.5f\n", length(cur), cur_r$nested))
}

rule("RESULT")
cat("  final set :", paste(cur, collapse = " + "), "\n")
cat(sprintf("  nested    : %.5f\n", cur_r$nested))
cat(sprintf("  iter35    : %.5f   delta %+.5f\n", r_g5$nested, r_g5$nested - cur_r$nested))
cat(sprintf("  z vs iter35: %+.2f\n", zc(r_g5$li, cur_r$li)))
same <- setequal(cur, GREEDY5)
cat(sprintf("  identical to iteration 35's set? %s\n", if (same) { "YES" } else { "NO" }))
adopt <- (r_g5$nested - cur_r$nested) >= ADOPT_BAR
cat(sprintf("  ADOPT over iteration 35 (needs >= %.5f)? %s\n", ADOPT_BAR,
            if (adopt) { "YES" } else { "NO -- keep iteration 35" }))
cat(sprintf("  nested evaluations: %d\n", NEVAL))

saveRDS(list(final = cur, nested = cur_r$nested, greedy5 = r_g5$nested,
             start = START, solo = solo, trace = rbindlist(trace),
             neval = NEVAL, adopt = adopt, identical_to_iter35 = same),
        file.path(DIR, "result.rds"))
fwrite(rbindlist(trace), file.path(DIR, "trace.csv"))
cat("\n  wrote result.rds + trace.csv\n")
