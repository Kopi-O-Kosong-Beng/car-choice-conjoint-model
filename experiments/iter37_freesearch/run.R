# =============================================================================
# ITERATION 37 -- SYSTEMATIC SEARCH OF THE FREE-SIGN MEMBER SPACE
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# WHY THERE IS A SPACE TO SEARCH AT ALL
# -----------------------------------------------------------------------------
# Iteration 35 found that model/06_blend.R constrained the pool weights to the
# simplex, making NEGATIVE coefficients unrepresentable. Every member-selection
# decision in this project's history was made under that constraint, so the
# whole negative half-space is unexplored. A scan of all 46 OOF artifacts found
# 31 price negative and 15 of those improve the 2-member blend individually.
#
# Iteration 35 then took the greedy top three (xgb_long, xgb_wide, xgb_2stage).
# That was a grab, not a search: COMBINATIONS were never examined, and a set
# chosen for individual marginal value is not the set that works best jointly --
# especially for control variates, whose whole purpose is to span a shared error
# direction, which is a property of the SET.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# A forward-selected member set will beat iteration 35's greedy set by more than
# the blend-level seed sd (0.00048), because the greedy pick optimised each
# member's solo contribution rather than the set's joint coverage of the tree
# family's shared error direction.
#
# -----------------------------------------------------------------------------
# THE REAL RISK, NAMED UP FRONT
# -----------------------------------------------------------------------------
# This is a SELECTION-HEAVY procedure on a fold structure that has already been
# read 35+ times. Forward selection over ~46 candidates for several rounds is
# hundreds of selection events. EXPERIMENTS.md is full of results that died
# exactly this way (iteration 08's monotone constraint survived eighteen
# iterations before being retracted for precisely this reason).
#
# Guards, all pre-registered:
#   G1  CAP AT 6 MEMBERS. Blend parameters are fitted on OOF rows, and
#       blend-level overfitting is what the ~1,500-row private board is exposed
#       to. 2 -> 5 already tripled the parameter count.
#   G2  Each addition must improve nested OOF by >= 0.00100, i.e. 2x the
#       blend-level seed sd of 0.00048. A smaller "gain" is noise.
#   G3  The winning set must be re-validated on folds_b AND folds_c (weight
#       level) and retain >= 80% of its gain over the 2-member baseline, the
#       repo's standing replication bar.
#   G4  Bonferroni over the number of nested evaluations actually performed:
#       the final set's respondent-clustered z must clear qnorm(1 - 0.025/N).
#   G5  Report the shipped test none-rate for every candidate set. Iteration 36
#       measured the true rate at 0.2665; a set that drags the margin further
#       from it is buying OOF logloss with deployment loss.
#
# EXCLUSIONS, fixed before running:
#   * xgb_pt        -- experiments/iter30_decorr/run.R states in its own header
#                      that the run was killed at fold 3 and any xgb_pt artifact
#                      is misnamed and must not be cited. It ranks FIRST on
#                      marginal value, so this exclusion is load-bearing.
#   * xgb_resenc*   -- proven 100% label leakage, iteration 15.
#   * *_b / *_c     -- artifacts built on the INDEPENDENT fold structures. Using
#                      them as members of a folds.rds blend would leak the
#                      validation split into the thing being validated.
#   * blend*        -- blends of blends.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
# ADOPT the searched set over iteration 35's 5-member set only if ALL of:
#   1. nested OOF improves by >= 0.00100 over 1.12341;
#   2. respondent-clustered z vs the 5-member set clears the G4 Bonferroni bar;
#   3. folds_b AND folds_c retention both >= 80% (G3);
#   4. the shipped test none-rate does not move FURTHER from 0.2665 than the
#      5-member set's 0.2377 (G5).
# Otherwise KEEP iteration 35's set and record the search as a negative result.
# A negative here is a genuinely useful finding: it would say the greedy pick
# was already at the frontier, which is worth knowing before spending more time.
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")

DIR <- "experiments/iter37_freesearch"
MAXM <- 6L; MIN_GAIN <- 0.00100; TRUE_NONE <- 0.26651
BASE <- c("xgb_lw2bag", "lcmnl3_both")
GREEDY5 <- c(BASE, "xgb_long", "xgb_wide", "xgb_2stage")

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

# ---- candidate pool ---------------------------------------------------------
avail <- gsub("^oof_|\\.rds$", "", list.files("model/artifacts", pattern = "^oof_.*\\.rds$"))
bad <- grepl("^blend|resenc|_b$|_c$|^xgb_pt$", avail)
pool <- setdiff(avail[!bad], BASE)
pool <- pool[sapply(pool, function(m) tryCatch(nrow(LG(m)) == n, error = function(e) FALSE))]

rule("SETUP")
cat(sprintf("  candidate pool: %d artifacts (after exclusions)\n", length(pool)))
cat(sprintf("  base: %s\n  cap: %d members | min gain per step: %.5f\n",
            paste(BASE, collapse = " + "), MAXM, MIN_GAIN))

set.seed(37)
r_base <- nested(BASE); r_g5 <- nested(GREEDY5)
cat(sprintf("\n  2-member base   %.5f\n  iter35 greedy-5 %.5f\n", r_base$nested, r_g5$nested))

# ---- forward selection ------------------------------------------------------
rule("FORWARD SELECTION")
cur <- BASE; cur_r <- r_base; trace <- list()
while (length(cur) < MAXM) {
  cands <- setdiff(pool, cur)
  sc <- vapply(cands, function(m) tryCatch(nested(c(cur, m))$nested, error = function(e) NA_real_), 0)
  sc <- sc[!is.na(sc)]
  best <- names(sc)[which.min(sc)]; gain <- cur_r$nested - min(sc)
  cat(sprintf("  round %d: best add = %-16s nested %.5f  gain %+.5f  %s\n",
              length(cur) - 1L, best, min(sc), gain,
              if (gain >= MIN_GAIN) { "ACCEPT" } else { "STOP (below min gain)" }))
  trace[[length(trace) + 1L]] <- data.table(round = length(cur) - 1L, added = best,
                                            nested = min(sc), gain = gain,
                                            accepted = gain >= MIN_GAIN)
  if (gain < MIN_GAIN) break
  cur <- c(cur, best); cur_r <- nested(cur)
}
rule("SELECTED SET")
cat("  ", paste(cur, collapse = " + "), "\n")
cat(sprintf("  nested %.5f   vs greedy-5 %.5f   delta %+.5f\n",
            cur_r$nested, r_g5$nested, r_g5$nested - cur_r$nested))
cat(sprintf("  nested evaluations so far: %d\n", NEVAL))

saveRDS(list(selected = cur, trace = rbindlist(trace), nested = cur_r$nested,
             greedy5 = r_g5$nested, base = r_base$nested, neval = NEVAL),
        file.path(DIR, "search.rds"))
fwrite(rbindlist(trace), file.path(DIR, "trace.csv"))
cat("\n  wrote search.rds + trace.csv (checkpoint before validation)\n")
