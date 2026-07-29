# =============================================================================
# THE CENTRAL FINDING, VERIFIED INDEPENDENTLY.
#
# CLAIM UNDER TEST: v7 is the BEST model on plain OOF and one of the WORST on the
# population we are actually graded on. Our wide 4-class xgboost is the culprit -
# it scores 1.1659 plain but ~1.248 on the test mix - and it currently carries
# real weight in the blend. Dropping it and equal-weighting the shared-utility
# family should be free on plain OOF and worth ~0.006 on the graded population.
#
# WHY THIS IS PLAUSIBLE A PRIORI: 60.7% of training respondents come from segments
# that make up 3.4% of the test set, and 68.8% of test comes from segments that are
# 9.4% of train. Plain OOF is dominated by respondents who barely appear in the
# grading population, so a model can win locally by being good at the wrong people.
#
# HONESTY ABOUT THE EVIDENCE: on the test-mix metric the effective sample is ~208
# respondents, so the paired-bootstrap MDE is ~0.012. A 0.006 improvement CANNOT
# reach significance there. This is therefore a STRUCTURAL decision justified by a
# mechanism (a component that is bad on the graded population is being given
# weight), not a significance-gated adoption. The gate we can still apply is: it
# must not COST anything on plain OOF, where we do have resolution.
# =============================================================================
source("exp_harness.R")

mnl    <- readRDS(file.path(CACHE, "mnl.rds"));   ourxgb <- readRDS(file.path(CACHE, "ourxgb.rds"))
mA     <- readRDS(file.path(CACHE, "mateA.rds")); mB     <- readRDS(file.path(CACHE, "mateB.rds"))
base   <- readRDS(file.path(CACHE, "v7_baseline.rds"))
getc <- function(n) { p <- file.path(CACHE, paste0(n, ".rds")); if (file.exists(p)) readRDS(p) else NULL }

# ---- Test-mix weights and weighted metric ----
seg_case <- tapply(as.character(train$segment), train$Case, function(z) z[1])
tr_sh <- prop.table(table(seg_case))
te_sh <- prop.table(table(tapply(as.character(test$segment), test$Case, function(z) z[1])))
w_seg <- setNames(rep(0, length(tr_sh)), names(tr_sh))
cm <- intersect(names(tr_sh), names(te_sh))
w_seg[cm] <- as.numeric(te_sh[cm]) / as.numeric(tr_sh[cm])
row_w <- as.numeric(w_seg[as.character(train$segment)]); row_w[is.na(row_w)] <- 0

wll <- function(P) {
  eps <- 1e-15; P <- pmin(pmax(as.matrix(P), eps), 1 - eps)
  l <- -log(P[cbind(seq_len(nrow(P)), train$Choice)])
  sum(l * row_w) / sum(row_w)
}
idx_by_case <- split(seq_len(nrow(train)), train$Case); uc <- names(idx_by_case)
wboot <- function(new, cur, B = 2000, seed = 777) {
  set.seed(seed); d <- numeric(B)
  for (b in seq_len(B)) {
    idx <- unlist(idx_by_case[sample(uc, length(uc), replace = TRUE)], use.names = FALSE)
    w <- row_w[idx]; if (sum(w) == 0) { d[b] <- 0; next }
    y <- train$Choice[idx]
    ln <- -log(pmin(pmax(new[idx, ], 1e-15), 1))[cbind(seq_along(y), y)]
    lc <- -log(pmin(pmax(cur[idx, ], 1e-15), 1))[cbind(seq_along(y), y)]
    d[b] <- sum((ln - lc) * w) / sum(w)
  }
  list(mean = mean(d), ci = unname(quantile(d, c(0.025, 0.975))), p = mean(d < 0))
}
show <- function(nm, P, ref = base) {
  b <- wboot(P, ref)
  cat(sprintf("%-46s plain %.4f   testmix %.4f   %+.4f [%+.4f,%+.4f] P=%.3f\n",
              nm, competition_logloss(train$Choice, P), wll(P), b$mean, b$ci[1], b$ci[2], b$p))
}

# ---- STEP 1: verify the per-component claim ----
cat("=== components: plain vs test-mix ===\n")
comps <- list(mnl = mnl, ourxgb = ourxgb, mateA = mA, mateB = mB)
for (nm in c("C1_searchwin","C2_softmax_searchwin","sm_eta03_r1500")) {
  P <- getc(nm); if (!is.null(P)) comps[[nm]] <- P
}
for (nm in names(comps))
  cat(sprintf("  %-24s plain %.4f   testmix %.4f\n", nm,
              competition_logloss(train$Choice, comps[[nm]]), wll(comps[[nm]])))
cat(sprintf("  %-24s plain %.4f   testmix %.4f\n", "v7 (blend)",
            competition_logloss(train$Choice, base), wll(base)))

# ---- STEP 2: per-segment breakdown - where is the model actually weak? ----
cat("\n=== v7 by segment ===\n")
ll_row <- -log(pmin(pmax(base[cbind(seq_len(nrow(base)), train$Choice)], 1e-15), 1))
for (s in names(sort(w_seg, decreasing = TRUE))) {
  k <- train$segment == s
  if (!any(k)) next
  cat(sprintf("  %-32s n_resp %4d  weight %5.2f  LogLoss %.4f  alt4 obs %.3f pred %.3f\n",
              s, length(unique(train$Case[k])), w_seg[[s]], mean(ll_row[k]),
              mean(train$Choice[k] == 4), mean(base[k, 4])))
}

# ---- STEP 3: the proposed blend - shared-utility family only, plus a little MNL ----
su_lib <- list(mateA = mA, mateB = mB)
for (nm in c("C1_searchwin", "C2_softmax_searchwin", "sm_eta03_r1500")) {
  P <- getc(nm); if (!is.null(P)) su_lib[[nm]] <- P
}
# add the strongest cached search configs, if present
sf <- list.files(CACHE, pattern = "^su_s[0-9]+_c[0-9]+\\.rds$", full.names = TRUE)
if (length(sf)) {
  sc <- sapply(sf, function(f) { P <- readRDS(f); competition_logloss(train$Choice, norm_rows(P)) })
  for (f in names(sort(sc))[seq_len(min(3, length(sc)))])
    su_lib[[sub("\\.rds$", "", basename(f))]] <- norm_rows(readRDS(f))
}
cat(sprintf("\nShared-utility library: %d members\n", length(su_lib)))

geo_mean <- function(L) {
  Z <- Reduce(`+`, lapply(L, function(P) log(pmax(P, 1e-15)))) / length(L)
  norm_rows(exp(Z - apply(Z, 1, max)))
}
su_blend <- geo_mean(su_lib)

cat("\n=== candidate blends (paired bootstrap is on the TEST MIX, vs v7) ===\n")
show("v7 (reference)", base)
show("SU family only (no MNL, no wide xgb)", su_blend)
for (wm in c(0.05, 0.10, 0.15, 0.20)) {
  show(sprintf("SU family %.2f + MNL %.2f (wide xgb DROPPED)", 1 - wm, wm),
       geo_w(mnl, su_blend, wm))
}
# keep the wide xgboost for comparison, to isolate that it is the problem
show("SU family + MNL 0.10 + wide xgb 0.10",
     geo_w(geo_w(mnl, ourxgb, 0.5), su_blend, 0.20))

# ---- STEP 4: temperature on the graded population ----
cat("\n=== temperature sweep on the proposed blend ===\n")
prop <- geo_w(mnl, su_blend, 0.10)
for (T in c(0.90, 0.95, 1.00, 1.05, 1.10)) {
  P <- temper(prop, T)
  cat(sprintf("  T=%.2f   plain %.4f   testmix %.4f\n", T,
              competition_logloss(train$Choice, P), wll(P)))
}
saveRDS(prop, file.path(CACHE, "testmix_blend.rds"))
cat("\nSaved cache/testmix_blend.rds\n")
