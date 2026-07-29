# =============================================================================
# ITERATION 58 -- diag2.R  READ-ONLY. Emits NO artifacts into model/artifacts/.
#
# diag.R established:
#   * lcmnl3_both's membership softmax is NOWHERE NEAR saturated (mean max pi
#     0.478 on test, ZERO respondents above 0.9) and assigns TEST luxury
#     respondents almost exactly the class shares it assigns TRAIN luxury
#     respondents (0.441/0.384/0.176 vs 0.450/0.374/0.176). There is no
#     extrapolation happening on the luxury axis.
#   * OOF per-segment: lcmnl3_both lux 0.18252 vs observed 0.15986 (+0.023);
#     xgb_lw2bag lux 0.25903 vs observed 0.15986 (+0.099).
#
# So the LATENT-CLASS model is the one that reproduces the training population,
# and the TREE is the one with a measured in-population defect (iteration 29).
# Yet the tree ships the near-exact global test none-rate. This script tests the
# implication: if the tree's global accuracy is a COINCIDENCE produced by its own
# defect, then a tree WITHOUT the defect must ship a WORSE global none-rate.
# xgb_lw3 (iteration 29, alt4 x segment carriers) is exactly that tree.
#
# D5  panel of every available model: OOF and TEST none-rate by segment.
# D6  is the luxury effect a small-sample artefact? respondent-clustered SE and
#     the empirical-Bayes shrinkage factor it implies. This is the a-priori test
#     of whether ANY shrinkage of the demographic channel is warranted.
# D7  are TEST luxury respondents demographically different from TRAIN luxury
#     respondents? If not, no feature-based model can tell them apart.
# D8  direction check on "fit the membership model on segment-reweighted
#     respondents", done cheaply: refit G alone on the production soft targets
#     with respondent weights, holding betas and H fixed.
# =============================================================================
suppressMessages({ library(data.table) })
source("model/99_utils.R")
R_STAR <- (1.7918 - 1.499) / 1.0986

long <- readRDS("model/artifacts/long.rds"); setorder(long, No, alt)
tasks <- unique(long[, .(No, Case, Task, y, is_test, segmentind, incomea, agea,
                         educind, regionind, Urbind, milesa, nighta, genderind,
                         pparkind, yearind, milesind, nightind)])
setorder(tasks, No)
tasks[, lux := as.integer(segmentind %in% c(3, 5))]
tr_t <- which(tasks$is_test == FALSE); te_t <- which(tasks$is_test == TRUE)
lux_tr <- tasks$lux[tr_t]; lux_te <- tasks$lux[te_t]
ytr <- tasks$y[tr_t]

grab <- function(f) {
  d <- readRDS(f)
  if (is.matrix(d)) return(d[, c("p1","p2","p3","p4"), drop = FALSE])
  setorder(d, No); as.matrix(d[, .(p1,p2,p3,p4)])
}

# =============================================================================
# D5 -- the panel
# =============================================================================
cat("=== D5: none-rate by segment, every available model ===\n")
cat(sprintf("observed TRAIN            all %.5f  lux %.5f  non %.5f\n",
            mean(ytr==4), mean(ytr[lux_tr==1]==4), mean(ytr[lux_tr==0]==4)))
cat(sprintf("measured TEST r*          all %.5f  lux ?        non ?\n\n", R_STAR))
cat(sprintf("%-16s | %-27s | %-27s\n", "model", "OOF (train population)", "TEST (shipped)"))
cat(sprintf("%-16s | %8s %8s %8s | %8s %8s %8s | %9s\n",
            "", "all", "lux", "non", "all", "lux", "non", "OOF ll"))
cat(strrep("-", 90), "\n")
mods <- c("lcmnl3_both","lcmnl3_tilt","lcmnl3_drift","lcmnl3","mnl_pw",
          "xgb_lw2bag","xgb_lw2","xgb_lw3","xgb_mono","hbmnl_nod","blend")
tab <- data.table()
for (m in mods) {
  fo <- sprintf("model/artifacts/oof_%s.rds", m); ft <- sprintf("model/artifacts/test_%s.rds", m)
  if (!file.exists(fo) || !file.exists(ft)) next
  O <- grab(fo); T <- grab(ft)
  if (nrow(O) != length(tr_t) || nrow(T) != length(te_t)) next
  r <- data.table(model = m,
    oof_all = mean(O[,4]), oof_lux = mean(O[lux_tr==1,4]), oof_non = mean(O[lux_tr==0,4]),
    te_all  = mean(T[,4]), te_lux  = mean(T[lux_te==1,4]), te_non  = mean(T[lux_te==0,4]),
    oof_ll  = logloss(ytr, O))
  tab <- rbind(tab, r)
  cat(sprintf("%-16s | %8.5f %8.5f %8.5f | %8.5f %8.5f %8.5f | %9.5f\n",
              m, r$oof_all, r$oof_lux, r$oof_non, r$te_all, r$te_lux, r$te_non, r$oof_ll))
}
cat("\n  OOF-luxury MISS (model - observed 0.15986) vs global TEST MISS (model - r*):\n")
tab[, oof_lux_miss := oof_lux - mean(ytr[lux_tr==1]==4)]
tab[, te_all_miss  := te_all - R_STAR]
for (i in seq_len(nrow(tab)))
  cat(sprintf("    %-16s in-population defect %+.5f   ->   global test miss %+.5f\n",
              tab$model[i], tab$oof_lux_miss[i], tab$te_all_miss[i]))
cc <- cor(tab$oof_lux_miss, tab$te_all_miss)
cat(sprintf("\n  correlation across %d models: %.4f\n", nrow(tab), cc))
cat("  (a POSITIVE correlation means the models closest to the measured global\n")
cat("   test none-rate are exactly the ones MOST wrong about luxury in-population)\n\n")

# =============================================================================
# D6 -- is the luxury effect precisely enough estimated to deserve shrinkage?
# =============================================================================
cat("=== D6: respondent-clustered precision of the training luxury effect ===\n")
per_resp <- tasks[is_test == FALSE, .(nr = mean(y == 4), lux = lux[1], seg = segmentind[1]),
                  by = Case]
a <- per_resp[lux == 1]$nr; b <- per_resp[lux == 0]$nr
se <- sqrt(var(a)/length(a) + var(b)/length(b))
d  <- mean(a) - mean(b)
cat(sprintf("  luxury resp n=%d  mean none-rate %.5f (sd %.4f)\n", length(a), mean(a), sd(a)))
cat(sprintf("  other  resp n=%d  mean none-rate %.5f (sd %.4f)\n", length(b), mean(b), sd(b)))
cat(sprintf("  difference %+.5f   clustered SE %.5f   z = %.2f\n", d, se, d/se))
Bshrink <- d^2 / (d^2 + se^2)
cat(sprintf("  empirical-Bayes retention factor  d^2/(d^2+SE^2) = %.4f\n", Bshrink))
cat("  => shrinking this effect toward zero is NOT warranted by its own precision.\n")
cat(sprintf("  95%% CI on the training luxury none-rate: [%.4f, %.4f]\n",
            mean(a) - 1.96*sd(a)/sqrt(length(a)), mean(a) + 1.96*sd(a)/sqrt(length(a))))
r_lux_B <- (R_STAR - (1 - mean(lux_te)) * mean(ytr[lux_tr==0]==4)) / mean(lux_te)
cat(sprintf("  story-B implied TEST luxury rate %.4f is %.1f clustered SEs from the\n",
            r_lux_B, (r_lux_B - mean(a)) / (sd(a)/sqrt(length(a)))))
cat("  training luxury rate -- a POPULATION difference, not estimation error.\n\n")

cat("  per-segment observed training none-rate (respondent-clustered):\n")
ps <- per_resp[, .(n = .N, nr = mean(nr), se = sd(nr)/sqrt(.N)), by = seg][order(seg)]
te_resp <- unique(tasks[is_test == TRUE, .(Case, segmentind)])
ps <- merge(ps, te_resp[, .(seg = segmentind, n_test = .N)][, .(n_test = sum(n_test)), by = seg],
            by = "seg", all.x = TRUE)
print(ps, digits = 4)
cat("\n")

# =============================================================================
# D7 -- can a feature-based model TELL test luxury respondents apart?
# =============================================================================
cat("=== D7: TRAIN-luxury vs TEST-luxury demographics ===\n")
resp <- unique(tasks[, .(Case, is_test, lux, segmentind, incomea, agea, educind,
                         regionind, Urbind, milesa, nighta, genderind, pparkind,
                         yearind, milesind, nightind)])
V <- c("incomea","agea","educind","regionind","Urbind","milesa","nighta",
       "genderind","pparkind","yearind","milesind","nightind")
L <- resp[lux == 1]
cat(sprintf("  %-12s %10s %10s %10s %8s\n", "variable", "train-lux", "TEST-lux", "diff", "|d|/sd"))
for (v in V) {
  x <- L[is_test == FALSE][[v]]; z <- L[is_test == TRUE][[v]]
  s <- sd(c(x, z))
  cat(sprintf("  %-12s %10.3f %10.3f %+10.3f %8.3f\n", v, mean(x), mean(z),
              mean(z) - mean(x), abs(mean(z) - mean(x)) / s))
}
cat("  segmentind mix within luxury:\n")
cat(sprintf("    train  seg3 %.3f  seg5 %.3f\n",
            mean(L[is_test==FALSE]$segmentind == 3), mean(L[is_test==FALSE]$segmentind == 5)))
cat(sprintf("    TEST   seg3 %.3f  seg5 %.3f\n",
            mean(L[is_test==TRUE]$segmentind == 3), mean(L[is_test==TRUE]$segmentind == 5)))
cat("\n")

# =============================================================================
# D8 -- direction of "membership fitted on segment-reweighted respondents"
# =============================================================================
cat("=== D8: refit G alone under respondent reweighting (betas and H frozen) ===\n")
cat("  This is NOT a full EM refit. It answers ONE question cheaply: which way\n")
cat("  does upweighting luxury respondents push the shipped none-rate?\n")
fit <- readRDS("experiments/iter11_latent_class/fit_C3.rds")
keep <- fit$keep
pw <- character(0)
for (aa in ATTRS) {
  lv <- sort(unique(long[alt != 4][[aa]])); ref <- lv[1]
  for (l in setdiff(lv, ref)) {
    nm <- sprintf("%s_L%s", aa, l); long[, (nm) := as.numeric(get(aa) == l)]; pw <- c(pw, nm)
  }
}
long[, Task_c := (as.numeric(Task) - 10) / 9]
long[, none_x_Task := asc4 * Task_c]; long[, Price_x_Task := Price * Task_c]
Xall <- as.matrix(long[, ..keep])
DEMO_CAT <- c("segmentind","pparkind","genderind","educind","regionind","Urbind")
DEMO_NUM <- c("agea","incomea","milesa","nighta","yearind","milesind","nightind")
dem <- unique(long[, c("Case","is_test",DEMO_CAT,DEMO_NUM), with = FALSE]); setorder(dem, Case)
Zl <- list(Intercept = rep(1, nrow(dem)))
for (v in DEMO_CAT) { lv <- sort(unique(dem[[v]])); for (l in lv[-1])
  Zl[[sprintf("%s_%s", v, l)]] <- as.numeric(dem[[v]] == l) }
tf <- function(v, x) if (v %in% c("incomea","milesa","nighta")) log1p(x) else x
for (v in DEMO_NUM) { x <- tf(v, as.numeric(dem[[v]])); Zl[[v]] <- (x - mean(x))/sd(x) }
Z <- do.call(cbind, Zl); rownames(Z) <- as.character(dem$Case)
stopifnot(identical(colnames(Z), fit$zcols))

cl_prob <- function(X, beta) {
  Vm <- matrix(as.vector(X %*% beta), ncol = 4, byrow = TRUE)
  mx <- pmax(Vm[,1],Vm[,2],Vm[,3],Vm[,4]); E <- exp(Vm - mx); E / rowSums(E)
}
row_softmax <- function(eta) { mx <- do.call(pmax, lapply(seq_len(ncol(eta)), function(j) eta[,j]))
  E <- exp(eta - mx); E / rowSums(E) }
row_lse <- function(eta) { mx <- do.call(pmax, lapply(seq_len(ncol(eta)), function(j) eta[,j]))
  mx + log(rowSums(exp(eta - mx))) }
rows_of <- function(ti) as.vector(rbind(4L*ti-3L,4L*ti-2L,4L*ti-1L,4L*ti))

perms <- list(c(1,2,3),c(1,3,2),c(2,1,3),c(2,3,1),c(3,1,2),c(3,2,1))
ref <- readRDS("model/artifacts/test_lcmnl3_both.rds"); setorder(ref, No)
predP <- function(ti, G, betas, scale = 1) {
  X <- Xall[rows_of(ti),,drop=FALSE]
  Pi <- row_softmax(cbind(0, scale * (Z[as.character(tasks$Case[ti]),,drop=FALSE] %*% G)))
  out <- matrix(0, length(ti), 4)
  for (cc in seq_along(betas)) out <- out + Pi[,cc] * cl_prob(X, betas[[cc]]); out
}
bd <- sapply(perms, function(p) max(abs(clip_norm(predP(te_t, fit$G, fit$betas[p])) -
                                        as.matrix(ref[,.(p1,p2,p3,p4)]))))
BETAS <- fit$betas[perms[[which.min(bd)]]]
stopifnot(min(bd) < 1e-6)

# recover the production posterior H on training respondents (E-step at the fitted params)
ucase <- sort(unique(tasks$Case[tr_t]))
respi <- match(tasks$Case[tr_t], ucase)
Ztr <- Z[as.character(ucase),,drop=FALSE]
LLr <- matrix(0, length(ucase), 3)
Xtr <- Xall[rows_of(tr_t),,drop=FALSE]
for (cc in 1:3) {
  P <- cl_prob(Xtr, BETAS[[cc]])
  llt <- log(P[cbind(seq_along(ytr), ytr)])
  LLr[,cc] <- as.vector(rowsum(llt, respi, reorder = TRUE))
}
logpi <- log(pmax(row_softmax(cbind(0, Ztr %*% fit$G)), 1e-12))
H <- row_softmax(logpi + LLr)
cat(sprintf("  recovered posterior shares: %s (log says 0.367 0.328 0.305 after relabel)\n",
            paste(sprintf("%.3f", colMeans(H)), collapse = " ")))

fit_G <- function(Z, H, w, ridge, G0) {
  p <- ncol(Z); C <- ncol(H)
  fn <- function(g) { G <- matrix(g,p,C-1); eta <- cbind(0, Z %*% G)
    -(sum(w * H * (eta - row_lse(eta))) - 0.5*ridge*sum(g^2)) }
  gr <- function(g) { G <- matrix(g,p,C-1); eta <- cbind(0, Z %*% G); Pi <- row_softmax(eta)
    -as.vector(crossprod(Z, (w * (H - Pi))[,-1,drop=FALSE]) - ridge*G) }
  matrix(optim(as.vector(G0), fn, gr, method = "BFGS", control = list(maxit = 500))$par, p, C-1)
}
lux_resp <- as.integer(unique(tasks[is_test==FALSE, .(Case, lux)])[match(ucase, Case)]$lux)
cat(sprintf("\n  %-34s %9s %9s %9s %9s\n", "membership variant", "test all", "test lux", "test non", "trainLL"))
scen <- list(
  list(nm = "production  (lambda_g = 2, w = 1)",      w = rep(1, length(ucase)), lam = 2),
  list(nm = "lambda_g = 8",                            w = rep(1, length(ucase)), lam = 8),
  list(nm = "lambda_g = 32",                           w = rep(1, length(ucase)), lam = 32),
  list(nm = "lambda_g = 128",                          w = rep(1, length(ucase)), lam = 128),
  list(nm = "lambda_g = 512",                          w = rep(1, length(ucase)), lam = 512),
  list(nm = "reweight lux to TEST share (w=21.4x)",    w = ifelse(lux_resp==1, 0.688/0.09427, (1-0.688)/(1-0.09427)), lam = 2),
  list(nm = "reweight lux 5x",                         w = ifelse(lux_resp==1, 5, 1), lam = 2))
for (s in scen) {
  G2 <- fit_G(Ztr, H, s$w, s$lam, fit$G)
  Pt <- predP(te_t, G2, BETAS); Po <- clip_norm(predP(tr_t, G2, BETAS))
  cat(sprintf("  %-34s %9.5f %9.5f %9.5f %9.5f\n", s$nm,
              mean(Pt[,4]), mean(Pt[lux_te==1,4]), mean(Pt[lux_te==0,4]), logloss(ytr, Po)))
}
cat(sprintf("\n  (trainLL is IN-SAMPLE and only shows direction; the honest number needs a\n",
            ""))
cat("   nested 5-fold refit, which run.R does.)\n")
cat("\nOK\n")
