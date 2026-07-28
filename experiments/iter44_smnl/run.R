# =============================================================================
# ITERATION 44 -- SCALE HETEROGENEITY (S-MNL) VIA gmnl
#
# Everything in this header is written BEFORE any result is looked at.
#
# -----------------------------------------------------------------------------
# THE GAP THIS FILLS -- the only unmodelled dimension left in the logit family
# -----------------------------------------------------------------------------
# Random utility is U_ij = mu_i * (X_ij' beta_i) + eps_ij. Two respondents can
# differ in TWO independent ways:
#
#   beta_i   WHAT they want          (tastes)
#   mu_i     HOW DECISIVELY they choose (scale / inverse error variance)
#
# Every model in this repository varies the first and holds the second fixed:
#   iteration 05  mixed logit          -> continuous taste variation
#   iteration 11  latent class (3)     -> discrete taste classes  [PRODUCTION]
#   iteration 14  4 and 5 classes      -> more taste classes, worse
#   iteration 17  hierarchical Bayes   -> continuous tastes + demographics
#   iteration 04  glmnet interactions  -> taste x demographic
#
# NOTHING has ever let mu_i vary. Every respondent is fitted at one average
# decisiveness. In stated-preference work this is often the LARGER effect: some
# people answer nearly deterministically, others nearly at random, and a model
# forced to average over both is misspecified for each.
#
# WHY IT COULD MATTER HERE SPECIFICALLY. Iteration 18 measured that alternative 4
# is 33% between-respondent while alternatives 1-3 are only 5-7%, and that
# demographics reach just 11.2% of the true none-propensity heterogeneity -- i.e.
# ~89% of what drives the outside option is unobserved. A respondent who answers
# noisily has a FLATTER predicted distribution over all four alternatives, which
# raises their none-probability toward 1/4 without any taste story. Scale is a
# candidate explanation for part of that 89% that no taste model can express.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# An S-MNL (scale heterogeneity, tastes fixed) will beat mnl_pw's 1.15686,
# because it adds a real dimension of heterogeneity that the part-worth logit
# cannot represent at any parameter setting.
#
# -----------------------------------------------------------------------------
# THE HONEST PROBLEM, STATED BEFORE RUNNING -- this is likely to fail
# -----------------------------------------------------------------------------
# Every test respondent is a STRANGER. We never observe their choices, so mu_i
# cannot be estimated for them; the only object we can emit is the
# population-averaged probability INT softmax(mu X beta) dF(mu). Iteration 17
# derived exactly this for tastes and measured the consequence: integrating a
# smooth F over a stranger BLURS the prediction, and hierarchical Bayes came out
# at 1.23703 -- the worst model in the repo.
#
# Scale heterogeneity faces the same structural obstacle. The counter-argument,
# and the reason this is still worth 2 hours: mixing over mu is NOT the same
# operation as mixing over beta. Mixing over beta smears the utility DIRECTION,
# which destroys information. Mixing over mu smears only the TEMPERATURE, and a
# temperature mixture is exactly the shape that a single logit cannot fit but
# that a population of heterogeneously-decisive people generates. It may
# therefore survive the population-averaging that killed HB.
#
# The honest prior is nonetheless BELOW even: iteration 18 already measured that
# per-task-position temperature is worth ~0 honest, and that
# dispersion-conditional temperature is -0.00041 -- i.e. conditioning the
# temperature on OBSERVABLES makes things worse, and the global temperature is
# already right. S-MNL conditions it on an UNOBSERVED mixing distribution
# instead, which is a different object, but the prior evidence is not encouraging.
#
# -----------------------------------------------------------------------------
# DECISION RULE -- fixed before running
# -----------------------------------------------------------------------------
# Compared against mnl_pw (1.15686), the same-family model with the same
# part-worths and no scale heterogeneity. That is the ONLY admissible comparator:
# comparing to the blend or to a tree would confound the model class.
#   1. member level: paired respondent-clustered z >= 2 vs mnl_pw. The
#      model-level seed sd is 0.00283, so eyeballing the OOF difference is not
#      admissible evidence.
#   2. blend level: adding it to the production 2-member pool must improve the
#      nested blend by more than the blend-level seed sd of 0.00048.
#   3. sanity: the estimated scale-heterogeneity parameter (sigma / tau) must be
#      significantly non-zero. If it is ~0 the model has collapsed to plain MNL
#      and any score difference is optimiser noise, not a finding.
# ADOPT only if all three pass. Otherwise record as a negative -- which is the
# expected outcome and is worth having, because it closes the last unexplored
# dimension of the logit family.
#
# ARTIFACTS: oof_smnl.rds / test_smnl.rds  (NEW names -- grep confirmed no
# existing artifact uses "smnl"; an earlier iteration overwrote a live blend
# member by inheriting a name from a copied script).
# =============================================================================
suppressMessages({ library(data.table); library(mlogit); library(dfidx); library(gmnl) })
source("model/99_utils.R")

DIR <- "experiments/iter44_smnl"
dir.create(DIR, showWarnings = FALSE, recursive = TRUE)
R_DRAWS <- 50L          # Halton draws for the scale mixture
rule <- function(s) cat("\n", strrep("=", 78), "\n", s, "\n", strrep("=", 78), "\n", sep = "")

long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

# ---- rebuild mnl_pw's exact design, so the comparison is same-features --------
# Copied verbatim from model/02_mnl_partworth.R rather than eval'ing a slice of
# it, so this script is self-contained and cannot silently drift if that file
# changes. Reference level = the LOWEST level occurring on a REAL bundle
# (alternatives 1-3). For most attributes that is 0; for Price it is 1, because
# Price is 0 only on the all-zero none option and using 0 as reference makes the
# price dummies collinear with the none constant (singular Hessian -- this is a
# documented trap in CLAUDE.md).
ATTRS_L <- character(0)
for (a in ATTRS) {
  lv_real <- sort(unique(long[alt != 4][[a]]))
  ref <- lv_real[1]
  for (l in setdiff(lv_real, ref)) {
    nm <- sprintf("%s_L%s", a, l)
    long[, (nm) := as.numeric(get(a) == l)]
    ATTRS_L <- c(ATTRS_L, nm)
  }
}
cat("part-worth columns built:", length(ATTRS_L), "\n")

cand <- c("asc2", "asc3", "asc4", ATTRS_L)
cand <- cand[cand %in% names(long)]
X  <- as.matrix(long[, ..cand])
Xc <- X - as.matrix(long[, lapply(.SD, mean), by = No, .SDcols = cand][
  match(long$No, unique(long$No)), ..cand])
qrx  <- qr(Xc, tol = 1e-7)
keep <- cand[sort(qrx$pivot[seq_len(qrx$rank)])]
cat("identified parameters:", length(keep), "of", length(cand), "\n")

fml <- as.formula(paste("chosen ~", paste(keep, collapse = " + "), "| 0"))
trl <- merge(long[is_test == FALSE], folds[, .(No, fold)], by = "No")
tel <- long[is_test == TRUE]
setorder(trl, No, alt); setorder(tel, No, alt)

fit_predict <- function(dtr, dva, model = "smnl") {
  dtr_x <- dfidx(as.data.frame(dtr), idx = list(c("No", "Case"), "alt"), choice = "chosen")
  fit <- gmnl(fml, data = dtr_x, model = model, R = R_DRAWS, panel = TRUE,
              halton = NA, method = "bfgs", print.level = 0)
  dva2 <- copy(dva); dva2[, chosen := FALSE]; dva2[alt == 1, chosen := TRUE]
  dva_x <- dfidx(as.data.frame(dva2), idx = list(c("No", "Case"), "alt"), choice = "chosen")
  P <- predict(fit, newdata = dva_x)
  list(P = matrix(P, ncol = 4, byrow = TRUE), nos = sort(unique(dva$No)), fit = fit)
}

rule(sprintf("S-MNL, %d Halton draws, nested over 5 respondent-grouped folds", R_DRAWS))
oof <- data.table(No = sort(unique(trl$No)), p1 = NA_real_, p2 = NA_real_,
                  p3 = NA_real_, p4 = NA_real_)
taus <- numeric(0)
for (k in 1:5) {
  t0 <- Sys.time()
  r <- tryCatch(fit_predict(trl[fold != k], trl[fold == k]),
                error = function(e) { cat("  fold", k, "FAILED:", conditionMessage(e), "\n"); NULL })
  if (is.null(r)) next
  oof[match(r$nos, No), `:=`(p1 = r$P[, 1], p2 = r$P[, 2], p3 = r$P[, 3], p4 = r$P[, 4])]
  cf <- tryCatch(coef(r$fit), error = function(e) numeric(0))
  tau <- cf[grep("tau|sigma|het", names(cf), ignore.case = TRUE)]
  if (length(tau)) taus <- c(taus, tau[1])
  cat(sprintf("  fold %d done in %.1f min%s\n", k,
              as.numeric(difftime(Sys.time(), t0, units = "mins")),
              if (length(tau)) sprintf("   scale param %.4f", tau[1]) else ""))
}

ytr <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
M <- as.matrix(oof[, .(p1, p2, p3, p4)])
if (anyNA(M)) {
  cat("\n!! some folds failed; OOF incomplete -- reporting what exists and stopping\n")
} else {
  ll <- logloss(ytr, M)
  rule("RESULT")
  cat(sprintf("  S-MNL   OOF %.5f\n", ll))
  cat(sprintf("  mnl_pw  OOF %.5f   (the same-family comparator)\n", 1.15686))
  cat(sprintf("  delta       %+.5f\n", 1.15686 - ll))
  if (length(taus)) cat(sprintf("  GATE 3 scale parameter per fold: %s\n",
                                paste(sprintf("%.4f", taus), collapse = ", ")))
  saveRDS(cbind(data.table(No = oof$No), as.data.table(M)),
          "model/artifacts/oof_smnl.rds")
  cat("  wrote model/artifacts/oof_smnl.rds\n")
  rf <- tryCatch(fit_predict(trl, tel), error = function(e) NULL)
  if (!is.null(rf)) {
    saveRDS(data.table(No = rf$nos, p1 = rf$P[,1], p2 = rf$P[,2], p3 = rf$P[,3], p4 = rf$P[,4]),
            "model/artifacts/test_smnl.rds")
    cat("  wrote model/artifacts/test_smnl.rds\n")
  }
}
cat("\nOK iter44\n")
