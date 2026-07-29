# =============================================================================
# ITERATION 28 — adversarial validation: LEARN the train->test shift.
# HYPOTHESIS BEFORE RESULTS.
#
# The repo reweights on ONE marginal at a time: income (shift_audit.R, and a probe
# showed income does not predict none-propensity at all, R^2 = 0.002) or segment
# (the veto metric). Neither is the multivariate density ratio. If respondents are
# drawn from densities p_tr(x) and p_te(x), the correct importance weight is
# w(x) = p_te(x)/p_tr(x), and by Bayes with equal priors that is exactly
# f(x)/(1-f(x)) where f(x) = P(test | x) from a classifier separating the two
# populations. Segment reweighting is the special case where x = segment alone.
#
# HYPOTHESIS: (a) the classifier separates well above chance -- the shift is real
# and multivariate; (b) segment dominates the learned weights, so the existing
# veto metric is a decent approximation; (c) any covariate that adds separation
# BEYOND segment is a shift axis nobody is currently auditing.
# FALSIFIED IF: AUC ~ 0.5 (no learnable shift), or the learned weights correlate
# ~1.0 with segment weights (nothing new to learn).
#
# Respondent-level, not row-level: 1,135 train + 263 test respondents.
# =============================================================================
suppressMessages({ library(data.table); library(glmnet) })
wide <- readRDS("model/artifacts/wide.rds")
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
r <- unique(wide[, c("Case","is_test", demo), with = FALSE])
cat("respondents: train", sum(!r$is_test), " test", sum(r$is_test), "\n\n")

# CORRECTED 26 Jul: segmentind and regionind are CATEGORICAL. The first run of this
# script encoded them as numeric 1..6, which cannot express "segment 6 is 27% of train
# and 0% of test", and produced AUC 0.677 with the false conclusion that segment
# contributes almost nothing. With dummies: AUC 0.908 vs 0.672 without segment.
num <- setdiff(demo, c("segmentind", "regionind"))
X <- as.matrix(r[, ..num]); X[, "incomea"] <- log1p(pmax(X[, "incomea"], 0)); X <- scale(X)
X <- cbind(X, model.matrix(~factor(regionind) - 1, r)[, -1, drop = FALSE],
              model.matrix(~factor(segmentind) - 1, r)[, -1, drop = FALSE])
yv <- as.numeric(r$is_test)
set.seed(42)
cv <- cv.glmnet(X, yv, family = "binomial", alpha = 0.5, nfolds = 10)
pr <- as.numeric(predict(cv, X, s = "lambda.min", type = "response"))
auc <- { o <- order(pr); rk <- rank(pr); (sum(rk[yv==1]) - sum(yv)*(sum(yv)+1)/2)/(sum(yv)*sum(1-yv)) }
cat("=== (a) is the shift learnable? ===\n")
cat("AUC separating test from train respondents:", round(auc, 4), "  (0.5 = no shift)\n\n")

cat("=== which covariates drive it? (non-zero coefficients) ===\n")
co <- as.matrix(coef(cv, s = "lambda.min")); co <- co[co[,1] != 0, , drop = FALSE]
print(round(co[order(-abs(co[,1])), , drop = FALSE], 3))

cat("\n=== (b) how well does SEGMENT alone approximate the learned weights? ===\n")
w_learn <- pr / pmax(1 - pr, 1e-6)
seg <- merge(r[is_test==FALSE, .(ptr=.N/sum(!r$is_test)), by=segmentind],
             r[is_test==TRUE,  .(pte=.N/sum(r$is_test)),  by=segmentind], by="segmentind", all.x=TRUE)
seg[is.na(pte), pte := 0]; seg[, wt := pte/ptr]
w_seg <- merge(r[, .(Case, segmentind)], seg[, .(segmentind, wt)], by="segmentind")[order(Case), wt]
w_l   <- w_learn[order(r$Case)]
tr_idx <- !r$is_test[order(r$Case)]
cat("cor(learned, segment) on TRAIN respondents:", round(cor(w_l[tr_idx], w_seg[tr_idx]), 4), "\n")
cat("Spearman:", round(cor(w_l[tr_idx], w_seg[tr_idx], method="spearman"), 4), "\n")
ess <- function(w) round(sum(w)^2/sum(w^2))
cat("ESS  learned:", ess(w_l[tr_idx]), "  segment:", ess(w_seg[tr_idx]), "  of", sum(tr_idx), "\n")

cat("\n=== (c) what does segment MISS? separation using non-segment covariates only ===\n")
d2 <- setdiff(demo, c("segmentind", "regionind"))
X2 <- scale(as.matrix(r[, ..d2]))
X2 <- cbind(X2, model.matrix(~factor(regionind) - 1, r)[, -1, drop = FALSE]); set.seed(42)
cv2 <- cv.glmnet(X2, yv, family="binomial", alpha=0.5, nfolds=10)
p2 <- as.numeric(predict(cv2, X2, s="lambda.min", type="response"))
auc2 <- { rk <- rank(p2); (sum(rk[yv==1]) - sum(yv)*(sum(yv)+1)/2)/(sum(yv)*sum(1-yv)) }
cat("AUC WITHOUT segment:", round(auc2,4), "   (with segment:", round(auc,4), ")\n")
saveRDS(list(auc=auc, auc_noseg=auc2, coef=co, w_learn=w_l, w_seg=w_seg),
        "experiments/iter28_advval/results.rds")
cat("\nOK\n")
