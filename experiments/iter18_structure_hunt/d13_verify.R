# Small confirmations of numbers quoted in the write-up.
suppressMessages(library(data.table))
source("model/99_utils.R")
res <- readRDS("experiments/iter18_structure_hunt/anova_assignment.rds")
cat("== H1: design assignment is a fixed VERSION, so the 19 task-wise ANOVAs per variable\n")
cat("   are the SAME partition repeated. Effective test count = 11 (one per demographic).\n")
u <- res[Task==1][order(p), .(var, F=round(F,3), p=round(p,4))]
print(u)
cat("identical across tasks?", all(res[, uniqueN(round(p,10)), by=var]$V1 == 1), "\n")
cat("Bonferroni-corrected min p over 11 tests:", round(min(u$p)*11, 3), "\n")

long <- readRDS("model/artifacts/long.rds"); tr <- long[is_test==FALSE]
tk <- unique(tr[, .(No, Case, y)])
rl <- tk[, .(o = mean(y==4)), by=Case]
p <- mean(tk$y==4); tot <- p*(1-p)
v_obs <- var(rl$o); v_true <- v_obs - tot/19
dem <- unique(tr[, .(Case, incomea, agea, genderind, educind, milesa, segmentind,
                     regionind, Urbind, nighta, pparkind, yearind)])
rl <- merge(rl, dem, by="Case"); rl[, loginc := log(pmax(incomea,1))]
r2 <- summary(lm(o ~ loginc + agea + factor(genderind) + educind + milesa + factor(segmentind) +
      factor(regionind) + factor(Urbind) + nighta + factor(pparkind) + yearind, rl))$r.squared
cat(sprintf("\n== how much respondent heterogeneity in the NONE decision do demographics reach? ==\n"))
cat(sprintf("   observed var of per-respondent none rate %.4f ; binomial share %.4f ; TRUE %.4f\n",
            v_obs, tot/19, v_true))
cat(sprintf("   demographics R2 on the observed rate %.4f -> explains %.4f of the true variance = %.1f%%\n",
            r2, r2*v_obs, 100*r2*v_obs/v_true))
cat(sprintf("   => ~%.0f%% of the none-propensity heterogeneity is unobservable, and every test\n",
            100*(1-r2*v_obs/v_true)))
cat("      respondent is a stranger, so it can never be conditioned on.\n")
