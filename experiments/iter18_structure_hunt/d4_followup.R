suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); wide <- readRDS("model/artifacts/wide.rds")
NP <- setdiff(ATTRS, "Price")

cat("== 1. does level 0 ever occur on a real bundle for an ACTIVE attribute? ==\n")
real <- long[alt <= 3]
cat("richness (count of nonzero non-price attrs) on real bundles:\n"); print(table(real$richness))
cat("Price on real bundles, min:", min(real$Price), " on none:", unique(long[alt==4]$Price), "\n")
cat("=> level 0 means 'attribute not part of this choice task' (partial profile)\n")

cat("\n== 2. number of levels used per attribute ==\n")
lv <- sapply(ATTRS, function(a) max(real[[a]]))
print(lv)

cat("\n== 3. do versions span train and test? ==\n")
acols3 <- unlist(lapply(1:3, function(j) paste0(ATTRS, j)))
dk <- data.table(Case = wide$Case, Task = wide$Task, is_test = wide$is_test,
                 dkey = do.call(paste, c(wide[, ..acols3], sep="_")))
v <- dk[Task == 1, .(Case, ver = dkey, is_test)]
vs <- v[, .(n = .N, ntest = sum(is_test), ntrain = sum(!is_test)), by = ver]
cat("versions:", nrow(vs), "\n"); print(table(train = vs$ntrain, test = vs$ntest))
cat("test respondents whose version has >=1 TRAIN respondent:",
    round(mean(v[is_test == TRUE]$ver %in% vs[ntrain > 0]$ver), 4), "\n")

cat("\n== 4. does the model already absorb the demographic gradient in the none-rate? ==\n")
tr <- long[is_test == FALSE]; tk <- unique(tr[, .(No, Case, Task, y)]); setorder(tk, No)
oof <- readRDS("model/artifacts/oof_xgb_lw2.rds"); setorder(oof, No)
P <- as.matrix(oof[, .(p1,p2,p3,p4)]); tk[, ph4 := P[,4]]
rl <- tk[, .(nobs = mean(y==4), nhat = mean(ph4)), by = Case]
dem <- unique(tr[, .(Case, incomea, agea, genderind, educind, milesa, segmentind,
                     regionind, Urbind, nighta, pparkind, yearind)])
rl <- merge(rl, dem, by="Case"); rl[, loginc := log(pmax(incomea,1))]; rl[, resid := nobs - nhat]
fml <- ~ loginc + agea + factor(genderind) + educind + milesa + factor(segmentind) +
  factor(regionind) + factor(Urbind) + nighta + factor(pparkind) + yearind
cat("R2 demographics -> observed none rate :", round(summary(lm(update(fml, nobs ~ .), rl))$r.squared,4), "\n")
cat("R2 demographics -> predicted none rate:", round(summary(lm(update(fml, nhat ~ .), rl))$r.squared,4), "\n")
cat("R2 demographics -> RESIDUAL none rate :", round(summary(lm(update(fml, resid ~ .), rl))$r.squared,4),
    " F-p", signif(anova(lm(update(fml, resid ~ .), rl))$`Pr(>F)`[1],3), "\n")

cat("\n== 5. joint support for attribute PAIRS (how many tasks show both) ==\n")
act <- real[, lapply(.SD, function(x) as.integer(any(x>0))), by=No, .SDcols=NP]
A <- as.matrix(act[, ..NP]); co <- crossprod(A)
cat("tasks total:", nrow(A), "; pair co-occurrence min/median/max:",
    min(co[upper.tri(co)]), median(co[upper.tri(co)]), max(co[upper.tri(co)]), "\n")
cat("per-attribute presence rate:", round(mean(diag(co))/nrow(A),3), "\n")
