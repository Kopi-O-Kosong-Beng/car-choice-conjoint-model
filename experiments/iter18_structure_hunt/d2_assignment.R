# Diagnostic 2: how are designs assigned to respondents? Version blocks? Random?
# And does assignment correlate with demographics (a transductive edge if so)?
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds"); wide <- readRDS("model/artifacts/wide.rds")

acols3 <- unlist(lapply(1:3, function(j) paste0(ATTRS, j)))
dk <- data.table(No = wide$No, Case = wide$Case, Task = wide$Task, is_test = wide$is_test,
                 dkey = do.call(paste, c(wide[, ..acols3], sep = "_")))
cat("unique designs total:", uniqueN(dk$dkey), " rows:", nrow(dk), "\n")
cat("designs per task position:\n"); print(dk[, uniqueN(dkey), by = Task][order(Task)]$V1)
cat("does a design appear at more than one task position? ")
print(table(dk[, uniqueN(Task), by = dkey]$V1))

# ---- version structure: does design at task1 predict design at task2? -------
d1 <- dk[Task == 1, .(Case, k1 = dkey)]; d2 <- dk[Task == 2, .(Case, k2 = dkey)]
m <- merge(d1, d2, by = "Case")
cat("\nrespondents:", nrow(m), " distinct k1:", uniqueN(m$k1), " distinct k2:", uniqueN(m$k2), "\n")
cat("distinct (k1,k2) pairs:", uniqueN(m[, paste(k1,k2)]), "\n")
cat("  -> if == distinct k1, the questionnaire is a fixed VERSION block\n")

# generalise: mutual information style -- given k1 group, how many distinct k_t
for (t in c(2,5,10,19)) {
  dt <- dk[Task == t, .(Case, kt = dkey)]
  mm <- merge(d1, dt, by = "Case")
  s <- mm[, .(nd = uniqueN(kt), n = .N), by = k1]
  cat(sprintf("task %2d: mean distinct designs within a task-1 group = %.2f (group size %.2f)\n",
              t, mean(s$nd), mean(s$n)))
}

# ---- version id -------------------------------------------------------------
dk[, vid := .GRP, by = dkey]
vwide <- dcast(dk, Case ~ Task, value.var = "vid")
setnames(vwide, c("Case", paste0("T", 1:19)))
vwide[, sig := do.call(paste, c(.SD, sep="_")), .SDcols = paste0("T", 1:19)]
cat("\ndistinct full 19-task signatures:", uniqueN(vwide$sig), "of", nrow(vwide), "respondents\n")
print(head(sort(table(vwide$sig), decreasing = TRUE), 5))

# ---- demographic dependence: one-way ANOVA of demo on design, per task ------
dem <- unique(wide[, .(Case, incomea, agea, genderind, educind, milesa, segmentind,
                       regionind, Urbind, nighta, pparkind, yearind, is_test)])
dem[, loginc := log(pmax(incomea, 1))]
dk2 <- merge(dk, dem, by = "Case")
vars <- c("loginc","agea","genderind","educind","milesa","segmentind","regionind","Urbind","nighta","pparkind","yearind")
res <- rbindlist(lapply(1:19, function(t) {
  d <- dk2[Task == t]
  rbindlist(lapply(vars, function(v) {
    f <- tryCatch(summary(aov(as.formula(paste(v, "~ factor(dkey)")), data = d))[[1]], error=function(e) NULL)
    if (is.null(f)) return(NULL)
    data.table(Task = t, var = v, F = f$`F value`[1], p = f$`Pr(>F)`[1])
  }))
}))
cat("\n== one-way ANOVA of demographic on design, 19 tasks x 11 vars ==\n")
cat("p-values should be ~Uniform(0,1) under random assignment\n")
print(res[, .(meanF = round(mean(F),3), frac_p05 = round(mean(p < .05),3),
              min_p = signif(min(p),3)), by = var])
cat("\noverall frac p<0.05:", round(mean(res$p < .05), 4), " (expect 0.05)\n")
cat("KS test of p-values vs uniform: ")
print(suppressWarnings(ks.test(res$p, "punif"))$p.value)
saveRDS(res, "experiments/iter18_structure_hunt/anova_assignment.rds")
