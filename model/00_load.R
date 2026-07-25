suppressMessages(library(data.table))
source("model/99_utils.R")

tr <- fread("Raw Dump/Competition Data/train2024.csv")
te <- fread("Raw Dump/Competition Data/test2024.csv")

# --- integrity checks (fail loudly, they encode assumptions) ---
stopifnot(nrow(tr) == 21565, nrow(te) == 4997)
stopifnot(max(tr$Case) < min(te$Case))                       # disjoint respondents
stopifnot(all(rowSums(tr[, .(Ch1, Ch2, Ch3, Ch4)]) == 1L))   # exactly one choice
a4 <- paste0(ATTRS, "4")
cat("alt-4 attribute max over all rows (expect 0):",
    max(as.matrix(rbind(tr, te, fill = TRUE)[, ..a4]), na.rm = TRUE), "\n")

tr[, y := max.col(cbind(Ch1, Ch2, Ch3, Ch4))]
te[, y := NA_integer_]
te[, intersect(c("Ch1","Ch2","Ch3","Ch4"), names(te)) := NULL]
all_dt <- rbind(tr, te, fill = TRUE)
all_dt[, is_test := No > 21565L]

demo_keep <- c("segment","segmentind","yearind","milesind","milesa","nightind","nighta",
               "ppark","pparkind","genderind","ageind","agea","educind",
               "region","regionind","Urbind","incomeind","incomea")
idv <- c("No","Case","Task","y","is_test", demo_keep)
mv  <- setNames(lapply(ATTRS, function(a) paste0(a, 1:4)), ATTRS)
long <- melt(all_dt, id.vars = idv, measure.vars = mv, variable.name = "alt")
long[, alt := as.integer(alt)]
long[, chosen := (alt == y)]          # NA for test rows
setorder(long, No, alt)

# --- engineered features (within task group No) ---
np <- setdiff(ATTRS, "Price")
long[, richness := rowSums(.SD != 0), .SDcols = np]
long[, lvlsum   := rowSums(.SD),      .SDcols = np]

# matrix layout: rows sorted by (No, alt) => each task is 4 consecutive rows
PM <- matrix(long$Price, ncol = 4, byrow = TRUE)
rank_m <- t(apply(PM, 1, rank, ties.method = "average"))
minr_m <- sapply(1:4, function(j) do.call(pmin, as.data.frame(PM[, -j, drop = FALSE])))
long[, price_rank      := as.vector(t(rank_m))]
long[, price_min_rival := as.vector(t(minr_m))]
long[, `:=`(price_task_mean = mean(Price), rich_task_mean = mean(richness)), by = No]

# task-centered attributes via one fast grouped-mean join
tm <- long[, lapply(.SD, mean), by = No, .SDcols = ATTRS]
setnames(tm, ATTRS, paste0(ATTRS, "_tm"))
long <- tm[long, on = "No"]
for (a in ATTRS) set(long, j = paste0(a, "_c"), value = long[[a]] - long[[paste0(a, "_tm")]])
long[, paste0(ATTRS, "_tm") := NULL]
setorder(long, No, alt)

for (j in 1:4) long[, paste0("alt", j) := as.numeric(alt == j)]
long[, `:=`(asc2 = alt2, asc3 = alt3, asc4 = alt4)]

# --- MNL interaction columns (explicit, so glmnet/manual X reuse them) ---
long[, Price_x_age   := Price * agea / 100]
long[, Price_x_ppark := Price * pparkind]
long[, Price_x_inc   := Price * incomea / 1e5]   # test respondents skew rich; get this slope right
for (s in sort(unique(long$segmentind))[-1]) long[, paste0("Price_x_seg", s) := Price * (segmentind == s)]
for (r in sort(unique(long$regionind))[-1])  long[, paste0("Price_x_reg", r) := Price * (regionind == r)]

dir.create("model/artifacts", recursive = TRUE, showWarnings = FALSE)
saveRDS(long,  "model/artifacts/long.rds")
saveRDS(all_dt, "model/artifacts/wide.rds")
cat("train tasks:", uniqueN(long[is_test == FALSE, No]), " test tasks:", uniqueN(long[is_test == TRUE, No]), "\n")
cat("choice shares train:", tr[, round(colMeans(cbind(Ch1, Ch2, Ch3, Ch4)), 4)], "\n")
cat("tasks per respondent (train):", tr[, .N, by = Case][, paste(range(N), collapse = "-")], "\n")
cat("OK: artifacts written\n")
