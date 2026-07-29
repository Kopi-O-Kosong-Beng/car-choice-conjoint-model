# Collects every el48_oof_* artifact, scores it, and prints the leak table.
suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")
y <- unique(long[is_test == FALSE, .(No, y)])[order(No), y]
fmap <- folds[order(No), fold]
fs <- list.files("experiments/iter48_encleak", pattern = "^el48_oof_.*\\.rds$", full.names = TRUE)
if (!length(fs)) stop("nothing yet")
rows <- rbindlist(lapply(fs, function(f) {
  b <- sub("^el48_oof_", "", sub("\\.rds$", "", basename(f)))
  p <- strsplit(b, "_")[[1]]
  d <- readRDS(f); setorder(d, No); P <- as.matrix(d[, .(p1,p2,p3,p4)])
  ll <- logloss(y, P)
  pf <- sapply(1:5, function(k) logloss(y[fmap == k], P[fmap == k, ]))
  data.table(arm = p[1], depth = as.integer(sub("d","",p[2])), mcw = as.integer(sub("m","",p[3])),
             eta = p[4], nr = as.integer(sub("n","",p[5])), seed = as.integer(sub("s","",p[6])),
             oof = ll, f1=pf[1], f2=pf[2], f3=pf[3], f4=pf[4], f5=pf[5])
}))
setorder(rows, depth, mcw, eta, nr, seed, arm)
cat("\n=== raw ===\n"); print(rows[, .(arm, depth, mcw, eta, nr, seed, oof = round(oof,5))], nrows = 200)

w <- dcast(rows, depth + mcw + eta + nr + seed ~ arm, value.var = "oof")
for (cc in c("prod","noenc","honest","leaky")) if (!cc %in% names(w)) w[[cc]] <- NA_real_
w[, `:=`(prod_gain   = noenc - prod,       # what production claims the encoding is worth
         honest_gain = noenc - honest,     # airtight value of the encoding, 3-fold support
         leaky_gain  = noenc - leaky,      # same support, leak channel open
         leak_3fold  = honest - leaky,     # >0 => leaky is better => that much is LEAK
         support_eff = leaky - prod)]      # >0 => prod better than leaky => support, not leak
cat("\n=== encoding value decomposition (positive = better/lower logloss) ===\n")
print(w[, .(depth, mcw, eta, nr, seed,
            noenc = round(noenc,5), prod = round(prod,5), leaky = round(leaky,5), honest = round(honest,5),
            prod_gain = round(prod_gain,5), honest_gain = round(honest_gain,5),
            leak_3fold = round(leak_3fold,5), support_eff = round(support_eff,5))], nrows = 200)

# per-fold consistency of the leak (CLAUDE.md: a real gain shows in every fold)
cat("\n=== per-fold: leaky minus honest (positive = leak helps that fold) ===\n")
pf <- merge(rows[arm == "leaky", .(depth, seed, l1=f1,l2=f2,l3=f3,l4=f4,l5=f5)],
            rows[arm == "honest", .(depth, seed, h1=f1,h2=f2,h3=f3,h4=f4,h5=f5)],
            by = c("depth","seed"))
if (nrow(pf)) print(pf[, .(depth, seed, k1 = round(h1-l1,4), k2 = round(h2-l2,4), k3 = round(h3-l3,4),
                           k4 = round(h4-l4,4), k5 = round(h5-l5,4))])
saveRDS(w, "experiments/iter48_encleak/el48_summary.rds")
cat("\nCOLLECT_DONE\n")
