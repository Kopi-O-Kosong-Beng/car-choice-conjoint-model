# =============================================================================
# ITERATION 82 — MODEL-DISTANCE AUDIT ACROSS OUR TWO TRACKS
#
# The 50/50 cross-track pool is built in LOG-PROBABILITY space, where w = 0.5 is
# exactly halfway by construction. Two questions that w does not answer:
#
#   (a) Is the pool halfway in MODEL space, or does one track dominate it?
#   (b) Is pooling a genuine diversity hedge at all -- i.e. are our two tracks
#       actually far apart, or are they near-duplicates wearing different names?
#
# A single distance cannot settle either, because two files can be far apart
# purely because of CALIBRATION (a one-parameter margin shift anyone can apply)
# while being the same MODEL underneath. So every comparison is decomposed:
#
#   MARGIN       mean p4 and its per-row spread   -- one scalar, trivially changed
#   CONDITIONAL  (p1,p2,p3)/(1-p4)                -- this is the model
#
# and read against a NULL: how far apart do models *we* built independently sit?
# Without that scale, no distance means anything.
#
# Every artifact read here is tracked in this repository. There is no external
# input and nothing is fitted -- this script only measures.
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter82_provenance/track_distances.R
# =============================================================================
suppressMessages(library(data.table))
setwd("d:/SUTD/Term5/Analytics Edge/Competition")

norm  <- function(P) { P <- pmax(as.matrix(P), 1e-12); P / rowSums(P) }
rdcsv <- function(p) { d <- fread(p); setorder(d, No); norm(d[, .(Ch1,Ch2,Ch3,Ch4)]) }
rdrds <- function(p) { d <- readRDS(p); setorder(d, No); norm(d[, .(p1,p2,p3,p4)]) }
kl    <- function(P,Q) mean(rowSums(P*log(pmax(P,1e-12)/pmax(Q,1e-12))))
sym   <- function(P,Q) 0.5*(kl(P,Q)+kl(Q,P))
cond  <- function(Q) { C <- Q[,1:3]/(1-Q[,4]); C/rowSums(C) }

# The two tracks, both anchored to the probe-measured r*, so the margin is held
# equal and any remaining distance is the model.
MAIN  <- rdcsv("submissions/sub_20260730_final00_reconstructed.csv")  # track 1
TRK2  <- rdcsv("submissions/cand_nnblend_anchored.csv")               # track 2
POOL  <- rdcsv("submissions/cand_pool5050_final00.csv")               # the 50/50 of them

cat("=========================================================================\n")
cat("PART 1 -- HOW FAR APART ARE OUR TWO TRACKS?\n")
cat("=========================================================================\n")
cat(sprintf("\n%-42s %11s %11s\n", "pair", "symKL_full", "symKL_cond"))
cat(sprintf("%-42s %11.5f %11.5f\n", "main track (final00) vs track 2 (nnblend)",
            sym(MAIN,TRK2), sym(cond(MAIN),cond(TRK2))))
cat("\nper-class correlation of the CONDITIONAL, main track vs track 2:\n")
for (j in 1:3) cat(sprintf("   cond p%d : %.6f\n", j, cor(cond(MAIN)[,j], cond(TRK2)[,j])))
cat(sprintf("   p4 (margin, both anchored to r*) : %.6f\n", cor(MAIN[,4], TRK2[,4])))
cat(sprintf("\ntop-ranked bundle differs on %.2f%% of the 4,997 test rows\n",
            100*mean(max.col(MAIN)!=max.col(TRK2))))

cat("\n=========================================================================\n")
cat("PART 2 -- WHERE DOES THE 50/50 POOL ACTUALLY SIT?\n")
cat("=========================================================================\n")
cat("w = 0.5 is halfway in LOG space. In model space it need not be.\n\n")
cat(sprintf("%-42s %11s %11s\n", "pool vs", "symKL_full", "symKL_cond"))
for (nm in c("main track (final00)","track 2 (nnblend anchored)")) {
  X <- if (nm == "main track (final00)") MAIN else TRK2
  cat(sprintf("%-42s %11.5f %11.5f\n", nm, sym(POOL,X), sym(cond(POOL),cond(X))))
}
sc_m <- sym(cond(POOL),cond(MAIN)); sc_t <- sym(cond(POOL),cond(TRK2))
cat(sprintf("\nratio (distance to main) / (distance to track 2) = %.2f\n", sc_m/sc_t))
cat("1.00 would mean the pool is genuinely central in model space.\n")

cat("\n=========================================================================\n")
cat("PART 3 -- THE NULL: how far apart are models WE built independently?\n")
cat("=========================================================================\n")
cat("Without this scale, Part 1's number is uninterpretable.\n")
pairs <- list(
  c("xgb_lw2bag","xgb_lw2"),       # same model, 10-seed bag vs single seed
  c("xgb_lw2bag","xgb_syn"),       # both listwise xgb, separately built
  c("xgb_long","xgb_wide"),
  c("lcmnl3_both","mnl_pw"),
  c("xgb_lw2bag","xgb_wide"),
  c("mixl2","lcmnl3_both"),
  c("xgb_lw2bag","lcmnl3_both"),   # the widest axis the blend spans
  c("xgb_lw2bag","mnl_pw")
)
cat(sprintf("\n%-34s %11s %11s\n", "our own pair", "symKL_full", "symKL_cond"))
nullc <- c()
for (p in pairs) {
  fa <- sprintf("model/artifacts/test_%s.rds", p[1]); fb <- sprintf("model/artifacts/test_%s.rds", p[2])
  if (!file.exists(fa) || !file.exists(fb)) next
  A <- rdrds(fa); B <- rdrds(fb)
  sc <- sym(cond(A),cond(B)); nullc <- c(nullc, sc)
  cat(sprintf("%-34s %11.5f %11.5f\n", paste(p, collapse=" vs "), sym(A,B), sc))
}
cat(sprintf("\nSMALLEST conditional distance between two models we built : %.5f\n", min(nullc)))
cat(sprintf("WIDEST   conditional distance between two models we built : %.5f\n", max(nullc)))
cat(sprintf("our two TRACKS sit at                                     : %.5f\n",
            sym(cond(MAIN),cond(TRK2))))
cat("\nREADING: if the two tracks sat below the same-model-reseeded floor, the\n")
cat("pool would be buying nothing. Above the floor, it is a real hedge.\n")

cat("\n=========================================================================\n")
cat("PART 4 -- FULL RANKING: every artifact this repo owns, vs the main track\n")
cat("=========================================================================\n")
memb <- gsub("^test_|\\.rds$", "", list.files("model/artifacts", pattern = "^test_.*\\.rds$"))
rows <- list()
for (m in memb) {
  P <- try(rdrds(sprintf("model/artifacts/test_%s.rds", m)), silent = TRUE)
  if (inherits(P,"try-error") || nrow(P) != 4997L) next
  rows[[length(rows)+1]] <- data.table(artifact = m, kind = "member",
    symKL_full = sym(P,MAIN), symKL_cond = sym(cond(P),cond(MAIN)))
}
for (f in list.files("submissions", pattern="\\.csv$", full.names=TRUE)) {
  P <- try(rdcsv(f), silent = TRUE)
  if (inherits(P,"try-error") || nrow(P) != 4997L) next
  rows[[length(rows)+1]] <- data.table(artifact = basename(f), kind = "submission",
    symKL_full = sym(P,MAIN), symKL_cond = sym(cond(P),cond(MAIN)))
}
NN <- rbindlist(rows); setorder(NN, symKL_cond)
cat(sprintf("\n%d artifacts compared (%d members, %d submissions).\n",
            nrow(NN), sum(NN$kind=="member"), sum(NN$kind=="submission")))
cat("\nCLOSEST 12 to the main track, by CONDITIONAL (calibration removed):\n")
print(head(NN, 12))
cat("\nFURTHEST 5, for scale:\n")
print(tail(NN, 5))

cat("\n=========================================================================\n")
cat("PART 5 -- IS THE TWO-TRACK DIFFERENCE STRUCTURED OR RANDOM?\n")
cat("=========================================================================\n")
D <- cond(TRK2) - cond(MAIN)
cat(sprintf("per-class mean signed diff : %s\n", paste(sprintf("%+.5f", colMeans(D)), collapse="  ")))
cat(sprintf("per-class sd of diff       : %s\n", paste(sprintf("%.5f",  apply(D,2,sd)),  collapse="  ")))
cat(sprintf("max |diff| in conditional  : %.5f\n", max(abs(D))))
cat(sprintf("rows where |diff| > 0.01   : %.2f%%\n", 100*mean(apply(abs(D),1,max) > 0.01)))
cat("\nA near-zero column mean with a large sd means the tracks disagree row by row\n")
cat("but not on average -- which is exactly the shape that pooling can exploit.\n")

cat("\ndone\n")
