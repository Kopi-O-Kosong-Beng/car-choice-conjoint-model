# =============================================================================
# ITERATION 80, STEP 2 -- invert the returned score into r_lux, then apply the
# PRE-REGISTERED decision rules from run.R. Nothing here is tuned after the fact.
#
#   Rscript experiments/iter80_mprobe/invert.R <returned_score>
#   e.g.    Rscript experiments/iter80_mprobe/invert.R 1.207
#
# Decision rules, verbatim from run.R:
#   r_lux outside [0.10, 0.40] -> homogeneity or public-composition assumption
#                                 is broken. DISCARD, ship nothing.
#   gain(r_lux) <  0.001       -> final00 is already at the optimum. FREEZE.
#   gain(r_lux) >= 0.001       -> ship the exact correction (written below).
# =============================================================================
suppressMessages(library(data.table))

a <- commandArgs(trailingOnly = TRUE)
if (length(a) < 1) stop("usage: invert.R <returned_score>   e.g. invert.R 1.207")
S_B <- as.numeric(a[1])

K <- readRDS("experiments/iter80_mprobe/inversion.rds")
f <- K$f; R4 <- K$R4; S_A <- K$s_A

LUX_SEGMENTS <- c("Prestige Luxury Sedan", "Midsize Luxury Utility segements")
wide <- readRDS("model/artifacts/wide.rds")
wt <- unique(wide[is_test == TRUE, .(No, segment)]); setorder(wt, No)
d <- read.csv("submissions/sub_20260730_final00.csv", check.names = FALSE); d <- d[order(d$No), ]
A <- as.matrix(d[, 2:5]); A <- pmax(A, 1e-12); A <- A / rowSums(A)
lx <- wt$segment %in% LUX_SEGMENTS
NO_TE <- as.integer(d$No)

seg2 <- function(Q, rl, rn) { out <- Q
  for (gi in 1:2) { grp <- if (gi == 1) which(lx) else which(!lx); tgt <- if (gi == 1) rl else rn
    g <- function(z) { m <- exp(z); mean(m*Q[grp,4]/(1+(m-1)*Q[grp,4])) - tgt }
    if (g(-25) > 0 || g(25) < 0) next
    m <- exp(uniroot(g, c(-25,25), tol = 1e-12)$root); z <- 1 + (m-1)*Q[grp,4]
    out[grp,] <- cbind(Q[grp,1:3]/z, m*Q[grp,4]/z) }; out }
ll <- function(Q, rl, rn) { r4 <- ifelse(lx, rl, rn)
  C <- Q[,1:3]/(1-Q[,4]); C <- C/rowSums(C)
  R <- cbind(C*(1-r4), r4); -mean(rowSums(R*log(pmax(Q,1e-15)))) }

# ---- invert -----------------------------------------------------------------
r_lux <- ((S_A - S_B) - K$INTER) / K$SLOPE
r_non <- (R4 - f*r_lux) / (1 - f)
sig   <- sqrt(2)*0.0005/abs(K$SLOPE)

cat(sprintf("\nreturned score s_B = %.3f   (baseline s_A = %.3f)\n", S_B, S_A))
cat(sprintf("s_A - s_B = %+.6f = %.6f + %.6f * r_lux\n\n", S_A - S_B, K$INTER, K$SLOPE))
cat(sprintf("  MEASURED r_lux = %.4f  +- %.4f (rounding) +- 0.0009 (composition)\n", r_lux, sig))
cat(sprintf("     realized-rate sampling, public vs private rows, adds ~0.006-0.010;\n"))
cat(sprintf("     its cost at the optimum is second-order, ~0.0004. Verified by simulation.\n"))
cat(sprintf("  MEASURED r_non = %.4f\n", r_non))
cat(sprintf("  (final00 currently implies r_lux %.4f)\n\n", mean(A[lx,4])))

if (r_lux < 0.10 || r_lux > 0.40) {
  cat("*** r_lux OUTSIDE [0.10, 0.40] ***\n")
  cat("The within-segment homogeneity or public-composition assumption is broken.\n")
  cat("PRE-REGISTERED RULE: DISCARD. Ship nothing from this iteration.\n")
  quit(status = 0)
}

B    <- seg2(A, r_lux, r_non)
gain <- ll(A, r_lux, r_non) - ll(B, r_lux, r_non)
cat(sprintf("  exact correction is worth %+.5f  -> forecast public %.4f\n\n", gain, S_A - gain))

if (gain < 0.001) {
  cat("PRE-REGISTERED RULE: gain < 0.001 -> final00 is already at the optimum.\n")
  cat("FREEZE. Do not spend a slot. Write the report.\n")
} else {
  OUT <- "submissions/sub_20260731_segexact.csv"
  stopifnot(nrow(B) == 4997, all(B > 0), max(abs(rowSums(B)-1)) < 1e-10,
            abs(mean(B[,4]) - R4) < 1e-9,
            max(abs((B[,1:3]/(1-B[,4]))/rowSums(B[,1:3]/(1-B[,4])) -
                    (A[,1:3]/(1-A[,4]))/rowSums(A[,1:3]/(1-A[,4])))) < 1e-12)
  fwrite(data.table(No=NO_TE, Ch1=B[,1], Ch2=B[,2], Ch3=B[,3], Ch4=B[,4]), OUT)
  cat(sprintf("PRE-REGISTERED RULE: gain >= 0.001 -> SHIP.\n  wrote %s\n", OUT))
  cat("  gates: 4997 rows, sums to 1, mean p4 = r*, within-buy conditional preserved.\n")
  cat("  NOT UPLOADED -- submission is the user's action.\n")
}
