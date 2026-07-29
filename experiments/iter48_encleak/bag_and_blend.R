# =============================================================================
# ITERATION 48 -- THE DECISIVE ARM.  What does the production 2-member blend score
# when its tree member carries NO design-share encoding?
#
# Bags the available seeds of each arm at depth 8 (arithmetic mean of the
# probability matrices -- the same rule xgb_lw2bag uses), then runs the nested
# simplex log-opinion pool from model/06_blend.R against lcmnl3_both, unchanged.
# Nothing is written to model/artifacts and members.txt is never read.
# Also reports the income-reweighted OOF, which is the quantity that has tracked
# the public board (06_blend.R prints it as a diagnostic).
# =============================================================================
suppressMessages(library(data.table))
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
wide  <- readRDS("model/artifacts/wide.rds")
folds <- readRDS("model/artifacts/folds.rds")
ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No); y <- ymap$y
fmap <- folds[order(No), fold]

resp <- unique(wide[, .(Case, is_test, incomeind)])
wt <- merge(resp[is_test == FALSE, .(ptr = .N / sum(!resp$is_test)), by = incomeind],
            resp[is_test == TRUE,  .(pte = .N / sum(resp$is_test)),  by = incomeind],
            by = "incomeind", all.x = TRUE)
wt[is.na(pte), pte := 0][, w := pmin(pmax(pte / ptr, 0.2), 5)]
rw <- merge(unique(wide[is_test == FALSE, .(No, incomeind)]), wt[, .(incomeind, w)],
            by = "incomeind")[order(No), w]

getP <- function(f) { d <- readRDS(f); setorder(d, No); stopifnot(nrow(d) == 21565); as.matrix(d[, .(p1,p2,p3,p4)]) }
bag <- function(arm, depth = 8) {
  fs <- list.files("experiments/iter48_encleak",
                   pattern = sprintf("^el48_oof_%s_d%d_m20_e003_n540_s.*\\.rds$", arm, depth),
                   full.names = TRUE)
  stopifnot(length(fs) > 0)
  cat(sprintf("  %-7s d%d: %d seed(s)  singles %s\n", arm, depth, length(fs),
              paste(sprintf("%.5f", sapply(fs, function(f) logloss(y, getP(f)))), collapse = " ")))
  A <- Reduce(`+`, lapply(fs, getP)) / length(fs)
  A / rowSums(A)
}

M <- 2; eps0 <- 1e-12
blendP <- function(theta, Ps, rows) {
  w <- exp(theta[1:M]); w <- w / sum(w)
  Tt <- exp(theta[M+1]); eA <- plogis(theta[M+2]) * 0.10
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows,,drop=FALSE], eps0)), w, Ps))
  L <- L / Tt; P <- exp(L - apply(L,1,max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
run_blend <- function(lab, PX) {
  Ps <- list(PX, getP("model/artifacts/oof_lcmnl3_both.rds"))
  obj <- function(th, rows) logloss(y[rows], blendP(th, Ps, rows))
  fitw <- function(rows) optim(c(rep(0,M),0,-3), obj, rows = rows, method = "Nelder-Mead",
                               control = list(maxit = 3000))
  nested <- sapply(1:5, function(k) obj(fitw(fmap != k)$par, fmap == k))
  o <- fitw(rep(TRUE, length(y)))
  w <- exp(o$par[1:M]); w <- w / sum(w)
  Pf <- blendP(o$par, Ps, rep(TRUE, length(y)))
  li <- -log(pmax(Pf[cbind(seq_along(y), y)], 1e-15))
  cat(sprintf("%-26s member %.5f | nested %.5f | plain %.5f | w_tree %.3f | income-rw %.5f\n",
              lab, logloss(y, PX), mean(nested), o$value, w[1], sum(rw*li)/sum(rw)))
  invisible(c(member = logloss(y, PX), nested = mean(nested), rw = sum(rw*li)/sum(rw)))
}

cat("bags assembled from experiments/iter48_encleak (depth 8, mcw 20, eta .03, 540 rounds):\n")
Pp <- bag("prod"); Pn <- bag("noenc"); Ph <- bag("honest"); Pl <- bag("leaky")
cat("\n--- nested blend with lcmnl3_both (unchanged) ---\n")
run_blend("PRODUCTION encoding", Pp)
run_blend("NO encoding", Pn)
run_blend("HONEST encoding (3-fold)", Ph)
run_blend("LEAKY encoding (3-fold)", Pl)
cat("\n--- reference: the live production blend ---\n")
run_blend("xgb_lw2bag (live member)", getP("model/artifacts/oof_xgb_lw2bag.rds"))
cat("\nBAGBLEND_DONE\n")
