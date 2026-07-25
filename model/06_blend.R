suppressMessages(library(data.table))
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

avail <- gsub("^oof_|\\.rds$", "", list.files("model/artifacts", pattern = "^oof_.*\\.rds$"))
avail <- avail[file.exists(sprintf("model/artifacts/test_%s.rds", avail))]
if (file.exists("model/members.txt")) {
  want <- sub("#.*$", "", readLines("model/members.txt"))   # strip trailing comments
  want <- trimws(want)
  want <- want[nzchar(want)]
  missing <- setdiff(want, avail)
  if (length(missing)) stop("members.txt lists models with no artifacts: ",
                            paste(missing, collapse = ", "))
  skipped <- setdiff(avail, want)
  if (length(skipped)) cat("NOT in blend (absent from members.txt):",
                           paste(skipped, collapse = ", "), "\n")
  memb <- want
} else {
  cat("WARNING: no model/members.txt -- auto-discovering every artifact\n")
  memb <- avail
}
cat("members:", paste(memb, collapse = ", "), "\n")
stopifnot(length(memb) >= 2)
OOF <- lapply(memb, function(m) { d <- readRDS(sprintf("model/artifacts/oof_%s.rds", m)); setorder(d, No); as.matrix(d[, .(p1, p2, p3, p4)]) })
TST <- lapply(memb, function(m) { d <- readRDS(sprintf("model/artifacts/test_%s.rds", m)); setorder(d, No); as.matrix(d[, .(p1, p2, p3, p4)]) })
names(OOF) <- names(TST) <- memb
stopifnot(all(sapply(OOF, nrow) == 21565), all(sapply(TST, nrow) == 4997))

ymap <- unique(long[is_test == FALSE, .(No, y)]); setorder(ymap, No)
y <- ymap$y
fmap <- folds[order(No), fold]
M <- length(memb); eps0 <- 1e-12

blend <- function(theta, Ps, rows) {
  w <- exp(theta[1:M]); w <- w / sum(w)
  Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, Ps))
  L <- L / Tt
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
obj <- function(theta, rows) logloss(y[rows], blend(theta, OOF, rows))
fit_w <- function(rows) optim(c(rep(0, M), 0, -3), obj, rows = rows,
                              method = "Nelder-Mead", control = list(maxit = 3000))

nested <- numeric(5)
for (k in 1:5) {
  o <- fit_w(fmap != k)
  nested[k] <- obj(o$par, fmap == k)
  cat("fold", k, "held-out blend logloss:", round(nested[k], 5), "\n")
}
cat(">>> NESTED blend OOF (decision number):", round(mean(nested), 5), "+-", round(sd(nested), 5), "\n")

o <- fit_w(rep(TRUE, length(y)))
w <- exp(o$par[1:M]); w <- w / sum(w)
cat("plain OOF:", round(o$value, 5), "\n")
cat("weights:", paste(memb, round(w, 3), collapse = " | "),
    "  T:", round(exp(o$par[M + 1]), 3), "  eps:", round(plogis(o$par[M + 2]) * 0.10, 4), "\n")
for (m in memb) cat(sprintf("  single %-14s OOF: %.5f\n", m, logloss(y, OOF[[m]])))
saveRDS(list(members = memb, par = o$par, nested = mean(nested)), "model/artifacts/blend.rds")
saveRDS(clip_norm(blend(o$par, TST, rows = seq_len(4997))), "model/artifacts/test_blend.rds")

# Diagnostic only: OOF logloss reweighted so train income distribution mimics test's
wide <- readRDS("model/artifacts/wide.rds")
resp <- unique(wide[, .(Case, is_test, incomeind)])
wtab <- merge(resp[is_test == FALSE, .(ptr = .N / sum(!resp$is_test)), by = incomeind],
              resp[is_test == TRUE,  .(pte = .N / sum(resp$is_test)),  by = incomeind],
              by = "incomeind", all.x = TRUE)
wtab[is.na(pte), pte := 0]
wtab[, wt := pmin(pmax(pte / ptr, 0.2), 5)]
rw <- merge(unique(wide[is_test == FALSE, .(No, incomeind)]), wtab[, .(incomeind, wt)],
            by = "incomeind")[order(No), wt]
Pfin <- blend(o$par, OOF, rows = rep(TRUE, length(y)))
ll_i <- -log(pmax(Pfin[cbind(seq_along(y), y)], 1e-15))
cat("income-reweighted OOF (diagnostic):", round(sum(rw * ll_i) / sum(rw), 5), "\n")
cat("OK\n")
