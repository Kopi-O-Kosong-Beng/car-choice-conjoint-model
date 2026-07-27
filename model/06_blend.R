suppressMessages(library(data.table))
source("model/99_utils.R")
long  <- readRDS("model/artifacts/long.rds")
folds <- readRDS("model/artifacts/folds.rds")

# ---- run modes (all default to the historical production behaviour) --------------
#   BLEND_WEIGHTS  "simplex" (default) | "free"
#   BLEND_MEMBERS  space-separated member names, overriding model/members.txt
#   BLEND_OUT      output prefix; writes <prefix>.rds / test_<prefix>.rds instead of
#                  blend.rds / test_blend.rds, so an experiment never clobbers production
MODE <- Sys.getenv("BLEND_WEIGHTS", "simplex")
stopifnot(MODE %in% c("simplex", "free"))
OUT  <- trimws(Sys.getenv("BLEND_OUT"))
out_blend <- if (nzchar(OUT)) sprintf("model/artifacts/%s.rds", OUT) else "model/artifacts/blend.rds"
out_test  <- if (nzchar(OUT)) sprintf("model/artifacts/test_%s.rds", OUT) else "model/artifacts/test_blend.rds"

avail <- gsub("^oof_|\\.rds$", "", list.files("model/artifacts", pattern = "^oof_.*\\.rds$"))
avail <- avail[file.exists(sprintf("model/artifacts/test_%s.rds", avail))]
env_memb <- trimws(Sys.getenv("BLEND_MEMBERS"))
if (nzchar(env_memb) || file.exists("model/members.txt")) {
  want <- if (nzchar(env_memb)) {
    strsplit(env_memb, "[[:space:]]+")[[1]]
  } else {
    sub("#.*$", "", readLines("model/members.txt"))         # strip trailing comments
  }
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

# theta layout: simplex = (M softmax logits, log T, logit eps); free = (M betas, logit eps).
#
# WHY the mode matters. The simplex line `w <- exp(.)/sum(exp(.))` forces every weight
# non-negative and summing to 1, so a member can only ever be *added* to the pool. A
# member that is individually worse but whose errors are correlated with the family's
# shared bias is useful precisely as a CONTROL VARIATE -- it wants a NEGATIVE
# coefficient, subtracting the common component. That was unrepresentable, so the
# simplex fit could only price such a member at ~0 and drop it. Free mode lifts the
# constraint. The temperature is then redundant (dividing all betas by T is the same
# reparameterisation), so free mode drops it and fits (beta_1..beta_M, eps) only.
blend <- function(theta, Ps, rows) {
  if (MODE == "simplex") {
    w <- exp(theta[1:M]); w <- w / sum(w)
    Tt <- exp(theta[M + 1]); eA <- plogis(theta[M + 2]) * 0.10
  } else {
    w <- theta[1:M]; Tt <- 1; eA <- plogis(theta[M + 1]) * 0.10
  }
  L <- Reduce(`+`, Map(function(wi, P) wi * log(pmax(P[rows, , drop = FALSE], eps0)), w, Ps))
  L <- L / Tt
  P <- exp(L - apply(L, 1, max)); P <- P / rowSums(P)
  (1 - eA) * P + eA * 0.25
}
obj <- function(theta, rows) logloss(y[rows], blend(theta, OOF, rows))
fit_free <- function(rows) {
  # Nelder-Mead alone is unreliable once signs are free and M grows: BFGS to get into
  # the basin, NM to polish, several starts kept honest by a fixed seed.
  set.seed(42)
  starts <- list(c(rep(1 / M, M), -3), c(rep(1, M), -3), c(rep(0.5, M), -3))
  for (i in 1:3) starts[[length(starts) + 1L]] <- c(rnorm(M, 1 / M, 0.5), -3)
  best <- NULL
  for (s in starts) {
    o <- optim(s, obj, rows = rows, method = "BFGS", control = list(maxit = 500))
    o <- optim(o$par, obj, rows = rows, method = "Nelder-Mead",
               control = list(maxit = 5000, reltol = 1e-12))
    if (is.null(best) || o$value < best$value) best <- o
  }
  best
}
fit_w <- function(rows) {
  if (MODE == "free") { fit_free(rows) } else {
    optim(c(rep(0, M), 0, -3), obj, rows = rows,
          method = "Nelder-Mead", control = list(maxit = 3000))
  }
}

nested <- numeric(5)
for (k in 1:5) {
  o <- fit_w(fmap != k)
  nested[k] <- obj(o$par, fmap == k)
  cat("fold", k, "held-out blend logloss:", round(nested[k], 5), "\n")
}
cat(">>> NESTED blend OOF (decision number):", round(mean(nested), 5), "+-", round(sd(nested), 5), "\n")

o <- fit_w(rep(TRUE, length(y)))
cat("plain OOF:", round(o$value, 5), "\n")
if (MODE == "simplex") {
  w <- exp(o$par[1:M]); w <- w / sum(w)
  cat("weights:", paste(memb, round(w, 3), collapse = " | "),
      "  T:", round(exp(o$par[M + 1]), 3), "  eps:", round(plogis(o$par[M + 2]) * 0.10, 4), "\n")
} else {
  w <- o$par[1:M]
  cat("free coefs:", paste(memb, round(w, 3), collapse = " | "),
      "  eps:", round(plogis(o$par[M + 1]) * 0.10, 4), "\n")
}
for (m in memb) cat(sprintf("  single %-14s OOF: %.5f\n", m, logloss(y, OOF[[m]])))
bl <- list(members = memb, par = o$par, nested = mean(nested))
if (MODE != "simplex") bl$mode <- MODE   # keep the simplex payload byte-identical to history
saveRDS(bl, out_blend)
saveRDS(clip_norm(blend(o$par, TST, rows = seq_len(4997))), out_test)
cat("wrote:", out_blend, "and", out_test, "\n")

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
