# Per-segment Ch4 logit-shift utilities (shared by confirm/submit scripts).

shift_ch4 <- function(p, b) {
  z <- log(pmax(p, 1e-15))
  z[, 4] <- z[, 4] + b
  ez <- exp(z - apply(z, 1, max))
  ez / rowSums(ez)
}

fit_b <- function(p, yy, lo = -1.5, hi = 1.5, tol = 1e-3) {
  f <- function(b) multiclass_log_loss(yy, shift_ch4(p, b))
  gr <- (sqrt(5) - 1) / 2
  a <- lo; d <- hi
  c1 <- d - gr * (d - a); c2 <- a + gr * (d - a)
  while (d - a > tol) {
    if (f(c1) < f(c2)) { d <- c2; c2 <- c1; c1 <- d - gr * (d - a) }
    else               { a <- c1; c1 <- c2; c2 <- a + gr * (d - a) }
  }
  (a + d) / 2
}

# Fit shifts per segment on (oof_fit, y) and apply them to probs_apply rows
# grouped by seg_apply. Returns list(probs, b).
fit_apply_shifts <- function(oof_fit, y_fit, seg_fit, probs_apply, seg_apply,
                             shrink = 1) {
  segs <- unique(seg_fit)
  b <- setNames(numeric(length(segs)), segs)
  out <- probs_apply
  for (s in segs) {
    b[[s]] <- shrink * fit_b(oof_fit[seg_fit == s, , drop = FALSE], y_fit[seg_fit == s])
    idx <- seg_apply == s
    if (any(idx)) out[idx, ] <- shift_ch4(probs_apply[idx, , drop = FALSE], b[[s]])
  }
  list(probs = out, b = b)
}
