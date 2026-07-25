# Shared design-level encoder (validated in experiments/iter01_design_encoding).
#
# Adds shrunk empirical choice shares per (design, alternative). For a training
# row in fold k the shares come ONLY from respondents in other folds; test rows
# use all training respondents. Respondents live entirely within one fold, so a
# person never contributes to their own encoding.
ENC_ALPHAS <- c(1, 5, 20)
ENC_COLS <- c(paste0("share_a", ENC_ALPHAS), "design_n")

add_design_key <- function(long, wide, attrs) {
  acols3 <- unlist(lapply(1:3, function(j) paste0(attrs, j)))
  dk <- data.table::data.table(No = wide$No,
                               dkey = do.call(paste, c(wide[, ..acols3], sep = "_")))
  merge(long, dk, by = "No")
}

encode_aligned <- function(ref, target) {
  t2 <- data.table::copy(target)[, .rid := .I]
  prior <- ref[, .(pr = sum(chosen) / .N), by = alt]
  cnt   <- ref[, .(n_ch = sum(chosen), n = .N), by = .(dkey, alt)]
  t2 <- merge(t2[, .(.rid, dkey, alt)], cnt, by = c("dkey", "alt"), all.x = TRUE)
  t2 <- merge(t2, prior, by = "alt", all.x = TRUE)
  t2[is.na(n), `:=`(n = 0, n_ch = 0)]
  for (a in ENC_ALPHAS) t2[, paste0("share_a", a) := (n_ch + a * pr) / (n + a)]
  t2[, design_n := n]
  data.table::setorder(t2, .rid)
  t2[, .SD, .SDcols = ENC_COLS]
}

# Adds ENC_COLS to trl (fold-aware) and tel (uses all of trl). Modifies in place.
apply_design_encoding <- function(trl, tel) {
  for (cc in ENC_COLS) { trl[, (cc) := NA_real_]; tel[, (cc) := NA_real_] }
  for (k in sort(unique(trl$fold))) {
    idx <- which(trl$fold == k)
    e <- encode_aligned(trl[fold != k], trl[fold == k])
    for (cc in ENC_COLS) data.table::set(trl, idx, cc, e[[cc]])
  }
  e <- encode_aligned(trl, tel)
  for (cc in ENC_COLS) data.table::set(tel, seq_len(nrow(tel)), cc, e[[cc]])
  invisible(NULL)
}
