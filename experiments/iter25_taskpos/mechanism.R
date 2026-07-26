# Does the PRICE-TILT term alone reproduce the rising none-rate?
#
# Attribution came out as: tilt alone +0.00515, none-drift alone +0.00165, both
# +0.00534. So the drift term adds almost nothing once the tilt is present. The
# hypothesis that explains this: the two observed phenomena are ONE mechanism.
# The none option has Price = 0, so it is always the cheapest alternative in its
# choice set. If price sensitivity rises with task position, the none option
# becomes relatively more attractive later in the questionnaire ON ITS OWN --
# no separate fatigue term is needed to generate a rising decline rate.
#
# If that is right, oof_lcmnl3_tilt should track the observed none-rate slope
# without ever having been given a none-specific task term.
suppressMessages(library(data.table))
source("model/99_utils.R")

long <- readRDS("model/artifacts/long.rds")
tk <- unique(long[is_test == FALSE, .(No, Task, y)])
setorder(tk, No)
tk[, obs := as.numeric(y == 4)]

get_p4 <- function(n) {
  d <- readRDS(sprintf("model/artifacts/oof_%s.rds", n))
  setorder(d, No)
  stopifnot(identical(d$No, tk$No))
  d$p4
}
MODELS <- c("lcmnl3", "lcmnl3_drift", "lcmnl3_tilt", "lcmnl3_both",
            "xgb_mono", "mnl_pw")
for (n in MODELS) tk[, (n) := get_p4(n)]

a <- tk[, lapply(.SD, mean), by = Task, .SDcols = c("obs", MODELS)][order(Task)]

cat("mean none-rate by task position, and the fitted linear slope per task\n")
cat("(observed slope is +0.00544; a model with no Task term predicts ~+0.0002)\n\n")
for (n in c("obs", MODELS)) {
  s <- coef(lm(a[[n]] ~ a$Task))[2]
  cat(sprintf("  %-14s  task1 %.3f -> task19 %.3f   slope %+.5f\n",
              n, a[[n]][1], a[[n]][19], s))
}

cat("\nfraction of the OBSERVED slope each model reproduces:\n")
so <- coef(lm(a$obs ~ a$Task))[2]
for (n in MODELS) {
  s <- coef(lm(a[[n]] ~ a$Task))[2]
  cat(sprintf("  %-14s  %5.1f%%\n", n, 100 * s / so))
}
