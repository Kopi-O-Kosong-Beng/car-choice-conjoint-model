# =============================================================================
# ENTRY POINT — runs the full pipeline in order and writes a Kaggle submission.
#
# From the repository root:
#   & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" model/run_all.R
#
# Run only some stages, by name:
#   ... model/run_all.R blend submit        # re-blend and re-submit only
#   ... model/run_all.R data folds mnl_pw   # rebuild data, folds, then the logit
#
# Stages:  data  folds  mnl_pw  xgb_lw  blend  submit        (default: all)
# Extras:  tests  audit                                       (opt-in)
#
# Every stage writes to model/artifacts/, so a later run can start midway as
# long as the earlier artifacts still exist. Full cold run ~30 min.
# =============================================================================

t_start <- Sys.time()
args <- commandArgs(trailingOnly = TRUE)

STAGES <- list(
  tests  = list(file = "model/tests.R",            label = "Unit tests (utils + data integrity)"),
  data   = list(file = "model/00_load.R",          label = "Load data, reshape to long, build features"),
  folds  = list(file = "model/01_folds.R",         label = "Create respondent-grouped CV folds"),
  mnl_pw = list(file = "model/02_mnl_partworth.R", label = "Model 1/2: part-worth conditional logit"),
  xgb_lw = list(file = "model/03_xgb_listwise.R",  label = "Model 2/2: listwise-objective xgboost"),
  blend  = list(file = "model/06_blend.R",         label = "Blend members, nested evaluation"),
  submit = list(file = "model/07_submit.R",        label = "Write Kaggle submission CSV"),
  audit  = list(file = "model/shift_audit.R",      label = "Train->test shift robustness audit")
)
DEFAULT <- c("data", "folds", "mnl_pw", "xgb_lw", "blend", "submit")
stages <- if (length(args)) args else DEFAULT

unknown <- setdiff(stages, names(STAGES))
if (length(unknown))
  stop("Unknown stage(s): ", paste(unknown, collapse = ", "),
       "\nValid: ", paste(names(STAGES), collapse = " "))

if (!file.exists("Raw Dump/Competition Data/train2024.csv"))
  stop("Cannot find the data. Run from the repository root (the folder containing\n",
       "'Raw Dump' and 'model'). Current working directory: ", getwd())

run_stage <- function(key) {
  s <- STAGES[[key]]
  cat("\n", strrep("=", 72), "\n[", format(Sys.time(), "%H:%M:%S"), "] ", s$label, "\n",
      strrep("=", 72), "\n", sep = "")
  t0 <- Sys.time()
  status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(s$file))
  mins <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
  if (status != 0) stop(s$label, " FAILED (exit ", status, ") after ", mins, " min")
  cat("[done in ", mins, " min]\n", sep = "")
}

# run in canonical order regardless of the order given on the command line
for (key in names(STAGES)) if (key %in% stages) run_stage(key)

cat("\n", strrep("=", 72), "\nPIPELINE COMPLETE in ",
    round(as.numeric(difftime(t_start <- Sys.time(), t_start, units = "mins")), 1),
    " min\n", strrep("=", 72), "\n", sep = "")

if ("submit" %in% stages) {
  subs <- list.files("submissions", pattern = "^sub_.*\\.csv$", full.names = TRUE)
  if (length(subs)) {
    newest <- subs[which.max(file.mtime(subs))]
    cat("Upload to Kaggle (team account only):\n  ", newest, "\n", sep = "")
    cat("Then record the public score in submissions/log.md\n")
  }
}
