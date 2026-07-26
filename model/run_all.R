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
# Stages:  data  folds  seedbag  lcmnl_task  blend  submit    (default)
# Extras:  tests  mnl_pw  xgb_lw  audit                        (opt-in)
#
# Every stage writes to model/artifacts/, so a later run can start midway as
# long as the earlier artifacts still exist.
#
# COLD RUN IS ~2.5 HOURS, almost all of it `seedbag`, which fits the listwise
# xgboost ten times under ten different seeds and averages them. That averaging
# is not an optimisation -- it removes Monte-Carlo variance that a single seed
# leaves in the predictor (measured sd 0.00283, larger than most margins in
# EXPERIMENTS.md). Each seed writes its own artifact and is skipped if present,
# so an interrupted run resumes rather than restarting.
#
# `mnl_pw` and `xgb_lw` are no longer blend members and are therefore opt-in.
# They remain because compare.R baselines and the report's ablation table use
# them:  Rscript model/run_all.R data folds mnl_pw xgb_lw
# =============================================================================

t_start <- Sys.time()
args <- commandArgs(trailingOnly = TRUE)

# Both production members live under experiments/. model/members.txt is the source of
# truth for blend membership; these stages produce exactly the artifacts it names.
# `argsets` is a list of argument vectors, each run as a separate process in order.
STAGES <- list(
  tests    = list(file = "model/tests.R",            label = "Unit tests (utils + data integrity)"),
  data     = list(file = "model/00_load.R",          label = "Load data, reshape to long, build features"),
  folds    = list(file = "model/01_folds.R",         label = "Create respondent-grouped CV folds"),
  mnl_pw   = list(file = "model/02_mnl_partworth.R", label = "part-worth conditional logit (reference, not a member)"),
  xgb_lw   = list(file = "model/03_xgb_listwise.R",  label = "listwise xgboost, single seed (reference, not a member)"),
  seedbag  = list(file = "experiments/iter26_seedbag/run.R",
                  argsets = c(lapply(1:10, function(s) c(as.character(s), "lw2")),
                              list(c("combine", "lw2"))),
                  label = "MEMBER 1: listwise xgboost, 10 seeds averaged -> xgb_lw2bag (weight 0.528)"),
  lcmnl_task = list(file = "experiments/iter25_taskpos/run.R", argsets = list(c("3", "both")),
                  label = "MEMBER 2: latent class + task position -> lcmnl3_both (weight 0.472)"),
  blend    = list(file = "model/06_blend.R",         label = "Blend members, nested evaluation"),
  submit   = list(file = "model/07_submit.R",        label = "Write Kaggle submission CSV"),
  audit    = list(file = "model/shift_audit.R",      label = "Train->test shift robustness audit")
)
DEFAULT <- c("data", "folds", "seedbag", "lcmnl_task", "blend", "submit")
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
  # a stage is one or more processes; argsets = NULL means a single bare invocation
  sets <- if (is.null(s$argsets)) list(character(0)) else s$argsets
  for (a in sets) {
    if (length(a)) cat("--> ", basename(s$file), " ", paste(a, collapse = " "), "\n", sep = "")
    status <- system2(file.path(R.home("bin"), "Rscript"),
                      c(shQuote(s$file), shQuote(a)))
    if (status != 0) {
      mins <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
      stop(s$label, " FAILED (exit ", status, ") on args '",
           paste(a, collapse = " "), "' after ", mins, " min")
    }
  }
  mins <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
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
