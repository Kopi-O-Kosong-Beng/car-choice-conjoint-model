# =============================================================================
# ITERATION 18 — STRUCTURE HUNT (diagnostic only; NO model was built, by design)
#
# QUESTION (stated before running): seventeen iterations have optimised estimators.
# Is there structure in the experimental design or in respondent behaviour that no
# model has been told about?
#
# HYPOTHESES TESTED, in the order the brief ranked them:
#   H1  design assignment correlates with demographics  -> transductive edge
#   H2  task-order / fatigue drift not captured by the models
#   H3  respondent predictability is itself predictable -> confidence adjustment
#   H4  attribute PAIRS matter beyond their separate part-worths
#   H5  anything else the data suggests
#
# VERDICT: H1-H5 all REFUTED as sources of predictive gain. Two new structural
# facts were established (299-version block design; partial-profile 9-of-19
# attribute presentation) and both turn out to be already absorbed by the
# incumbent models. The remaining loss is quantified as respondent heterogeneity
# that is structurally inaccessible: every test respondent is a stranger.
#
# HEADLINE NUMBERS (all measured, none estimated):
#   - 299 questionnaire versions; design at task 1 determines all 19 (1398/299 = 4.68 per version)
#   - assignment vs 11 demographics: min p = 0.123, none below 0.05
#   - none-rate rises 0.237 (task 1) -> 0.337 (task 19), logit slope +0.0259, z = 9.51
#     ... but the incumbent already predicts 0.252 -> 0.338; per-task temperature
#     buys 0.00072 IN-SAMPLE, i.e. nothing
#   - 190 pairwise attribute interactions, >=5501 tasks of joint support each,
#     ridge lambda picked by an inner Case-grouped split so nothing is self-tuned:
#     5-fold OOF 1.17566 (main effects) -> 1.17530. +0.00036, i.e. nothing.
#   - respondent-level shrinkage: honest +0.00095 on xgb_lw2, -0.00072 on
#     xgb_resenc2, in-sample ceiling +0.00037 on the blend
#   - price sensitivity STRENGTHENS across the sequence: conditional-logit price
#     coefficient -0.178 (task 1) -> -0.331 (task 19), weighted trend z = -6.29,
#     with the design balanced across positions. Residual after the incumbent
#     z = -2.50 but worth ~3e-5 of logloss.
#   - ORACLE respondent rates would take the blend 1.12974 -> 0.98568 (gap 0.144)
#
# Run: & "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" experiments/iter18_structure_hunt/run.R
# Runtime ~35 min total. Individual scripts can be run alone; each is self-contained.
# NOTHING in model/ is touched and no artifact is written.
# =============================================================================
scripts <- c(
  d1  = "d1_blocks.R",      # is the design blocked?  -> partial profile, 9 of 19 attrs
  d2  = "d2_assignment.R",  # H1: version structure + demographic dependence of assignment
  d3  = "d3_behaviour.R",   # H2/H3: fatigue, per-respondent slopes, dominance, scale
  d4  = "d4_followup.R",    # partial-profile semantics, version/train-test overlap
  d5  = "d5_pilot.R",       # H4: 190 pairwise interactions + calibration slopes
  d6  = "d6_respshrink.R",  # H3: respondent-level over-dispersion shrinkage (SLOW ~20 min)
  d7  = "d7_profile.R",     # H5: does the none row miss the design's attribute set?
  d8  = "d8_debias_enc.R",  # H5: version-mate-debiased design encoding
  d9  = "d9_headroom.R",    # where the remaining loss lives; oracle bound
  d10 = "d10_rankfeats.R",  # H5: normalised / rank bundle summaries vs residual
  d11 = "d11_interactions_full.R", # H4: full 5-fold OOF, inner-picked lambda
  d12 = "d12_anchoring.R",  # H5: sequential anchoring, with a next-task placebo
  d13 = "d13_verify.R",     # confirmations of quoted numbers
  d14 = "d14_task_price.R", # H2: does task order interact with price sensitivity?
  d15 = "d15_pricedrift.R"  # H2: design balance + clustered residual test
)
base <- "experiments/iter18_structure_hunt"
for (s in scripts) {
  cat("\n\n########## ", s, " ##########\n")
  source(file.path(base, s), echo = FALSE)
}
