# Data loading + senior team's Version-8 feature engineering.
#
# Ports python src/data.py (itself ported from senior's XGB_Ver3 (Features).Rmd).
# Formulas:
#   Price_avg     = rowMeans(Price1, Price2, Price3, na.rm=TRUE)
#   Price_age     = Price_avg * ageind
#   Price_income  = Price_avg * incomeind
#   Price_squared = Price_avg^2
#   Price_edu     = Price_avg * educind
#   Price_gender  = Price_avg * genderind
#   Price_Diff_1  = Price1 - pmin(Price2, Price3, na.rm=TRUE)
#   Price_Diff_2  = Price2 - pmin(Price1, Price3, na.rm=TRUE)
#   Price_Diff_3  = Price3 - pmin(Price1, Price2, na.rm=TRUE)
#   Price_Diff_4  =    0   - pmin(Price1, Price2, Price3, na.rm=TRUE)

library(dplyr)

DEMO_BLOCK <- c(
  "segment", "segmentind", "year", "yearind",
  "miles", "milesind", "milesa", "night", "nightind", "nighta",
  "ppark", "pparkind", "gender", "genderind",
  "age", "ageind", "agea", "educ", "educind",
  "region", "regionind", "Urb", "Urbind",
  "income", "incomeind", "incomea"
)

ALT_PREFIXES <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
                   "PP","KA","SC","TS","NV","MA","LB","AF","HU","Price")
# alt-major order (all of alt1's cols, then alt2's, ...) — matches raw csv
# column order / select(CC1:Price4) in the senior Rmd.
ALT_BLOCK <- as.vector(sapply(1:4, function(i) paste0(ALT_PREFIXES, i)))

ENGINEERED_COLS <- c(
  "Price_avg", "Price_age", "Price_income",
  "Price_squared", "Price_edu", "Price_gender",
  "Price_Diff_1", "Price_Diff_2", "Price_Diff_3", "Price_Diff_4"
)

# String demographic columns -> natural categoricals for CatBoost (as factors).
CAT_STR_COLS <- c("segment", "miles", "night", "ppark", "gender",
                   "age", "educ", "region", "Urb", "income")

add_engineered <- function(df) {
  df %>%
    mutate(
      Price_avg      = rowMeans(select(., Price1, Price2, Price3), na.rm = TRUE),
      Price_age      = Price_avg * ageind,
      Price_income   = Price_avg * incomeind,
      Price_squared  = Price_avg^2,
      Price_edu      = Price_avg * educind,
      Price_gender   = Price_avg * genderind,
      Price_Diff_1   = Price1 - pmin(Price2, Price3, na.rm = TRUE),
      Price_Diff_2   = Price2 - pmin(Price1, Price3, na.rm = TRUE),
      Price_Diff_3   = Price3 - pmin(Price1, Price2, na.rm = TRUE),
      Price_Diff_4   =    0   - pmin(Price1, Price2, Price3, na.rm = TRUE)
    )
}

make_label <- function(df) {
  # 0-based label (0..3), matching Python / xgboost multi:softprob convention.
  as.integer(ifelse(df$Ch1 == 1, 0,
              ifelse(df$Ch2 == 1, 1,
               ifelse(df$Ch3 == 1, 2, 3))))
}

feature_cols <- function() {
  c(ALT_BLOCK, DEMO_BLOCK, ENGINEERED_COLS)
}

load_train <- function(path) {
  train <- read.csv(path, stringsAsFactors = FALSE)
  train <- add_engineered(train)
  y <- make_label(train)
  case <- train$Case
  list(df = train, y = y, case = case)
}

load_test <- function(path) {
  test <- read.csv(path, stringsAsFactors = FALSE)
  add_engineered(test)
}

# Build a plain numeric matrix for xgboost, matching R's
# mutate_if(is.character, as.factor) %>% mutate_if(is.factor, as.numeric)
# i.e. label-encode string columns 1..k (alphabetical), same as senior code.
build_xgb_matrix <- function(df) {
  cols <- feature_cols()
  X <- df[, cols, drop = FALSE]
  for (c in cols) {
    if (is.character(X[[c]])) {
      X[[c]] <- as.numeric(as.factor(X[[c]]))
    }
  }
  as.matrix(X)
}

# Build a data.frame for catboost with the 10 categorical string columns
# converted to factor (CatBoost auto-detects factor columns as categorical
# when data is a data.frame; character columns would instead be treated as
# TEXT features, so this conversion is required, not optional).
build_catboost_frame <- function(df) {
  cols <- feature_cols()
  X <- df[, cols, drop = FALSE]
  for (c in CAT_STR_COLS) {
    if (c %in% names(X)) {
      X[[c]] <- as.factor(X[[c]])
    }
  }
  X
}
