suppressMessages(library(data.table))
wide <- readRDS("model/artifacts/wide.rds")
demo <- c("segmentind","yearind","milesind","milesa","nightind","nighta","pparkind",
          "genderind","ageind","agea","educind","regionind","Urbind","incomeind","incomea")
resp <- unique(wide[, c("Case", "is_test", demo), with = FALSE])
cat("respondents: train", resp[is_test == FALSE, .N], " test", resp[is_test == TRUE, .N], "\n")
tab <- resp[, lapply(.SD, mean), by = is_test, .SDcols = demo]
print(t(tab), digits = 3)
fwrite(tab, "model/artifacts/shift_check.csv")
