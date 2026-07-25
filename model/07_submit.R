suppressMessages(library(data.table))
source("model/99_utils.R")
long <- readRDS("model/artifacts/long.rds")
P <- readRDS("model/artifacts/test_blend.rds")
b <- readRDS("model/artifacts/blend.rds")
nos <- sort(unique(long[is_test == TRUE, No]))
stopifnot(nrow(P) == 4997, length(nos) == 4997, min(nos) == 21566, max(nos) == 26562)
P <- clip_norm(P)
stopifnot(all(abs(rowSums(P) - 1) < 1e-8), all(P > 0), all(P < 1))
sub <- data.table(No = nos, Ch1 = P[, 1], Ch2 = P[, 2], Ch3 = P[, 3], Ch4 = P[, 4])
dir.create("submissions", showWarnings = FALSE)
f <- sprintf("submissions/sub_%s.csv", format(Sys.time(), "%Y%m%d_%H%M"))
fwrite(sub, f)
cat(sprintf("| %s | %s | %s | %.5f | expect public ~%.3f | (pending) |\n",
            format(Sys.time(), "%Y-%m-%d %H:%M"), basename(f),
            paste(b$members, collapse = "+"), b$nested, b$nested + 0.05),
    file = "submissions/log.md", append = TRUE)
cat("wrote", f, " nested OOF:", round(b$nested, 5), "\n")
