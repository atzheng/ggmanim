# Generates the horizontal (orientation = "y") stacked-area fixtures used by
# tests/test_flipped_area.py.
#   Rscript tests/fixtures/flipped_area_r.R   (from the repo root)
suppressPackageStartupMessages(library(ggplot2))
d <- data.frame(
  t = c(0, 1, 2, 3, 4, 0.5, 1.5, 2.5, 3.5, 0, 2, 4),
  v = c(1, 3, 2, 4, 3, 2, -1, 1, 2, 1, 2, 1),
  g = c(rep("a", 5), rep("b", 4), rep("c", 3))
)
write.csv(d, "tests/fixtures/flipped_area_data.csv", row.names = FALSE, quote = FALSE)
shared <- data.frame(t = rep(c(0, 1, 2), 3), v = c(1, 2, 3, 2, 2, 1, 0.5, 1, 4),
                     g = rep(c("a", "b", "c"), each = 3))
write.csv(shared, "tests/fixtures/flipped_area_shared.csv", row.names = FALSE, quote = FALSE)
grab <- function(p, case) {
  ld <- layer_data(p)
  ld$g <- c("a", "b", "c")[ld$group]
  ld$case <- case
  ld[, c("case", "g", "x", "xmin", "xmax", "y", "flipped_aes")]
}
out <- rbind(
  grab(ggplot(d, aes(x = v, y = t, fill = g)) + geom_area(orientation = "y"), "align"),
  grab(ggplot(shared, aes(x = v, y = t, fill = g)) +
         geom_area(stat = "identity", orientation = "y"), "identity"),
  grab(ggplot(shared, aes(x = v, y = t, fill = g)) +
         geom_area(stat = "identity", position = "fill", orientation = "y"), "fill"),
  grab(ggplot(shared, aes(x = v, y = t, fill = g)) +
         geom_area(stat = "identity", position = position_stack(reverse = TRUE),
                   orientation = "y"), "reverse")
)
num <- c("x", "xmin", "xmax", "y")
out[num] <- lapply(out[num], function(v) sprintf("%.10g", v))
write.csv(out, "tests/fixtures/flipped_area_r.csv", row.names = FALSE, quote = FALSE)
cat(R.version.string, "ggplot2", as.character(packageVersion("ggplot2")), "\n")
