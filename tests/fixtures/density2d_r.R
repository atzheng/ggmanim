# Generates the 2-D density / bin2d reference fixtures used by tests/test_density2d.py
# and tests/test_bin2d.py.
#   Rscript tests/fixtures/density2d_r.R   (from the repo root)
suppressPackageStartupMessages({library(MASS); library(ggplot2)})
set.seed(20261114)
n <- 60
x <- rnorm(n, 1, 2)
y <- 0.5 * x + rnorm(n, 0, 1)
write.csv(data.frame(x = sprintf("%.17g", x), y = sprintf("%.17g", y)),
          "tests/fixtures/density2d_data.csv", row.names = FALSE, quote = FALSE)

h <- c(bandwidth.nrd(x), bandwidth.nrd(y))
d <- kde2d(x, y, n = 20)
g <- expand.grid(x = d$x, y = d$y)
g$density <- as.vector(d$z)
g$hx <- h[1]; g$hy <- h[2]
g[] <- lapply(g, function(v) sprintf("%.12g", v))
write.csv(g, "tests/fixtures/kde2d_r.csv", row.names = FALSE, quote = FALSE)

df <- data.frame(x = x, y = y)
lev <- layer_data(ggplot(df, aes(x, y)) + geom_density_2d())
lev <- aggregate(cbind(n = x) ~ level, data = lev, FUN = length)
bins <- layer_data(ggplot(df, aes(x, y)) + geom_bin_2d(bins = 8))
out <- rbind(
  data.frame(kind = "density_level", a = lev$level, b = NA, c = lev$n),
  data.frame(kind = "bin2d", a = bins$x, b = bins$y, c = bins$count)
)
out$a <- sprintf("%.10g", out$a); out$b <- sprintf("%.10g", out$b)
write.csv(out, "tests/fixtures/ggplot2d_r.csv", row.names = FALSE, quote = FALSE)
cat(R.version.string, "ggplot2", format(packageVersion("ggplot2")), "\n")
