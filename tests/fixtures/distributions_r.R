# Reference fixtures for tests/test_boxplot.py, test_density.py, test_violin.py.
#   Rscript tests/fixtures/distributions_r.R   (from the repo root; needs ggplot2 >= 4.0)
# Statistics are taken from ggplot2's layer_data(), so they include ggplot2's
# defaults (quantile type 7, 1.5 IQR whiskers, bw.nrd0, 512-point grids ...).
suppressPackageStartupMessages(library(ggplot2))
set.seed(20261114)
g <- rep(c("a", "b", "c"), times = c(25, 40, 18))
v <- c(rnorm(25, 2, 1), rexp(40, 0.6), runif(18, -1, 3))
v[c(3, 30, 70)] <- c(7.5, 12, -4)  # outliers
w <- round(runif(length(v), 0.2, 2), 3)
d <- data.frame(g = g, v = v, w = w)
f <- "tests/fixtures/"
fmt <- function(df) { df[] <- lapply(df, function(c) if (is.numeric(c)) sprintf("%.12g", c) else c); df }
write.csv(fmt(d), paste0(f, "dist_data.csv"), row.names = FALSE, quote = FALSE)

# boxplots (one per level), notches, outliers ";"-joined
b <- layer_data(ggplot(d, aes(g, v)) + geom_boxplot())
b$outliers <- vapply(b$outliers, function(o) paste(sprintf("%.12g", o), collapse = ";"), "")
write.csv(fmt(b[c("x", "ymin", "lower", "middle", "upper", "ymax", "notchlower", "notchupper",
                  "xmin", "xmax", "outliers")]),
          paste0(f, "dist_boxplot_r.csv"), row.names = FALSE, quote = FALSE)

# bandwidths per group
bw <- do.call(rbind, lapply(split(d$v, d$g), function(x) data.frame(nrd0 = bw.nrd0(x), nrd = bw.nrd(x))))
bw$g <- rownames(bw)
write.csv(fmt(bw), paste0(f, "dist_bw_r.csv"), row.names = FALSE, quote = FALSE)

# densities: default; weighted + adjust + epanechnikov; bounded
dens <- function(p, case) {
  ld <- layer_data(p)
  ld$case <- case
  ld[c("case", "group", "x", "density", "count", "scaled", "n")]
}
p1 <- ggplot(d, aes(v, colour = g)) + geom_density(n = 64)
p2 <- ggplot(d, aes(v, colour = g, weight = w)) +
  geom_density(n = 64, adjust = 0.7, kernel = "epanechnikov")
p3 <- ggplot(d[d$g == "b", ], aes(v)) + geom_density(n = 64, bounds = c(0, Inf))
write.csv(fmt(rbind(dens(p1, "default"), dens(p2, "weighted"), dens(p3, "bounded"))),
          paste0(f, "dist_density_r.csv"), row.names = FALSE, quote = FALSE)

# violins (every 16th grid row + the quantile rows), area and count scaling
vio <- function(p, case) {
  ld <- layer_data(p)
  keep <- !is.na(ld$quantile) | (seq_len(nrow(ld)) %% 16 == 1)
  ld <- ld[keep, ]
  ld$case <- case
  ld[c("case", "group", "x", "y", "density", "violinwidth", "xmin", "xmax", "quantile")]
}
q1 <- ggplot(d, aes(g, v)) + geom_violin()
q2 <- ggplot(d, aes(g, v)) + geom_violin(scale = "count", trim = FALSE)
write.csv(fmt(rbind(vio(q1, "area"), vio(q2, "count"))),
          paste0(f, "dist_violin_r.csv"), row.names = FALSE, quote = FALSE)
cat(R.version.string, "ggplot2", as.character(packageVersion("ggplot2")), "\n")
