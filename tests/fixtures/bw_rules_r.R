# Reference fixtures for the bw = "ucv" / "bcv" / "SJ" rules in
# tests/test_distribution_followups.py.
#   Rscript tests/fixtures/bw_rules_r.R   (from the repo root; needs ggplot2 >= 4.0)
# Samples: the three groups of dist_data.csv (n = 25, 40, 18; unbinned pair
# counts) plus a 700-point mixture with negative values (n > nb / 2: R's binned
# pair counts) and a 300-point skewed sample.
suppressPackageStartupMessages(library(ggplot2))
set.seed(20261115)
f <- "tests/fixtures/"
d <- read.csv(paste0(f, "dist_data.csv"))
big <- data.frame(case = "mix700", v = c(rnorm(400, -2, 1), rnorm(300, 3, 1.5)))
skew <- data.frame(case = "exp300", v = rexp(300, 0.4))
samples <- rbind(data.frame(case = paste0("dist_", d$g), v = d$v), big, skew)
fmt <- function(df) { df[] <- lapply(df, function(c) if (is.numeric(c)) sprintf("%.17g", c) else c); df }
write.csv(fmt(samples[samples$case %in% c("mix700", "exp300"), ]),
          paste0(f, "bw_rules_data.csv"), row.names = FALSE, quote = FALSE)

bw <- do.call(rbind, lapply(split(samples$v, samples$case), function(x) {
  q <- function(e) suppressWarnings(e)
  data.frame(n = length(x), ucv = q(bw.ucv(x)), bcv = q(bw.bcv(x)),
             sj = bw.SJ(x), sj_dpi = bw.SJ(x, method = "dpi"))
}))
bw$case <- rownames(bw)
write.csv(fmt(bw[c("case", "n", "ucv", "bcv", "sj", "sj_dpi")]),
          paste0(f, "bw_rules_r.csv"), row.names = FALSE, quote = FALSE)

# ggplot2 end to end: geom_density(bw = "SJ") and geom_violin(bw = "bcv")
p <- layer_data(ggplot(d, aes(v, colour = g)) + geom_density(bw = "SJ", n = 32))
v <- suppressWarnings(layer_data(ggplot(d, aes(g, v)) + geom_violin(bw = "bcv")))
v <- v[is.na(v$quantile), ]
v <- v[seq_len(nrow(v)) %% 32 == 1, ]
out <- rbind(data.frame(case = "density_SJ", group = p$group, x = p$x, y = p$density),
             data.frame(case = "violin_bcv", group = v$group, x = v$y, y = v$density))
write.csv(fmt(out), paste0(f, "bw_rules_ggplot_r.csv"), row.names = FALSE, quote = FALSE)
cat(R.version.string, "ggplot2", as.character(packageVersion("ggplot2")), "\n")
