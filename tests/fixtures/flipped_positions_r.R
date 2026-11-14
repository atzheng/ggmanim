# Reference fixtures for horizontal (flipped_aes) positions in
# tests/test_distribution_followups.py.
#   Rscript tests/fixtures/flipped_positions_r.R   (from the repo root; needs ggplot2 >= 4.0)
# Horizontal boxplots (dodge2, and stack, which leaves single boxes in place),
# violins (dodge), and points with position_jitterdodge(jitter.width = 0) (so the
# dodge is deterministic), all with y = g discrete and fill = h.
suppressPackageStartupMessages(library(ggplot2))
f <- "tests/fixtures/"
d <- read.csv(paste0(f, "dist_data.csv"))
d$h <- ifelse(seq_len(nrow(d)) %% 2 == 0, "q", "p")
fmt <- function(df) { df[] <- lapply(df, function(c) if (is.numeric(c)) sprintf("%.12g", c) else c); df }
pick <- function(ld, case) {
  ld <- ld[!duplicated(ld$group), ]
  data.frame(case = case, group = ld$group, y = ld$y, ymin = ld$ymin, ymax = ld$ymax,
             xmin = ld$xmin, xmax = ld$xmax)
}
b1 <- layer_data(ggplot(d, aes(v, g, fill = h)) + geom_boxplot())
b2 <- suppressWarnings(layer_data(ggplot(d, aes(v, g)) + geom_boxplot(position = "stack")))
v1 <- layer_data(ggplot(d, aes(v, g, fill = h)) + geom_violin())
v1 <- do.call(rbind, lapply(split(v1, v1$group), function(s) {
  s$xmin <- min(s$x); s$xmax <- max(s$x); s[1, ]
}))
p1 <- layer_data(ggplot(d, aes(v, g, fill = h)) +
                   geom_point(position = position_jitterdodge(jitter.width = 0)))
p1 <- data.frame(case = "jitterdodge", group = p1$group, y = p1$y, ymin = NA, ymax = NA,
                 xmin = p1$x, xmax = p1$x)
out <- rbind(pick(b1, "boxplot_dodge2"), pick(b2, "boxplot_stack"), pick(v1, "violin_dodge"), p1)
write.csv(fmt(out), paste0(f, "flipped_positions_r.csv"), row.names = FALSE, quote = FALSE,
          na = "")
cat(R.version.string, "ggplot2", as.character(packageVersion("ggplot2")), "\n")
