# Generates the loess reference fixtures used by tests/test_loess.py.
#   Rscript tests/fixtures/loess_r.R   (from the repo root)
# Exact statistics + direct surface: what ggmanim's numpy loess computes.
# The *_interp columns use R's defaults (kd-tree interpolation, approximate
# deltas), as ggplot2's geom_smooth(method = "loess") does.
set.seed(20261114)
n <- 50
x <- sort(runif(n, 0, 10))
y <- sin(x) + 0.3 * x + rnorm(n, sd = 0.4)
write.csv(data.frame(x = sprintf("%.17g", x), y = sprintf("%.17g", y)),
          "tests/fixtures/loess_data.csv", row.names = FALSE, quote = FALSE)

grid <- data.frame(x = seq(min(x), max(x), length.out = 80))
ctl <- loess.control(surface = "direct", statistics = "exact")
fit <- function(deg, control) loess(y ~ x, span = 0.75, degree = deg, family = "gaussian",
                                    control = control)
p2 <- predict(fit(2, ctl), grid, se = TRUE)
p1 <- predict(fit(1, ctl), grid, se = TRUE)
pi <- predict(fit(2, loess.control()), grid, se = TRUE)
out <- data.frame(x = grid$x,
                  fit2 = p2$fit, se2 = p2$se.fit, df2 = p2$df, s2 = p2$residual.scale,
                  fit1 = p1$fit, se1 = p1$se.fit, df1 = p1$df, s1 = p1$residual.scale,
                  fit2_interp = pi$fit, se2_interp = pi$se.fit, df2_interp = pi$df)
out[] <- lapply(out, function(v) sprintf("%.10g", v))
write.csv(out, "tests/fixtures/loess_r.csv", row.names = FALSE, quote = FALSE)
cat(R.version.string, "\n")
