# Themes

`theme_default()` (= `theme_manim()`) is drawn straight on the scene background,
with foreground colours that follow it. The ggplot2 looks `theme_gray()` (alias
`theme_grey()`), `theme_bw()`, `theme_linedraw()`, `theme_light()` and
`theme_dark()` bring their own white plot background; `theme_minimal()`,
`theme_classic()` and `theme_void()` are background-agnostic. Complete themes
replace the current theme; `theme(...)` overrides single elements on top of it:


```python
import pandas as pd
from ggmanim import aes, ggplot, theme, theme_bw

cities = pd.DataFrame({"city": ["Lima", "Oslo", "Pune", "Kyiv"],
                       "share": [0.42, 0.17, 0.30, 0.11]})

themed = (ggplot(cities, aes("city", "share"))
          .geom_col()
          .add(theme_bw(), theme(axis_text_x_angle=30, panel_grid_minor=False,
                                 panel_background="#F4F1EA", plot_title_position="plot"))
          .labs(title="Share by city")
          .build())
```

Elements include text sizes and fonts, per-axis switches (`axis_text_x=False`,
`axis_ticks_y=False`, ...), tick-label rotation and justification
(`axis_text_x_angle`, `axis_text_x_hjust`), `panel_background`, `panel_border`,
major/minor grid per direction, `plot_background`, `plot_margin`,
`panel_width`/`panel_height` and all the `legend_*` elements above. See the
`Theme` docstring for the full list.

ggplot2's dotted element names become underscores and `element_*()` objects
become flat fields: `theme(legend.position = "bottom")` is
`theme(legend_position="bottom")`, `theme(axis.text.x = element_text(angle = 45))`
is `theme(axis_text_x_angle=45)`.

## Background and foreground

`theme_default()` takes its foreground colour (axes, ticks, text, legends) from
manim's `config.background_color` when the plot is built: white on manim's
default black background, black on a white one. To drop the white "paper" of a
ggplot2 look and draw it straight on the scene, add
`theme(plot_background=None, foreground=None, axis_text_color=None,
axis_title_color=None, title_color=None)`.

## Panel size

`theme(panel_width=..., panel_height=...)` sets the size of the panel (the whole
grid when faceted) in scene units. When unset, the panel fills the frame less a margin for tick labels and titles
(and room for the legend). `examples/gallery.py` (`ThemeGallery`) puts `theme_gray`, `theme_bw`,
`theme_dark` and `theme_void` next to each other.
