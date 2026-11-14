"""Live plots (``ggmanim.live``): updaters and ``ValueTracker`` s drive a built plot.

Render (``-s`` gives only the last frame)::

    manim -ql examples/live_plot.py SineTracker     # geom_function follows a ValueTracker
    manim -ql examples/live_plot.py StreamingLine   # a line growing point by point
    manim -ql examples/live_plot.py MovingDots      # update_data moves the same Dots

Every scene builds the plot once; per frame only the layers are recomputed and the
existing marks are moved, so axes, tick labels and legends are never laid out again.
"""

import numpy as np
import pandas as pd
from manim import Scene, ValueTracker, linear

from ggmanim import aes, ggplot, labs
from ggmanim.live import LivePlot, always_rebuild, build_live


class SineTracker(Scene):
    """``always_rebuild``: the curve's amplitude and phase read two trackers."""

    def construct(self):
        amp, phase = ValueTracker(0.5), ValueTracker(0.0)
        spec = (
            ggplot()
            .geom_function(
                fun=lambda x, a=amp, p=phase: a.get_value() * np.sin(x + p.get_value()),
                n=200,
                colour="#1f77b4",
                linewidth=4,
            )
            .geom_hline(yintercept=0, linetype="dashed")
            .xlim(0, 2 * np.pi)
            .ylim(-2, 2)
            .labs(x="x", y="a · sin(x + φ)", title="geom_function bound to ValueTrackers")
        )
        plot = always_rebuild(spec, layers=[0])  # only the curve is redrawn
        self.add(plot)
        self.play(amp.animate.set_value(1.8), run_time=2)
        self.play(phase.animate.set_value(np.pi), run_time=2, rate_func=linear)


def _signal(n: int = 120) -> pd.DataFrame:
    t = np.linspace(0, 10, n)
    rng = np.random.default_rng(3)
    return pd.DataFrame({"t": t, "value": np.sin(t) + 0.15 * rng.normal(size=n)})


class StreamingLine(Scene):
    """``update_data`` in an updater: rows are appended as a tracker advances."""

    def construct(self):
        full = _signal()
        now = ValueTracker(0.3)
        spec = (
            ggplot(full.iloc[:3], aes("t", "value"))
            .geom_line(colour="#d62728", linewidth=3)
            .xlim(0, 10)
            .ylim(-1.6, 1.6)
            .add(labs(title="Streaming line", x="time", y="signal"))
        )
        plot = build_live(spec)
        plot.add_updater(lambda m: m.update_data(full[full["t"] <= now.get_value()]))
        self.add(plot)
        self.play(now.animate.set_value(10), run_time=2, rate_func=linear)


class MovingDots(Scene):
    """``LivePlot`` with a spec factory: the same Dot objects glide around."""

    def construct(self):
        t = ValueTracker(0.0)
        base = pd.DataFrame({"id": list("abcdefgh"), "k": np.arange(8)})

        def frame() -> pd.DataFrame:
            ang = t.get_value() + base["k"] * np.pi / 4
            return base.assign(
                x=np.cos(ang) * (1 + 0.1 * base["k"]),
                y=np.sin(2 * ang) * 0.6 + 0.12 * (base["k"] - 3.5),
            )

        live = LivePlot(
            lambda: (
                ggplot(frame(), aes("x", "y", colour="id", key="id"))
                .geom_point(size=0.12)
                .xlim(-2, 2)
                .ylim(-1.2, 1.2)
                .labs(title="update_data keeps mark identity")
            )
        )
        live.plot.add_updater(live.updater())
        self.add(live.plot)
        self.play(t.animate.set_value(2 * np.pi), run_time=2, rate_func=linear)
