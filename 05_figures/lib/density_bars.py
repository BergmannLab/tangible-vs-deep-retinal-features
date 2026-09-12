"""Overlaid distributions drawn as kernel-smoothed per-bin proportions plus a KDE line.

Shared by Figure 3 panels g and h, because the geometry is not obvious and the two panels
must not drift apart.

Author: Michael Beyeler (github.com/mjbeyeler)

WHAT MAKES THIS NOT A HISTOGRAM. The bar heights are not empirical bin counts: each bar is
the *KDE's* mass in that bin, `gaussian_kde.integrate_box_1d(lo, hi)`, i.e. the smoothed
proportion of the feature set's traits falling in the bin. So the bars and the line
describe the same estimate at two resolutions, and the bars of a 17-trait series stay
legible where a 17-sample histogram would be mostly empty bins. Do not "simplify" it into
`hist(density=True)`, which would change what is plotted.

Y-AXIS UNITS. Both the bars and the curve are multiplied by the bin width, so the bars are
per-bin proportions -- the axis reads "Trait proportions" -- and the curve is the same
estimate drawn continuously.

The estimator is reflected at zero (see REFLECT_AT), because both quantities it is used for
are bounded below by zero. At the upper end the curve stops at the largest observation
(seaborn 0.11.2 `histplot(kde=True)` sets `cut = 0`, gridsize 200).

The series are drawn side by side within each bin rather than overlaid, so that three
distributions with very different spreads stay readable without transparency tricks.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

BAR_ALPHA = 0.4
KDE_GRIDSIZE = 200      # seaborn's default gridsize
# Both quantities plotted with these bars -- SNP heritability and a count of significant
# genes -- are bounded below by zero, so the kernel is REFLECTED at zero: the estimate at
# x is f(x) + f(-x). Without it a Gaussian kernel sitting on the pile of traits at or near
# zero throws half its weight onto impossible negative values, which does two visible
# things: the bars stop summing to 1, and the curve RISES away from zero before
# decaying, because near the boundary the estimate is missing the mass that fell off the left edge. The true
# distributions are monotone decreasing from zero. Reflection folds that mass back where
# it belongs; it is the standard boundary correction for a density on a half-line and it
# changes nothing away from the boundary.
REFLECT_AT = 0.0


def kde_bar_heights(values, edges):
    """Smoothed proportion of the series in each bin -- the bar heights.

    Bins wholly below the smallest or at/above the largest observed value are zero, as
    in the published construction.
    """
    kde = gaussian_kde(values)
    heights = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # ... plus the mass the kernel put on the mirror image of the bin, below the
        # boundary, where the quantity cannot go (see REFLECT_AT).
        mirrored = kde.integrate_box_1d(2 * REFLECT_AT - hi, 2 * REFLECT_AT - lo)
        heights.append(kde.integrate_box_1d(lo, hi) + mirrored)
    return np.array(heights), kde


def draw_density_bars(ax, series, edges, colors, line_width):
    """Draw one bar group and one KDE line per series.

    Parameters
    ----------
    series : mapping of label -> 1-D values, in the order they should be drawn
    edges  : bin edges shared by every series (uniform width)
    colors : mapping of label -> colour

    Returns the per-series sum of the bar heights, for the caller to report.
    """
    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)
    if not np.allclose(widths, widths[0]):
        raise ValueError("density_bars needs uniform bins")
    bin_width = float(widths[0])
    n = len(series)
    # Each bin is split into n slots so the bars sit beside each other, not on top.
    slot = widths / n

    sums = {}
    for i, (label, values) in enumerate(series.items()):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        heights, kde = kde_bar_heights(values, edges)
        sums[label] = float(heights.sum())
        ax.bar(
            edges[:-1] + slot * (i + 0.5),
            heights,
            width=slot,
            color=colors[label],
            alpha=BAR_ALPHA,
            linewidth=0,
            zorder=1,
        )
        # From the boundary, not from the smallest observation: seaborn's `cut = 0` starts
        # the curve at the data minimum, which in panel g left the TIF curves beginning
        # abruptly at h2 = 0.05 with empty axis beneath them. With
        # the kernel reflected at zero the estimate below the smallest observation is
        # real -- it is where the mass folded back went -- so the curve is drawn there.
        # The upper end keeps `cut = 0` and stops at the largest observation.
        grid = np.linspace(min(REFLECT_AT, values.min()), values.max(), KDE_GRIDSIZE)
        curve = (kde(grid) + kde(2 * REFLECT_AT - grid)) * bin_width
        ax.plot(grid, curve, color=colors[label], linewidth=line_width, zorder=2)
    return sums
