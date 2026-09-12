#!/usr/bin/env python
"""Figure 3, third strip: heritability (g), polygenicity (h), pleiotropy (i), pathway
overlap (j).

Author: Michael Beyeler (github.com/mjbeyeler)

Self-contained: fetches its inputs from the public data deposit by DOI and draws all four
panels. No access-controlled data.

PANEL j LIVES HERE, not with panel k, because the manuscript puts g-j on one row and gives
k a row of its own. Its data comes from the pathway tables rather than the gene tables, so
this script fetches from both -- the strip is a layout unit, not a single analysis.

    pixi run python 05_figures/main/fig3_polygenicity.py

The tables are built upstream in the analysis pipeline.

Python rather than R, unlike the a-c and d-f strips: the bars are a KDE-integrated
construct rather than a histogram (see 05_figures/lib/density_bars.py), and src/plot_style.py
carries the shared panel-letter and typography helpers. The g/h values are per-bin
proportions, the KDE line is 2 pt, and the top/right spines are removed as in Figure 2.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from matplotlib.ticker import MultipleLocator  # noqa: E402

from src.fetch_data import fetch  # noqa: E402
from src.plot_style import load_plot_style  # noqa: E402
from src.display_item import figure_dir  # noqa: E402 -- config: output.figures_dir

_DB_PATH = REPO_ROOT / "05_figures" / "lib" / "density_bars.py"
_spec = importlib.util.spec_from_file_location("density_bars", str(_DB_PATH))
density_bars = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(density_bars)

_RV_PATH = REPO_ROOT / "05_figures" / "lib" / "render_venn3.py"
_rv_spec = importlib.util.spec_from_file_location("render_venn3", str(_RV_PATH))
render_venn3 = importlib.util.module_from_spec(_rv_spec)
_rv_spec.loader.exec_module(render_venn3)

# Drawing order, and therefore z-order: the LVs are the tallest, narrowest distribution and
# go down first so the two TIF series stay visible in front of them.
SERIES = ("LV", "mTIF", "dTIF")
COLOR_KEY = {"LV": "lvs", "mTIF": "mtifs", "dTIF": "dtifs"}

# Bin edges. The published panels stopped at h2 = 0.30 and 290 genes; the bars are
# proportions of the whole feature set, so the bins have to reach past the estimate's
# upper tail or the missing mass simply vanishes from the sum -- the TIF series in g lost
# 11 % above 0.30 (4 Sep 2026). The bins now run further and the PLOTTED range is set
# separately, so the panels look as published while the arithmetic is complete.
H2_EDGES = np.arange(0.0, 0.55, 0.05)
H2_XLIM = (0.0, 0.30)
GENES_EDGES = np.arange(0, 300, 10)
GENES_XLIM = (0, 40)

# Panel i buckets: 1, 2, 3, 4, 5, then everything above pooled.
PLEIO_BUCKETS = ["1", "2", "3", "4", "5", ">5"]

# The published panel i came from sns.barplot, which quietly desaturates bar colours to
# 75 %: its orange was #DF7F20 where every other panel in the paper draws #FF7F00. That
# was reproduced during the fidelity pass and dropped again on 4 Sep 2026 -- one palette,
# defined in config, used verbatim everywhere.

# Clear space kept above the tallest panel letter, so an ascender cannot touch the strip's
# top edge (the assembler stacks the strips, so anything above it is simply cut).
LETTER_TOP_MARGIN_MM = 0.15


# Y AXIS: three ticks, and a top that clears the data. The published
# panels used a fixed MultipleLocator per panel, which puts the top tick wherever the step
# happens to land -- sometimes above the tallest bar, sometimes below it, and labelled
# "0.0" where "0" is meant. Instead the top is the smallest round number that is strictly
# above everything drawn in the panel, the middle tick is half of it, and the labels are
# formatted with {:g} so a whole number stays whole.
Y_TICK_MANTISSAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)


def drawn_max(ax):
    """Tallest thing drawn in `ax` within its current x limits -- bars and KDE lines."""
    lo, hi = sorted(ax.get_xlim())
    top = 0.0
    for patch in ax.patches:
        x = patch.get_x() + 0.5 * patch.get_width()
        if lo <= x <= hi:
            top = max(top, patch.get_y() + patch.get_height())
    for line in ax.lines:
        x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        inside = (x >= lo) & (x <= hi) & np.isfinite(y)
        if inside.any():
            top = max(top, float(y[inside].max()))
    return top


def three_y_ticks(ax, top=None):
    """0, half, top -- the top a round number strictly above the panel's tallest ink.

    `top` overrides that with a fixed ceiling, which CLIPS anything taller: panel h's LV
    curve peaks near 3 and squashes the other two series if the axis follows it.
    """
    if top is not None:
        ticks = [0.0, top / 2.0, top]
        ax.set_ylim(0.0, top)
        ax.set_yticks(ticks)
        ax.set_yticklabels(["{:g}".format(t) for t in ticks])
        return ticks
    target = drawn_max(ax)
    if target <= 0:
        raise SystemExit("panel with no positive data: cannot pick y ticks")
    # Half-steps are searched from small to large, so the top is the TIGHTEST round number
    # that still clears the data: 1.53 gives 0.8 / 1.6, not 1 / 2.
    exponent = int(np.floor(np.log10(target))) - 1
    for power in range(exponent, exponent + 4):
        for mantissa in Y_TICK_MANTISSAS:
            step = mantissa * 10.0 ** power
            if 2.0 * step > target * (1.0 + 1e-9):
                ticks = [0.0, step, 2.0 * step]
                ax.set_ylim(0.0, ticks[-1])
                ax.set_yticks(ticks)
                ax.set_yticklabels(["{:g}".format(t) for t in ticks])
                return ticks
    raise SystemExit("could not find three round y ticks above {:g}".format(target))


def panel_letter(fig, ax, letter, style, renderer, x=None, y=None):
    """Top-left corner of the panel's ink -- ticks, labels and title included -- which is
    where the published figure put them (Fig 4 does the same). `x` and `y` (figure
    fractions) override either coordinate: g and j take the figure-wide letter columns so
    they line up with a/d/k and with b/c/e/f, and all four letters of this strip share one
    baseline."""
    box = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    fig.text(box.x0 if x is None else x, box.y1 if y is None else y, letter,
             ha="left", va="center",
             fontsize=style.panel_label_fontsize, fontweight=style.panel_label_fontweight)


def letter_top(fig, ax, renderer):
    """Figure-fraction y of the top of a panel's ink."""
    return ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y1


def bucket(n):
    return str(n) if n <= 5 else ">5"


def main() -> int:
    config_path = REPO_ROOT / "config.yaml"
    with open(str(config_path), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    colors = {name: config["colors"][key] for name, key in COLOR_KEY.items()}

    style = load_plot_style(str(config_path))
    style.apply_font()
    fig3 = config["plot_styles"]["figures"]["fig3"]

    title_pt = style.title_fontsize
    axis_pt = style.axis_label_fontsize
    tick_pt = style.tick_label_fontsize
    style.assert_text_sizes(title=title_pt, axis_label=axis_pt, tick_label=tick_pt)
    kde_lw = float(fig3["kde_linewidth"])
    axis_lw_pt = style.axis_linewidth_pt
    tick_len_pt = style.axis_tick_length_pt

    print("Fetching Figure 3 polygenicity inputs from the data deposit ...")
    inputs = fetch("figure_intermediates", files=[
        "Fig3_12_g_heritability.csv",
        "Fig3_13_h_polygenicity.csv",
        "Fig3_14_i_pleiotropy.csv",
        "Fig3_16_j_pathway_venn_counts.csv",
    ])
    h2 = pd.read_csv(str(inputs["Fig3_12_g_heritability.csv"]))
    poly = pd.read_csv(str(inputs["Fig3_13_h_polygenicity.csv"]))
    pleio = pd.read_csv(str(inputs["Fig3_14_i_pleiotropy.csv"]))
    venn_counts = render_venn3.read_counts_csv(
        inputs["Fig3_16_j_pathway_venn_counts.csv"])

    width_mm = style.display_item_width_mm
    height_mm = float(fig3["strip_height_mm"]["polygenicity"])
    # Panel j's cell is much wider than g/h/i's. Their axes are squares whose side is set
    # by the strip's height, so an equal-width split left ~6 mm of slack in each of their
    # cells and 15 mm of white between neighbouring panels; j meanwhile has to hold a Venn
    # drawn at the figure-wide millimetres-per-unit scale. So the three squares are packed
    # to the left and the width they give up goes to j.
    fig, axs = plt.subplots(
        1, 4, figsize=(width_mm / 25.4, height_mm / 25.4), layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.55]},
    )
    # Leave a band above the panels for the letters: they sit centred on the top edge of
    # each panel's ink, so half a letter needs to fit between that edge and the strip's
    # top.
    letter_band = (0.6 * style.panel_label_fontsize / 72.0 * 25.4) / height_mm
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 1.0 - letter_band))

    # --- g: heritability -------------------------------------------------------------
    draw = {name: h2.loc[h2.feature_set == name, "h2"].values for name in SERIES}
    sums_g = density_bars.draw_density_bars(axs[0], draw, H2_EDGES, colors, kde_lw)
    axs[0].set_title("Heritability", fontsize=title_pt)
    axs[0].set_xlabel(r"$h^2$", fontsize=axis_pt)
    axs[0].set_ylabel("Trait proportions", fontsize=axis_pt)
    axs[0].set_xlim(*H2_XLIM)
    axs[0].set_xticks([0.0, 0.1, 0.2, 0.3])

    # --- h: polygenicity -------------------------------------------------------------
    draw = {name: poly.loc[poly.feature_set == name, "n_significant_genes"].values
            for name in SERIES}
    sums_h = density_bars.draw_density_bars(axs[1], draw, GENES_EDGES, colors, kde_lw)
    axs[1].set_title("Polygenicity", fontsize=title_pt)
    axs[1].set_xlabel("Genes (N)", fontsize=axis_pt)
    axs[1].set_ylabel("Trait proportions", fontsize=axis_pt)
    axs[1].set_xlim(*GENES_XLIM)
    axs[1].xaxis.set_major_locator(MultipleLocator(10))

    # --- i: pleiotropy ---------------------------------------------------------------
    # Percentages WITHIN each feature set, so the three sets are comparable despite having
    # 187, 210 and 122 significant genes. Each set's bars therefore sum to 100 %.
    pleio = pleio.assign(bucket=pleio["n_significant_traits"].map(bucket))
    counts = (
        pleio.groupby(["feature_set", "bucket"]).size().unstack(fill_value=0)
        .reindex(columns=PLEIO_BUCKETS, fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100.0
    if not np.allclose(pct.sum(axis=1), 100.0):
        raise SystemExit("panel i: percentages do not sum to 100 within a feature set")

    x = np.arange(len(PLEIO_BUCKETS))
    bar_w = 0.8 / len(SERIES)
    for i, name in enumerate(SERIES):
        axs[2].bar(x + bar_w * (i + 0.5) - 0.4, pct.loc[name, PLEIO_BUCKETS].values,
                   width=bar_w, color=colors[name],
                   linewidth=0)
    axs[2].set_title("Pleiotropy", fontsize=title_pt)
    # "Traits per gene (N)", not the published "Traits (N)": panel i buckets GENES by how
    # many traits each one is significant for, and with "Genes (%)" on the other axis the
    # bare noun reads as a gene count. h keeps "Genes (N)", where
    # "Trait proportions" on the y axis already says what is counted over.
    axs[2].set_xlabel("Traits per gene (N)", fontsize=axis_pt)
    axs[2].set_ylabel("Genes (%)", fontsize=axis_pt)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(PLEIO_BUCKETS)

    # --- j: pathway overlap ------------------------------------------------------------
    # The axis is only reserved here; render_venn3.place_in_axis() fills it after the
    # layout freeze, because the diagram's size on the page is set in millimetres and that
    # needs the axis's final box. Same module, constants and count size as panels c and f.
    venn_pt = float(fig3["venn_label_fontsize"])
    style.assert_text_sizes(venn_count=venn_pt)
    axs[3].set_axis_off()

    h_top = fig3.get("h_y_max")
    for index, ax in enumerate(axs[:3]):
        # The published panels were squares -- g, h and i were drawn one per figure at the
        # same size and read as a row of them -- but a square is not
        # what the data needs: all three distributions are wide and low, and the top third
        # of each box was empty. The boxes are now ghi_box_aspect (0.8) as tall as they are
        # wide, and strip_height_mm.polygenicity came down by what they gave up, so the row
        # is shorter rather than the white space larger.
        ax.set_box_aspect(float(fig3["ghi_box_aspect"]))
        # Three y ticks, after all the data and the x limits are in place.
        three_y_ticks(ax, float(h_top) if index == 1 and h_top is not None else None)
        # Axis furniture from config, not written here: the same numbers reach panels
        # a-f through natgen_style.R, so the whole display item carries one weight.
        ax.tick_params(labelsize=tick_pt, width=axis_lw_pt, length=tick_len_pt)
        for spine in ax.spines.values():
            spine.set_linewidth(axis_lw_pt)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Freeze the layout before the panel letters are placed -- they are derived from the
    # solved axis geometry, so this must run last.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    # Panel j occupies the display item's RIGHT COLUMN exactly -- the same split as the
    # a-c and d-f strips -- so that its letter lines up with b, c, e and f and its Venn
    # sits under theirs. g, h and i keep the left column, which is why their cells are
    # narrower than a quarter of the strip.
    right_frac = float(fig3["right_width_weight"]) / (
        float(fig3["left_width_weight"]) + float(fig3["right_width_weight"]))

    # Panel g's left spine goes under the Manhattans'. This strip is
    # laid out by matplotlib and the two above it by ggplot, so nothing made their plot
    # areas start at the same x -- g's began 1.35 mm to the left of a's and the column of
    # panels read as ragged. g, h and i move together by the same amount, so the widths and
    # the gaps constrained layout solved are untouched; only the row's left edge moves.
    ghi_shift = (float(fig3["ghi_left_mm"]) - axs[0].get_position().x0 * width_mm) / width_mm
    for ax in axs[:3]:
        box = ax.get_position()
        ax.set_position([box.x0 + ghi_shift, box.y0, box.width, box.height])
    right_edge_mm = axs[2].get_position().x1 * width_mm
    if right_edge_mm > (1.0 - right_frac) * width_mm:
        raise SystemExit(
            "panel i now ends at {:.1f} mm, inside panel j's column ({:.1f} mm): "
            "plot_styles.figures.fig3.ghi_left_mm has pushed the row too far right"
            .format(right_edge_mm, (1.0 - right_frac) * width_mm))
    print("  panels g-i  left spine at {:.2f} mm (Manhattan panel edge); i ends at "
          "{:.2f} mm, j starts at {:.2f}".format(
              axs[0].get_position().x0 * width_mm, right_edge_mm,
              (1.0 - right_frac) * width_mm))

    box_ghi = axs[0].get_position()
    axs[3].set_position([1.0 - right_frac, box_ghi.y0, right_frac, box_ghi.height])

    # Panel j at the scale panels c and f are placed at, so the three Venns carry the
    # same physical area of ink (see plot_styles.figures.fig3.venn_scale_mm_per_unit).
    # After the freeze, because it needs the axis's final size.
    venn_scale = float(fig3["venn_scale_mm_per_unit"])
    venn_j, (span_x, span_y) = render_venn3.place_in_axis(
        venn_counts, render_venn3.venn_colors(config),
        axs[3], venn_pt, venn_scale, str(fig3["venn_label_fontweight"]))
    diagram = render_venn3.patch_extent_units(venn_j, axs[3])
    if diagram[0] > span_x or diagram[1] > span_y:
        raise SystemExit(
            "panel j: the Venn needs {:.1f} x {:.1f} mm at {:.2f} mm/unit but its axis is "
            "{:.1f} x {:.1f} mm -- lower plot_styles.figures.fig3.venn_scale_mm_per_unit "
            "(shared with panels c and f)".format(
                diagram[0] * venn_scale, diagram[1] * venn_scale, venn_scale,
                span_x * venn_scale, span_y * venn_scale))

    renderer = fig.canvas.get_renderer()
    # Two letter columns, shared with the strips above: g at panel_letter_x_mm from the
    # left edge (with a, d and k), j the same distance into the right column (with b, c,
    # e and f). h and i keep the left edge of their own ink. All four sit on ONE baseline,
    # the highest of the four, so the row reads as a row.
    letter_x_mm = float(fig3["panel_letter_x_mm"])
    letter_x = {"g": letter_x_mm / width_mm,
                "j": (1.0 - right_frac) + letter_x_mm / width_mm}
    # ... but never so high that the letter runs off the top of the strip. The letters are
    # centred on that baseline, so half a letter sits above it; g survives on x-height
    # alone while h, i and j have ascenders and were being cut by ~0.3 mm (measured off
    # the built PDF, 4 Sep 2026 -- their ink reached row 0). The band reserved above the
    # panels is a fraction of the strip height and stopped being enough when the strip
    # came down to 38.7 mm; this clamp is in millimetres and cannot drift with it.
    half_letter_mm = 0.5 * style.panel_label_fontsize / 72.0 * 25.4
    ceiling = 1.0 - (half_letter_mm + LETTER_TOP_MARGIN_MM) / height_mm
    letter_y = min(max(letter_top(fig, ax, renderer) for ax in axs), ceiling)
    # ... except j, which sits at the top-left corner of its VENN rather than on the row's
    # baseline. The Venn is much shorter than the axis it is centred
    # in, so the shared baseline left the letter floating a long way above the diagram it
    # labels. Only the height moves: j keeps the right column's letter x, so b, c, e, f and
    # j still line up vertically down the display item.
    letter_y_by_panel = {"j": render_venn3.patch_top_figure_fraction(venn_j, fig)}
    for ax, letter in zip(axs, ["g", "h", "i", "j"]):
        panel_letter(fig, ax, letter, style, renderer,
                     x=letter_x.get(letter),
                     y=letter_y_by_panel.get(letter, letter_y))

    out_dir = figure_dir("main_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight" anywhere: it would silently change the locked strip width and
    # break the assembler's width check.
    fig.savefig(str(out_dir / "fig3_panel_3_ghij.pdf"), transparent=True)
    fig.savefig(str(out_dir / "fig3_panel_3_ghij.png"),
                dpi=int(fig3["hi_dpi"]), transparent=True)
    fig.savefig(str(out_dir / "fig3_panel_3_ghij__docs.png"),
                dpi=int(fig3["docs_dpi"]), transparent=True)
    plt.close(fig)

    box = axs[0].get_position()
    print("  panels g-i  {:.1f} x {:.1f} mm boxes (aspect {:g}); strip ink {:.1f} of "
          "{:.1f} mm tall".format(
              box.width * width_mm, box.height * height_mm,
              float(fig3["ghi_box_aspect"]),
              (max(a.get_tightbbox(renderer).y1 for a in axs)
               - min(a.get_tightbbox(renderer).y0 for a in axs)) / fig.dpi * 25.4,
              height_mm))
    print("  panel g  {} traits; bar sums per set  {}".format(
        len(h2), "  ".join("{} {:.3f}".format(k, v) for k, v in sums_g.items())))
    print("  panel h  bar sums per set  {}".format(
        "  ".join("{} {:.3f}".format(k, v) for k, v in sums_h.items())))
    print("  panel h  mean significant genes per trait  " + "  ".join(
        "{} {:.2f}".format(n, poly.loc[poly.feature_set == n, "n_significant_genes"].mean())
        for n in SERIES))
    print("  panel i  " + "  ".join(
        "{} {} genes".format(n, int(counts.loc[n].sum())) for n in SERIES))
    print("  panel j  regions: {}; diagram {:.1f} x {:.1f} mm at {:.2f} mm/unit "
          "({:.0f} mm2 of circles)".format(
              ", ".join(str(venn_counts[k]) for k in render_venn3.SUBSET_ORDER),
              diagram[0] * venn_scale, diagram[1] * venn_scale, venn_scale,
              venn_scale ** 2 * render_venn3.NORMALIZE_TO))
    print("Wrote Figure 3 polygenicity strip to {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
