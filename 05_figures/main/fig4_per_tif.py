#!/usr/bin/env python
"""Figure 4: each measured trait against its deep counterpart, one row per trait.

  a  SNP heritability of the mTIF and of the dTIF, and the difference dTIF - mTIF
  b  study-wide significant genes of the mTIF and of the dTIF, and how many of them are
     exclusive to each version

Author: Michael Beyeler (github.com/mjbeyeler)

Self-contained: fetches its three inputs from the public data deposit by DOI and draws the
four annotated heat-map strips. No access-controlled data. The tables are written by the
upstream analysis pipeline, which is where the PascalX and LDSC outputs are read.

    pixi run python 05_figures/main/fig4_per_tif.py

The rendering is pixel-faithful to the published panel except where a journal rule or a
recorded decision supersedes it:

  - the journal's 179 mm display-item width (was 210 mm) at the published aspect ratio,
    every type size from config.yaml and checked against the 5-7 pt band (titles were 9 pt,
    panel letters 13 pt), Nimbus Sans throughout (was DejaVu Sans), and vector PDF with
    editable text beside the PNG;
  - seaborn's heatmap is gone but its rendering rules are kept: the cells are pcolormesh
    quads and the numbers are text (so the PDF carries no bitmap), the colour bars have no
    outline and are de-rasterised, dark cell text is grey `.15` at seaborn's luminance
    cut-off, and the delta strip slices the diverging map to the data range exactly as
    seaborn's `center=0` does, so zero sits at the grey and the bar spans the data;
  - the published colour maps stay: a green ramp on the magnitude strips, coolwarm on the
    signed strips;
  - numeric text 6 pt with the trait names at 7 pt, sentence-case "delta heritability"
    with italic h2 as in Figure 3g, "0.00" rather than "-0.00", and thin white separators
    between the cells;
  - panel b's first strip is titled "Significant genes": it counts the study-wide
    significant genes (P < 1.14e-07), and it pairs with "Exclusive genes" as panel a's
    Heritability pairs with delta heritability;
  - the exclusive-gene key is one split bar, not the published signed one (4 Sep 2026,
    revised 5 Sep 2026): the published panel put both series on a single signed bar running
    -26..36 and labelled both ends with the count, which read as one scale with two zeros.
    The cell colours are the published ones -- the blue mTIF ramp and the red dTIF ramp,
    the two halves of the same coolwarm map, so a count of n sits where -n and +n sat --
    and the key is now ONE bar of the figure's own 1.57 x 17.17 mm, split down the middle
    into the two ramps (blue left, red right, in the columns' order), with no rule between
    them and one 0..36 scale on the right. Under the borderless colour-bar design
    (plot_styles.colorbar_ticks.outline_pt = 0, 5 Sep 2026) nothing is drawn around the
    pair either; the two ramps meet edge to edge and the bar reads as one.

Every number drawn is in the deposit: the h2 with its standard error (not shown on the
panel), and for panel b the identity and both P values of every gene counted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from seaborn.utils import relative_luminance  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.fetch_data import fetch  # noqa: E402
from src.display_item import figure_dir, ink_margins_mm, render_to_width, save, verify  # noqa: E402
from src.plot_style import draw_colorbar_ticks, load_plot_style  # noqa: E402

H2_FILE = "Fig4_01_a_heritability_per_tif.csv"
COUNTS_FILE = "Fig4_02_b_gene_counts_per_tif.csv"
MEMBERS_FILE = "Fig4_03_b_gene_membership_per_tif.csv"

OUT_DIR = figure_dir("main_figures")
OUT_NAME = "fig4_per_tif"
MM = 25.4  # millimetres per inch

# seaborn.matrix._annotate_heatmap: dark text is ".15", not black, above this luminance.
TEXT_LUMINANCE_CUTOFF = 0.408


def centred_slice(cmap, vmin, vmax):
    """seaborn's `center=0` behaviour (matrix.py, _determine_cmap_params).

    The diverging map is laid over the symmetric range +-max(|vmin|, |vmax|) so that zero is
    its neutral centre, and then only the part between vmin and vmax is kept as a new
    colormap with a plain Normalize(vmin, vmax). The colour bar therefore spans the data
    range, with zero off-centre -- which is how the published panel looks.
    """
    vrange = max(vmax, -vmin)
    symmetric = Normalize(-vrange, vrange)
    cmin, cmax = symmetric([vmin, vmax])
    return ListedColormap(cmap(np.linspace(cmin, cmax, 256))), Normalize(vmin, vmax)


def half_ramp(cmap, half):
    """One half of a diverging map, as a ramp running from its neutral centre outwards.

    `half` is "low" (the centre down to the map's first colour) or "high" (the centre up to
    its last). On Normalize(0, vmax) a value v then lands on exactly the colour the full map
    gives -v or +v on the symmetric range, so a pair of these ramps keeps every cell colour
    of the published signed strip while giving each series its own 0-based key.
    """
    return ListedColormap(cmap(np.linspace(0.5, 0.0 if half == "low" else 1.0, 256)))


def heat_strip(ax, values, col_labels, cmap, norm, fmt, annot_pt, tick_pt, edge_lw,
               axis_lw_pt, tick_len_pt, shown=None):
    """One strip: rows are traits, top to bottom; one column per series; a number in
    every cell. `cmap` and `norm` may be given per column -- the exclusive-gene strip gives
    each series its own ramp over the same range -- and the strip is then drawn as one mesh
    per column, one per colour bar. `shown` overrides the printed value."""
    n_rows, n_cols = values.shape
    per_column = isinstance(cmap, (list, tuple))
    cmaps = list(cmap) if per_column else [cmap] * n_cols
    norms = list(norm) if per_column else [norm] * n_cols
    if per_column:
        # The edges shared by two columns are drawn by both meshes, at the same place and
        # the same width, so the white separators come out as they do on a single mesh.
        mesh = [ax.pcolormesh(np.arange(j, j + 2), np.arange(n_rows + 1),
                              values[:, j:j + 1], cmap=cmaps[j], norm=norms[j],
                              edgecolors="white", linewidth=edge_lw)
                for j in range(n_cols)]
    else:
        mesh = ax.pcolormesh(np.arange(n_cols + 1), np.arange(n_rows + 1), values,
                             cmap=cmap, norm=norm, edgecolors="white", linewidth=edge_lw)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)  # first trait at the top, as published
    shown = values if shown is None else shown
    for i in range(n_rows):
        for j in range(n_cols):
            value = shown[i, j]
            if abs(value) < 0.005:
                value = 0.0  # a rounded-to-zero negative would print as "-0.00"
            lum = relative_luminance(cmaps[j](norms[j](values[i, j])))
            ax.text(j + 0.5, i + 0.5, fmt.format(value), ha="center", va="center",
                    fontsize=annot_pt, color=".15" if lum > TEXT_LUMINANCE_CUTOFF else "w")
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=tick_pt)
    ax.set_yticks([])
    # The tick marks are this panel's only axis furniture -- the spines come off below --
    # so they carry the paper-wide weight and length from config, not matplotlib's
    # 0.8 x 3.5 pt defaults, which left them heavier than Figures 2 and 3 (5 Sep 2026).
    ax.tick_params(width=axis_lw_pt, length=tick_len_pt)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def add_colorbar(fig, mesh, ax, fig4, ticks_cfg, tick_pt, strip_width_mm, strip_height_mm,
                 fmt="{:.2f}"):
    """The colour bar(s) of one strip, the metric size of Figure 2's. Returns them in a
    list, in the order the meshes came.

    A bar is `colorbar_mm` (width, height) in physical millimetres -- Figure 2's bar
    measured off fig2_lv.py and turned vertical, so the figures' bars match -- first
    solved into matplotlib's shrink/aspect so the layout reserves the room, then pinned
    to the exact box once the layout is frozen (see build()). `fmt` formats the labels.

    A strip whose columns carry their own ramp hands over one mesh per column and gets one
    half-width bar each, pinned by build() side by side inside ONE bar's box, in the
    strip's own column order: the key is the same size as every other
    bar in the figure and reads as one bar split down the middle. Nothing divides the two
    ramps -- their own outlines are off, and build() draws the single outline around the
    pair only when one is asked for (outline_pt > 0) -- and only the rightmost is ticked,
    since the halves share a range and one scale on the right is what the reader needs.
    """
    meshes = list(mesh) if isinstance(mesh, (list, tuple)) else [mesh]
    width_mm, height_mm = [float(v) for v in fig4["colorbar_mm"]]
    # `fraction` is the slice of the parent's width handed to the bar; the default 15 %
    # is too little for the one-column delta strip, and the bar would come out narrower
    # than asked. The bar's box aspect then fits the height and width inside that slice.
    fraction = min(0.5, 1.1 * width_mm / strip_width_mm) / len(meshes)
    cbars = []
    for index, one in enumerate(meshes):
        cbar = fig.colorbar(one, ax=ax, shrink=height_mm / strip_height_mm,
                            aspect=height_mm / width_mm, fraction=fraction,
                            pad=float(fig4["colorbar_pad"]) if index == 0 else 0.0)
        cbars.append(cbar)
        # Colorbar.solids is created rasterized=True, which would drop a bitmap into an
        # otherwise vector PDF (see the conventions in 05_figures/README.md).
        cbar.solids.set_rasterized(False)
        # Paper-wide colour-bar rule (config plot_styles.colorbar_ticks): the box's outline
        # is outline_pt wide and OFF at 0, which is the borderless design now in force; the
        # tick MARKS are drawn by draw_colorbar_ticks() once the bar is pinned (see
        # build()), on the label side, so the labels keep the pad the marks take up.
        # A split bar is outlined once, around the pair, by build(): an outline per half
        # would put a rule down the middle, which is exactly what the split must not have.
        outline_pt = float(ticks_cfg["outline_pt"])
        cbar.outline.set_visible(outline_pt > 0 and len(meshes) == 1)
        cbar.outline.set_linewidth(outline_pt)
        cbar.outline.set_edgecolor("black")
        if index < len(meshes) - 1:
            cbar.set_ticks([])
            continue
        label_side_pad = (float(ticks_cfg["length_pt"])
                          if ticks_cfg["vertical_side"] == "right" else 0.0)
        cbar.ax.tick_params(labelsize=tick_pt, length=0,
                            pad=float(fig4["colorbar_tick_pad"]) + label_side_pad)
        # Two ticks per bar, at its two ends (decided 3 Sep 2026): the bar spans the range
        # it is normalised to, so the ends are the range.
        positions = [one.norm.vmin, one.norm.vmax]
        cbar.set_ticks(positions)
        cbar.set_ticklabels([fmt.format(t) for t in positions])
    return cbars


def panel_letter(fig, ax, letter, style, renderer, gap_mm=0.0):
    """Top-left corner of the panel's ink -- ticks, labels and title included -- which is
    where the published figure put them. Size and weight are the global ones. With
    `gap_mm` the letter is set right-aligned that far LEFT of the ink instead, clear of
    it: panel b's ink starts at its heat map, and a letter sitting on the cells' edge
    reads as part of the table."""
    box = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    if gap_mm > 0:
        fig.text(box.x0 - gap_mm / 25.4 / fig.get_size_inches()[0], box.y1, letter,
                 ha="right", va="center", fontsize=style.panel_label_fontsize,
                 fontweight=style.panel_label_fontweight)
        return
    fig.text(box.x0, box.y1, letter, ha="left", va="center",
             fontsize=style.panel_label_fontsize, fontweight=style.panel_label_fontweight)


def main() -> int:
    config_path = REPO_ROOT / "config.yaml"
    with open(str(config_path), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    fig4 = config["plot_styles"]["figures"]["fig4"]

    style = load_plot_style(str(config_path))
    style.apply_font()

    title_pt = style.title_fontsize
    # The trait names are the row-axis labels, the counterpart of the column labels'
    # heading, so they take the axis-label size (the Figure 2 rule); the numeric text --
    # cell values, column labels, colour-bar ticks -- takes the tick size.
    trait_pt = style.axis_label_fontsize
    tick_pt = style.tick_label_fontsize
    # Axis furniture, paper-wide: one weight and one tick length across Figures 2-4.
    axis_lw_pt = style.axis_linewidth_pt
    tick_len_pt = style.axis_tick_length_pt
    annot_pt = float(fig4["annotation_fontsize"])
    style.assert_text_sizes(title=title_pt, trait_label=trait_pt, tick_label=tick_pt,
                            annotation=annot_pt)

    print("Fetching Figure 4 inputs from the data deposit ...")
    inputs = fetch("figure_intermediates", files=[H2_FILE, COUNTS_FILE, MEMBERS_FILE])
    h2 = pd.read_csv(str(inputs[H2_FILE]))
    counts = pd.read_csv(str(inputs[COUNTS_FILE]))
    members = pd.read_csv(str(inputs[MEMBERS_FILE]))

    if list(h2["trait"]) != list(counts["trait"]):
        raise SystemExit("panels a and b list the traits in different orders")
    traits = list(h2["trait_label"])
    n_rows = len(traits)

    # The membership table must reproduce the counts it sits beside.
    from_members = members.groupby("trait").agg(
        n_mtif=("in_mtif", "sum"), n_dtif=("in_dtif", "sum"))
    # fill_value: a trait with no significant gene in either version has no rows here.
    from_members = from_members.reindex(counts["trait"], fill_value=0)
    if not (from_members["n_mtif"].values == counts["n_mtif"].values).all() or \
            not (from_members["n_dtif"].values == counts["n_dtif"].values).all():
        raise SystemExit("gene membership table does not reproduce the per-trait counts")

    # --- colour scales, as published ------------------------------------------------------
    sequential = plt.get_cmap(str(fig4["sequential_cmap"]))
    diverging = plt.get_cmap(str(fig4["diverging_cmap"]))

    h2_values = h2[["h2_mtif", "h2_dtif"]].to_numpy()
    delta = h2[["delta_h2"]].to_numpy()
    gene_values = counts[["n_mtif", "n_dtif"]].to_numpy(dtype=float)
    exclusive = counts[["n_exclusive_mtif", "n_exclusive_dtif"]].to_numpy(dtype=float)

    # The h2 ramp runs over a fixed 0-0.3 (config), not the data range, so the two end
    # ticks are round numbers; the other three bars span their data.
    h2_norm = Normalize(*[float(v) for v in fig4["h2_range"]])
    if h2_values.min() < h2_norm.vmin or h2_values.max() > h2_norm.vmax:
        raise SystemExit("h2 values fall outside plot_styles.figures.fig4.h2_range")
    gene_norm = Normalize(gene_values.min(), gene_values.max())
    delta_cmap, delta_norm = centred_slice(diverging, delta.min(), delta.max())
    # Exclusive genes: the published colours, one ramp per series. The ramps are the two
    # halves of the coolwarm map the published panel used, so every cell keeps the colour
    # it had; both run from zero to the same maximum, and each gets its own full-size bar,
    # the two stacked in the columns' order.
    excl_norm = Normalize(0.0, exclusive.max())
    excl_cmaps = [half_ramp(diverging, "low"), half_ramp(diverging, "high")]

    # --- figure ---------------------------------------------------------------------------
    # Drawn on a canvas and cropped to its ink, like Figure 2: the journal's display-item
    # rules want the artwork to FILL exactly 179 mm with no white margin, and constrained
    # layout leaves ~1 mm of padding on every side. render_to_width() re-runs build() until
    # the ink is the display-item width; the canvas keeps the published aspect ratio.
    width_mm = style.display_item_width_mm
    height_mm = float(fig4["height_mm"])
    edge_lw = float(fig4["cell_edge_linewidth"])
    # Singular: each cell is one trait's mTIF or dTIF, and the delta column's
    # "dTIF − mTIF" then reads consistently with the headers.
    series = ["mTIF", "dTIF"]

    def build(canvas_w_in):
        canvas_h_in = canvas_w_in * height_mm / width_mm
        fig = plt.figure(figsize=(canvas_w_in, canvas_h_in), layout="constrained")
        # Five columns: the four strips and, between panels a and b, an empty spacer
        # column whose only job is the visual separation of the two panels.
        ratios = [float(w) for w in fig4["width_ratios"]]
        ratios.insert(2, float(fig4["panel_gap_ratio"]))
        grid = fig.add_gridspec(1, 5, width_ratios=ratios)
        axs = [fig.add_subplot(grid[0, c]) for c in (0, 1, 3, 4)]

        meshes = [
            heat_strip(axs[0], h2_values, series, sequential, h2_norm, "{:.2f}",
                       annot_pt, tick_pt, edge_lw, axis_lw_pt, tick_len_pt),
            heat_strip(axs[1], delta, ["dTIF − mTIF"], delta_cmap, delta_norm, "{:.2f}",
                       annot_pt, tick_pt, edge_lw, axis_lw_pt, tick_len_pt),
            heat_strip(axs[2], gene_values, series, sequential, gene_norm, "{:.0f}",
                       annot_pt, tick_pt, edge_lw, axis_lw_pt, tick_len_pt),
            heat_strip(axs[3], exclusive, series, excl_cmaps, [excl_norm, excl_norm],
                       "{:.0f}", annot_pt, tick_pt, edge_lw, axis_lw_pt,
                       tick_len_pt),
        ]
        axs[0].set_title("Heritability", fontsize=title_pt)
        axs[0].set_yticks(np.arange(n_rows) + 0.5)
        axs[0].set_yticklabels(traits, fontsize=trait_pt)
        axs[1].set_title("Δ heritability", fontsize=title_pt)
        axs[2].set_title("Significant genes", fontsize=title_pt)
        axs[3].set_title("Exclusive genes", fontsize=title_pt)

        # The colour bars are sized in millimetres, so the strip height they are a
        # fraction of has to be known first: solve the layout once without them.
        fig.canvas.draw()
        strip_h_mm = axs[0].get_position().height * canvas_h_in * MM
        # The h2 bar's ends are the round numbers 0 and 0.3, so no trailing zeros; the
        # delta bar's ends are data values and keep the cells' two decimals.
        formats = ["{:g}", "{:.2f}", "{:.0f}", "{:.0f}"]
        cbars = []
        for ax, mesh, fmt in zip(axs, meshes, formats):
            strip_w_mm = ax.get_position().width * canvas_w_in * MM
            cbars.append(add_colorbar(fig, mesh, ax, fig4,
                                      config["plot_styles"]["colorbar_ticks"], tick_pt,
                                      strip_w_mm, strip_h_mm, fmt=fmt))

        # Freeze the layout before the panel letters are placed; they are derived from
        # the solved geometry, so they must come last (the conventions in 05_figures/README.md).
        fig.canvas.draw()
        fig.set_layout_engine("none")
        # Constrained layout has reserved room for the bars but renegotiates their size
        # (all four came out at 0.81x the request). With the layout frozen, pin each bar
        # to the exact millimetre box: at the strip's right edge plus the pad, centred on
        # the strip's height, as matplotlib itself would place it. A strip with two ramps
        # divides that one box between them, left to right in the columns' order, and gets
        # a single outline drawn around the whole so no rule falls on the split.
        bar_w_mm, bar_h_mm = [float(v) for v in fig4["colorbar_mm"]]
        bar_w, bar_h = bar_w_mm / MM / canvas_w_in, bar_h_mm / MM / canvas_h_in
        ticks_cfg = config["plot_styles"]["colorbar_ticks"]
        for ax, group in zip(axs, cbars):
            box = ax.get_position()
            x0 = box.x1 + float(fig4["colorbar_pad"]) * box.width
            y0 = box.y0 + (box.height - bar_h) / 2
            share = bar_w / len(group)
            for index, cbar in enumerate(group):
                # fig.colorbar fixes the axes' BOX ASPECT, so a position whose aspect
                # differs from the request is re-fitted inside it at draw time -- a
                # half-width bar would come back half height too. Release it: the box is
                # being given in millimetres here, and nothing is left to fit.
                cbar.ax.set_box_aspect(None)
                cbar.ax.set_aspect("auto")
                cbar.ax.set_position([x0 + index * share, y0, share, bar_h])
            if len(group) > 1 and float(ticks_cfg["outline_pt"]) > 0:
                fig.add_artist(Rectangle((x0, y0), bar_w, bar_h, transform=fig.transFigure,
                                         fill=False, edgecolor="black",
                                         linewidth=float(ticks_cfg["outline_pt"])))
            draw_colorbar_ticks(group[-1].ax, [0.0, 1.0], str(ticks_cfg["vertical_side"]),
                                float(ticks_cfg["length_pt"]), float(ticks_cfg["width_pt"]),
                                outline_pt=float(ticks_cfg["outline_pt"]))
        # A thin vertical rule down the middle of the spacer column, from the top of the
        # strips to the bottom of their column labels, to separate panel a from panel b.
        divider_lw = float(fig4["divider_linewidth"])
        if divider_lw > 0:
            left, right = cbars[1][-1].ax.get_position(), axs[2].get_position()
            x = 0.5 * (left.x1 + right.x0)
            strips = axs[0].get_position()
            fig.add_artist(plt.Line2D([x, x], [strips.y0, strips.y1],
                                      transform=fig.transFigure, color=fig4["divider_color"],
                                      linewidth=divider_lw, solid_capstyle="butt"))
        renderer = fig.canvas.get_renderer()
        panel_letter(fig, axs[0], "a", style, renderer)
        panel_letter(fig, axs[2], "b", style, renderer,
                     gap_mm=float(fig4["panel_letter_gap_mm"]))
        fig._fig4_axes = axs  # for the size report below
        return fig

    fig, ink = render_to_width(build, style)
    out_stem = str(OUT_DIR / OUT_NAME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # matplotlib's bbox for the 45-degree column labels is the box of the rotated
    # rectangle, whose empty corners leave ~0.5 mm of white under the labels that the
    # journal's zero-margin rule (and the staging gate) rejects. Save once, measure the
    # real ink with ghostscript, trim top and bottom, save again. Left and right are left
    # alone: the width is already the display-item width to within tolerance.
    probe = out_stem + "__probe.pdf"
    fig.savefig(probe, bbox_inches=ink, pad_inches=0)
    _, bottom_mm, _, top_mm = ink_margins_mm(probe)
    Path(probe).unlink()
    ink = Bbox.from_extents(ink.x0, ink.y0 + bottom_mm / MM, ink.x1, ink.y1 - top_mm / MM)

    fig.savefig(out_stem + "__docs.png", bbox_inches=ink, pad_inches=0,
                dpi=int(fig4["docs_dpi"]))
    save(fig, ink, out_stem, dpi=int(fig4["hi_dpi"]))   # PDF + PNG cropped to the ink
    ink_w_mm, ink_h_mm = ink.width * MM, ink.height * MM

    problems = verify(out_stem + ".pdf", style, geometry=True,
                      expect_text=("Heritability", "Exclusive genes", "A temporal angle"))
    if problems:
        raise SystemExit("display-item rules violated:\n  " + "\n  ".join(problems))
    print("  display-item check passed (width, margins, font, no rasters, text)")

    higher = int((h2["delta_h2"] > 0).sum())
    print("  panel a  {} traits; dTIF h2 higher for {} of them; largest |delta| {} "
          "({:+.2f}); bar {:.3f}..{:.3f}, delta bar {:.3f}..{:.3f}".format(
              n_rows, higher, h2.loc[h2["delta_h2"].abs().idxmax(), "trait_label"],
              h2.loc[h2["delta_h2"].abs().idxmax(), "delta_h2"],
              h2_norm.vmin, h2_norm.vmax, delta_norm.vmin, delta_norm.vmax))
    print("  panel b  threshold P < {:.3g}; genes per trait mTIF {:.1f}, dTIF {:.1f}; "
          "exclusive mTIF {} (max {:.0f}), dTIF {} (max {:.0f}); bars 0..{:.0f} and a "
          "split bar, both halves 0..{:.0f}".format(
              counts["threshold_p"].iloc[0], counts["n_mtif"].mean(),
              counts["n_dtif"].mean(), int(counts["n_exclusive_mtif"].sum()),
              exclusive[:, 0].max(), int(counts["n_exclusive_dtif"].sum()),
              exclusive[:, 1].max(), gene_norm.vmax, excl_norm.vmax))
    print("  {:.1f} x {:.1f} mm (ink; canvas aspect {:.0f} x {:.0f})".format(
        ink_w_mm, ink_h_mm, width_mm, height_mm))
    fig_w_in, fig_h_in = fig.get_size_inches()
    bars = [a for a in fig.axes if a not in fig._fig4_axes]
    print("  colour bars: " + ", ".join(
        "{:.2f} x {:.2f} mm".format(a.get_position().width * fig_w_in * MM,
                                     a.get_position().height * fig_h_in * MM)
        for a in bars) + "  (target {} x {} mm, Figure 2's bar)".format(*fig4["colorbar_mm"]))
    print("Wrote Figure 4 to {}".format(OUT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
