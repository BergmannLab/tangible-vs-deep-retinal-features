#!/usr/bin/env python3
"""Figure 2: what the RETFound representation captures of the measured retinal traits.

  a  Bi-clustered heatmap of the raw Pearson correlation between each of the 17 measured
     tangible image features (mTIFs) and each of the 1024 RETFound latent variables (LVs).
  b  Per-trait prediction accuracy (R2) of three models, in the row order panel a's
     clustering produces: the best single LV (SLR), all LVs (MLR), and the fine-tuned
     deep tangible image features (dTIFs).
  c  The attention maps of the 17 fine-tuned dTIF models.

Both panels rest on the reported analysis cohort of 41 526, so the figure states one n
(the handful of traits whose fine-tuned prediction is missing for a few participants are
exact in panel b's `n` column).

Authors: a, b  Michael Beyeler (github.com/mjbeyeler)
         c     David Presby (github.com/presbyd)

Self-contained: fetches its inputs from the public data deposit by DOI (downloading on
first run, reusing the local cache afterwards) and builds the figure from them. No
access-controlled data, no cluster, no GPU. The deposited files are written by the
upstream analysis pipeline.

    pixi run python 05_figures/main/fig2_lv.py

How the figure is built:

  - one figure, both panels, sharing one 17-row grid, at the journal's 179 mm display-item
    width and in vector PDF with editable text;
  - the dendrograms, heatmap and colourbar are drawn into explicit axes rather than by
    seaborn's clustermap, which owns its own Figure and so cannot share a canvas with
    panel b. The clustering is average linkage on the signed dissimilarity 1 - r, and it
    reproduces the published row order exactly;
  - panel b overlays the five cross-validation fold R2 on every bar, which is what the
    journal's data-presentation policy asks for when n <= 10: the observations themselves,
    not a summary interval drawn from them. All three series use the same cross-validation
    partition, so each bar is the mean of the five dots drawn on it and fold k is the same
    participants in all three series;
  - the panel letters are placed by the shared helper, and every type size is read from
    config.yaml and checked against the journal's band before anything is drawn.

The heatmap is vector, not a rasterised image: 17 x 1024 quads via pcolormesh, so
`pdfimages -list` on the output reports no images at all.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.cluster.hierarchy as hc  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.display_item import figure_dir, ink_margins_mm  # noqa: E402
from src.fetch_data import FetchError, fetch  # noqa: E402
from src.plot_style import (add_panel_labels_corner, draw_colorbar_ticks,  # noqa: E402
                            fold_marker_spec, load_plot_style, swarm_diameter,
                            swarm_offsets)

CONFIG = os.path.join(REPO_ROOT, "config.yaml")
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_config  # noqa: E402 -- machine paths: config.local.yaml

OUT_DIR = str(figure_dir("main_figures"))
OUT_NAME = "fig2_lv"

CORR_FILE = "Fig2_01_a_lv_mtif_correlation.csv"
R2_FILE = "Fig2_02_b_prediction_r2.csv"
ATTENTION_ARCHIVE = "Fig2_03_c_attention_maps.zip"  # panel c, holding one
ATTENTION_FILE = "Fig2_03_c_attention_map_{}.png"   # member per trait token

# ---------------------------------------------------------------------------
# Style. Every size comes from config.yaml; nothing here is a literal.
# ---------------------------------------------------------------------------
STYLE = load_plot_style(CONFIG)

_CFG = load_config(CONFIG) or {}

COLORS = _CFG.get("colors", {}) or {}
_CB_TICKS = (_CFG.get("plot_styles", {}) or {}).get("colorbar_ticks", {}) or {}
CB_TICK_LEN_PT = float(_CB_TICKS.get("length_pt", 2.0))
CB_TICK_W_PT = float(_CB_TICKS.get("width_pt", 0.4))
CB_OUTLINE_PT = float(_CB_TICKS.get("outline_pt", 0.0))
FIG2 = ((_CFG.get("plot_styles", {}) or {}).get("figures", {}) or {}).get("fig2", {}) or {}
_CFG_C = FIG2.get("panel_c", {}) or {}

# Axis furniture -- spines and tick marks -- from the paper-wide knobs, so Figures 2, 3
# and 4 carry one weight and one tick length. matplotlib's own defaults (0.8 x 3.5 pt)
# are never what we want here (5 Sep 2026).
AXIS_LW_PT = STYLE.axis_linewidth_pt          # 0.5 pt
AXIS_TICK_LEN_PT = STYLE.axis_tick_length_pt  # 2 pt

AXIS_LABEL_FS = STYLE.axis_label_fontsize     # 7 pt
TICK_FS = STYLE.tick_label_fontsize           # 6 pt -- numeric ticks only
LEGEND_FS = STYLE.info_fontsize               # 6 pt
# The trait names are panel a's row-axis LABELS, the counterpart of the column axis title,
# not numeric tick labels -- so they take the axis-label size, and the two axes of the
# heatmap are lettered consistently. The numeric ticks stay one step down.
TRAIT_FS = AXIS_LABEL_FS
STYLE.assert_text_sizes(axis_label=AXIS_LABEL_FS, tick_label=TICK_FS, legend=LEGEND_FS,
                        trait_label=TRAIT_FS)

MM = 25.4  # millimetres per inch

# Model series, drawn top to bottom within each trait row. Tokens match the deposit.
SERIES = [
    # No literal fallbacks: a default that shadows config is how two figures end up
    # drawing the same series in two different colours (4 Sep 2026).
    ("slr_top_lv", "Top LV (SLR)", COLORS["top_lvs"]),
    ("mlr_all_lvs", "All LVs (MLR)", COLORS["lvs"]),
    ("dtif", "Deep TIFs", COLORS["dtifs"]),
]


def _mm(key, default):
    return float(FIG2.get(key, default))


def _dendrogram_segments(linkage):
    """Leaf-space line segments of a dendrogram, without drawing it.

    scipy returns `icoord` in leaf space (leaf k centred on 5 + 10k) and `dcoord` as
    merge heights. Converting the leaf coordinate to k + 0.5 puts every segment in the
    same units as the heatmap's cell centres, which is what keeps the two aligned -- the
    alternative, letting scipy draw into an axis and matching limits afterwards, depends
    on orientation conventions that differ between scipy versions.
    """
    tree = hc.dendrogram(linkage, no_plot=True)
    for leaf_coords, heights in zip(tree["icoord"], tree["dcoord"]):
        yield [(c - 5.0) / 10.0 + 0.5 for c in leaf_coords], list(heights)


def _text_width_in(strings, fontsize):
    """Widest of `strings` in inches when rendered at `fontsize`.

    Type is set in points, so this width is fixed however the figure is scaled -- which
    is what lets the trait-name gutter be solved for rather than guessed at.
    """
    scratch = plt.figure(figsize=(1, 1))
    renderer = scratch.canvas.get_renderer()
    widest = 0.0
    for text in strings:
        artist = scratch.text(0, 0, text, fontsize=fontsize)
        widest = max(widest, artist.get_window_extent(renderer=renderer).width / scratch.dpi)
        artist.remove()
    plt.close(scratch)
    return widest


def load_attention_maps(order, tif_label_of):
    """Panel c's 17 attention maps, fetched from the data deposit.

    Returns [(display name, RGB array), ...] in `order`, or None when the deposit does not
    hold them, and the figure then builds panels a and b alone.
    """
    try:
        archive = fetch("figure_intermediates", files=[ATTENTION_ARCHIVE])[ATTENTION_ARCHIVE]
    except FetchError as exc:
        print("[panel c] attention maps not available ({}); drawing a and b only".format(exc))
        return None
    from PIL import Image

    with zipfile.ZipFile(str(archive)) as maps:
        return [(name, np.asarray(Image.open(io.BytesIO(
                    maps.read(ATTENTION_FILE.format(tif_label_of[name])))).convert("RGB")))
                for name in order]


def _to_dpi(image, width_in):
    """Cap a map at the configured dpi at its printed width.

    The deposited maps are below the cap and go in unchanged; this only keeps a larger
    map from putting more pixels in the PDF than print can use.
    """
    target = int(round(float(_CFG_C.get("target_dpi", 600)) * width_in))
    if target <= 0 or image.shape[1] <= target:
        return image
    from PIL import Image

    height = int(round(target * image.shape[0] / image.shape[1]))
    return np.asarray(Image.fromarray(image).resize((target, height), Image.LANCZOS))


def load_inputs(corr_csv, r2_csv):
    """The two deposited tables, in the shape the panels want.

    Also returns display name -> pipeline token, which panel c needs to find each trait's
    attention map by filename.
    """
    corr = pd.read_csv(corr_csv)
    labels = corr["trait_label"].tolist()
    token_of = dict(zip(corr["trait_label"], corr["trait"]))
    matrix = corr[[c for c in corr.columns if c.startswith("LV_")]]
    r2 = pd.read_csv(r2_csv)
    return matrix, labels, r2, token_of


def build_figure(corr_csv, r2_csv, out_dir, _canvas_w_in=None, _pass=0):
    """Build Figure 2. The private arguments are the width solver -- see the note at the
    savefig call: the figure is drawn on a slightly oversized canvas and then cropped to
    its ink, and `_canvas_w_in` is the canvas width that makes the cropped result exactly
    the journal's display-item width."""
    # Before anything measures text: the gutter is solved from the rendered width of the
    # trait names, and matplotlib's default DejaVu Sans is materially wider than the
    # document face, so measuring first and pinning the font second overestimates the
    # gutter by ~3 mm.
    STYLE.apply_font()

    matrix, trait_labels, r2, token_of = load_inputs(corr_csv, r2_csv)
    values = matrix.to_numpy(dtype=float)
    n_traits, n_lvs = values.shape

    # ---- clustering ------------------------------------------------------
    # Average linkage on the SIGNED dissimilarity 1 - r, not 1 - |r|: the signed one is
    # what the published figure was drawn from.
    dissimilarity = 1.0 - values
    row_linkage = hc.linkage(dissimilarity, method="average")
    col_linkage = hc.linkage(dissimilarity.T, method="average")
    row_order = hc.leaves_list(row_linkage)
    col_order = hc.leaves_list(col_linkage)

    ordered = values[np.ix_(row_order, col_order)]
    ordered_labels = [trait_labels[i] for i in row_order]
    print("panel a: {} mTIFs x {} LVs".format(n_traits, n_lvs))
    print("row order: {}".format(", ".join(ordered_labels)))

    # ---- geometry: the published layout, scaled ---------------------------
    # Every rectangle is taken from the positions the published figure was drawn with
    # (config: plot_styles.figures.fig2) and expressed in the coordinates of that
    # reference canvas, in inches. Only the last step -- one uniform scale factor onto
    # the journal's display-item width -- differs from the published figure, so no
    # proportion changes. Panel b's bar axes are the published 1.2 x 3.15 in and share the
    # heatmap's vertical extent exactly, which is how the two panels were aligned.
    canvas_w, canvas_h = [float(v) for v in FIG2["published_canvas_in"]]

    def frac_rect(key):
        """A published clustermap axes rectangle, in reference-canvas inches."""
        x0, y0, w, h = [float(v) for v in FIG2[key]]
        return [x0 * canvas_w, y0 * canvas_h, w * canvas_w, h * canvas_h]

    heat = frac_rect("heatmap_frac")
    rowd = frac_rect("row_dendrogram_frac")
    cold = frac_rect("col_dendrogram_frac")
    cbar_rect = frac_rect("colorbar_frac")

    bar_w_in, bar_h_in = [float(v) for v in FIG2["bar_panel_in"]]
    if abs(bar_h_in - heat[3]) > 1e-9:
        raise ValueError(
            "panel b is {} in tall but the heatmap is {} in: the published figure aligns "
            "the two exactly".format(bar_h_in, heat[3]))
    bottom_pad = _mm("bottom_pad_in", 0.15)
    right_pad = _mm("bar_right_pad_in", 0.15)
    edge_pad = _mm("edge_pad_in", 0.10)          # white margin, left and top
    heat_right = heat[0] + heat[2]

    # The trait-name gutter is measured, not inherited. Type is set in points, so the
    # names occupy a fixed PHYSICAL width however the figure is scaled -- which makes the
    # scale factor implicit: total = A + labels/scale and scale = target/total, so
    # total = A / (1 - labels/target). Solved rather than iterated.
    width_mm = STYLE.display_item_width_mm
    target_w_in = _canvas_w_in if _canvas_w_in else width_mm / MM
    tick_len_pt, tick_pad_pt = AXIS_TICK_LEN_PT, 1.5   # must match the tick_params below
    widest_label_in = _text_width_in(ordered_labels, TRAIT_FS)
    labels_in = (widest_label_in
                 + (tick_len_pt + tick_pad_pt) / 72.0
                 + _mm("label_to_bars_in", 0.015))
    fixed_in = edge_pad + heat_right + bar_w_in + right_pad
    total_w_in = fixed_in / (1.0 - labels_in / target_w_in)
    scale = target_w_in / total_w_in

    bar_x_in = heat_right + (total_w_in - fixed_in)

    # The colour bar lives in the trait-name gutter. Its published x anchor (0.77 of the
    # old canvas) was tied to a much wider 8 pt label band and now falls past the end of
    # the names, so it is sized against the names instead: a little narrower than the
    # longest one, and centred in the gutter.
    cbar_rect[2] = (widest_label_in * _mm("colorbar_width_frac_of_labels", 0.9)) / scale
    cbar_rect[3] = cbar_rect[2] / _mm("colorbar_aspect", 10.91)   # published proportions
    cbar_rect[1] += _mm("colorbar_lift_in", 0.07)                 # nudge off the bottom
    # Left-aligned with the trait names above it, i.e. starting where their glyphs start
    # (past the tick and its pad), rather than centred in the gutter.
    cbar_rect[0] = heat_right + ((tick_len_pt + tick_pad_pt) / 72.0) / scale
    # ---- panel c band ------------------------------------------------------
    # Laid out inside the same content width as a and b, so it cannot change the solved
    # width; only the height grows. Tiles are square-ish images, each captioned in
    # vector type, with the colour bar in the cell the 17th tile leaves free.
    # Tile order. "filename" reproduces the published arrangement: David's script listed
    # the directory and sorted it, so the panel runs in ASCII order of the trait tokens
    # (uppercase first: D_A_std, D_V_std, VD_orig_*, then bifurcations, eq_*, ...), which
    # is not the clustered order of panels a and b.
    _order_mode = str(_CFG_C.get("order", "filename"))
    if _order_mode == "clustered":
        c_order = ordered_labels
    elif _order_mode == "canonical":
        c_order = trait_labels
    else:
        c_order = sorted(trait_labels, key=lambda name: token_of[name])
    maps = load_attention_maps(c_order, token_of)

    # Panel b's last x tick label is CENTRED on its tick, so half of it hangs past the end
    # of the bar axes -- and with "1" as that label it was the only ink in the whole figure
    # reaching that far right. The page is cropped to its ink, so the crop landed on that
    # lone digit and the glyph sat hard against the paper edge, reading as cut off. The
    # fix is to give the right edge to panel c instead: the tile band runs to where the
    # centred label ends, which is half a label width past the axes. The digit then keeps
    # its own side bearing of white and the edge of the page is a tile, as the left edge is
    # the heatmap. Costs each of the 9 tiles ~0.06 mm of width. (4 Sep 2026.)
    tick_overhang_in = 0.5 * _text_width_in(["1"], TICK_FS) / scale
    content_left, content_right = rowd[0], bar_x_in + bar_w_in + tick_overhang_in
    c_ncols = int(_CFG_C.get("ncols", 9))
    gap_frac = float(_CFG_C.get("tile_gap_frac", 0.06))
    tile_w = (content_right - content_left) / (c_ncols + (c_ncols - 1) * gap_frac)
    tile_gap = tile_w * gap_frac
    if maps:
        aspect = maps[0][1].shape[0] / maps[0][1].shape[1]
        tile_h = tile_w * aspect
        c_nrows = int(np.ceil(len(maps) / float(c_ncols)))
        # the caption band is physical (type is in points), so convert through the scale
        label_fs = next((pt for pt in (7, 6, 5)
                         if _text_width_in(c_order, pt) <= tile_w * scale), 5)
        label_band = (label_fs * 1.6 / 72.0) / scale
        c_band = (float(_CFG_C.get("band_gap_frac", 0.35)) * tile_w
                  + c_nrows * (tile_h + label_band) + (c_nrows - 1) * tile_gap)
        print("panel c: {} maps, {} x {} grid, tiles {:.1f} mm, names at {} pt".format(
            len(maps), c_nrows, c_ncols, tile_w * scale * MM, label_fs))
    else:
        c_band = 0.0
        print("panel c: attention maps unavailable; drawing panels a and b only")

    # the legend hangs below the canvas; the letters sit in the top margin
    total_h_in = canvas_h + bottom_pad + edge_pad + c_band
    canvas_w_mm = target_w_in * MM      # drawing canvas; cropped to its ink when saved
    height_mm = total_h_in * MM * scale

    fig = plt.figure(figsize=(canvas_w_mm / MM, height_mm / MM))

    def axes_in(x, y, w, h):
        """Reference-canvas inches -> figure fractions, inside the edge margins."""
        return fig.add_axes([((x + edge_pad) * MM * scale) / canvas_w_mm,
                             ((y + bottom_pad + c_band) * MM * scale) / height_mm,
                             (w * MM * scale) / canvas_w_mm,
                             (h * MM * scale) / height_mm])

    ax_rowdend = axes_in(*rowd)
    ax_heat = axes_in(*heat)
    ax_coldend = axes_in(*cold)
    ax_cbar = axes_in(*cbar_rect)
    ax_bar = axes_in(bar_x_in, heat[1], bar_w_in, bar_h_in)
    # The key is anchored to a point rather than given its own axes: an empty axes claims
    # its whole rectangle in the figure's tight bounding box, which left ~2 mm of white
    # below the key once the page was cropped to the artwork.
    # The key's band is measured, not the published 0.945 in: that constant sized a boxed
    # key whose frame we have since removed, and reserving it left ~10 mm of dead space
    # between panel b and panel c. Three rows plus a title, at the key's own point size.
    legend_h_in = ((len(SERIES) + 1) * LEGEND_FS * 1.7 / 72.0) / scale
    # The key and panel a's colour bar sit on ONE line. The alignment is to the key's
    # MIDDLE ENTRY, not to its box: the box carries a title row on top, which puts its
    # centre half a row above the middle swatch. So the box is lifted by half a row.
    legend_row = legend_h_in / (len(SERIES) + 1)
    legend_centre = (bar_x_in + 0.5 * bar_w_in,
                     cbar_rect[1] + 0.5 * cbar_rect[3] + 0.5 * legend_row)

    heat_w = heat[2] * MM * scale               # for the panel-letter offset, in mm
    row_dend_w = rowd[2] * MM * scale

    # ---- panel a: heatmap ------------------------------------------------
    vmax = float(np.nanmax(np.abs(ordered)))
    mesh = ax_heat.pcolormesh(
        np.arange(n_lvs + 1), np.arange(n_traits + 1), ordered,
        cmap=str(FIG2["heatmap_cmap"]), vmin=-vmax, vmax=vmax,
        linewidth=0, edgecolors="none", rasterized=False, snap=True)
    ax_heat.set_xlim(0, n_lvs)
    ax_heat.set_ylim(n_traits, 0)          # first clustered trait at the top
    ax_heat.set_xticks([])
    ax_heat.set_yticks(np.arange(n_traits) + 0.5)
    ax_heat.set_yticklabels(ordered_labels, fontsize=TRAIT_FS)
    ax_heat.yaxis.tick_right()
    ax_heat.tick_params(axis="y", length=tick_len_pt, width=AXIS_LW_PT, pad=tick_pad_pt)
    # RETFound is set roman: Nature reserves italics for gene symbols, species and
    # statistical symbols, and a model name is a proper noun.
    ax_heat.set_xlabel("RETFound latent variables (LVs)", fontsize=AXIS_LABEL_FS,
                       labelpad=_mm("heatmap_title_pad_pt", 6.0))
    ax_heat.xaxis.set_label_position("top")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # Both dendrograms must MEET the heatmap: their zero-distance leaves sit exactly on
    # the shared edge. Setting the limits explicitly is what does it -- matplotlib's
    # default 5 % autoscale margin otherwise leaves a hairline of white between the
    # leaf tips and the heatmap.
    line_w = _mm("dendrogram_linewidth", 0.4)
    row_segments = list(_dendrogram_segments(row_linkage))
    for leaf_coords, heights in row_segments:
        ax_rowdend.plot(heights, leaf_coords, color="black", linewidth=line_w,
                        solid_capstyle="butt")
    # The root end needs headroom or it is clipped against the axes boundary -- verified by
    # integrating the rendered ink: the roots came out at 2.00 and 1.89 px where a 0.4 pt
    # stroke at 500 dpi is 2.78. Half a stroke is not enough, because that lands the line's
    # outer edge exactly on the boundary and the crop then shaves its outermost antialiased
    # row; a full stroke leaves the line whole, at the cost of ~0.07 mm of white outside it.
    def _stroke_margin(axis_extent_in, data_range):
        return (line_w / 72.0) / axis_extent_in * data_range

    ax_rowdend.set_ylim(n_traits, 0)
    row_max = max(max(h) for _, h in row_segments)
    ax_rowdend.set_xlim(row_max + _stroke_margin(rowd[2] * scale, row_max), 0)
    ax_rowdend.axis("off")

    col_segments = list(_dendrogram_segments(col_linkage))
    for leaf_coords, heights in col_segments:
        ax_coldend.plot(leaf_coords, heights, color="black", linewidth=line_w,
                        solid_capstyle="butt")
    ax_coldend.set_xlim(0, n_lvs)
    col_max = max(max(h) for _, h in col_segments)
    ax_coldend.set_ylim(col_max + _stroke_margin(cold[3] * scale, col_max), 0)
    ax_coldend.axis("off")

    cbar = fig.colorbar(mesh, cax=ax_cbar, orientation="horizontal")
    # matplotlib rasterises the colour bar's solids by default, which puts a small bitmap
    # into an otherwise fully vector PDF (checked: `pdfimages -list` listed one 102 x 10
    # image). The bar is 256 quads; drawing it as vector costs nothing.
    cbar.solids.set_rasterized(False)
    cb_ticks = [-0.5, 0.0, 0.5]
    cbar.set_ticks(cb_ticks)
    # Tick marks are drawn by draw_colorbar_ticks() (thinner and longer than matplotlib's);
    # the labels keep the pad the built-in marks would have had.
    cbar.ax.tick_params(labelsize=TICK_FS, length=0, pad=1.0 + CB_TICK_LEN_PT)
    # A mark under each LABEL, at its position along the ramp. This bar runs to +-max|r|,
    # which is not +-0.5, so marks at the ramp's two ends stood next to labels that meant
    # something else -- only the middle one lined up. Panel c's bar is the case where
    # end-flush marks are right, because its labels ARE the ends (min, max); it is the
    # exception, not the rule. (Fixed 4 Sep 2026.)
    draw_colorbar_ticks(cbar.ax, [(v + vmax) / (2.0 * vmax) for v in cb_ticks], "bottom",
                        CB_TICK_LEN_PT, CB_TICK_W_PT, outline_pt=CB_OUTLINE_PT)
    # The box's outline is the paper-wide knob plot_styles.colorbar_ticks.outline_pt, and
    # 0 (the borderless design, 5 Sep 2026) turns it off; the ramp's own edge is then the
    # drawn edge, which is what draw_colorbar_ticks() insets the end marks against.
    cbar.outline.set_visible(CB_OUTLINE_PT > 0)
    cbar.outline.set_linewidth(CB_OUTLINE_PT)
    cbar.outline.set_edgecolor("black")
    ax_cbar.set_title("Pearson correlation", fontsize=TICK_FS,
                      pad=_mm("colorbar_title_pad_pt", 4.0))

    # ---- panel b: prediction R2 -----------------------------------------
    r2 = r2.set_index(["trait_label", "model"])
    group_fraction = _mm("bar_group_fraction", 0.82)
    bar_h = group_fraction / len(SERIES)
    fold_columns = [c for c in r2.columns if c.startswith("fold_")]
    # The overlaid folds are a paper-wide mark shared with Figure 6, so their geometry and
    # colour treatment come from config and from src/plot_style rather than from here --
    # one spec, one beeswarm, one deepening rule, and the two figures cannot drift apart.
    fold = fold_marker_spec(_CFG)
    marker_lw = float(fold["edge_linewidth_pt"])
    # GEOMETRY is Figure 6's, to the point: same spec, same diameter rule, same beeswarm,
    # so the two figures carry one mark. Only the FILL is Figure 2's own -- plain black
    # rather than the popped bar colour, and opaque (see the comment at the config key).
    # Both knobs live in config with no default: a fallback here is how a colour rule
    # silently forks.
    marker_face = FIG2["fold_marker_facecolor"]
    marker_alpha = float(FIG2["fold_marker_alpha"])
    marker_edge = FIG2["fold_marker_edgecolor"]
    marker_edge_lw = float(FIG2["fold_marker_linewidth"])

    # Points per data unit, both axes: the panel's size in inches is known here, so the
    # beeswarm can be solved at draw time without waiting for a layout pass.
    points_per_row = (bar_h_in * scale * 72.0) / n_traits
    bar_thickness_pt = bar_h * points_per_row
    pt_per_x = bar_w_in * scale * 72.0            # ax_bar spans x = 0..1

    # One absolute size for both figures, capped at a half-bar so a two-row swarm can never
    # leave the bar: both pass `diameter_pt` (0.92), so the two figures carry one dot.
    marker_size = swarm_diameter(bar_thickness_pt, marker_lw,
                                 float(fold["diameter_pt"]))
    print("panel b: bar {:.2f} pt thick; fold dots {:.2f} pt "
          "= {:.0f}% of the bar".format(bar_thickness_pt, marker_size,
                                        100 * (marker_size + marker_lw) / bar_thickness_pt))

    n_with_folds = 0
    swarm_x, swarm_y, swarm_face = [], [], []
    for row_index, label in enumerate(ordered_labels):
        for series_index, (model, _, color) in enumerate(SERIES):
            record = r2.loc[(label, model)]
            centre = row_index + 0.5 + (series_index - 1) * bar_h
            ax_bar.barh(centre, float(record["r2"]), height=bar_h, color=color,
                        linewidth=0, zorder=2)
            folds = pd.to_numeric(record[fold_columns], errors="coerce").dropna()
            if len(folds):
                values = folds.to_numpy(dtype=float)
                # Upper half of the 95% CI over the folds, drawn under the points as a
                # PLAIN LINE in the bar's own colour -- no caps. It
                # continues the bar rather than annotating it, so at 0.39 pt on a ~3 pt bar
                # it reads as the bar's reach and leaves the dots the only black marks in
                # the panel. Figure 6 keeps its capped black whisker; the mark that must
                # match between the two figures is the fold dot, not the interval.
                if len(values) > 1:
                    sem = values.std(ddof=1) / np.sqrt(len(values))
                    half = t_dist.ppf(0.975, df=len(values) - 1) * sem
                    mean_x = float(record["r2"])
                    ax_bar.plot([mean_x, mean_x + half], [centre, centre], color=color,
                                linewidth=float(fold["whisker_linewidth_pt"]),
                                solid_capstyle="butt", zorder=3)
                # Beeswarm rather than a fixed ladder: points move aside only where they
                # would overprint, so a tight cluster stays a tight cluster.
                room = max(0.0, (bar_thickness_pt - marker_size - marker_lw) / 2.0)
                offsets = swarm_offsets(values * pt_per_x, marker_size, room_pt=room)
                swarm_x.append(values)
                swarm_y.append(centre + offsets / points_per_row)
                swarm_face.extend([marker_face] * len(values))
                n_with_folds += 1

    if swarm_x:
        # All 255 fold points as one collection rather than 51 Line2D artists.
        # `marker_lw` sized the dot above, so the diameter matches Figure 6's; here the
        # stroke itself is this figure's own and is not drawn.
        ax_bar.scatter(np.concatenate(swarm_x), np.concatenate(swarm_y),
                       s=marker_size ** 2, facecolors=swarm_face, edgecolors=marker_edge,
                       linewidths=marker_edge_lw, alpha=marker_alpha, zorder=4)

    ax_bar.set_ylim(n_traits, 0)
    # Tick marks on the y axis, one per trait row, unlabelled -- as published (seaborn
    # kept the ticks and only blanked the text, since the names are panel a's).
    ax_bar.set_yticks(np.arange(n_traits) + 0.5)
    ax_bar.set_yticklabels([])
    ax_bar.tick_params(axis="y", length=AXIS_TICK_LEN_PT, width=AXIS_LW_PT, pad=1.0)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xticks([0.0, 0.5, 1.0])
    # "0", "0.5", "1" -- the trailing zeros of 0.0 and 1.0 claim a precision the axis does
    # not have and cost width at 5-7 pt; 0.5 keeps its decimal because it needs one.
    ax_bar.set_xticklabels(["0", "0.5", "1"])
    ax_bar.tick_params(axis="x", labelsize=TICK_FS, length=AXIS_TICK_LEN_PT,
                       width=AXIS_LW_PT, pad=1.5)
    ax_bar.xaxis.tick_top()
    ax_bar.xaxis.set_label_position("top")
    ax_bar.set_xlabel("$R^2$", fontsize=AXIS_LABEL_FS,
                      labelpad=_mm("bar_title_pad_pt", 5.0))
    # No gridlines and no frame beyond the two tick-bearing axes. The author guidance asks
    # that "borders / shading / gridlines are removed from all figures (unless they convey
    # specific information that is defined in the legend)": with ticks at 0, 0.5 and 1.0
    # and no threshold to mark, the gridlines duplicated the ticks, and the right and
    # bottom spines carried nothing -- this panel's x axis is on top and its y ticks are
    # on the left.
    ax_bar.grid(False)
    for side in ("right", "bottom"):
        ax_bar.spines[side].set_visible(False)
    for side in ("top", "left"):
        ax_bar.spines[side].set_linewidth(AXIS_LW_PT)
    print("panel b: {} bars, {} of them with fold points".format(
        n_traits * len(SERIES), n_with_folds))

    # Three rows, as published. The fold markers are defined in the figure legend prose
    # ("outlined circles are the five individual fold values") rather than in the key.
    handles = [Patch(facecolor=color, label=label) for _, label, color in SERIES]
    # Unboxed: the frame is a border carrying no information. The key itself stays inside
    # the panel, which is what the author guidance prefers over defining colours in prose.
    # Handle size and spacing are matplotlib's defaults, which is what the published key
    # used -- wide swatches (handlelength 2.0 font units), not the narrow ones an earlier
    # version specified.
    anchor = ((legend_centre[0] + edge_pad) * MM * scale / canvas_w_mm,
              (legend_centre[1] + bottom_pad + c_band) * MM * scale / height_mm)
    key = fig.legend(handles=handles, loc="center", title="Model", frameon=False,
                     fontsize=LEGEND_FS, title_fontsize=LEGEND_FS,
                     bbox_to_anchor=anchor, bbox_transform=fig.transFigure)

    # Align the key's MIDDLE swatch with the colour bar's centre. The row pitch cannot be
    # predicted from the font size -- it depends on the title, labelspacing and handle
    # height -- so the key is drawn, measured, and shifted by the residual.
    target_y = ax_cbar.get_position().y0 + 0.5 * ax_cbar.get_position().height
    anchor_y = anchor[1]
    # Measure at output resolution, not the figure's default 100 dpi: there one pixel is
    # 0.25 mm, so a quarter-millimetre misalignment is invisible to the measurement and
    # the correction stalls with the swatch visibly off the bar.
    draft_dpi = fig.dpi
    fig.set_dpi(int(FIG2.get("dpi", 500)))
    for _ in range(4):     # one correction overshoots slightly; this converges
        fig.canvas.draw()
        middle = key.legend_handles[len(SERIES) // 2].get_window_extent(
            fig.canvas.get_renderer())
        middle_y = 0.5 * (middle.y0 + middle.y1) / (fig.get_size_inches()[1] * fig.dpi)
        residual = target_y - middle_y
        if abs(residual) * height_mm < 0.02:      # 0.02 mm is a tenth of a pixel here
            break
        anchor_y += residual
        key.set_bbox_to_anchor((anchor[0], anchor_y), transform=fig.transFigure)
    fig.set_dpi(draft_dpi)

    # ---- panel c: the attention maps ------------------------------------
    ax_first_tile = None
    if maps:
        # panel c starts below whichever of the a/b block reaches lowest -- the column
        # dendrogram (which ends at y = 0) or the key -- not below a reserved band
        ab_bottom = min(0.0, heat[1] - _mm("legend_top_gap_in", 0.055) - legend_h_in)
        band_top = ab_bottom - float(_CFG_C.get("band_gap_frac", 0.35)) * tile_w
        for index, (name, image) in enumerate(maps):
            row, col = divmod(index, c_ncols)
            x = content_left + col * (tile_w + tile_gap)
            y = band_top - (row + 1) * (tile_h + label_band) - row * tile_gap
            # the tile sits at the bottom of its cell; the trait name is a TITLE above it
            ax_tile = axes_in(x, y, tile_w, tile_h)
            # The map goes in as a bitmap while its caption below stays vector type.
            # interpolation="none" is load-bearing: any other value makes matplotlib
            # RESAMPLE the array to the figure's render resolution before embedding, which
            # silently downsampled these maps to 100 dpi (caught by the display-item
            # check). "none" embeds exactly the pixels handed to it -- so the resampling
            # is done here instead, deliberately, to a stated dpi.
            ax_tile.imshow(_to_dpi(image, tile_w * scale), interpolation="none")
            ax_tile.set_xticks([])
            ax_tile.set_yticks([])
            for spine in ax_tile.spines.values():
                spine.set_visible(False)
            ax_tile.set_title(name, fontsize=label_fs,
                              pad=float(_CFG_C.get("label_pad_frac", 0.05)) * 72
                              * tile_w * scale)
            ax_first_tile = ax_first_tile or ax_tile

        # The free cell after the 17th tile carries the scale. Same size and shape as
        # panel a's colour bar -- one figure, one kind of colour scale -- so it reuses that
        # rectangle's width and height rather than defining its own.
        row, col = divmod(len(maps), c_ncols)
        cb_w, cb_h = cbar_rect[2], cbar_rect[3]
        cb_x = content_left + col * (tile_w + tile_gap) + 0.5 * (tile_w - cb_w)
        cb_y = band_top - (row + 1) * (tile_h + label_band) - row * tile_gap \
            + 0.5 * tile_h
        ax_attn = axes_in(cb_x, cb_y, cb_w, cb_h)
        # drawn as quads, not as an image: a one-pixel-tall imshow would embed a raster
        # with a nonsensical resolution, and this is line art
        steps = 256
        ax_attn.pcolormesh(np.arange(steps + 1), np.arange(2),
                           np.linspace(0, 1, steps).reshape(1, -1),
                           cmap=str(FIG2["attention_cmap"]), linewidth=0,
                           edgecolors="none", rasterized=False)
        ax_attn.set_yticks([])
        ax_attn.set_xticks([0, steps])
        ax_attn.set_xticklabels(["min", "max"], fontsize=TICK_FS)
        # bound to the ends of the ramp rather than centred on them, so neither label
        # overhangs the bar
        end_labels = ax_attn.get_xticklabels()
        end_labels[0].set_horizontalalignment("left")
        end_labels[-1].set_horizontalalignment("right")
        ax_attn.tick_params(axis="x", length=0, pad=1.0 + CB_TICK_LEN_PT)
        # flush with the bar's two ends, not centred on them (see draw_colorbar_ticks)
        draw_colorbar_ticks(ax_attn, [0.0, 1.0], "bottom", CB_TICK_LEN_PT, CB_TICK_W_PT,
                            outline_pt=CB_OUTLINE_PT)
        for spine in ax_attn.spines.values():     # same outline knob as panel a's bar
            spine.set_visible(CB_OUTLINE_PT > 0)
            spine.set_linewidth(CB_OUTLINE_PT)
            spine.set_edgecolor("black")
        ax_attn.set_title("Relative attention", fontsize=TICK_FS,
                          pad=_mm("colorbar_title_pad_pt", 4.0))

    # ---- panel letters, after the layout is final ------------------------
    # Positions measured off the published rendering rather than chosen: there, the glyph
    # tops sit 0.130 heatmap heights above the heatmap, 'a' is flush with the left edge of
    # the row dendrogram (-0.248 heatmap widths from the heatmap) and 'b' with the start
    # of the bars (+1.224). The published letters are inset from the canvas corner because
    # bbox_inches="tight" added 0.1 in of padding around everything; edge_pad reproduces
    # that margin, so the letters land in it the same way.
    fig.canvas.draw()
    heat_box = ax_heat.get_position()
    letter_y = heat_box.y1 + _mm("panel_letter_top_frac", 0.130) * heat_box.height
    letters = [("a", ax_rowdend.get_position().x0, letter_y),
               ("b", ax_bar.get_position().x0, letter_y)]
    if ax_first_tile is not None:
        # clear of the tile's title, then up by the letter's own height so it reads as a
        # panel marker rather than as part of the first caption
        tile_box = ax_first_tile.get_position()
        title_band_frac = (label_fs * 1.6 / 72.0) / (height_mm / MM)
        letter_frac = (STYLE.panel_label_fontsize / 72.0) / (height_mm / MM)
        letters.append(("c", ax_rowdend.get_position().x0,
                        tile_box.y1 + title_band_frac
                        + _mm("panel_letter_top_frac", 0.130) * tile_box.height
                        + letter_frac))
    for letter, x, y in letters:
        fig.text(x, y, letter, ha="left", va="top",
                 fontsize=STYLE.panel_label_fontsize,
                 fontweight=STYLE.panel_label_fontweight)

    # ---- crop to the ink, then write --------------------------------------
    # The page must BE the artwork: exactly the display-item width with no white margin
    # on any side. So the figure is drawn on a slightly oversized canvas and saved to its
    # tight bounding box with zero padding. The canvas width that makes the cropped result
    # land on the target is solved rather than guessed: ink = a * canvas + b, where b is
    # the text overhang (fixed, because type is set in points), so two passes determine it
    # exactly and a third confirms.
    ink = fig.get_tightbbox(fig.canvas.get_renderer())
    # matplotlib's tight bbox is the text LAYOUT box: it carries each glyph's side
    # bearings, which ghostscript -- and the journal -- do not see as ink. With "1" as
    # panel b's last x tick label (a narrow glyph in a wide advance) that left 0.44 mm of
    # white at the right edge, enough to fail the zero-margin rule; with "1.0" the round
    # final glyph happened to hide it. So probe the saved PDF for the real ink and trim to
    # it, and let the pass loop below rescale the canvas until the TRIMMED artwork is the
    # display-item width. (Added 4 Sep 2026 with the 0/0.5/1 tick labels.)
    os.makedirs(out_dir, exist_ok=True)
    probe_path = os.path.join(out_dir, OUT_NAME + "__probe.pdf")
    fig.savefig(probe_path, bbox_inches=ink, pad_inches=0)
    left_mm, bottom_mm, right_mm, top_mm = ink_margins_mm(probe_path)
    os.remove(probe_path)
    ink = Bbox.from_extents(ink.x0 + left_mm / MM, ink.y0 + bottom_mm / MM,
                            ink.x1 - right_mm / MM, ink.y1 - top_mm / MM)

    target_in = width_mm / MM
    if abs(ink.width - target_in) > 1e-4 and _pass < 4:
        plt.close(fig)
        return build_figure(corr_csv, r2_csv, out_dir,
                            _canvas_w_in=target_w_in * target_in / ink.width,
                            _pass=_pass + 1)

    final_h_mm = ink.height * MM
    if final_h_mm > STYLE.display_item_max_height_mm:
        raise ValueError("figure is {:.1f} mm tall, over the {} mm display-item cap".format(
            final_h_mm, STYLE.display_item_max_height_mm))

    pdf_path = os.path.join(out_dir, OUT_NAME + ".pdf")
    png_path = os.path.join(out_dir, OUT_NAME + ".png")
    fig.savefig(pdf_path, bbox_inches=ink, pad_inches=0)     # vector, editable text
    fig.savefig(png_path, bbox_inches=ink, pad_inches=0,
                dpi=int(FIG2.get("dpi", 500)))               # same size, for drafts
    plt.close(fig)
    print("wrote {}".format(pdf_path))
    print("wrote {} ({:.2f} mm x {:.2f} mm, cropped to the artwork after {} pass(es))".format(
        png_path, ink.width * MM, final_h_mm, _pass + 1))
    return pdf_path, png_path


def main():
    print("Fetching Figure 2 inputs from the data deposit ...")
    inputs = fetch("figure_intermediates", files=[CORR_FILE, R2_FILE])
    build_figure(str(inputs[CORR_FILE]), str(inputs[R2_FILE]), OUT_DIR)


if __name__ == "__main__":
    main()
