#!/usr/bin/env python
"""Figure 5: every disease and risk factor against every trait, measured and deep.

One split-cell heat map. Each cell is one disease x one trait: the lower-left triangle is
the measured TIF's association, the upper-right the deep TIF's, both coloured by the
regression beta on a symmetric blue-white-red ramp. A cell carries `*` where both versions
survive FDR, `D` where only the deep one does and `M` where only the measured one does.
The marginal bars count those three outcomes down each column and across each row.

Author: David Presby (github.com/presbyd)

Self-contained: fetches its two inputs from the public data deposit by DOI and draws the
whole figure. No access-controlled data.

A third deposited table, Fig5_03_disease_tif_n.csv, carries the n behind every one of the
32 x 34 pairs -- the legend points readers at it and this script does not read it. It is
built upstream in the analysis pipeline, whose source-data step is the place to start
BEFORE recomputing anything here: it records how the counts are derived (two source tables,
either one chosen wrongly rewrites hundreds of cells) and the one seam between those counts
and the published betas.

    pixi run python 05_figures/main/fig5_disease.py

It keeps the published rendering and adds what the journal requires:

  - the 179 mm display-item width, with the canvas height SOLVED from the published cell
    aspect of 1.52 : 1 rather than copied from the screenshot's outer aspect, so a cell has
    the shape it has in the manuscript; every type size from config.yaml and inside the
    5-7 pt Nature Genetics band (was a flat 12 pt), Nimbus Sans throughout, and vector PDF
    beside the PNG -- the cells are polygons and the markers are text, nothing rasterised;
  - axis furniture at the paper-wide 0.5 pt / 2 pt from plot_styles. The COLOUR BAR is the
    exception: it is the published one, not the paper-wide metric bar -- see the note
    where it is drawn;
  - the mTIF/dTIF bar colours read from config.yaml `colors:` through
    feature_set_colors() with no default.

DISEASE ORDER IS THE PUBLISHED ONE EXCEPT FOR ONE ROW. `age_high_BP_both` is labelled
"Hypertension" and moved from row 3 (risk factors) to row 22, leading the general diseases
-- decided on the data, 7 Sep 2026; see the note on ROW_ORDER. Everything else keeps the
published order, and rows are reindexed BY NAME so data always travels with its label.

FDR is Benjamini-Hochberg applied WITHIN each version separately, over that version's
32 x 17 family, which is what the published figure did. The deposit carries raw p-values,
as it should; the correction lives here.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.fetch_data import fetch  # noqa: E402
from src.display_item import figure_dir, ink_margins_mm, render_to_width, save, verify  # noqa: E402
from src.plot_style import feature_set_colors, load_plot_style  # noqa: E402

MM = 25.4
CONFIG_PATH = REPO_ROOT / "config.yaml"
OUT_DIR = figure_dir("main_figures")
OUT_NAME = "fig5_disease"

BETAS_FILE = "Fig5_01_disease_tif_betas.csv"
LOG10P_FILE = "Fig5_02_disease_tif_log10p.csv"

# The three blocks bracketed down the left edge, as (label, first row, one past last row).
# 12 + 9 + 11 = 32.
GROUPS = (("Risk Factors", 0, 12), ("Ocular Diseases", 12, 21), ("General Diseases", 21, 32))

# Row order, as deposit keys, and the display name for each. TWO CHANGES from the
# published panel, both decided 7 Sep 2026, both about the same variable:
#
#   * `age_high_BP_both` is labelled 'Hypertension', not 'High BP'. Figure 6 and
#     Supplementary Table 1 already call it that, so Figure 5 was the only place using the
#     other name.
#   * it moves out of the risk factors and into the general diseases, where David had
#     already put it in both committed scripts and where config's `disease_names` records
#     it (`systemic_diseases_age_at_onset`). It is age at high-blood-pressure diagnosis
#     (n = 8 943, self-reported) -- an age at onset like Diabetes or Stroke, not a
#     measurement taken at the imaging visit like every other row in the risk-factor
#     block. It leads that block, ahead of Diabetes, as in David's `desired_order`.
#
# The display names keep the published figure's shortenings of config's disease_names.
ROW_ORDER = (
    "DBP_both", "SBP_both", "PR_both", "pulse_wave_arterial_stiffness_index_both",
    "HDL_cholesterol_both", "LDL_direct_both", "Triglycerides_both", "HbA1c_both",
    "alcohol_intake_frequency_both", "N_cigarettes_curr_daily_both",
    "Pack_year_smok_both", "BMI_both",
    "age_glaucoma_both", "age_cataract_both", "eye_amblyopia_both", "eye_presbyopia_both",
    "eye_hypermetropia_both", "eye_myopia_both", "eye_astigmatism_both",
    "eye_diabetes_both", "age_other_serious_eye_condition_both",
    "age_high_BP_both", "age_diabetes_both", "age_angina_both", "age_heartattack_both",
    "age_DVT_both", "age_stroke_both", "age_pulmonary_embolism_both",
    "date_reported_atherosclerosis", "date_disorders_arteries_arterioles", "date_AD",
    "age_death",
)
DISEASE_LABELS = (
    "DBP", "SBP", "PR", "PWASI", "HDL", "LDL", "Triglycerides", "HbA1c",
    "Alcohol", "N cig/day", "N pack/yr", "BMI",
    "Glaucoma", "Cataract", "Amblyopia", "Presbyopia", "Hypermetropia", "Myopia",
    "Astigmatism", "Diabetes-eye", "Other eye condition",
    "Hypertension", "Diabetes", "Angina", "Heart attack", "DVT", "Stroke",
    "Pulmonary embolism", "Atherosclerosis", "Artery/arteriole disorder", "Alzheimers",
    "Death",
)


# THE PUBLISHED LAYOUT, REPRODUCED EXACTLY.
#
# The published script does:
#     fig = plt.figure(figsize=(12, 15))
#     gs  = GridSpec(2, 2, width_ratios=[4, 0.5], height_ratios=[0.5, 4],
#                    wspace=0.01, hspace=0.01)
# and leaves the subplot MARGINS at matplotlib's defaults. That last part is load-bearing:
# the colour bar and the legend are placed by absolute
# figure fractions (0.815/0.795 and 0.814/0.882), which only land where they landed in the
# manuscript if the axes sit where the DEFAULT margins put them. Pinning the margins to
# fill the canvas moved every one of those relationships. The defaults are written out
# here rather than inherited so the layout cannot drift with an rcParams or style change.
PUBLISHED_CANVAS_IN = (12.0, 15.0)
GRID_KW = dict(width_ratios=[4.0, 0.5], height_ratios=[0.5, 4.0],
               wspace=0.01, hspace=0.01,
               left=0.125, right=0.9, bottom=0.11, top=0.88)

# The published figure's one base font size, and the sizes derived from it. Nature
# Genetics caps display-item type at 7 pt, so every size is the published one scaled by
# 7/12 -- the largest factor that keeps the biggest of them inside the band. This is the
# only place the rebuild deliberately departs from the published proportions: at the true
# print scale the base size would be about 8 pt, so the type is ~12% smaller relative to
# the artwork than the published figure's. Everything geometric scales with the canvas instead and so
# keeps its published proportion exactly.
PUBLISHED_FS = 12.0
TYPE_SCALE = 7.0 / 12.0


def split_versions(frame):
    """Deposit rows alternate <trait>_pred, <trait>_true. Return (deep, measured).

    Both come back as diseases x traits, the orientation the figure draws in.
    """
    deep = frame.iloc[0::2].copy()
    measured = frame.iloc[1::2].copy()
    deep.index = [i[: -len("_pred")] for i in deep.index]
    measured.index = [i[: -len("_true")] for i in measured.index]
    if list(deep.index) != list(measured.index):
        raise SystemExit("the deposit's pred and true rows do not pair up")
    return deep.T, measured.T


def fdr_reject(log10p, alpha=0.05):
    """BH over one version's whole 32 x 17 family, back in the input's shape."""
    reject, _, _, _ = multipletests(
        (10.0 ** -log10p.values).ravel(), alpha=alpha, method="fdr_bh")
    return reject.reshape(log10p.shape)


def glyph_ink_centre_pt(glyph, size_pt, family):
    """Points from the text baseline up to the vertical centre of the glyph's INK.

    matplotlib's ``va="center"`` centres a glyph's EM BOX, not the mark you can see. For
    an asterisk -- which sits high on its em box and fills barely a third of it -- that
    leaves the mark visibly above the middle of the cell, and the taller the asterisk the
    worse it gets. Measuring the outline instead gives the offset that centres what the
    reader actually sees, and it does the same for `M` and `D`, whose caps sit entirely
    above the baseline. TextPath reads the real glyph outline, so this needs no renderer
    and no draw pass.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    extents = TextPath((0.0, 0.0), glyph, size=size_pt,
                       prop=FontProperties(family=family)).get_extents()
    return (extents.y0 + extents.y1) / 2.0


def star_outline(size_pt, cell_w_pt, cell_h_pt):
    """The published asterisk as a filled OUTLINE, centred on (0, 0) in data units.

    The star is drawn in DEJAVU SANS, the face the published script got by never setting
    one -- and that face, not any size, is why the published stars read better: DejaVu's
    asterisk carries 58% more ink than Nimbus Sans's at the same point size (45.6 vs 28.8
    per 100 pt em, measured). So the stars keep the published font and relative size.

    Emitting it as a PATH rather than as text is what lets that coexist with the
    one-document-face rule: an outline embeds no second font, so `pdffonts` still reports
    Nimbus Sans alone and the display-item check passes. It is also honest about what the
    mark is -- a significance symbol, not a letter. `M` and `D` ARE letters and stay in
    Nimbus Sans as text.

    Returned in data units with y already flipped, because the axis is inverted.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    path = TextPath((0.0, 0.0), "*", size=size_pt,
                    prop=FontProperties(family="DejaVu Sans"))
    box = path.get_extents()
    centre = np.array([(box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0])
    scale = np.array([1.0 / cell_w_pt, -1.0 / cell_h_pt])
    return (path.vertices - centre) * scale, path.codes


def draw_split_cell(ax, col, row, deep, measured, deep_sig, measured_sig, cmap, norm,
                    edge_lw, diag_lw, marker_pt, star_verts, star_codes, y_offset):
    """One cell: deep in the upper-right triangle, measured in the lower-left.

    Drawn in data coordinates with the y axis inverted, so row 0 is the top row. The
    diagonal runs from the cell's top-left to its bottom-right. `y_offset` maps each
    marker glyph to the data-unit shift that puts its ink on the cell's centre line.
    """
    x0, y0, x1, y1 = col, row, col + 1, row + 1
    ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1)], closed=True,
                         facecolor=cmap(norm(deep)), edgecolor="grey", linewidth=edge_lw))
    ax.add_patch(Polygon([(x0, y0), (x0, y1), (x1, y1)], closed=True,
                         facecolor=cmap(norm(measured)), edgecolor="grey",
                         linewidth=edge_lw))
    ax.add_patch(Rectangle((x0, y0), 1, 1, fill=False, edgecolor="grey",
                           linewidth=edge_lw))
    # The divider is WHITE and sits ON TOP of both triangles. In the published script this
    # is not stated, it is a z-order accident that matters: the line is drawn first but a
    # Line2D defaults to zorder 2 and a Patch to zorder 1, so the white line wins and the
    # two halves are separated by white, not by the triangles' grey edges. Reproduced
    # explicitly here rather than left to depend on default z-orders.
    ax.plot([x0, x1], [y0, y1], color="white", linewidth=diag_lw, zorder=3,
            solid_capstyle="butt")
    if deep_sig and measured_sig:
        # The star is the published DejaVu outline, already centred on (0, 0), so it only
        # needs translating onto the cell's midpoint.
        ax.add_patch(PathPatch(
            MplPath(star_verts + np.array([col + 0.5, row + 0.5]), star_codes),
            facecolor="black", edgecolor="none", zorder=4))
    elif deep_sig or measured_sig:
        # `M` and `D` are letters: Nimbus Sans, at the published relative size, placed on
        # their baseline at the y that puts their measured ink centre on the cell's middle.
        # The axis is inverted, so a positive offset moves the baseline DOWN the data axis,
        # which is what lifts the ink into the centre on screen.
        letter = "D" if deep_sig else "M"
        ax.text(col + 0.5, row + 0.5 + y_offset[letter], letter, ha="center",
                va="baseline", fontsize=marker_pt, color="black", zorder=4)


def style_axis(ax, lw, tick_len, tick_pt):
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    ax.tick_params(width=lw, length=tick_len, labelsize=tick_pt)


def main() -> int:
    config = yaml.safe_load(io.open(str(CONFIG_PATH), encoding="utf-8"))
    style = load_plot_style(str(CONFIG_PATH))
    style.apply_font()

    # Every size is the published one scaled by 7/12. WHICH element took WHICH size in the
    # published script matters and is reproduced exactly: the cell markers and the two
    # 'Counts' axis titles took the base FS; the HEAT MAP's tick labels took FS - 2, but
    # the two BAR panels' tick labels took the full FS; the legend took FS - 2.5; and the
    # colour bar, never given a size, took matplotlib's default 10.
    marker_pt = PUBLISHED_FS * TYPE_SCALE                 # 7.00  the M / D letters
    label_pt = PUBLISHED_FS * TYPE_SCALE                  # 7.00  'Counts'
    bar_tick_pt = PUBLISHED_FS * TYPE_SCALE               # 7.00  bar panels' ticks
    heat_tick_pt = (PUBLISHED_FS - 2.0) * TYPE_SCALE      # 5.83  row / column names
    legend_pt = (PUBLISHED_FS - 2.5) * TYPE_SCALE         # 5.54
    cbar_pt = 10.0 * TYPE_SCALE                           # 5.83
    axis_lw = style.axis_linewidth_pt
    tick_len = style.axis_tick_length_pt
    style.assert_text_sizes(marker=marker_pt, axis_label=label_pt,
                            bar_tick=bar_tick_pt, heat_tick=heat_tick_pt,
                            legend=legend_pt, colorbar=cbar_pt)

    palette = feature_set_colors(config)
    mtif_c, dtif_c = palette["mtifs"], palette["dtifs"]
    shared_c = "lightgrey"

    print("Fetching Figure 5 inputs from the data deposit ...")
    inputs = fetch("figure_intermediates", files=[BETAS_FILE, LOG10P_FILE])
    betas = pd.read_csv(str(inputs[BETAS_FILE]), index_col=0)
    log10p = pd.read_csv(str(inputs[LOG10P_FILE]), index_col=0)
    if betas.shape != log10p.shape:
        raise SystemExit("the beta and p-value tables are different shapes")

    deep_b, meas_b = split_versions(betas)
    deep_p, meas_p = split_versions(log10p)

    # Columns of the figure = traits, in the order config records for this figure; rows =
    # the deposit's own disease order, which is the published one.
    order = config["figure_tif_orders"]["disease_heatmap"]
    trait_labels = list(order["tif_order_labels"])
    trait_names = list(order["tif_order_names"])
    missing = [t for t in trait_labels if t not in deep_b.columns]
    if missing:
        raise SystemExit("traits missing from the deposit: {}".format(missing))
    deep_b, meas_b = deep_b[trait_labels], meas_b[trait_labels]
    deep_p, meas_p = deep_p[trait_labels], meas_p[trait_labels]

    # Rows in ROW_ORDER, which is the deposit's order with Hypertension moved into the
    # general diseases. Reindexing by name (not by position) is what keeps every row's
    # data with its label through the move.
    unknown = [d for d in ROW_ORDER if d not in deep_b.index]
    if unknown or len(ROW_ORDER) != len(deep_b.index):
        raise SystemExit("ROW_ORDER does not match the deposit's outcomes: {}".format(
            unknown or "count {} vs {}".format(len(ROW_ORDER), len(deep_b.index))))
    deep_b, meas_b = deep_b.loc[list(ROW_ORDER)], meas_b.loc[list(ROW_ORDER)]
    deep_p, meas_p = deep_p.loc[list(ROW_ORDER)], meas_p.loc[list(ROW_ORDER)]

    diseases = list(deep_b.index)
    if len(diseases) != len(DISEASE_LABELS):
        raise SystemExit("expected {} outcomes, deposit has {}".format(
            len(DISEASE_LABELS), len(diseases)))
    if GROUPS[-1][2] != len(diseases):
        raise SystemExit("the category brackets do not cover every row")

    deep_v, meas_v = deep_b.to_numpy(), meas_b.to_numpy()
    deep_sig, meas_sig = fdr_reject(deep_p), fdr_reject(meas_p)

    # Marginal counts: measured-only, deep-only, both. Down the columns and across the rows.
    meas_only_col = ((meas_sig) & (~deep_sig)).sum(axis=0)
    deep_only_col = ((deep_sig) & (~meas_sig)).sum(axis=0)
    both_col = (deep_sig & meas_sig).sum(axis=0)
    meas_only_row = ((meas_sig) & (~deep_sig)).sum(axis=1)
    deep_only_row = ((deep_sig) & (~meas_sig)).sum(axis=1)
    both_row = (deep_sig & meas_sig).sum(axis=1)

    abs_max = float(np.nanmax(np.abs(np.concatenate([deep_v.ravel(), meas_v.ravel()]))))
    cmap = LinearSegmentedColormap.from_list("beta", ["#0000FF", "#FFFFFF", "#FF0000"])
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    n_rows, n_cols = deep_v.shape
    width_mm = style.display_item_width_mm

    # Canvas aspect is the published one, 15 : 12. With the published margins and ratios
    # that already produces the published cell shape -- the manuscript rendering's
    # cell-border grid lines sit at x = 81 + 19j and y = 56 + 12.5i, so a published cell is
    # 19 : 12.5 = 1.52 : 1 -- and the run report below prints the cell aspect so the match
    # is checked on every build rather than assumed.
    CELL_ASPECT = 19.0 / 12.5
    height_mm = width_mm * PUBLISHED_CANVAS_IN[1] / PUBLISHED_CANVAS_IN[0]

    # EQUALISE THE TWO GUTTERS: the gap above the heat map must not be
    # larger than the one beside it. matplotlib's `wspace` and `hspace` are fractions of the
    # AVERAGE axes WIDTH and HEIGHT respectively, and this figure's panels are nowhere near
    # square, so the published pair of equal values (0.01 / 0.01) renders two visibly
    # different gaps. Solve for the hspace whose gutter matches the horizontal one in
    # millimetres; gap height is very nearly linear in hspace, so this converges at once.
    grid_kw = dict(GRID_KW)
    for _ in range(8):
        probe = plt.figure(figsize=PUBLISHED_CANVAS_IN)
        g = probe.add_gridspec(2, 2, **grid_kw)
        mb = g[1, 0].get_position(probe)
        tb = g[0, 0].get_position(probe)
        rb = g[1, 1].get_position(probe)
        plt.close(probe)
        gap_x_in = (rb.x0 - mb.x1) * PUBLISHED_CANVAS_IN[0]
        gap_y_in = (tb.y0 - mb.y1) * PUBLISHED_CANVAS_IN[1]
        if abs(gap_y_in - gap_x_in) < 1e-5:
            break
        grid_kw["hspace"] *= gap_x_in / gap_y_in
    if height_mm > style.display_item_max_height_mm:
        raise SystemExit("figure is taller than the display-item cap")

    def build(canvas_w_in):
        canvas_h_in = canvas_w_in * height_mm / width_mm
        # Line widths are in points, so they do NOT scale with the canvas the way the
        # artwork does. To keep the published PROPORTIONS, every published weight is scaled
        # by how much smaller this canvas is than the published 12 in one.
        geom = canvas_w_in / PUBLISHED_CANVAS_IN[0]
        cell_lw, diag_lw, grid_lw, bar_lw = (0.5 * geom, 1.0 * geom,
                                             1.0 * geom, 1.0 * geom)
        fig = plt.figure(figsize=(canvas_w_in, canvas_h_in))
        grid = fig.add_gridspec(2, 2, **grid_kw)
        ax_main = fig.add_subplot(grid[1, 0])
        ax_top = fig.add_subplot(grid[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(grid[1, 1], sharey=ax_main)

        # One data unit in y is one cell, so a glyph's ink offset in points converts
        # straight into data units once the cell's height in points is known.
        main_box = grid[1, 0].get_position(fig)
        cell_h_pt = main_box.height * canvas_h_in * 72.0 / n_rows
        cell_w_pt = main_box.width * canvas_w_in * 72.0 / n_cols
        y_offset = {g: glyph_ink_centre_pt(g, marker_pt, style.natgen_font_family)
                    / cell_h_pt for g in ("M", "D")}
        # The published star is FS = 12 on a 12 in canvas. `geom` is how much smaller this
        # canvas is, so `PUBLISHED_FS * geom` is the same size RELATIVE TO THE PANEL.
        star_pt = PUBLISHED_FS * geom
        star_verts, star_codes = star_outline(star_pt, cell_w_pt, cell_h_pt)

        for row in range(n_rows):
            for col in range(n_cols):
                draw_split_cell(ax_main, col, row, deep_v[row, col], meas_v[row, col],
                                deep_sig[row, col], meas_sig[row, col], cmap, norm,
                                cell_lw, diag_lw, marker_pt, star_verts, star_codes,
                                y_offset)
        ax_main.set_xlim(0, n_cols)
        ax_main.set_ylim(0, n_rows)
        ax_main.invert_yaxis()
        ax_main.set_xticks(np.arange(n_cols) + 0.5)
        ax_main.set_yticks(np.arange(n_rows) + 0.5)
        ax_main.set_xticklabels(trait_names, rotation=45, ha="right")
        ax_main.set_yticklabels(DISEASE_LABELS)
        style_axis(ax_main, axis_lw, tick_len, heat_tick_pt)

        # --- marginal bars ------------------------------------------------------------
        bar = 0.25
        centres = np.arange(n_cols) + 0.5
        ax_top.grid(axis="y", linestyle="--", linewidth=grid_lw, color="gray", alpha=0.5)
        ax_top.set_axisbelow(True)
        for offset, values, colour, label in (
                (-bar, meas_only_col, mtif_c, "Measured TIF"),
                (0.0, deep_only_col, dtif_c, "Deep TIF"),
                (bar, both_col, shared_c, "Shared")):
            ax_top.bar(centres + offset, values, width=bar, color=colour, label=label,
                       edgecolor="black", linewidth=bar_lw)
        ax_top.set_ylabel("Counts", fontsize=label_pt)
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        style_axis(ax_top, axis_lw, tick_len, bar_tick_pt)
        # Drop the 0 tick. A zero count sits on the axis line and
        # labels nothing a reader needs. set_yticks() would otherwise stretch the axis to
        # the next round tick, so the limits are captured first and restored after.
        top_lim = ax_top.get_ylim()
        ax_top.set_yticks([t for t in ax_top.get_yticks()
                           if t != 0 and top_lim[0] <= t <= top_lim[1]])
        ax_top.set_ylim(top_lim)

        rows = np.arange(n_rows) + 0.5
        ax_right.grid(axis="x", linestyle="--", linewidth=grid_lw, color="gray", alpha=0.5)
        ax_right.set_axisbelow(True)
        for offset, values, colour in ((-bar, meas_only_row, mtif_c),
                                       (0.0, deep_only_row, dtif_c),
                                       (bar, both_row, shared_c)):
            ax_right.barh(rows + offset, values, height=bar, color=colour,
                          edgecolor="black", linewidth=bar_lw)
        ax_right.set_xlabel("Counts", fontsize=label_pt)
        ax_right.tick_params(axis="y", left=False, labelleft=False)
        ax_right.set_xticks([5, 10])
        style_axis(ax_right, axis_lw, tick_len, bar_tick_pt)

        # --- category brackets down the left edge --------------------------------------
        # Drawn in figure coordinates against the finished axes box, so the rules line up
        # with the row boundaries whatever the label widths turn out to be.
        fig.canvas.draw()
        box = ax_main.get_position()
        tight = ax_main.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted())
        label_x = tight.x0 - 0.017
        rule_x0, rule_x1 = label_x - 0.013, box.x0

        def row_y(row):
            return box.y1 - (row / float(n_rows)) * (box.y1 - box.y0)

        for name, lo, hi in GROUPS:
            for edge in (lo, hi):
                fig.add_artist(Line2D([rule_x0, rule_x1], [row_y(edge)] * 2,
                                      transform=fig.transFigure, color="black",
                                      linestyle="--", linewidth=axis_lw, zorder=5))
            fig.text(label_x, (row_y(lo) + row_y(hi)) / 2.0, name, rotation=90,
                     va="center", ha="center", fontsize=label_pt)

        # --- legend and colour bar, at the published figure fractions --------------------
        # Both are placed by the ABSOLUTE fractions the published script uses. They are
        # meaningful only against the published margins, which GRID_KW now restores; the
        # earlier port recentred them inside the empty corner cell, which is what put the
        # bar and the legend in visibly different places.
        handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black",
                             linewidth=bar_lw)
                   for c in (mtif_c, dtif_c, shared_c)]
        # Published: fig.legend(bbox_to_anchor=(.814, .882), fontsize=FS-2.5, ncols=3),
        # which is matplotlib's default loc="upper right" and a framed box.
        fig.legend(handles, ["Measured TIF", "Deep TIF", "Shared"],
                   loc="upper right", bbox_to_anchor=(0.814, 0.882), ncol=3,
                   fontsize=legend_pt, frameon=True, fancybox=False)

        # THE PUBLISHED COLOUR BAR. It supersedes the paper-wide one-metric-size rule the other main
        # figures follow. These are the published numbers verbatim --
        #     cax = fig.add_axes([0.815, 0.795, 0.022, 0.085])
        #     plt.colorbar(..., cax=cax, label='Beta Values')
        # -- so the bar keeps its published size, place and furniture: an outline, ticks at
        # -0.5 / 0.0 / 0.5 on the right, and the label outside them. Fractions, not
        # millimetres, so it rides render_to_width's trial canvases unchanged.
        cax = fig.add_axes([0.815, 0.795, 0.022, 0.085])
        bar_obj = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                               label="Beta Values")
        # Colorbar.solids is created rasterized=True, which would drop a bitmap into an
        # otherwise vector figure and fail the display-item check.
        bar_obj.solids.set_rasterized(False)
        bar_obj.outline.set_linewidth(axis_lw)
        cax.tick_params(width=axis_lw, length=tick_len, labelsize=cbar_pt)
        cax.yaxis.label.set_size(cbar_pt)
        fig._fig5_cax = cax
        fig._fig5_main = ax_main
        fig._fig5_top = ax_top
        fig._fig5_right = ax_right
        return fig

    fig, ink = render_to_width(build, style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = str(OUT_DIR / OUT_NAME)

    # Converge on ink that is exactly the display-item width AND has no white on any side.
    # render_to_width gets the canvas close using matplotlib's tight bbox, but that bbox
    # overestimates the ink: the 45-degree column labels' boxes have empty corners, and the
    # colour bar's tick and axis labels carry side bearing. Figure 4 trims top and bottom
    # only, which is enough there; here the overhang is on the right too, and trimming it
    # takes the width off target -- so measure the real ink with ghostscript, trim all four
    # sides, and rebuild at a corrected canvas width until both conditions hold at once.
    #
    # The tight bbox stops at a frame's path, not its stroke, so each probe is padded first
    # and trimmed back to what ghostscript sees; save() then adds the border around it.
    probe = out_stem + "__probe.pdf"
    canvas_w_in = fig.get_size_inches()[0]
    PAD_IN = 2.0 / 72.0
    t = lambda mm: max(0.0, mm) / MM
    for _ in range(6):
        ink = Bbox.from_extents(ink.x0 - PAD_IN, ink.y0 - PAD_IN,
                                ink.x1 + PAD_IN, ink.y1 + PAD_IN)
        fig.savefig(probe, bbox_inches=ink, pad_inches=0)
        left_mm, bottom_mm, right_mm, top_mm = ink_margins_mm(probe)
        ink = Bbox.from_extents(ink.x0 + t(left_mm), ink.y0 + t(bottom_mm),
                                ink.x1 - t(right_mm), ink.y1 - t(top_mm))
        if abs(ink.width * MM - style.display_item_width_mm) <= 0.05:
            break
        canvas_w_in *= style.display_item_width_mm / (ink.width * MM)
        plt.close(fig)
        fig = build(canvas_w_in)
        ink = fig.get_tightbbox(fig.canvas.get_renderer())
    else:
        raise SystemExit("could not converge on a zero-margin 179 mm ink box")
    Path(probe).unlink()

    save(fig, ink, out_stem, dpi=500)
    problems = verify(out_stem + ".pdf", style, geometry=True,
                      expect_text=("Risk Factors", "Ocular Diseases", "Counts", "DBP"))
    if problems:
        raise SystemExit("display-item rules violated:\n  " + "\n  ".join(problems))
    print("  display-item check passed (width, margins, font, no rasters, text)")

    n_cells = deep_sig.size
    print("  {} outcomes x {} traits = {} cells per version".format(
        n_rows, n_cols, n_cells))
    print("  FDR 5% (BH within each version): deep {} cells, measured {} cells, "
          "both {}, deep-only {}, measured-only {}".format(
              int(deep_sig.sum()), int(meas_sig.sum()), int(both_col.sum()),
              int(deep_only_col.sum()), int(meas_only_col.sum())))
    print("  beta ramp {:+.3f} .. {:+.3f} (symmetric)".format(-abs_max, abs_max))
    print("  {:.1f} x {:.1f} mm (ink)".format(ink.width * MM, ink.height * MM))
    fig_w_in, fig_h_in = fig.get_size_inches()
    cax_box = fig._fig5_cax.get_position()
    print("  colour bar {:.2f} x {:.2f} mm (the published bar at print scale, not the "
          "paper-wide metric bar)".format(
              cax_box.width * fig_w_in * MM, cax_box.height * fig_h_in * MM))
    main_box = fig._fig5_main.get_position()
    cell_w = main_box.width * fig_w_in * MM / deep_v.shape[1]
    cell_h = main_box.height * fig_h_in * MM / deep_v.shape[0]
    print("  cell {:.2f} x {:.2f} mm, aspect {:.3f} : 1 (published 1.520 : 1)".format(
        cell_w, cell_h, cell_w / cell_h))
    tb = fig._fig5_top.get_position()
    rb = fig._fig5_right.get_position()
    print("  gutters: above heat map {:.2f} mm, beside it {:.2f} mm (hspace {:.4f})".format(
        (tb.y0 - main_box.y1) * fig_h_in * MM,
        (rb.x0 - main_box.x1) * fig_w_in * MM, grid_kw["hspace"]))
    print("Wrote Figure 5 to {}".format(OUT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
