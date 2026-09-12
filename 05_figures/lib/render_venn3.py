#!/usr/bin/env python
"""Draw the three-way feature-set Venn from its deposited region counts.

This is the single definition of how that Venn looks, and the reason the data deposit
can ship a table instead of a picture.

Author: Michael Beyeler (github.com/mjbeyeler)

WHY COUNTS ARE ENOUGH. matplotlib_venn derives the entire area-proportional layout --
circle radii, centres, and the position of every label -- from the seven region sizes
alone. ``venn3(subsets=<seven counts>)`` produces byte-identical path vertices and
identical label text and positions to ``venn3([set_a, set_b, set_c])``; the sets
themselves never reach the geometry solver. So the counts are a complete description of
the panel, and a deposited SVG would add nothing but a rendering that no reader can
check: deposit the numbers, not the rendering.

Every Venn in the paper comes from here, sharing the styling constants below and the
count size in config (plot_styles.figures.fig3.venn_label_fontsize):

  05_figures/main/fig3_snp.R (c)                public path: fetches the counts CSV from
  05_figures/main/fig3_gene.R (f)               the deposit and shells out to this file's
                                                CLI, then imports the SVG as vector paths
                                                and draws the counts itself.
  05_figures/main/fig3_polygenicity.py (j)      in-figure path: place_in_axis() draws the
                                                same venn3 straight into a matplotlib
                                                axis at the same mm-per-unit scale.
  the upstream analysis pipeline                pipeline path: builds the sets from
                                                access-controlled association results and
                                                calls style_and_save() on its own venn3.

So c, f and j differ in exactly one thing -- which library finally sets the text -- and
that is forced by the R import chain (see WHY THE SVG CARRIES NO TEXT below).

Keeping the pipeline script's venn3 call separate (from sets, not from counts) is what
makes the two SVGs comparable -- diffing them is the regression test that the deposited
counts really do reproduce the panel.

WHY THE SVG CARRIES NO TEXT. Nature requires vector figures whose "text ... should remain
editable" (Guide to preparing final artwork). matplotlib can emit real <text> elements
(``svg.fonttype = "none"``), but the R import chain destroys them: ``rsvg::rsvg_svg()``
converts every glyph to a path on its way to the Cairo flavour of SVG that grImport2 can
read -- verified, the re-emitted file has zero <text> elements. So the region counts are
NOT drawn into the SVG at all. Instead this script records where each count belongs, as a
fraction of the saved image box, and the R caller draws the numbers itself with
``cowplot::draw_label()``. The text is then native to the PDF, in the document's own font,
at a size we set in points rather than one that rides on the diagram's scale factor.

Usage:
    python 05_figures/lib/render_venn3.py <counts.csv> <out.svg> [--config config.yaml]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib_venn import venn3  # noqa: E402

VENN_FIG_INCHES = 5.0
VENN_DPI = 300
# 10 % transparent, over the panel's own background: the feature-set colours themselves,
# just taken off full strength. A tenth is what takes the edge off three overlapping fills
# without lightening the diagram the way the pale variants below did.
VENN_ALPHA = 0.90
# Borderless: matplotlib_venn's own default, and what the published panel shows.
VENN_EDGE_WIDTH = 0
VENN_EDGE_COLOR = "none"
# Only the PNG preview shows these; the SVG carries no text and the final panel's numbers
# are set in points by the R caller (see the module docstring).
VENN_SET_LABEL_FONTSIZE = 12
VENN_SUBSET_LABEL_FONTSIZE = 12
VENN_FONTWEIGHT = "normal"
# savefig's default when bbox_inches="tight". Applied explicitly here so the PNG, the SVG
# and the recorded label fractions are all framed by the same box.
VENN_PAD_INCHES = 0.1
# Transparent so the diagram sits on whatever the surrounding panel provides. With an
# opaque figure patch matplotlib bakes a full-canvas white rectangle into the SVG, which
# then shows up as a visible box around the Venn.
VENN_TRANSPARENT = True

# Set order, and therefore colour order, is [LV, dTIF, mTIF] -- as in the manuscript panel.
# SET_ORDER names are the caller's set keys; COLOR_KEYS index config.yaml's `colors`.
SET_ORDER = ("LV", "dTIF", "mTIF")
COLOR_KEYS = ("lvs", "dtifs", "mtifs")

# matplotlib_venn's subset order: Abc, aBc, ABc, abC, AbC, aBC, ABC, with A/B/C the three
# sets in SET_ORDER. Each key is (in_lv, in_dtif, in_mtif).
SUBSET_ORDER = (
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)

# Region names used in the counts CSV, in the same order.
REGION_NAMES = (
    "LV",
    "dTIF",
    "LV&dTIF",
    "mTIF",
    "LV&mTIF",
    "dTIF&mTIF",
    "LV&dTIF&mTIF",
)

COUNTS_COLUMNS = ["region", "in_lv", "in_dtif", "in_mtif", "n_snps"]
# The diagram's own coordinate system, written beside the SVG so a caller can place it at
# a SIZE IT CHOOSES rather than at whatever scale happens to fill its panel box.
#
# matplotlib_venn normalises the total area of the three circles to 1.0 square data unit
# (`normalize_to`, _venn3.py:50), whatever the counts are. So the physical area of a
# placed diagram is exactly (mm per data unit)^2, and placing all three of Figure 3's
# Venns at ONE mm-per-unit scale makes their total areas identical -- which is what makes
# the three panels comparable at a glance. Fitting each diagram to its own box does not:
# the boxes and the diagrams' aspect ratios differ, so panel c came out 3.6 % larger in
# linear scale than f (7 % in area) and panel j, drawn into a wider axis, 30 % larger.
GEOMETRY_COLUMNS = ["box_width_units", "box_height_units", "total_circle_area_units2"]
NORMALIZE_TO = 1.0
LOCI_COLUMNS = ["rsid", "chr", "pos", "in_lv", "in_dtif", "in_mtif"]


def region_key(sets, rsid):
    """Membership key (in_lv, in_dtif, in_mtif) of one locus."""
    return tuple(1 if rsid in sets[name] else 0 for name in SET_ORDER)


def counts_from_sets(sets):
    """Seven region counts, keyed by membership tuple. Empty regions are present as 0."""
    counts = {key: 0 for key in SUBSET_ORDER}
    for rsid in set().union(*(sets[name] for name in SET_ORDER)):
        counts[region_key(sets, rsid)] += 1
    return counts


def read_counts_csv(path):
    """Read a counts CSV into {(in_lv, in_dtif, in_mtif): n_snps}.

    Row order is not trusted -- the membership flags decide which region a row is, so a
    reader who re-sorts the file still gets the same figure.
    """
    counts = {}
    with open(str(path), newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in COUNTS_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                "{} lacks column(s): {}".format(path, ", ".join(missing))
            )
        for row in reader:
            key = tuple(int(row[c]) for c in ("in_lv", "in_dtif", "in_mtif"))
            if key not in SUBSET_ORDER:
                raise SystemExit(
                    "{}: row '{}' has membership {} -- the empty region (0,0,0) is not a "
                    "Venn region".format(path, row["region"], key)
                )
            if key in counts:
                raise SystemExit("{}: membership {} appears twice".format(path, key))
            counts[key] = int(row["n_snps"])

    absent = [k for k in SUBSET_ORDER if k not in counts]
    if absent:
        # Not a convenience default: venn3 draws a label in every region, zero included,
        # so a missing row would silently change the panel rather than leave a gap.
        raise SystemExit(
            "{} is missing {} of the 7 regions: {}".format(
                path, len(absent), ", ".join(str(k) for k in absent)
            )
        )
    return counts


def write_counts_csv(path, counts):
    with open(str(path), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COUNTS_COLUMNS)
        for name, key in zip(REGION_NAMES, SUBSET_ORDER):
            writer.writerow([name, key[0], key[1], key[2], counts[key]])


def subset_tuple(counts):
    return tuple(counts[key] for key in SUBSET_ORDER)


def venn_colors(config):
    """The three fill colours, [LV, dTIF, mTIF]. `config` is a path or an already-read dict.

    The feature-set palette itself -- the same colour the Manhattans give a feature set for
    its dark chromosomes -- at VENN_ALPHA. Two paler schemes were tried and rejected on
    4 Sep 2026, in this order: the Manhattans' alternate-chromosome tint (0.6 towards
    white) and the midpoint of that (0.3). Do not reintroduce either without asking. They
    also broke the diagram's mixing: matplotlib_venn colours an overlap by ADDING its
    members (0.7 * (c1 + c2) in _common.mix_colors), and with pale inputs the sum clips at
    white, so the largest intersection in panel c -- the 43 loci shared by the mTIFs and
    the dTIFs -- came out pure white and the overlap vanished. At full strength the
    addition behaves and matplotlib_venn's own mixing is what draws the seven regions.

    Every caller goes through here (panels c and f via the CLI below, panel j via
    fig3_polygenicity.py, and the upstream pipeline through its own venn steps) so
    the three Venns cannot drift apart.
    """
    if not isinstance(config, dict):
        with open(str(config)) as handle:
            config = yaml.safe_load(handle)
    colors = config["colors"]
    return [colors[key] for key in COLOR_KEYS]


LABEL_COLUMNS = ["text", "x_frac", "y_frac"]


def apply_styling(venn, ax):
    """Colour, weight and orientation. Shared by every caller so there is one definition."""
    for label in list(venn.set_labels) + list(venn.subset_labels):
        if label is None:
            continue
        label.set_color("black")
        label.set_fontweight(VENN_FONTWEIGHT)
        label.set_fontsize(
            VENN_SET_LABEL_FONTSIZE if label in venn.set_labels
            else VENN_SUBSET_LABEL_FONTSIZE
        )
    for patch in venn.patches:
        if patch is not None:
            patch.set_linewidth(VENN_EDGE_WIDTH)
            patch.set_edgecolor(VENN_EDGE_COLOR)

    # Orientation of the published panel.
    ax.invert_xaxis()
    ax.invert_yaxis()


def label_fractions(venn, figure, box):
    """Where each region count sits, as a fraction of `box` (x from left, y from bottom).

    The R caller redraws the counts itself, so it needs their positions in the same frame
    it places the imported diagram in. Fractions of the saved image box are exactly that
    frame and survive any later rescaling.
    """
    renderer = figure.canvas.get_renderer()
    rows = []
    for label in venn.subset_labels:
        if label is None:
            continue
        extent = label.get_window_extent(renderer)
        cx = (extent.x0 + extent.x1) / 2.0 / figure.dpi
        cy = (extent.y0 + extent.y1) / 2.0 / figure.dpi
        rows.append([
            label.get_text(),
            (cx - box.x0) / box.width,
            (cy - box.y0) / box.height,
        ])
    return rows


def units_per_inch(ax):
    """Data units per inch of figure, from the axis transform (aspect is equal)."""
    inverse = ax.transData.inverted()
    origin = inverse.transform((0.0, 0.0))
    one_inch = inverse.transform((ax.figure.dpi, 0.0))
    return abs(one_inch[0] - origin[0])


def write_geometry_csv(path, box, ax):
    """Record the saved image box in the diagram's own data units."""
    per_inch = units_per_inch(ax)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(GEOMETRY_COLUMNS)
        writer.writerow(["{:.6f}".format(box.width * per_inch),
                         "{:.6f}".format(box.height * per_inch),
                         "{:.6f}".format(NORMALIZE_TO)])


def data_centre(venn, ax):
    """Centre of the drawn circles, in data units."""
    inverse = ax.transData.inverted()
    xs, ys = [], []
    for patch in venn.patches:
        if patch is None:
            continue
        extent = patch.get_path().transformed(patch.get_transform()).get_extents()
        for point in ((extent.x0, extent.y0), (extent.x1, extent.y1)):
            x, y = inverse.transform(point)
            xs.append(x)
            ys.append(y)
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def ink_bbox(venn):
    """Display-coordinate bounding box of the drawn circles.

    A Venn axis is a plain box with the diagram floating inside it, so `ax.get_tightbbox`
    is the whole cell and is the wrong anchor for a panel letter.
    """
    from matplotlib.transforms import Bbox

    boxes = [patch.get_window_extent() for patch in venn.patches if patch is not None]
    return Bbox.union(boxes)


def patch_extent_units(venn, ax):
    """(width, height) of the drawn circles, in data units."""
    inverse = ax.transData.inverted()
    xs, ys = [], []
    for patch in venn.patches:
        if patch is None:
            continue
        extent = patch.get_path().transformed(patch.get_transform()).get_extents()
        for point in ((extent.x0, extent.y0), (extent.x1, extent.y1)):
            x, y = inverse.transform(point)
            xs.append(x)
            ys.append(y)
    return max(xs) - min(xs), max(ys) - min(ys)


def patch_top_figure_fraction(venn, fig):
    """Figure-fraction y of the TOP of the drawn circles.

    A Venn does not fill its axis: the diagram is wider than it is tall and sits centred,
    so the axis's own top is well above the ink. A caller that wants to put something at
    the top of the DIAGRAM -- panel j's letter -- needs the circles, not the box.
    """
    tops = [patch.get_path().transformed(patch.get_transform()).get_extents().y1
            for patch in venn.patches if patch is not None]
    return fig.transFigure.inverted().transform((0.0, max(tops)))[1]


def axis_box_mm(ax):
    """The axis's box in millimetres, as the LAYOUT placed it.

    `get_position(original=True)` and not the active position: matplotlib_venn leaves the
    axis on `set_aspect("equal")`, and an equal-aspect axis with the default
    `adjustable="box"` reports a SHRUNKEN active box. Sizing the data window from that
    while drawing into the full box is what turned panel j's circles into ellipses.
    """
    box = ax.get_position(original=True)
    fig_w_in, fig_h_in = ax.figure.get_size_inches()
    return box.width * fig_w_in * 25.4, box.height * fig_h_in * 25.4


def scale_axis_to(ax, venn, mm_per_unit):
    """Window the axis so one data unit is `mm_per_unit` millimetres on the page.

    The counterpart of the caller-chosen size on the SVG route: with the same
    mm_per_unit, a Venn drawn into an axis has the same physical area as one imported
    from an SVG (see GEOMETRY_COLUMNS).

    The spans are set exactly proportional to the axis box, so x and y share one scale --
    circles stay circles -- without matplotlib_venn's `set_aspect("equal")` being allowed
    to resize the box (adjustable="box" would shrink the axis to the new aspect instead).
    """
    cx, cy = data_centre(venn, ax)

    axis_width_mm, axis_height_mm = axis_box_mm(ax)
    half_w = axis_width_mm / mm_per_unit / 2.0
    half_h = axis_height_mm / mm_per_unit / 2.0
    ax.set_aspect("auto")
    ax.set_xlim(cx + half_w, cx - half_w)      # inverted, as apply_styling() leaves it
    ax.set_ylim(cy + half_h, cy - half_h)
    return half_w * 2.0, half_h * 2.0


def mm_per_unit_xy(ax):
    """Millimetres per data unit along x and y -- equal iff the circles are circles."""
    forward = ax.transData
    origin = forward.transform((0.0, 0.0))
    return (abs(forward.transform((1.0, 0.0))[0] - origin[0]) / ax.figure.dpi * 25.4,
            abs(forward.transform((0.0, 1.0))[1] - origin[1]) / ax.figure.dpi * 25.4)


def write_labels_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        for text, x_frac, y_frac in rows:
            writer.writerow([text, "{:.6f}".format(x_frac), "{:.6f}".format(y_frac)])


def style_and_save(venn, figure, svg_path, png_path=None, labels_path=None,
                   geometry_path=None):
    """Apply the panel's styling to a drawn venn3 and write it out.

    The SVG is written with every region count BLANKED. That is not an oversight: the R
    import chain (rsvg::rsvg_svg -> grImport2) converts any <text> to outlines, so text
    left in the SVG could never be editable in the final PDF. The counts travel beside it
    in `labels_path` instead and are drawn as real text by the caller. The PNG is a visual
    check, so it keeps its text.

    Both files are framed by the same explicitly computed box, so the label fractions
    describe the SVG exactly.
    """
    ax = plt.gca()
    apply_styling(venn, ax)

    # Compute the crop ONCE, with the text still visible, and reuse it for both outputs.
    # Passing "tight" to each savefig separately would re-solve it after the text is
    # blanked and could shift the frame out from under the recorded fractions.
    figure.canvas.draw()
    box = figure.get_tightbbox(figure.canvas.get_renderer()).padded(VENN_PAD_INCHES)
    labels = label_fractions(venn, figure, box)

    if png_path is not None:
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            str(png_path), bbox_inches=box, transparent=VENN_TRANSPARENT, dpi=VENN_DPI
        )

    for label in venn.subset_labels:
        if label is not None:
            label.set_text("")
    Path(svg_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(svg_path), bbox_inches=box, transparent=VENN_TRANSPARENT, format="svg")

    if labels_path is not None:
        write_labels_csv(labels_path, labels)
    if geometry_path is not None:
        write_geometry_csv(geometry_path, box, ax)

    plt.close(figure)
    return labels


def new_figure():
    return plt.figure(figsize=(VENN_FIG_INCHES, VENN_FIG_INCHES), dpi=VENN_DPI)


def render_from_counts(counts, svg_path, colors, png_path=None, labels_path=None,
                       geometry_path=None):
    figure = new_figure()
    venn = venn3(
        subsets=subset_tuple(counts),
        set_colors=colors,
        set_labels=["", "", ""],
        alpha=VENN_ALPHA,
    )
    return style_and_save(venn, figure, svg_path, png_path, labels_path, geometry_path)


def place_in_axis(counts, colors, ax, fontsize, mm_per_unit,
                  fontweight=VENN_FONTWEIGHT):
    """Draw the Venn into `ax` at `mm_per_unit` and check that it came out circular.

    THE one entry point for an in-figure Venn, so that panel j is built by exactly the
    same code as the SVG that panels c and f import: same venn3 call, same subset order,
    same colours and alpha, same borderless patches, same inversion, and the same count
    font size handed in from config. The only difference between the routes is where the
    text is finally set (matplotlib here, cowplot::draw_label in R), which the R chain
    forces (see the module docstring).
    """
    venn = draw_into_axis(counts, colors, ax, fontsize, fontweight)
    spans = scale_axis_to(ax, venn, mm_per_unit)
    per_mm_x, per_mm_y = mm_per_unit_xy(ax)
    if abs(per_mm_x - per_mm_y) > 1e-6 * mm_per_unit:
        raise SystemExit(
            "the Venn axis is anisotropic ({:.4f} vs {:.4f} mm per unit): its circles "
            "would be drawn as ellipses".format(per_mm_x, per_mm_y))
    return venn, spans


def draw_into_axis(counts, colors, ax, fontsize, fontweight=VENN_FONTWEIGHT):
    """Draw the same Venn straight into an existing axis, keeping its text.

    For callers that are already building a matplotlib figure and export it to PDF
    directly (Figure 3's j/k strip). There is no SVG round-trip, so matplotlib's own
    text survives into the PDF as real text -- provided the caller set
    ``pdf.fonttype = 42``, which ``PlotStyle.apply_font()`` does.

    Same geometry solver and same styling constants as the SVG route, so the Venn in one
    strip is visually identical in construction to the Venns in the others.
    """
    venn = venn3(
        subsets=subset_tuple(counts),
        set_colors=colors,
        set_labels=["", "", ""],
        alpha=VENN_ALPHA,
        ax=ax,
    )
    apply_styling(venn, ax)
    for label in venn.subset_labels:
        if label is not None:
            label.set_fontsize(fontsize)
            label.set_fontweight(fontweight)
    return venn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", type=Path, help="region-counts CSV from the data deposit")
    parser.add_argument("out_svg", type=Path, help="SVG to write")
    parser.add_argument("--png", type=Path, default=None, help="also write a PNG here")
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help="write the region counts and their positions (as fractions of the SVG box) "
             "here, for a caller that draws the text itself",
    )
    parser.add_argument(
        "--geometry-out",
        type=Path,
        default=None,
        help="write the SVG box's size in the diagram's own data units here, so the "
             "caller can place it at a chosen mm-per-unit scale",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config.yaml",
        help="config.yaml supplying the feature-set colours",
    )
    args = parser.parse_args()

    counts = read_counts_csv(args.counts)
    render_from_counts(
        counts, args.out_svg, venn_colors(args.config), args.png, args.labels_out,
        args.geometry_out,
    )
    print("  wrote {}".format(args.out_svg))
    for extra in (args.labels_out, args.geometry_out):
        if extra is not None:
            print("  wrote {}".format(extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
