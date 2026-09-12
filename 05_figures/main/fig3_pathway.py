#!/usr/bin/env python
"""Figure 3, fourth strip: the significant-pathway map (k), full width.

Author: Michael Beyeler (github.com/mjbeyeler)

Self-contained: fetches its inputs from the public data deposit by DOI.

    pixi run python 05_figures/main/fig3_pathway.py

The analysis -- nestedness de-duplication, thresholds, category collapsing, display labels
-- is done upstream in the analysis pipeline; this file only draws.

All four category blocks are drawn as one figure, with square cells at 6.63 mm (shrunk only
if the width cannot hold them), columns ordered by best P within a category, pathway names
at 60 degrees and 6 pt, and the full category names as headings. The colour ramp is read
from config (plot_styles.figures.fig3.pathway_cmap_rgb).

Two renderings. The default is the glow rendering: a ramp that climbs through the violet
to white on the strongest evidence, with a glow on the strongest cell of each feature
set; see draw_cell_bloom for how the glow is built. `--no-glow` draws the flat
white -> #730FF0 ramp with the #e0e0e0 grid between cells and seaborn's .15-grey/white
asterisk rule.

Panel j is NOT here -- the manuscript puts it on the g-i row, so it is drawn by
05_figures/main/fig3_polygenicity.py. That leaves k the full page width.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
import matplotlib.patheffects as patheffects  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from scipy.ndimage import distance_transform_cdt  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402
from seaborn.utils import relative_luminance  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.display_item import figure_dir, ink_margins_mm  # noqa: E402
from src.fetch_data import fetch  # noqa: E402
from src.plot_style import draw_colorbar_ticks, load_plot_style  # noqa: E402

# Left to right. The order is a display choice and lives here, not in the deposit,
# which is grouped by category and sorted by best P within one.
CATEGORY_ORDER = ["Vascular Remodeling & Endothelial Biology",
                  "Pigmentation & Melanogenesis", "Neuronal and ocular", "Other"]
ROWS = [("mTIFs", "mtif_p", "threshold_mtif"),
        ("dTIFs", "dtif_p", "threshold_dtif"),
        ("LVs", "lv_p", "threshold_lv")]

# Geometry of the published panel, in millimetres of the printed strip: square cells (the
# size is capped by config at the published 6.63 mm and shrunk only if the width cannot
# hold 23 of them), a colour bar beside the last block, and the four category blocks close
# together -- the published composite abutted them, so spare width goes into the cells,
# not into the gaps. The bar is the paper-wide one (plot_styles.figures.fig3.colorbar_mm,
# the Figure 2 size -- see the colour-bar rule in 05_figures/README.md) in WIDTH; its
# configured height is a cap, and it is shortened to the heat map's own height so the
# bar's ends line up with the map's (the paper-wide 17.17 mm overhangs
# a 14.4 mm map top and bottom). That is the published behaviour again, and the one
# sanctioned exception to the paper-wide bar size.
# Space between the last block and the colour bar comes from config
# (pathway_colorbar_gap_mm); see the note there on how it trades against the block gaps.
TITLE_PAD_PT = 4.0
# Room above the heat map for the category headings. One line needs about 3 mm at 7 pt;
# the band is recomputed from the number of lines the headings actually wrap to.
HEADING_LINE_MM = 2.6
HEADING_BAND_MM = 5.0
# seaborn's annotation rule (matrix.py, 0.11.2): dark grey `.15` on cells whose relative
# luminance exceeds 0.408, white otherwise. The published asterisks followed it.
ANNOT_LUMINANCE_CUT = 0.408
ANNOT_DARK = ".15"


def star_ink_path(mark, size_pt, family):
    """The asterisks as a filled OUTLINE, centred on their own ink at (0, 0), in points.

    This is Figure 5's technique (fig5_disease.star_outline) and it is copied deliberately,
    because the face alone is not the point. Setting DejaVu Sans as a text font EMBEDS
    DejaVu in the PDF, and Figure 3's assembler checks that the whole figure uses one family
    -- verified: drawing the stars as text turned "one family" into
    FAIL ['DejaVuSans', 'NimbusSans'], while Figure 5 embeds NimbusSans only.
    Converting the glyph to a path gives DejaVu's shape as geometry and embeds no font at
    all, which is how Figure 5 gets away with it.

    Centring on the ink also replaces the baseline-offset correction the text version needed:
    an asterisk sits high on its em box, so centring the box leaves the mark riding high.
    """
    path = TextPath((0.0, 0.0), mark, size=size_pt,
                    prop=FontProperties(family=family))
    box = path.get_extents()
    centre = np.array([(box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0])
    return path.vertices - centre, path.codes


def local_module():
    """The local companion to this script, or None.

    Anything in private/ is outside this repository, so a checkout without it runs
    the script's own defaults throughout.
    """
    path = REPO_ROOT / "private" / "fig3_pathway_local.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("fig3_pathway_local", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annot_ink_offset_pt(fontsize, family=None):
    """How far the ink of "*" sits above its baseline, in points.

    An asterisk occupies only the top of its line box, so `va="center"` centres the BOX
    and leaves the glyph riding high in the cell. Measuring the
    glyph outline gives the offset that centres the ink itself; it is the same for *, **
    and ***, which share one vertical extent.
    """
    path = TextPath((0.0, 0.0), "*", size=fontsize,
                    prop=FontProperties(family=family or plt.rcParams["font.family"]))
    box = path.get_extents()
    return (box.y0 + box.y1) / 2.0


def heading(text):
    """The category as a heading: first word capitalised, "and" written as "&".

    Three of the four categories were named with an ampersand and one with the word, so
    the row read inconsistently. The deposit keeps the category
    exactly as the analysis wrote it; case and ampersand are display choices and are
    applied here, like the pathway display labels.
    """
    return (text[:1].upper() + text[1:].lower()).replace(" and ", " & ")


def wrap_to_width(fig, text, width_mm, fontsize, max_lines=3):
    """Greedy word wrap so a heading fits over its block rather than overhanging it.

    Measured, not guessed: a throwaway text artist is laid out with the real renderer,
    so the break points follow the font actually in use.
    """
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0.0, 0.0, "", fontsize=fontsize)

    def width_of(line):
        probe.set_text(line)
        return probe.get_window_extent(renderer).width / fig.dpi * 25.4

    words = text.split()
    if width_of(text) <= width_mm:
        probe.remove()
        return [text]
    # Preferred break: straight after an "&". "Vascular remodeling &" / "endothelial
    # biology" reads as the two halves of one name, where greedy wrapping breaks inside
    # "endothelial biology".
    for index, word in enumerate(words):
        if word == "&" and 0 < index < len(words) - 1:
            head, tail = " ".join(words[:index + 1]), " ".join(words[index + 1:])
            if max(width_of(head), width_of(tail)) <= width_mm:
                probe.remove()
                return [head, tail]

    lines, current = [], ""
    for word in words:
        trial = word if not current else current + " " + word
        if current and width_of(trial) > width_mm and len(lines) < max_lines - 1:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    probe.remove()
    return lines


def glow_cmap(stops):
    """The published ramp extended into the glow rendering's colour map.

    `stops` is the config list of (position, hex) pairs, and it keeps #730FF0 at its
    middle: a strongly significant cell is still exactly the colour it is in the published
    panel. What the ramp adds is a dark surround below that anchor and a white-hot core
    above it, so the map stays monotonic in luminance -- brighter is more significant all
    the way up, with no hue reversal for the eye to read against.
    """
    return LinearSegmentedColormap.from_list(
        "glow", [(float(pos), str(color)) for pos, color in stops])


def halo_opacity_at(d_mm, cfg):
    """Accumulated opacity of the block halo `d_mm` outside the heat map.

    This is the profile the coats are solved to reproduce, so it is also what the halo
    actually paints. Used to decide whether the row labels, which sit in the gutter the
    halo reaches into, have to go white.
    """
    if d_mm >= float(cfg["halo_spread_mm"]):
        return 0.0
    sigma = float(cfg["halo_sigma_mm"])
    return float(cfg["halo_peak"]) * np.exp(-0.5 * (max(d_mm, 0.0) / sigma) ** 2)


def halo_visible_reach_mm(cfg, floor=0.01):
    """How far out the halo can still be seen, in millimetres.

    NOT halo_spread_mm, which is only where the last coat is drawn: with a Gaussian of
    sigma 1.3 mm the profile is already down to a fifth of a percent there, and comparing
    block gaps against it would condemn a layout whose canvas is plainly white. This
    inverts the profile for the opacity `floor` below which nothing is visible on paper.
    """
    peak = float(cfg["halo_peak"])
    if peak <= floor:
        return 0.0
    reach = float(cfg["halo_sigma_mm"]) * np.sqrt(2.0 * np.log(peak / floor))
    return min(reach, float(cfg["halo_spread_mm"]))


def draw_cell_bloom_blur(ax, matrix, norm, cfg):
    """A real glow: stamp the emitters into a field, blur it, composite it over the cells.

    The ring version below approximates a blur with concentric stroked annuli, and at 5 mm
    cells the approximation shows -- the rings read as nested squares rather than as light.
    A Gaussian filter is the thing itself, at the cost of the layer being a BITMAP; see the
    note in config, and bloom_render: rings to go back.

    The light travels OUTWARD FROM THE CELL'S EDGE, not outward from its centre. That
    distinction is the whole shape of the effect and a Gaussian blur gets it wrong: blurring
    a filled square pulls its iso-contours towards a disc and is weakest exactly at the four
    corners, so the glow reads as a round blob sitting behind a square.

    A DISTANCE TRANSFORM is the right tool: every pixel outside the emitters is given its
    distance to the nearest emitting pixel, and brightness is a function of that distance
    alone, so the glow's contours are offsets of the cell outline and its width is the same
    the whole way round.

    The metric is CHESSBOARD (Chebyshev), not Euclidean, and that choice is the look. Under
    Chebyshev distance the set of points a fixed distance from a square is another square,
    so the glow grows as concentric squares with square corners -- what the earlier vector
    version drew as stroked rings, now continuous. Euclidean distance was tried first: it
    rounds every corner, because a Euclidean offset of a square genuinely is round there.

    return_indices carries each pixel's NEAREST emitter with it, so cells of different
    -log10(P) keep their own brightness instead of being averaged into one field.
    """
    start = float(cfg["bloom_start"])
    gamma = float(cfg["bloom_gamma"])
    ppc = int(cfg["blur_px_per_cell"])
    gain = float(cfg["blur_gain"])
    max_alpha = float(cfg["blur_max_alpha"])
    n_rows, n_cols = matrix.shape

    emitters = np.zeros((n_rows * ppc, n_cols * ppc), dtype=float)
    lit = 0
    for row in range(n_rows):
        for col in range(n_cols):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            t = float(np.clip(norm(value), 0.0, 1.0))
            if t <= start:
                continue
            strength = ((t - start) / (1.0 - start)) ** gamma
            emitters[row * ppc:(row + 1) * ppc, col * ppc:(col + 1) * ppc] = strength
            lit += 1
    if not lit:
        return 0

    lit_mask = emitters > 0.0
    # Distance to the nearest LIT pixel, plus which one it was.
    distance, (near_y, near_x) = distance_transform_cdt(
        ~lit_mask, metric="chessboard", return_distances=True, return_indices=True)
    falloff = np.exp(-0.5 * (distance / (float(cfg["blur_sigma_cells"]) * ppc)) ** 2)
    intensity = np.clip(emitters[near_y, near_x] * falloff * gain, 0.0, 1.0)
    # A cell does not glow over itself: it would wash out the near-white top of the ramp,
    # which is what a reader grades the panel by.
    intensity[lit_mask] = 0.0

    # Blow the bright part of the glow out towards white. Compositing is "over", not
    # additive, so a constant amethyst laid on violet cells is nearly invisible; letting
    # the colour ride up to white with intensity is how a real glow photographs and is what
    # makes the light read at all. The faint outer aura keeps bloom_color unaltered.
    hot = np.clip(intensity ** float(cfg["hot_gamma"]), 0.0, 1.0) \
        * float(cfg["hot_white"])
    glow_rgb = np.array(to_rgb(str(cfg["bloom_color"])))
    rgba = np.zeros(intensity.shape + (4,), dtype=float)
    rgba[..., :3] = (glow_rgb[None, None, :] * (1.0 - hot)[..., None]
                     + hot[..., None])
    rgba[..., 3] = intensity * max_alpha
    # aspect="auto" is REQUIRED: imshow otherwise forces an equal aspect on an axes whose
    # box was placed by hand in millimetres, and the whole strip geometry moves.
    ax.imshow(rgba, extent=(0, n_cols, n_rows, 0), origin="upper", aspect="auto",
              interpolation="bilinear", zorder=2.5)
    return lit


def best_cell_per_row(matrices, cfg):
    """Which single cell in each row is the panel's strongest, across every block.

    The category blocks are separate axes and each only sees its own columns, so a per-block
    maximum would light four cells a row instead of one. Returns {block index: {row: col}},
    carrying only cells that also clear `bloom_start`.
    """
    if not matrices:
        return {}
    n_rows = matrices[0].shape[0]
    winners = {}
    for row in range(n_rows):
        best = None
        for index, matrix in enumerate(matrices):
            values = matrix[row]
            if not np.any(np.isfinite(values)):
                continue
            col = int(np.nanargmax(values))
            if best is None or values[col] > best[0]:
                best = (float(values[col]), index, col)
        if best is not None:
            winners[row] = best
    # RANKED ACROSS BLOCKS, because the three winners sit in different category blocks --
    # ordering them within a block would rank the Vascular pair against each other and miss
    # the Pigmentation one entirely. Rank 0 is the strongest cell in the panel.
    allowed = {}
    for rank, (row, (value, index, col)) in enumerate(
            sorted(winners.items(), key=lambda item: -item[1][0])):
        allowed.setdefault(index, {})[row] = (col, rank)
    return allowed


def draw_cell_bloom(ax, matrix, norm, cell_mm, cfg, allowed=None):
    """Spill light out of each cell in proportion to the evidence.

    The rings are STROKED rectangles, not filled ones. A filled rectangle large enough to
    glow would also cover the cell it came from, and lavender over a white-hot core turns
    the core lavender -- the opposite of the effect. A stroke is centred on its path, so
    each ring paints an annulus of its own thickness and leaves the cell alone.

    Ring k of `bloom_rings` sits at a fraction f of the spill distance and carries
    `bloom_alpha` * (1-f)^`bloom_falloff_power`. Raising that exponent pushes the light
    against the cell's edge and thins the outer aura without moving the reach, which is how
    the glow is made to read hotter at a fixed size.
    Everything is in CELL WIDTHS, so it survives the cell-size iteration; only the stroke
    weight needs the final millimetre size, which is why this runs after place() settles.
    """
    start = float(cfg["bloom_start"])
    gamma = float(cfg["bloom_gamma"])
    rings = int(cfg["bloom_rings"])
    spill = float(cfg["bloom_spill_cells"])
    # Reach by RANK: the panel's strongest cell throws light furthest, the weakest of the
    # three least. Equal *strength* still holds -- what differs is how far it carries.
    spills = [float(v) for v in cfg.get("bloom_spill_cells_by_rank") or []]
    alpha0 = float(cfg["bloom_alpha"])
    equal = bool(cfg.get("glow_equal_strength"))
    falloff = float(cfg.get("bloom_falloff_power", 2.0))
    hold = float(cfg.get("bloom_hold_frac", 0.0))
    glow_rgb = np.array(to_rgb(str(cfg["bloom_color"])))
    hot_white = float(cfg["hot_white"])
    hot_gamma = float(cfg["hot_gamma"])
    pt_per_cell = cell_mm / 25.4 * 72.0
    # Collect first, then draw STRONGEST LAST, because glows that reach each other have to
    # resolve in favour of the more significant cell. Two things enforce that: a weaker glow is CLIPPED out of every stronger glow's
    # territory so it cannot bleed in, and what remains is drawn underneath it.
    emitters = []
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            if not np.isfinite(value):
                continue
            # allowed maps row -> (col, rank); comparing it to a bare column index
            # silently matched nothing and switched the whole glow off.
            if allowed is not None and allowed.get(r, (None, None))[0] != c:
                continue
            t = float(norm(value))
            if t <= start:
                continue
            strength = 1.0 if equal else ((t - start) / (1.0 - start)) ** gamma
            rank = allowed[r][1] if allowed is not None else 0
            reach_here = spills[min(rank, len(spills) - 1)] if spills else spill
            emitters.append((float(value), r, c, strength, reach_here))
    emitters.sort(key=lambda item: -item[0])          # strongest first

    def reach(row, col, strength, span):
        """The square a cell's glow can occupy, in data coordinates."""
        e = span * strength
        return (col - e, row - e, col + 1.0 + e, row + 1.0 + e)

    for rank, (value, r, c, strength, span) in enumerate(emitters):
        # Two kinds of hole. A stronger glow's whole REACH is off limits, so the weaker one
        # cannot bleed into it. And every other winner's CELL is off limits to everyone,
        # in both directions -- light does not fall inside another winning cell. The second is what the
        # STRONGEST glow needed: nothing outranked it, so nothing had been stopping it
        # painting straight over a weaker winner's cell.
        holes = [reach(rr, cc, ss, sp) for _, rr, cc, ss, sp in emitters[:rank]]
        holes += [(cc, rr, cc + 1.0, rr + 1.0)
                  for index, (_, rr, cc, _, _) in enumerate(emitters) if index != rank]
        clip = None
        if holes:
            # The outer boundary is deliberately WIDE, so this path clips nothing except
            # the holes. The glow is allowed past the heat map's edge; the
            # only thing still cut out of a weaker glow is the territory of a stronger one.
            outer = (-2.0, -2.0,
                     float(matrix.shape[1]) + 2.0, float(matrix.shape[0]) + 2.0)
            verts, codes = [], []
            for index, (x0, y0, x1, y1) in enumerate([outer] + holes):
                corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
                if index:                              # holes wind the other way
                    corners = corners[::-1]
                verts.extend(corners)
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * 3
                             + [MplPath.CLOSEPOLY])
            clip = MplPath(verts, codes)
        thickness = span * strength / rings
        for k in range(1, rings + 1):
            f = (k - 0.5) / rings
            e = span * strength * f
            # The ride towards white is deliberately slight: the glow has to stay a
            # different colour from the near-white winning cell it surrounds.
            # A PLATEAU, then the falloff. The light holds full strength across the first
            # `hold` of the band and only then decays, so the glow reads as a solid rim that
            # fades at its outer edge rather than as something dying from the cell wall
            # outwards -- and the reach is untouched, because the decay is simply compressed
            # into what is left.
            after = 0.0 if f <= hold else (f - hold) / max(1.0 - hold, 1e-6)
            level = strength * (1.0 - after) ** falloff
            hot = min(level ** hot_gamma, 1.0) * hot_white
            ring = Rectangle(
                (c - e, r - e), 1.0 + 2.0 * e, 1.0 + 2.0 * e,
                fill=False,
                edgecolor=tuple(glow_rgb * (1.0 - hot) + hot),
                linewidth=thickness * pt_per_cell,
                alpha=alpha0 * level,
                zorder=2.5 - 0.01 * rank)            # stronger sits on top
            if clip is not None:
                ring.set_clip_path(clip, ax.transData)
            else:
                # add_patch would otherwise clip this to the axes, i.e. to the block.
                ring.set_clip_on(False)
            ax.add_patch(ring)
    return len(emitters)


def draw_block_halo(fig, axes, width_mm, height_mm, cfg):
    """A soft violet surround for each category block.

    ONE HALO PER BLOCK, not one around their common bounding box. The union was tried
    first and is wrong: the gaps between categories are 9 mm wide, so filling them at the
    halo's full opacity turned a third of the strip's canvas solid violet and the panel
    read as a dark purple background with cells on it. Per block, each aura dies out well before the next block starts and the
    canvas between categories stays white.

    It lives in an invisible full-figure axes scaled in MILLIMETRES, because figure
    fractions are not square on a strip this wide and the rounded corners would otherwise
    come out as ellipses.

    Coats go largest first, so what the page ends up showing at distance d is the product
    of everything still to be painted over it. Each coat's alpha is therefore SOLVED
    rather than set: to leave opacity O(d_k) after coat k, given O(d_k+1) is already down,
    coat k needs 1 - (1-O(d_k)) / (1-O(d_k+1)). That makes the accumulated profile exactly
    halo_opacity_at -- a Gaussian -- instead of the geometric series a constant alpha
    gives, which is what a blur would produce and what neither earlier attempt could.
    """
    rings = int(cfg["halo_rings"])
    spread = float(cfg["halo_spread_mm"])
    color = str(cfg["halo_color"])
    corner = float(cfg["halo_corner_mm"])

    ax_halo = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=-1)
    ax_halo.set_xlim(0.0, width_mm)
    ax_halo.set_ylim(0.0, height_mm)
    ax_halo.set_axis_off()
    ax_halo.patch.set_visible(False)

    boxes = [ax.get_position() for ax in axes]
    for box in boxes:
        x0, x1 = box.x0 * width_mm, box.x1 * width_mm
        y0, y1 = box.y0 * height_mm, box.y1 * height_mm
        outside = 1.0    # transparency still to account for, at the last coat drawn
        for k in range(rings, 0, -1):
            d = spread * k / float(rings)
            want = 1.0 - halo_opacity_at(d, cfg)
            alpha = 1.0 - want / outside
            outside = want
            if alpha <= 1e-4:
                continue
            ax_halo.add_patch(FancyBboxPatch(
                (x0 - d, y0 - d), (x1 - x0) + 2.0 * d, (y1 - y0) + 2.0 * d,
                boxstyle="round,pad=0,rounding_size={:.4f}".format(corner + d),
                facecolor=color, edgecolor="none", alpha=min(alpha, 1.0),
                linewidth=0.0))
    # The narrowest gap between two blocks, against twice the halo's VISIBLE reach: if the
    # auras meet, the canvas between those categories is not white any more and the union
    # problem is back in miniature. Reported so it cannot regress silently when the block
    # gap or the sigma is retuned.
    gaps = [(b.x0 - a.x1) * width_mm for a, b in zip(boxes, boxes[1:])]
    return ax_halo, boxes[0].x0 * width_mm, (min(gaps) if gaps else float("inf"))


def light_row_labels(fig, axes, block_left_mm, cfg):
    """Flip the row labels to white where the halo behind them is dark.

    Same rule the asterisks use on the cells (seaborn's relative-luminance cut), applied to
    the halo colour composited over the white page at whatever opacity the label's midpoint
    actually sits in -- so tuning halo_spread_mm cannot silently leave black text on a
    saturated violet ground.
    """
    ax = axes[0]
    halo_rgb = np.array(to_rgb(str(cfg["halo_color"])))
    cut = float(cfg["label_lum_cut"])
    fig.canvas.draw()
    flipped = []
    for label in list(ax.get_yticklabels()) + [ax.yaxis.label]:
        box = label.get_window_extent(fig.canvas.get_renderer())
        mid_mm = (box.x0 + box.x1) / 2.0 / fig.dpi * 25.4
        near_mm = block_left_mm - box.x1 / fig.dpi * 25.4
        far_mm = block_left_mm - box.x0 / fig.dpi * 25.4
        opacity = halo_opacity_at(block_left_mm - mid_mm, cfg)
        blend = opacity * halo_rgb + (1.0 - opacity) * np.ones(3)
        print("    label {!r}: {:.2f}-{:.2f} mm out, halo {:.0f}% at its middle".format(
            label.get_text(), near_mm, far_mm, 100.0 * opacity))
        if relative_luminance(tuple(blend)) <= cut:
            label.set_color("w")
            flipped.append(label.get_text())
    return flipped


def stars(p, threshold):
    """Significance tier: * at the study-wide pathway threshold, ** at a fifth of it, ***
    at a fiftieth."""
    if pd.isna(p):
        return ""
    if p < threshold / 50:
        return "***"
    if p < threshold / 5:
        return "**"
    if p < threshold:
        return "*"
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The glow rendering is the default; --no-glow draws the flat one.
    parser.add_argument("--presly", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-glow", dest="glow", action="store_false",
        help="draw panel k in the flat rendering (white to #730FF0) instead of the "
             "glow rendering")
    local = local_module()
    args = parser.parse_args(argv)

    config_path = REPO_ROOT / "config.yaml"
    with open(str(config_path), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    style = load_plot_style(str(config_path))
    style.apply_font()
    fig3 = config["plot_styles"]["figures"]["fig3"]
    glow = fig3["pathway_glow"] if args.glow else None

    # Panel k's headings are a point below the figure's other titles -- its own key, not
    # plot_styles.fonts.title (config, pathway_title_fontsize).
    title_pt = float(fig3["pathway_title_fontsize"])
    axis_pt = style.axis_label_fontsize
    tick_pt = style.tick_label_fontsize
    annot_pt = float(fig3["pathway_annot_fontsize"])
    annot_family = None
    if glow is not None and glow.get("star_match_fig5"):
        # Figure 5's rule, verbatim: its star is PUBLISHED_FS (12 pt) scaled by
        # canvas_width / PUBLISHED_CANVAS_IN[0] (12 in), which for a display-item-width
        # canvas reduces to the width in inches -- 7.05 pt at 179 mm. The face is DejaVu
        # Sans, and that is the part that matters: DejaVu's asterisk carries 58% more ink
        # than Nimbus Sans's at the same size (see fig5_disease.star_outline), which is why
        # the published stars read and a 6 pt Nimbus one does not.
        annot_pt = style.display_item_width_mm / 25.4
        annot_family = str(glow["star_font"])
    rotation = float(fig3["pathway_label_rotation"])
    edge = fig3["pathway_cell_edge"]
    cell_cap_mm = float(fig3["pathway_cell_mm"])
    gap_min_mm = float(fig3["pathway_block_gap_min_mm"])
    block_gap_mm = float(fig3["pathway_block_gap_mm"])
    bar_gap_mm = float(fig3["pathway_colorbar_gap_mm"])
    bar_w_mm, bar_h_mm = (float(v) for v in fig3["colorbar_mm"])
    bar_tick_pad_pt = float(fig3["colorbar_tick_pad"])
    bar_ticks = config["plot_styles"]["colorbar_ticks"]
    # The pathway names are the x-axis tick labels and the category headings are panel
    # titles, so both take the figure-wide sizes rather than a size of their own.
    # pathway_annot is checked ONLY when it is ordinary text. Matched to Figure 5 it is
    # 7.05 pt, a hair over Nature's 7 pt ceiling, and Figure 5 already holds that as a
    # considered exception rather than passing its star to this guard: the asterisk is a
    # symbol, not body text, and DejaVu's glyph fills about 45% of its em, so its actual
    # ink is nearer 3.2 pt. Everything else on this panel stays inside the band.
    checked = dict(title=title_pt, axis_label=axis_pt, tick_label=tick_pt)
    if annot_family is None:
        checked["pathway_annot"] = annot_pt
    style.assert_text_sizes(**checked)

    print("Fetching Figure 3 pathway inputs from the data deposit ...")
    inputs = fetch("figure_intermediates", files=["Fig3_15_jk_pathway_scores.csv"])
    scores = pd.read_csv(str(inputs["Fig3_15_jk_pathway_scores.csv"]))
    # Column order within a category, as published: by the pathway's best P across the
    # three feature sets. The deposit is written in this order; re-sorting here keeps the
    # panel independent of the table's row order.
    scores["_min_p"] = scores[["mtif_p", "dtif_p", "lv_p"]].min(axis=1)
    scores = scores.sort_values(["_min_p", "pathway"]).drop(columns="_min_p")

    if local is not None and hasattr(local, "prepare"):
        scores = local.prepare(scores, args)

    # The published ramp: white to RGB(115, 15, 240), from config. The glow rendering
    # swaps it for a ramp that keeps that same violet as its middle anchor.
    if glow is None:
        cmap = sns.light_palette(
            tuple(c / 255.0 for c in fig3["pathway_cmap_rgb"]), as_cmap=True)
    else:
        # Settle the map BEFORE fig.colorbar() below. Repointing a mappable's cmap
        # afterwards makes the bar rebuild its solids, and the rebuilt QuadMesh comes back
        # with matplotlib's rasterized default -- which puts a bitmap in the exported PDF
        # and breaks the vector-only requirement (caught by pdfimages -list, 7 Sep 2026).
        cmap = glow_cmap(glow["cmap_stops"])
        edge = dict(edge, color=str(glow["cell_edge_color"]))

    width_mm = style.display_item_width_mm
    height_mm = float(fig3["strip_height_mm"]["pathway"])
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))

    # --- panel k -----------------------------------------------------------------------
    present = [c for c in CATEGORY_ORDER if (scores["category"] == c).any()]
    blocks = [scores[scores["category"] == c] for c in present]

    # Axes are placed by hand, in millimetres, because the published geometry is a set of
    # physical sizes (square cells, a bar as tall as the heat map) and the rotated names
    # are the thing that has to fit. The heat map hangs from a fixed top edge; the names
    # fall below it and the strip height in config must be tall enough for them -- the
    # ink check at the end enforces that.
    # The y-label reaches x = 0 and the colour-bar label reaches the right edge, so this
    # row is what makes the assembled figure exactly the display-item width.
    K_LEFT_MM, K_RIGHT_MM = 0.075 * width_mm, 0.956 * width_mm
    # Room for the category headings, in a one-element list because it is re-derived once
    # the headings have been wrapped to their blocks and a two-line heading needs more.
    heat_top = [height_mm - HEADING_BAND_MM]
    bar_left = K_RIGHT_MM - bar_w_mm
    n_cols = sum(len(b) for b in blocks)

    def geometry(left_mm, bar_left_mm):
        """Square cells at the configured size, blocks at the configured gap.

        The blocks are laid out left to right at `pathway_block_gap_mm` and the width
        they do not use becomes the clearance before the colour bar -- so the two
        distances set in config are independent, and the cell is the published size
        unless 23 columns plus the gaps genuinely do not fit, in which case it shrinks
        (never below what `pathway_block_gap_min_mm` protects).
        """
        span = bar_left_mm - left_mm - bar_gap_mm
        gaps = (len(blocks) - 1) * block_gap_mm
        cell = min(cell_cap_mm, (span - gaps) / n_cols)
        if cell <= 0:
            cell = (span - (len(blocks) - 1) * gap_min_mm) / n_cols
        if cell <= 0:
            raise SystemExit("panel k: no width left for the heat map cells")
        widths = [cell * len(b) for b in blocks]
        return cell, cell * len(ROWS), widths, block_gap_mm

    def axes_mm(left, bottom, w, h):
        return fig.add_axes([left / width_mm, bottom / height_mm,
                             w / width_mm, h / height_mm])

    def place(left_mm, bar_left_mm=bar_left, bottom_mm=None):
        """Lay the blocks and the colour bar out for this pair of side edges."""
        cell, heat_h, widths, gap = geometry(left_mm, bar_left_mm)
        bottom = heat_top[0] - heat_h if bottom_mm is None else bottom_mm
        x = left_mm
        for ax, w in zip(axes, widths):
            ax.set_position([x / width_mm, bottom / height_mm,
                             w / width_mm, heat_h / height_mm])
            x += w + gap
        bar_h = min(bar_h_mm, heat_h)   # the heat map's height, capped at the paper-wide bar
        cax.set_position([bar_left_mm / width_mm,
                          (bottom + (heat_h - bar_h) / 2.0) / height_mm,
                          bar_w_mm / width_mm, bar_h / height_mm])
        return cell, heat_h, gap

    values = scores[[c for _, c, _ in ROWS]].apply(lambda s: -np.log10(s))
    # The scale ends on the next whole unit above the data (13 for a maximum of 12.19),
    # so the bar's top label is a round number.
    data_max = float(np.nanmax(values.values))
    norm = Normalize(vmin=0.0, vmax=float(np.ceil(data_max)))
    if glow is not None and glow.get("core_at_data_max"):
        # Squeeze the ramp so its top stop lands on the strongest cell rather than on the
        # rounded end of the axis; above that it stays the top colour. Rebuilt here
        # because it needs the data, which glow_cmap does not see.
        top = data_max / norm.vmax
        squeezed = [[float(pos) * top, str(color)]
                    for pos, color in glow["cmap_stops"]]
        squeezed.append([1.0, str(glow["cmap_stops"][-1][1])])
        cmap = glow_cmap(squeezed)
        print("  glow: ramp compressed to [0, {:.3f}] so -log10 P = {:.2f} is exactly the "
              "top colour; the bar still ends on {:g}".format(
                  top, data_max, norm.vmax))

    annot_offset_pt = annot_ink_offset_pt(annot_pt, annot_family)
    marks = []         # glow only: (axes, x, y, mark), drawn as outlines at the end
    mesh = None
    axes = []
    # Built BEFORE the drawing loop, because the winning cell of each feature set has to be
    # known while the cells are drawn -- it is what decides which cell keeps the dark grid
    # line and which takes its own colour. The bloom then reuses the same answer at the end,
    # where it needs the final cell size.
    matrices = [np.vstack([-np.log10(block[col].values) for _, col, _ in ROWS])
                for block in blocks]
    # Created at the starting geometry; place() sets the real boxes once the ink of the
    # rotated names is known, and re-derives the cell size for the width it then has.
    best_cells = (best_cell_per_row(matrices, glow)
                  if glow is not None and glow.get("glow_best_per_feature_set")
                  else None)
    cell0, heat_h0, block_w0, _ = geometry(K_LEFT_MM, bar_left)
    bar_h0 = min(bar_h_mm, heat_h0)
    cax = axes_mm(bar_left, heat_top[0] - heat_h0 + (heat_h0 - bar_h0) / 2.0,
                  bar_w_mm, bar_h0)
    for idx, (category, block) in enumerate(zip(present, blocks)):
        ax = axes_mm(K_LEFT_MM, heat_top[0] - heat_h0, block_w0[idx], heat_h0)
        axes.append(ax)
        matrix = matrices[idx]
        winners_here = {} if best_cells is None else best_cells.get(idx, {})
        # pcolormesh, NOT imshow. imshow embeds the heatmap as a BITMAP -- verified,
        # `pdfimages -list` listed a raster in the exported PDF -- and Nature requires the
        # figure to be vector throughout. pcolormesh draws one filled quad per cell; the
        # edge colour is the published between-cell grid.
        ax.set_xlim(0, matrix.shape[1])
        ax.set_ylim(matrix.shape[0], 0)          # row 0 (mTIFs) at the top, as published
        mesh = ax.pcolormesh(
            np.arange(matrix.shape[1] + 1), np.arange(matrix.shape[0] + 1),
            matrix, cmap=cmap, norm=norm, shading="flat",
            edgecolors=edge["color"], linewidth=float(edge["linewidth"]),
            rasterized=False,
        )

        for r, (_, col, thr_col) in enumerate(ROWS):
            for c_i, (p, thr) in enumerate(zip(block[col].values, block[thr_col].values)):
                mark = stars(p, thr)
                if (winners_here.get(r, (None, None))[0] == c_i and glow is not None
                        and glow.get("winner_edge_is_fill")):
                    # Overpaint this cell's perimeter in its own fill colour. A shared
                    # boundary carries two coincident strokes -- this cell's and its
                    # neighbour's -- so the repaint is winner_edge_width_mult times wider
                    # than the mesh line, or the neighbour's half stays visible as a dark
                    # fringe. zorder keeps it under the glow layer (2.5), so light still
                    # passes over it like anywhere else.
                    ax.add_patch(Rectangle(
                        (c_i, r), 1.0, 1.0, facecolor="none",
                        edgecolor=cmap(norm(-np.log10(p))),
                        linewidth=(float(edge["linewidth"])
                                   * float(glow["winner_edge_width_mult"])),
                        zorder=2.0))
                if mark and glow is not None and annot_family is not None:
                    # Outline route, used ONLY when matching Figure 5's DejaVu star. Drawn
                    # at the end, once the final cell size is known -- a path lives in data
                    # units, so it needs the millimetre scale.
                    marks.append((ax, c_i + 0.5, r + 0.5, mark))
                elif mark:
                    lum = relative_luminance(cmap(norm(-np.log10(p))))
                    # Published: seaborn's dark-or-white-by-luminance rule. Glow: one
                    # silver for every cell, which makes them read as marks rather than as
                    # part of the heat map.
                    star = (dict(color=ANNOT_DARK if lum > ANNOT_LUMINANCE_CUT else "w")
                            if glow is None else dict(
                                color=str(glow["star_color"]),
                                path_effects=([patheffects.withStroke(
                                    linewidth=float(glow["star_edge_pt"]),
                                    foreground=str(glow["star_edge_color"]))]
                                    if float(glow["star_edge_pt"]) > 0 else [])))
                    # Anchored on the baseline and pushed down by half the glyph's ink
                    # height, in points, so the asterisk is centred on the cell whatever
                    # the cell size.
                    ax.text(c_i + 0.5, r + 0.5, mark, ha="center", va="baseline",
                            fontsize=annot_pt,
                            transform=ax.transData + ScaledTranslation(
                                0.0, -annot_offset_pt / 72.0, fig.dpi_scale_trans),
                            **star)

        ax.set_xticks(np.arange(len(block)) + 0.5)
        # Rotation, alignment and the default rotation mode as published.
        ax.set_xticklabels(block["pathway_label"], rotation=rotation,
                           ha="right", va="top", fontsize=tick_pt)
        # Set for real once the block widths are final -- see the wrap step below.
        ax.set_title(heading(category), fontsize=title_pt, pad=TITLE_PAD_PT)
        if idx == 0:
            ax.set_yticks(np.arange(len(ROWS)) + 0.5)
            ax.set_yticklabels([name for name, _, _ in ROWS], fontsize=tick_pt)
            ax.set_ylabel("Feature set", fontsize=axis_pt)
        else:
            ax.set_yticks([])
        # The marks are the panel's only axis furniture -- see below -- so they carry the
        # display item's shared weight and length from config rather than matplotlib's
        # 0.8 x 3.5 pt defaults, which made them the heaviest lines in Figure 3.
        ax.tick_params(labelsize=tick_pt, width=style.axis_linewidth_pt,
                       length=style.axis_tick_length_pt)
        # seaborn's heatmap despines all four sides; the tick marks stay.
        for spine in ax.spines.values():
            spine.set_visible(False)

    bar = fig.colorbar(mesh, cax=cax)
    # matplotlib rasterises the colour bar's QuadMesh by default, to keep a continuous
    # gradient small. That put a bitmap in the exported PDF -- caught by `pdfimages -list`,
    # and a violation of the vector-only requirement. Turning it off costs one quad per
    # colour step and nothing visible.
    bar.solids.set_rasterized(False)
    bar.set_label(r"$-\log_{10}(P)$", fontsize=axis_pt,
                  labelpad=float(fig3["pathway_colorbar_label_pad_pt"]))
    # Paper-wide colour-bar rule (the colour-bar rule in 05_figures/README.md): the
    # box's outline is outline_pt wide and OFF at 0, the borderless design (5 Sep 2026);
    # labels at the two ENDS of the data range, as in Figures 2 and 4 (the published bar's
    # ticks every 2 units left the top mark short of the bar's end); the marks are NOT
    # matplotlib's but thinner, longer and flush with the drawn edge (draw_colorbar_ticks
    # insets the end marks), on the label side. The bar never changes size after this
    # point, only position, so the marks can be drawn now.
    bar.outline.set_visible(float(bar_ticks["outline_pt"]) > 0)
    bar.outline.set_linewidth(float(bar_ticks["outline_pt"]))
    bar.outline.set_edgecolor("black")
    bar.set_ticks([norm.vmin, norm.vmax])
    bar.set_ticklabels(["{:g}".format(norm.vmin), "{:g}".format(norm.vmax)])
    mark_side = str(bar_ticks["vertical_side"])
    bar.ax.tick_params(labelsize=tick_pt, length=0,
                       pad=bar_tick_pad_pt + (float(bar_ticks["length_pt"])
                                              if mark_side == "right" else 0.0))
    draw_colorbar_ticks(bar.ax, [0.0, 1.0], mark_side,
                        float(bar_ticks["length_pt"]), float(bar_ticks["width_pt"]),
                        outline_pt=float(bar_ticks["outline_pt"]))

    # Panel letter last, in figure coordinates. Its x is the figure-wide left-column
    # position (config panel_letter_x_mm) so k, a, d and g line up down the page.
    fig.text(float(fig3["panel_letter_x_mm"]) / width_mm, 0.985, "k",
             fontsize=style.panel_label_fontsize,
             fontweight=style.panel_label_fontweight,
             ha="left", va="top")

    # The strip must hold all of its ink: the rotated names are the tallest element and
    # only a render can measure them. Measure, then slide the whole panel up by whatever
    # the names overrun, spending the slack above the headings; refuse if that slack is
    # gone, because then the strip height in config is genuinely too small. A strip
    # taller than its ink is trimmed by the assembler.
    def ink_mm():
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())    # figure inches
        return box.x0 * 25.4, box.x1 * 25.4, box.y0 * 25.4, box.y1 * 25.4

    # Horizontal first: the rotated names of the first block reach left of it by most
    # of their length, so the blocks start wherever that overhang ends (the published
    # composite spaced them the same way) and the colour-bar label sets the right edge.
    # This has to ITERATE: moving a side edge changes the width available to the cells,
    # the cells change the block widths, and the block widths move the names again. Two
    # or three rounds settle it; each round is a fraction of a millimetre.
    EDGE_MM = 0.15

    def wrap_titles(widths):
        """Break each heading over its own block instead of letting it overhang.

        A heading wider than its block used to run out over the gaps beside it; with the
        cells this small, "Vascular remodeling & endothelial biology" is wider than its
        nine columns. Returns the tallest heading, in lines.
        """
        lines = 1
        for ax, category, w in zip(axes, present, widths):
            wrapped = wrap_to_width(fig, heading(category), w, title_pt)
            lines = max(lines, len(wrapped))
            ax.set_title("\n".join(wrapped), fontsize=title_pt, pad=TITLE_PAD_PT)
        return lines

    left_mm, bar_left_mm = K_LEFT_MM, bar_left
    for _ in range(5):
        place(left_mm, bar_left_mm)
        left, right, bottom, top = ink_mm()
        if left > EDGE_MM - 0.01 and right < width_mm - EDGE_MM + 0.01:
            break
        left_mm += EDGE_MM - left
        bar_left_mm -= max(0.0, right - (width_mm - EDGE_MM))
    else:
        raise SystemExit("panel k: the side edges did not settle")
    # matplotlib's tight bbox is the box of the ROTATED text rectangles, not the ink, so
    # the names' empty corners leave most of a millimetre of white at the sides. Save,
    # measure the real ink with ghostscript (src/display_item.ink_margins_mm, as Fig 4
    # does) and close the gap, so the strip reaches both side edges as the display-item
    # rule requires -- again iterating, for the same reason.
    for _ in range(4):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "k_probe.pdf"
            fig.savefig(str(probe), transparent=True)
            gs_left, _, gs_right, _ = ink_margins_mm(str(probe))
        if max(gs_left, gs_right) < 0.15:
            break
        left_mm -= gs_left - 0.05
        bar_left_mm += gs_right - 0.05
        place(left_mm, bar_left_mm)
    cell_mm, heat_h, gap_mm = place(left_mm, bar_left_mm)
    # Now that the block widths are final, wrap the headings to them and give the band
    # the height the tallest one needs, pushing the heat map down by the difference.
    n_lines = wrap_titles(geometry(left_mm, bar_left_mm)[2])
    band_mm = max(HEADING_BAND_MM,
                  n_lines * HEADING_LINE_MM + TITLE_PAD_PT / 72.0 * 25.4)
    heat_top[0] = height_mm - band_mm
    cell_mm, heat_h, gap_mm = place(left_mm, bar_left_mm)
    left, right, bottom, top = ink_mm()
    if bottom < 0:
        shift = -bottom + 0.2
        if top + shift > height_mm:
            raise SystemExit(
                "panel k: the pathway names overrun the strip by {:.1f} mm and there is "
                "no room above the headings -- raise "
                "plot_styles.figures.fig3.strip_height_mm.pathway to {:.0f}".format(
                    -bottom, np.ceil(height_mm - bottom + 0.5)))
        for ax in axes + [cax]:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + shift / height_mm, pos.width, pos.height])
        left, right, bottom, top = ink_mm()
    bar_clear_mm = (bar_left_mm - left_mm
                    - sum(geometry(left_mm, bar_left_mm)[2])
                    - (len(blocks) - 1) * gap_mm)
    print("  panel k cells {:.2f} mm square (cap {:.2f}), heat map {:.1f} mm tall, "
          "block gaps {:.1f} mm, {:.1f} mm before the colour bar".format(
              cell_mm, cell_cap_mm, heat_h, gap_mm, bar_clear_mm))
    print("  ink spans {:.1f}-{:.1f} mm of the {:.0f} mm strip, {:.1f}-{:.1f} mm of its "
          "{:.0f} mm width".format(bottom, top, height_mm, left, right, width_mm))

    # The glow goes on LAST, after every placement iteration: the halo would otherwise
    # enlarge the tight bbox the side-edge loop measures and push the blocks inward on
    # each round. The bloom is in cell units and only its stroke weight needs the final
    # cell size, so it waits here too.
    if glow is not None:
        if glow["bloom_render"] == "blur":
            lit = sum(draw_cell_bloom_blur(ax, matrix, norm, glow)
                      for ax, matrix in zip(axes, matrices))
        else:
            lit = sum(draw_cell_bloom(
                          ax, matrix, norm, cell_mm, glow,
                          None if best_cells is None else best_cells.get(index, {}))
                      for index, (ax, matrix) in enumerate(zip(axes, matrices)))
        floor = float(glow["bloom_start"])
        if best_cells:
            spans = glow.get("bloom_spill_cells_by_rank") or []
            for index, rows in sorted(best_cells.items()):
                for row, (col, rank) in sorted(rows.items(), key=lambda kv: kv[1][1]):
                    print("    rank {}: {:<6s} {:<44s} -log10 P = {:5.2f}  glow reach "
                          "{:.4f} cell".format(
                              rank, ROWS[row][0], present[index][:44],
                              float(matrices[index][row, col]),
                              float(spans[min(rank, len(spans) - 1)]) if spans
                              else float(glow["bloom_spill_cells"])))
        # The asterisks, as Figure 5 draws them: DejaVu outlines, no font embedded.
        star_scale = 1.0 / (cell_mm / 25.4 * 72.0)          # points -> cell widths
        star_rgb = str(glow["star_color"])
        cache = {}
        for ax_m, x, y, mark in marks:
            if mark not in cache:
                cache[mark] = star_ink_path(mark, annot_pt, annot_family)
            verts, codes = cache[mark]
            ax_m.add_patch(PathPatch(
                MplPath(verts * star_scale + np.array([x, y]), codes),
                facecolor=star_rgb, edgecolor="none", linewidth=0.0, zorder=4))
        if marks:
            print("    asterisks: {} marks as {} outlines at {:.2f} pt (Figure 5's rule), "
                  "no font embedded".format(len(marks), annot_family, annot_pt))
        else:
            print("    asterisks: panel k's own style, {:.2f} pt text".format(annot_pt))
        scope = ("the strongest cell in each feature set"
                 if glow.get("glow_best_per_feature_set")
                 else "every cell above {:g} of the scale".format(floor))
        print("  glow: {} of {} cells glow -- {}, i.e. -log10 P >= {:.2f}; the other {} "
              "throw no light at all [{}]".format(
                  lit, len(scores) * len(ROWS), scope, floor * norm.vmax,
                  len(scores) * len(ROWS) - lit,
                  "distance-transform glow, RASTER layer" if glow["bloom_render"] == "blur"
                  else "vector rings"))
        if glow["halo_enabled"]:
            _, block_left_mm, min_gap_mm = draw_block_halo(
                fig, axes, width_mm, height_mm, glow)
            flipped = light_row_labels(fig, axes, block_left_mm, glow)
            reach = 2.0 * halo_visible_reach_mm(glow)
            print("    one halo per block ({:.1f} mm reach, {} coats), row labels white: "
                  "{}".format(float(glow["halo_spread_mm"]), int(glow["halo_rings"]),
                              ", ".join(flipped) or "none"))
            print("    canvas between categories: narrowest block gap {:.1f} mm vs "
                  "{:.1f} mm of VISIBLE halo from both sides -- {}".format(
                      min_gap_mm, reach,
                      "stays white" if min_gap_mm > reach else "THE HALOS MEET"))

    # One output location. It was split while the glow was a variant that had to leave the
    # published strip untouched; now that the glow IS the strip, a second directory would
    # only be a way to stage the wrong file.
    out_dir = figure_dir("main_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": it would change the locked strip width.
    #
    # The published strip is saved TRANSPARENT, which is right for something the assembler
    # stacks onto a page. The glow rendering is not: it is looked at on its own, and a
    # transparent ground is shown as BLACK by most image viewers -- which, next to a dark
    # violet halo, reads as a figure with a black background. So it gets a real white ground written
    # into the file. The default keeps transparent=True untouched.
    # The PNG carries a real white ground, because it is looked at on its own and most
    # viewers paint transparency as black. The PDF must NOT: it is a strip the assembler
    # stacks, and fig3.py trims the finished figure to its ink, so an opaque page would
    # extend the ink to the strip's full height and change the assembled figure's size
    # (measured: it would cost the 1.78 mm bottom trim). Every other strip is transparent
    # for the same reason.
    pdf_page = dict(transparent=True)
    png_page = (dict(transparent=True) if glow is None
                else dict(transparent=False, facecolor="white", edgecolor="none"))
    png_dpi = int(fig3["hi_dpi"]) if glow is None else int(glow["png_dpi"])
    # dpi matters for the PDF TOO once the glow layer is a raster: savefig's dpi is what
    # sets the resolution of rasterised elements inside a vector page. Without it the blur
    # went into the PDF at matplotlib's default 100 dpi -- verified with pdfimages -list,
    # which reported the layer at 100 ppi (7 Sep 2026). The published strip has no raster
    # at all, so it is unaffected either way and keeps its original call.
    if glow is None:
        fig.savefig(str(out_dir / "fig3_panel_4_k.pdf"), **pdf_page)
    else:
        fig.savefig(str(out_dir / "fig3_panel_4_k.pdf"), dpi=png_dpi, **pdf_page)
    fig.savefig(str(out_dir / "fig3_panel_4_k.png"), dpi=png_dpi, **png_page)
    fig.savefig(str(out_dir / "fig3_panel_4_k__docs.png"),
                dpi=int(fig3["docs_dpi"]), **png_page)
    plt.close(fig)

    print("  panel k: {} pathways in {} categories ({})".format(
        len(scores), len(present),
        ", ".join("{} {}".format(c, len(b)) for c, b in zip(present, blocks))))
    print("Wrote Figure 3 pathway strip to {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
