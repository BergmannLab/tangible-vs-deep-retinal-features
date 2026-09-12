from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import colorsys

import matplotlib.colors as mcolors
import yaml


@dataclass(frozen=True)
class PlotStyle:
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.4
    grid_color: str = "#9A9A9A"
    grid_alpha: float = 0.7
    legend_fontsize: int = 8
    legend_alpha: float = 1.0
    tick_count: int = 3
    title_default: bool = False
    joint_width_cm: float = 8.5
    joint_height_cm: float = 8.5
    joint_dpi: int = 200
    joint_layout: Dict[str, float] = field(default_factory=dict)
    title_fontsize: float = 10.0
    axis_label_fontsize: float = 9.0
    tick_label_fontsize: float = 8.0
    info_fontsize: float = 8.0
    panel_title_fontsize: float = 6.0
    footer_fontsize: float = 9.0
    joint_title_fontsize: float = 10.0
    joint_axis_label_fontsize: float = 9.0
    joint_tick_label_fontsize: float = 8.0
    main_fig_size_inch: Tuple[float, float] = (5.0, 4.0)
    main_fig_dpi: int = 200
    supp_overview_width_cm: float = 18.0
    supp_overview_height_ratio: float = 0.75
    supp_overview_min_height_cm: float = 10.0
    supp_per_trait_width_cm: float = 18.0
    supp_per_trait_height_ratio: float = 1.1
    supp_per_trait_min_height_cm: float = 13.0
    supp_per_trait_wspace: float = 0.3
    supp_per_trait_hspace: float = 0.3
    identity_linestyle: str = "--"
    identity_color: str = "red"
    identity_alpha: float = 0.5
    page_width_cm: float = 21.0
    page_height_cm: float = 29.7
    page_margin_cm: float = 1.5
    panel_label_fontsize: float = 8.0
    panel_label_fontweight: str = "bold"
    # Axis furniture -- spines, axis lines, tick marks -- shared by every figure. In
    # points as rendered; matplotlib's `linewidth` and `tick_params(width=)` are already
    # in points, so these go straight in. See plot_styles.axis_linewidth_pt in config.yaml
    # for why the R side needs a two-step conversion to reach the same number.
    axis_linewidth_pt: float = 0.5
    axis_tick_length_pt: float = 2.0
    # Nature Genetics artwork limits -- see plot_styles.natgen in config.yaml for the
    # two documents these come from and why the usable band is 5-7 pt.
    natgen_text_min_pt: float = 5.0
    natgen_text_max_pt: float = 7.0
    natgen_font_family: str = "Nimbus Sans"
    display_item_max_width_mm: float = 179.0
    display_item_max_height_mm: float = 260.0
    display_item_border_mm: float = 1.0     # white inside the page edge, every side

    @property
    def full_panel_width_cm(self) -> float:
        """Full printable panel width = page width minus left/right margins."""
        return self.page_width_cm - 2.0 * self.page_margin_cm

    @property
    def display_item_page_width_mm(self) -> float:
        """Page width of a main figure.

        The page geometry derives 180 mm, Nature's artwork guide names 180 mm as the
        2-column width and this manuscript's author checklist caps a display item at
        179 mm; Michael set 178 mm (11 Sep 2026). Take the smallest.
        """
        return min(self.full_panel_width_cm * 10.0, self.display_item_max_width_mm)

    @property
    def display_item_width_mm(self) -> float:
        """Width to author a main figure's INK at: the page less the border on each side."""
        return self.display_item_page_width_mm - 2.0 * self.display_item_border_mm

    def assert_text_sizes(self, **sizes: float) -> None:
        """Fail loudly if any text size leaves Nature's 5-7 pt band.

        Called by the figure scripts on every size they are about to use, so a stray
        hardcoded font size is caught at build time rather than by a production editor.

        Do NOT pass ``panel_label_fontsize`` here. Panel letters are a deliberate
        exception at 8 pt bold (Nature's own house style for a panel locator, as opposed
        to body text); keeping them out of this check by name is what makes that a
        considered exception rather than a silent leak.
        """
        bad = {
            name: pt for name, pt in sizes.items()
            if not (self.natgen_text_min_pt <= pt <= self.natgen_text_max_pt)
        }
        if bad:
            raise ValueError(
                "text size(s) outside Nature's {:g}-{:g} pt band: {}".format(
                    self.natgen_text_min_pt, self.natgen_text_max_pt,
                    ", ".join("{} = {:g} pt".format(k, v) for k, v in sorted(bad.items())),
                )
            )

    def apply_font(self) -> None:
        """Pin matplotlib to the one sans face Nature accepts, with editable PDF text.

        `pdf.fonttype = 42` embeds TrueType rather than converting glyphs to Type 3
        outlines, which is what makes the text in the exported PDF selectable and
        editable -- a hard requirement of the artwork guide.

        The mathtext block is not optional decoration. matplotlib renders anything
        between dollar signs with its own math font set, which defaults to DejaVu --
        so a single `$R^2$` axis label or a `$1 \\times 10^{-9}$` p-value (this work's
        p-value convention) silently embeds DejaVu Sans next to Liberation Sans, and
        the artwork guide asks for one sans face. Verified: Figure 2 embedded
        DejaVuSans and DejaVuSans-Oblique from one `$R^2$` before this was set.
        """
        import matplotlib

        _register_vendored_fonts()
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [self.natgen_font_family]
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["mathtext.fontset"] = "custom"
        matplotlib.rcParams["mathtext.rm"] = self.natgen_font_family
        matplotlib.rcParams["mathtext.it"] = self.natgen_font_family + ":italic"
        matplotlib.rcParams["mathtext.bf"] = self.natgen_font_family + ":bold"
        # cal/sf/tt default to cursive/sans/monospace; left unset they emit a
        # "Generic family 'cursive' not found" warning on every mathtext string.
        matplotlib.rcParams["mathtext.cal"] = self.natgen_font_family + ":italic"
        matplotlib.rcParams["mathtext.sf"] = self.natgen_font_family
        matplotlib.rcParams["mathtext.tt"] = self.natgen_font_family
        matplotlib.rcParams["mathtext.default"] = "it"


# --------------------------------------------------------------------------- fonts

# The typeface travels with the repository: 05_figures/lib/fonts holds the four Nimbus
# Sans faces. See that directory's README for the licence and for why R finds them a
# different way.
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "05_figures", "lib", "fonts")


def _register_vendored_fonts() -> None:
    """Teach matplotlib about the fonts in this repository.

    matplotlib does NOT use fontconfig. It keeps its own index of system directories, so
    on a machine where Nimbus Sans is not installed system-wide -- any Mac, out of the box
    -- pointing fontconfig at these files does nothing for the Python figures. They have to
    be handed to the font manager by path.
    """
    import matplotlib.font_manager as fm

    if not os.path.isdir(FONT_DIR):
        return
    known = {os.path.abspath(f.fname) for f in fm.fontManager.ttflist}
    for name in sorted(os.listdir(FONT_DIR)):
        if not name.lower().endswith((".otf", ".ttf")):
            continue
        path = os.path.abspath(os.path.join(FONT_DIR, name))
        if path not in known:
            try:
                fm.fontManager.addfont(path)
            except (RuntimeError, OSError):
                pass


def verify_font(family: Optional[str] = None) -> str:
    """Stop unless the font that will be drawn is the font that was asked for.

    THE FAILURE THIS EXISTS FOR IS SILENT. Asked for a family it cannot find, matplotlib
    substitutes DejaVu Sans and says nothing louder than a warning nobody reads in a build
    log. The figure still renders, still looks like a figure, and is wrong in a way that
    survives review: different metrics, heavier ink, and a second face embedded in a PDF
    the journal wants set in one. It happened on a Mac clone of this repository on 9 Sep
    2026, which is why the fonts are now vendored and why this check exists.
    """
    import matplotlib
    import matplotlib.font_manager as fm

    _register_vendored_fonts()
    wanted = family or matplotlib.rcParams["font.sans-serif"][0]
    resolved = fm.FontProperties(family=wanted).get_name()
    if resolved != wanted:
        raise SystemExit(
            "font {!r} is not available: matplotlib resolved {!r} instead, and would have "
            "drawn the figure in it.\n"
            "  The four faces ship with this repository in {}; if that directory is intact, "
            "matplotlib's font cache may be stale -- delete ~/.cache/matplotlib and rerun.\n"
            "  Nimbus Sans is metric-compatible with Helvetica, which is what the journal "
            "asks for; substituting anything else changes the figure.".format(
                wanted, resolved, os.path.relpath(FONT_DIR))
        )
    return resolved


def _get_style_dict(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return data.get("plot_styles", {}) if isinstance(data, dict) else {}


FEATURE_SET_COLOR_KEYS = ("mtifs", "dtifs", "lvs", "top_lvs")


def fold_marker_spec(config: Any) -> Dict[str, float]:
    """The paper-wide overlaid-fold-point spec, from config.yaml `plot_styles.fold_markers`.

    Figures 2b and 6 both draw the individual cross-validation folds on top of their bars,
    because the journal asks for the observations themselves when n is small. They must
    look like the same mark on the same page, so the geometry lives here rather than in
    either script. Absolute points: these dots are not scaled with anything.

    Raises on a missing block, for the reason in feature_set_colors().
    """
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    spec = (config.get("plot_styles", {}) or {}).get("fold_markers")
    if not spec:
        raise KeyError("config.yaml has no plot_styles.fold_markers block")
    return spec


def pop_color(color, saturation: float, lightness: float, gray_lightness: float):
    """A series colour, deepened until a dot of it reads against the bar it sits on.

    Saturation alone cannot do this for the feature-set palette: several of its colours are
    already fully saturated and two are achromatic greys, where saturation is undefined. So
    lightness does the work and saturation is taken up only where there is room for it --
    the same perceptual move for every colour, each dot still its bar's hue.
    """
    hue, light, sat = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    if sat < 0.05:                       # grey: only lightness is available
        return colorsys.hls_to_rgb(hue, light * gray_lightness, sat)
    return colorsys.hls_to_rgb(hue, light * lightness, min(1.0, sat * saturation))


def swarm_diameter(bar_thickness_pt, edge_lw_pt, diameter_pt, rows=2):
    """Dot diameter for a swarm confined to `rows` rows inside a bar of this thickness.

    Demanding that no two dots ever touch AND that every dot stay inside the bar makes the
    dots unreadable: five folds at nearly the same value have to stack five diameters deep,
    which on a 3.2 pt bar caps the dot near 0.6 pt -- invisible at A4. Fixing the depth at
    two rows instead lets the dot be the full half-bar, and the only cost is that folds
    which genuinely coincide draw over one another, which is an honest thing for coincident
    values to do. Staying inside the bar is never traded away.
    """
    return min(diameter_pt, max(0.0, (bar_thickness_pt - edge_lw_pt) / float(rows)))


def swarm_offsets(x_pt, diameter_pt, room_pt=None):
    """Vertical offsets, in points, that stop overlapping points overprinting each other.

    Greedy beeswarm: take the points left to right and give each the smallest vertical
    displacement at which it clears everything already placed, then centre the result on
    the bar line. A handful of points per bar, so the naive O(n^2) test is free.

    `x_pt` must already be in points -- collision is a question about marks on paper, so
    the caller converts from data units once the axes are placed.

    `room_pt` bounds |offset|, so the swarm cannot leave its bar. Within that bound the
    placement is still best-effort: if no clear slot exists, the point takes the position
    that leaves the most space rather than being pushed outside.
    """
    import numpy as np

    offsets = np.zeros(len(x_pt))
    placed = []
    # A fine ladder of candidate offsets: the 2-D clearance test below is what actually
    # guarantees no overprint, so a smaller step just lets a point settle closer in, which
    # makes the whole swarm shallower and leaves room for a larger dot.
    step = diameter_pt * 0.35
    for i in np.argsort(x_pt):
        x = x_pt[i]
        candidates = [0.0]
        for k in range(1, 16):
            candidates += [k * step, -k * step]
        if room_pt is not None:
            candidates = [y for y in candidates if abs(y) <= room_pt + 1e-9] or [0.0]

        def clearance(y):
            if not placed:
                return float("inf")
            return min((x - px) ** 2 + (y - py) ** 2 for px, py in placed)

        for y in candidates:
            if clearance(y) >= diameter_pt ** 2 - 1e-9:
                offsets[i] = y
                break
        else:
            # bounded and crowded: take the roomiest slot instead of leaving the bar
            offsets[i] = max(candidates, key=clearance)
        placed.append((x, offsets[i]))
    return offsets - (offsets.max() + offsets.min()) / 2.0


def feature_set_colors(config: Any) -> Dict[str, str]:
    """The paper's feature-set palette, read from config.yaml's `colors:` block.

    `config` is the loaded config mapping, its `colors` sub-mapping, or a path to
    config.yaml -- whichever the caller already has in hand.

    This raises on a missing key instead of returning a default. Scripts used to write
    `colors.get("dtifs", "#FF7F00")`, which agrees with config right up to the moment it
    does not: rename or drop the key and the figure keeps rendering, silently, in a colour
    no longer in the palette. A KeyError here stops the run, which is the only outcome that
    cannot ship a wrong figure (4 Sep 2026).
    """
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError("feature_set_colors() wants config.yaml, its mapping, or its "
                        "`colors` block; got {!r}".format(type(config).__name__))
    colors = config.get("colors", config)
    missing = [key for key in FEATURE_SET_COLOR_KEYS if key not in colors]
    if missing:
        raise KeyError("config.yaml `colors:` is missing {} -- the feature-set palette is "
                       "the single source of truth and has no defaults".format(missing))
    return {key: str(colors[key]) for key in FEATURE_SET_COLOR_KEYS}


def load_plot_style(config_path: str) -> PlotStyle:
    style_dict = _get_style_dict(config_path)
    if not style_dict:
        return PlotStyle()
    fonts = style_dict.get("fonts", {})
    figures = style_dict.get("figures", {})
    lines = style_dict.get("lines", {})
    def _get_list(key: str, default: Sequence[float]) -> Tuple[float, float]:
        value = figures.get(key)
        if isinstance(value, Sequence) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except Exception:
                pass
        return float(default[0]), float(default[1])

    joint_width = float(figures.get("joint_width_cm", 8.5))
    joint_height = float(figures.get("joint_height_cm", joint_width))
    joint_size = figures.get("joint_size_cm")
    if isinstance(joint_size, Sequence) and len(joint_size) >= 2:
        try:
            joint_width = float(joint_size[0])
            joint_height = float(joint_size[1])
        except Exception:
            pass
    joint_dpi = int(
        figures.get(
            "joint_dpi",
            figures.get("main_fig_dpi", 200),
        )
    )
    page = figures.get("page", {}) if isinstance(figures.get("page", {}), dict) else {}
    panel_label = style_dict.get("panel_label", {})
    if not isinstance(panel_label, dict):
        panel_label = {}
    natgen = style_dict.get("natgen", {})
    if not isinstance(natgen, dict):
        natgen = {}
    item_max = natgen.get("display_item_max_mm", {})
    if not isinstance(item_max, dict):
        item_max = {}
    raw_layout = figures.get("joint_layout", {})
    joint_layout: Dict[str, float] = {}
    if isinstance(raw_layout, dict):
        for key in ("left", "right", "bottom", "top"):
            if key in raw_layout:
                try:
                    joint_layout[key] = float(raw_layout[key])
                except Exception:
                    continue

    return PlotStyle(
        grid_linestyle=str(style_dict.get("grid", {}).get("linestyle", "--")),
        grid_linewidth=float(style_dict.get("grid", {}).get("linewidth", 0.4)),
        grid_color=str(style_dict.get("grid", {}).get("color", "#9A9A9A")),
        grid_alpha=float(style_dict.get("grid", {}).get("alpha", 0.7)),
        legend_fontsize=int(style_dict.get("legend", {}).get("fontsize", 8)),
        legend_alpha=float(style_dict.get("legend", {}).get("alpha", 1.0)),
        tick_count=int(style_dict.get("ticks", {}).get("count", 3)),
        title_default=bool(style_dict.get("titles", {}).get("default_show", False)),
        joint_width_cm=joint_width,
        joint_height_cm=joint_height,
        joint_dpi=joint_dpi,
        joint_layout=joint_layout,
        title_fontsize=float(fonts.get("title", 10)),
        axis_label_fontsize=float(fonts.get("axis_label", 9)),
        tick_label_fontsize=float(fonts.get("tick_label", 8)),
        info_fontsize=float(fonts.get("info_box", 8)),
        panel_title_fontsize=float(fonts.get("panel_title", 6)),
        footer_fontsize=float(fonts.get("footer", 9)),
        joint_title_fontsize=float(fonts.get("joint_title", 10)),
        joint_axis_label_fontsize=float(fonts.get("joint_axis_label", 9)),
        joint_tick_label_fontsize=float(fonts.get("joint_tick_label", 8)),
        main_fig_size_inch=_get_list("main_fig_size_inch", (5.0, 4.0)),
        main_fig_dpi=int(figures.get("main_fig_dpi", 200)),
        supp_overview_width_cm=float(figures.get("supp_overview_width_cm", 18.0)),
        supp_overview_height_ratio=float(figures.get("supp_overview_height_ratio", 0.75)),
        supp_overview_min_height_cm=float(figures.get("supp_overview_min_height_cm", 10.0)),
        supp_per_trait_width_cm=float(figures.get("supp_per_trait_width_cm", 18.0)),
        supp_per_trait_height_ratio=float(figures.get("supp_per_trait_height_ratio", 1.1)),
        supp_per_trait_min_height_cm=float(figures.get("supp_per_trait_min_height_cm", 13.0)),
        supp_per_trait_wspace=float(figures.get("supp_per_trait_wspace", 0.3)),
        supp_per_trait_hspace=float(figures.get("supp_per_trait_hspace", 0.3)),
        identity_linestyle=str(lines.get("identity_linestyle", "--")),
        identity_color=str(lines.get("identity_color", "red")),
        identity_alpha=float(lines.get("identity_alpha", 0.5)),
        page_width_cm=float(page.get("width_cm", 21.0)),
        page_height_cm=float(page.get("height_cm", 29.7)),
        page_margin_cm=float(page.get("margin_cm", 1.5)),
        panel_label_fontsize=float(panel_label.get("fontsize", 8.0)),
        panel_label_fontweight=str(panel_label.get("fontweight", "bold")),
        axis_linewidth_pt=float(style_dict.get("axis_linewidth_pt", 0.5)),
        axis_tick_length_pt=float(style_dict.get("axis_tick_length_pt", 2.0)),
        natgen_text_min_pt=float(natgen.get("text_min_pt", 5.0)),
        natgen_text_max_pt=float(natgen.get("text_max_pt", 7.0)),
        natgen_font_family=str(natgen.get("font_family", "Nimbus Sans")),
        display_item_max_width_mm=float(item_max.get("width", 179.0)),
        display_item_max_height_mm=float(item_max.get("height", 260.0)),
        display_item_border_mm=float(natgen.get("display_item_border_mm", 1.0)),
    )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base`` (nested dicts merged,
    scalars/lists replaced). Used to layer per-figure settings on shared defaults."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_supp_panel_cfg(config_path: str, *figure_keys: str) -> Dict[str, Any]:
    """Return the merged settings dict for a supplementary figure: the shared
    ``plot_styles.supp_defaults`` block overlaid, in order, with each
    ``plot_styles.<figure_key>`` block (nested dicts deep-merged).

    Lets every supplementary figure share one set of defaults (fonts, marker/error
    sizing, info-box style, gridlines) while overriding only what it needs. Passing
    several keys layers them, e.g. a family block then a figure-specific block:
    ``load_supp_panel_cfg(cfg, "sf_contra_pertif", "sf_contra_pertif_beta")`` merges
    supp_defaults -> sf_contra_pertif -> sf_contra_pertif_beta. Returns an empty dict
    if the config is unreadable.
    """
    style_dict = _get_style_dict(config_path)
    if not style_dict:
        return {}
    merged = style_dict.get("supp_defaults", {}) or {}
    if not isinstance(merged, dict):
        merged = {}
    for key in figure_keys:
        block = style_dict.get(key, {}) or {}
        if isinstance(block, dict):
            merged = _deep_merge(merged, block)
    return merged


def add_panel_labels(fig: Any, axes: Sequence[Any], letters: Sequence[str],
                     style: PlotStyle) -> None:
    """Place panel letters (a, b, c, ...) on a multi-panel figure with the GLOBAL
    panel-label font size/weight (``style.panel_label_*``), positioned so that each
    letter's x-center matches the x-center of that axis's y-axis label and its
    y-center matches the y-center of the panel title.

    Call this AFTER the final layout (e.g. after ``fig.subplots_adjust``) so the axis
    labels and titles are in their final positions. Shared by all multi-panel figures
    so the letters are identical across this work.
    """
    fig.canvas.draw()  # realize label/title window extents
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        ylab = ax.yaxis.get_label().get_window_extent(renderer=renderer)
        title = ax.title.get_window_extent(renderer=renderer)
        cx = 0.5 * (ylab.x0 + ylab.x1)   # x-center of the y-axis label
        cy = 0.5 * (title.y0 + title.y1)  # y-center of the panel title
        fx, fy = inv.transform((cx, cy))
        fig.text(fx, fy, letter, ha="center", va="center",
                 fontsize=style.panel_label_fontsize,
                 fontweight=style.panel_label_fontweight)


def add_panel_labels_corner(fig: Any, axes: Sequence[Any], letters: Sequence[str],
                            style: PlotStyle, dx_mm: float = 0.0,
                            dy_mm: float = 1.5) -> None:
    """Panel letters anchored to the top-left corner of each axis's bounding box.

    The sibling of ``add_panel_labels()``, for panels that carry neither a y-axis label
    nor a title for it to derive a position from -- a heatmap, an imported picture, a
    dendrogram. Size and weight still come from ``style.panel_label_*``, so letters stay
    identical across the work and only the anchor differs.

    ``dx_mm`` / ``dy_mm`` offset the letter from the corner (left and up, respectively),
    in millimetres of the printed figure rather than in axis fractions, so the gap does
    not change when a panel is resized.

    Call after the layout is final (see the layout note in the conventions in 05_figures/README.md).
    """
    fig_w_in, fig_h_in = fig.get_size_inches()
    dx = (dx_mm / 25.4) / fig_w_in
    dy = (dy_mm / 25.4) / fig_h_in
    for ax, letter in zip(axes, letters):
        box = ax.get_position()
        fig.text(box.x0 - dx, box.y1 + dy, letter, ha="left", va="bottom",
                 fontsize=style.panel_label_fontsize,
                 fontweight=style.panel_label_fontweight)


def draw_colorbar_ticks(cax: Any, fractions: Sequence[float], side: str,
                        length_pt: float, width_pt: float, color: str = "black",
                        outline_pt: float = 0.0) -> None:
    """Tick marks for a colour bar, drawn as lines flush with the bar's visual box.

    matplotlib's own tick marks are centred on the tick position, so a mark at the end of
    the ramp overhangs the bar by half its line width. These are inset so the outermost
    marks end exactly where the bar visibly ends: on the ramp's edge when the bar has no
    outline, on the outline's outer edge when it has one (`outline_pt`, the outline's
    line width -- an outline is itself centred on the box edge, so it absorbs that much of
    the mark's overhang; with outline and mark the same width the inset is nil and mark
    and outline are flush). `fractions` are positions along the bar (0 = start, 1 = end);
    `side` is "bottom" for a horizontal bar, "left" or "right" for a vertical one. Call
    after the layout is final -- the inset is computed from the axes' size on the page.
    Style from config.yaml -> plot_styles.colorbar_ticks.
    """
    from matplotlib.lines import Line2D

    fig = cax.figure
    fig_w_in, fig_h_in = fig.get_size_inches()
    box = cax.get_position()
    ax_w_in, ax_h_in = box.width * fig_w_in, box.height * fig_h_in
    half_in = max(0.0, width_pt - outline_pt) / 2.0 / 72.0
    length_in = length_pt / 72.0
    for fraction in fractions:
        if side == "bottom":
            inset = half_in / ax_w_in
            x = min(max(float(fraction), inset), 1.0 - inset)
            xs, ys = [x, x], [0.0, -length_in / ax_h_in]
        else:
            inset = half_in / ax_h_in
            y = min(max(float(fraction), inset), 1.0 - inset)
            run = length_in / ax_w_in
            xs = [0.0, -run] if side == "left" else [1.0, 1.0 + run]
            ys = [y, y]
        line = Line2D(xs, ys, transform=cax.transAxes, color=color, linewidth=width_pt,
                      solid_capstyle="butt", clip_on=False)
        cax.add_line(line)


def apply_joint_layout(fig: Any, style: PlotStyle) -> None:
    """
    Apply a shared joint-figure layout to the provided Matplotlib figure.
    Falls back to constrained layout when no layout overrides are defined.
    """
    layout = getattr(style, "joint_layout", None)
    if isinstance(layout, dict) and layout:
        adjust_kwargs = {
            key: float(value)
            for key, value in layout.items()
            if value is not None
        }
        if adjust_kwargs:
            try:
                fig.subplots_adjust(**adjust_kwargs)
                return
            except Exception:
                pass
    try:
        fig.set_constrained_layout(True)
    except Exception:
        pass
