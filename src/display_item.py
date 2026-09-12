"""Display-item rules, enforced rather than remembered.

A submitted figure has to satisfy four things at once, and each of them has bitten this
project at least once:

  1. the page is exactly the display-item width (178 mm), and no taller than the cap;
  2. the ink sits exactly plot_styles.natgen.display_item_border_mm (1 mm) inside every
     page edge -- no less, because a frame on the edge reads as cut off in a PDF viewer,
     and no more (Michael, 11 Sep 2026);
  3. every glyph is the one document face, embedded (no DejaVu leaking in via mathtext);
  4. it is vector throughout -- no rasterised panels, colour bars or heatmaps.

Rules 1 and 2 are a layout problem, not a cropping problem: type is set in points, so a
figure cannot simply be scaled to width without changing its type size. `render_to_width`
solves it instead -- it rebuilds the figure on a trial canvas until the *ink* lands on the
target width, then saves cropped to that ink.

`verify` checks a finished PDF from any source, matplotlib or R, and is what the staging
script runs so a non-conforming figure cannot be staged.

    from src.display_item import render_to_width, verify

    fig, ink = render_to_width(build, style)          # build(canvas_w_in) -> Figure
    save(fig, ink, figure_dir("main_figures"), dpi=500)
"""
from __future__ import annotations

import os
import re
import subprocess

MM = 25.4

# gs reports a bounding box rounded outward, and an antialiased hairline reads a fraction
# of a pixel wide, so "zero margin" is enforced to a quarter of a millimetre rather than
# to the bit. At 500 dpi that is 5 px; a real layout margin is never this small.
MARGIN_TOL_MM = 0.25
WIDTH_TOL_MM = 0.20
# "should be directly saved/scanned in at least 300 dpi resolution" -- artwork guide.
MIN_IMAGE_PPI = 300.0


def render_to_width(build, style, max_passes=5):
    """Rebuild `build(canvas_w_in)` until its ink is exactly the display-item width.

    Returns (figure, ink_bbox). The ink width is linear in the canvas width -- the axes
    scale with the canvas while the text does not -- so this converges in two passes and
    the rest are confirmation.
    """
    import matplotlib.pyplot as plt

    target_in = style.display_item_width_mm / MM
    canvas_in = target_in
    for _ in range(max_passes):
        fig = build(canvas_in)
        ink = fig.get_tightbbox(fig.canvas.get_renderer())
        if abs(ink.width - target_in) <= WIDTH_TOL_MM / MM:
            return fig, ink
        canvas_in *= target_in / ink.width
        plt.close(fig)
    raise RuntimeError(
        "could not converge on the display-item width in {} passes".format(max_passes))


def _style():
    """The paper's PlotStyle, from the repository's config.yaml."""
    import sys as _sys
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    if str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    from src.plot_style import load_plot_style
    return load_plot_style(str(root / "config.yaml"))


def save(fig, ink, out_stem, dpi, border_mm=None):
    """Write PDF + PNG cropped to the ink plus the white border, at the same physical size.

    `border_mm` defaults to plot_styles.natgen.display_item_border_mm.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    if border_mm is None:
        border_mm = _style().display_item_border_mm
    pad = border_mm / MM
    box = Bbox.from_extents(ink.x0 - pad, ink.y0 - pad, ink.x1 + pad, ink.y1 + pad)
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    paths = []
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": int(dpi)})):
        path = out_stem + suffix
        fig.savefig(path, bbox_inches=box, pad_inches=0, facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _ink_box_mm(pdf_path):
    """(x0, y0, x1, y1) of the drawn content, in mm, via ghostscript."""
    out = subprocess.run(
        ["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=bbox", pdf_path],
        capture_output=True, text=True).stderr
    # A coordinate can be negative: a hairline drawn on the crop boundary lands a
    # hundredth of a point outside it, and ghostscript reports that faithfully. Without
    # the sign in the pattern the match fails and the figure looks like a broken PDF
    # (Figure 5, 7 Sep 2026: x0 = -0.0089 pt).
    match = re.search(
        r"%%HiResBoundingBox:\s*(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)", out)
    if not match:
        raise RuntimeError("ghostscript reported no bounding box for {}".format(pdf_path))
    return [float(v) * MM / 72.0 for v in match.groups()]


def ink_margins_mm(pdf_path):
    """(left, bottom, right, top) white margin of a saved PDF, in mm, as ghostscript sees it.

    matplotlib's tight bbox is not the ink: for rotated text it is the box of the rotated
    rectangle, whose empty corners leave a fraction of a millimetre of white -- enough to
    fail the zero-margin rule. A figure script saves once, trims its bbox by what this
    reports, and saves again (see 05_figures/main/fig4_per_tif.py).
    """
    x0, y0, x1, y1 = _ink_box_mm(pdf_path)
    page_w, page_h = _page_size_mm(pdf_path)
    return x0, y0, page_w - x1, page_h - y1


def _page_size_mm(pdf_path):
    out = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True).stdout
    match = re.search(r"Page size:\s*([\d.]+) x ([\d.]+)", out)
    return float(match.group(1)) * MM / 72.0, float(match.group(2)) * MM / 72.0


def extractable_text(pdf_path):
    """The figure's text as a reader's PDF viewer would select it."""
    out = subprocess.run(["pdftotext", pdf_path, "-"],
                         capture_output=True, text=True).stdout
    return re.sub(r"\s+", " ", out).strip()


def verify(pdf_path, style, geometry=True, allow_images=False, expect_text=(), typeface=True):
    """Return a list of rule violations for a finished figure. Empty list = conforming.

    `geometry=False` checks only the font and vector rules -- for a strip or a single
    panel staged as a component, which is not a page and has no page geometry to meet.

    `typeface=False` drops the "must be the paper's font family" rule while keeping the
    embedding check -- for a supplementary figure, which is not a display item and does
    not owe the manuscript its type band. An unembedded font is still a defect anywhere.
    """
    problems = []
    page_w, page_h = _page_size_mm(pdf_path)

    if geometry:
        x0, y0, x1, y1 = _ink_box_mm(pdf_path)
        border = style.display_item_border_mm
        want_w = style.display_item_page_width_mm
        if abs(page_w - want_w) > WIDTH_TOL_MM:
            problems.append("page is {:.2f} mm wide, not the display-item width {:.0f} mm"
                            .format(page_w, want_w))
        if page_h > style.display_item_max_height_mm + WIDTH_TOL_MM:
            problems.append("page is {:.1f} mm tall, over the {:.0f} mm cap".format(
                page_h, style.display_item_max_height_mm))
        # The border is a floor as much as a target: under it by more than gs's rounding
        # fails outright, over it by more than MARGIN_TOL_MM fails too.
        for name, margin in (("left", x0), ("bottom", y0),
                             ("right", page_w - x1), ("top", page_h - y1)):
            if margin < border - 0.02 or margin > border + MARGIN_TOL_MM:
                problems.append("{:.2f} mm of white margin at the {}, not {:g} mm".format(
                    margin, name, border))

    fonts = subprocess.run(["pdffonts", pdf_path], capture_output=True, text=True).stdout
    families, unembedded = set(), []
    for line in fonts.splitlines()[2:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        families.add(re.sub(r"^[A-Z]{6}\+", "", parts[0]).split("-")[0])
        if parts[-4] != "yes":
            unembedded.append(parts[0])
    expected = style.natgen_font_family.replace(" ", "")
    stray = sorted(f for f in families if f != expected)
    if stray and typeface:
        problems.append("font(s) other than {}: {}".format(expected, ", ".join(stray)))
    if unembedded:
        problems.append("font(s) not embedded: {}".format(", ".join(unembedded)))

    # An embedded font is NOT proof that the text is text. Glyphs converted to outlines --
    # which rsvg does to any SVG it touches, and which silently cost Figure 3c its editable
    # labels once already -- leave the other panels' fonts in place and pdfimages clean.
    # So check that named strings really come back out of the page.
    if expect_text:
        text = extractable_text(pdf_path)
        missing = [s for s in expect_text if s not in text]
        if missing:
            problems.append(
                "text not extractable (converted to outlines?): {}".format(
                    "; ".join(repr(s) for s in missing)))

    # Images are not banned -- the artwork guide asks for photographs and complex
    # illustrations as bitmaps "at least 300 dpi resolution", with the vector elements laid
    # over them. What is banned is rasterising line art (a heatmap, a colour bar, a panel
    # of a chart), which is why `allow_images` is off by default: a chart figure should have
    # none, and a figure with a photographic panel opts in and gets its resolution checked.
    images = [line for line in subprocess.run(
        ["pdfimages", "-list", pdf_path], capture_output=True, text=True
    ).stdout.splitlines()[2:] if line.strip()]
    if images and not allow_images:
        problems.append(
            "{} rasterised image(s); line art must stay vector. If this figure genuinely "
            "contains a photograph, verify with allow_images=True".format(len(images)))
    elif images:
        for line in images:
            parts = line.split()
            try:
                ppi = min(float(parts[12]), float(parts[13]))
            except (IndexError, ValueError):
                continue
            if ppi < MIN_IMAGE_PPI:
                problems.append("embedded image at {:.0f} dpi, under the {:.0f} dpi "
                                "minimum".format(ppi, MIN_IMAGE_PPI))

    return problems


def figures_root(config=None):
    """The directory `main_figures/` and `supplementary_figures/` sit in: `results/` in a
    public checkout."""
    from pathlib import Path as _P
    if config is None:
        import sys as _sys
        root = _P(__file__).resolve().parents[1]
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from src.config import load_config
        config = load_config()
    root = _P(__file__).resolve().parents[1]
    configured = ((config.get("output") or {}).get("figures_dir")) or "results"
    path = _P(configured)
    return path if path.is_absolute() else root / path


FIGURE_SECTIONS = ("main_figures", "supplementary_figures")


def figure_dir(section, config=None, create=True):
    """Where a figure's output belongs -- `results/main_figures/` in a public checkout.

    `section` is one of FIGURE_SECTIONS, and each is flat: every file name already says
    which figure and panel it holds, so a folder per figure only adds a level to click
    through. Read from config (`output.figures_dir`) rather than hardcoded, which is what
    lets our own runs keep writing into the working tree the staging script reads while a
    clone writes somewhere a person would look first.
    """
    if section not in FIGURE_SECTIONS:
        raise ValueError("figure_dir({!r}): expected one of {}".format(
            section, ", ".join(FIGURE_SECTIONS)))
    from pathlib import Path as _P
    import os as _os
    if config is None:
        import sys as _sys
        root = _P(__file__).resolve().parents[1]
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from src.config import load_config
        config = load_config()
    path = figures_root(config) / section
    if create:
        _os.makedirs(str(path), exist_ok=True)
    return path
