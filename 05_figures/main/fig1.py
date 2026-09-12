#!/usr/bin/env python
"""Render Figure 1 from the artwork, so that it is a figure script like any other.

    pixi run python 05_figures/main/fig1.py

Author: Michael Beyeler (github.com/mjbeyeler)

Figure 1 is drawn rather than computed: there is no analysis behind it and no data to
plot. That made it the one display item a reader could not produce: every other figure answers
one `pixi run` with a PDF and a PNG, and this one answered with a file to go and find. It
now answers the same way.

WHY rsvg AND NOT THE USUAL ROUTES. Elsewhere in this repository rsvg is avoided, because
it rasterises or mangles live text in an SVG. Here it is the right tool for exactly the
reason that would otherwise disqualify it: the drawing carries NO live text. Google's SVG
export converts every glyph to outlines -- 226 paths and zero <text> elements -- so there
is nothing left for a text engine to get wrong, and the check below asserts that rather
than trusting it. If a future export does carry text, this stops instead of quietly
flattening it.

The PDF is vector throughout. The PNG is a 300 dpi companion for looking at, like the one
every other figure script writes beside its PDF.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.display_item import figure_dir  # noqa: E402 -- config: output.figures_dir

ARTWORK = Path(__file__).resolve().parent / "fig1_overview" / "fig1_overview.svg"
PNG_DPI = 300


def main() -> int:
    print("START fig1.py")
    if not ARTWORK.is_file():
        raise SystemExit("the artwork is missing: {}".format(ARTWORK))

    svg = ARTWORK.read_text(encoding="utf-8", errors="replace")
    live_text = len(re.findall(r"<text[\s>]", svg))
    if live_text:
        raise SystemExit(
            "{} carries {} <text> element(s). This script renders with rsvg, which is "
            "only safe on this file because every glyph is an outline; live text would be "
            "re-shaped by a different font engine. Re-export the drawing with text "
            "converted to paths, or render it with a tool that embeds fonts.".format(
                ARTWORK.name, live_text)
        )

    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        raise SystemExit(
            "rsvg-convert is not on PATH. It ships with this project's environment: run "
            "`pixi install`, or `pixi run python 05_figures/main/fig1.py`."
        )

    out_dir = figure_dir("main_figures")
    pdf, png = out_dir / "fig1.pdf", out_dir / "fig1.png"
    subprocess.run([rsvg, "-f", "pdf", "-o", str(pdf), str(ARTWORK)], check=True)

    # The PNG is sized from the PDF's own page box, NOT with rsvg's --dpi flags: those set
    # the resolution at which the SVG is interpreted, not the scale of the raster, and
    # asking for 300 there quietly produced a 96 dpi image (1970 px across a 521 mm page).
    # Width in pixels is the only unambiguous instruction rsvg takes.
    from pypdf import PdfReader

    box = PdfReader(str(pdf)).pages[0].mediabox
    width_pt, height_pt = float(box.width), float(box.height)
    width_px = int(round(width_pt / 72.0 * PNG_DPI))
    subprocess.run([rsvg, "-f", "png", "-w", str(width_px), "-o", str(png), str(ARTWORK)],
                   check=True)

    for path in (pdf, png):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit("rsvg-convert reported success but {} is empty".format(path))
    print("  {}  ({:.1f} MB)".format(pdf, pdf.stat().st_size / 1e6))
    print("  {}  ({:.1f} MB, {} px wide = {} dpi)".format(
        png, png.stat().st_size / 1e6, width_px, PNG_DPI))
    # Over the journal's 179 x 260 mm display-item limit, and scaled at layout.
    print("  page: {:.0f} x {:.0f} mm".format(width_pt / 72 * 25.4, height_pt / 72 * 25.4))
    print("  no live text in the artwork: {} <text> elements".format(live_text))
    print("END fig1.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
