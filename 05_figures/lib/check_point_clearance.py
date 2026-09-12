#!/usr/bin/env python
"""Check that a Manhattan panel's points keep clear of the axis line beneath them.

    python 05_figures/lib/check_point_clearance.py <panel.png> [--min-gap 3]

Author: Michael Beyeler (github.com/mjbeyeler)

Two things can go wrong at the foot of a Manhattan facet, and both are invisible at draft
scale:

  1. the cloud runs past the bottom of the plot area and is sliced flat by it;
  2. the cloud sits flush on the axis line, so the lowest dots merge into it.

This measures both directly. It composites the (transparent) PNG on white, finds each
facet's axis line as a long near-black horizontal run, and then reports, per facet, how
many coloured pixels fall BELOW that line and how big the gap above it is.

An earlier version of this script guessed at (1) from how flat the bottom row of the cloud
looked, which gave a false "clipped" on the gene panel where the cloud is simply dense.
Measuring against the axis line instead is unambiguous.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CHROMA_MIN = 25      # a pixel is "data" when max-min across RGB exceeds this
DARK_MAX = 90        # a pixel is "ink" when its brightest channel is below this
LINE_COVERAGE = 0.6  # a row is an axis line when this fraction of it is ink


def on_white(path):
    image = Image.open(str(path)).convert("RGBA")
    backdrop = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(backdrop, image).convert("RGB")).astype(int)


def axis_lines(dark, width):
    """Row spans of the horizontal axis lines, top to bottom."""
    hits = [y for y in range(dark.shape[0]) if dark[y].sum() > LINE_COVERAGE * width]
    spans, start, prev = [], None, None
    for y in hits:
        if start is None:
            start = y
        elif y != prev + 1:
            spans.append((start, prev))
            start = y
        prev = y
    if start is not None:
        spans.append((start, prev))
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("png", type=Path)
    parser.add_argument("--min-gap", type=int, default=3,
                        help="pixels of clear space required between the lowest dot and "
                             "the axis line")
    args = parser.parse_args()

    image = on_white(args.png)
    height, width, _ = image.shape
    data = (image.max(axis=2) - image.min(axis=2)) > CHROMA_MIN
    dark = image.max(axis=2) < DARK_MAX

    spans = axis_lines(dark, width)
    if not spans:
        print("  no axis line found in {}".format(args.png.name))
        return 1

    print("  point clearance, {} ({} facet(s), need {} px)".format(
        args.png.name, len(spans), args.min_gap))
    failures = 0
    for n, (top, bottom) in enumerate(spans, 1):
        below = int(data[bottom + 1:bottom + 30].sum())
        above = [y for y in range(max(0, top - 80), top) if data[y].any()]
        gap = top - (above[-1] + 1) if above else None
        problems = []
        if below:
            problems.append("{} px below the line".format(below))
        if gap is not None and gap < args.min_gap:
            problems.append("only {} px of clearance".format(gap))
        failures += bool(problems)
        print("    facet {}: axis rows {}-{} | gap {} px | below {} px  {}".format(
            n, top, bottom, gap, below, "  ".join(problems) if problems else "ok"))

    print("  {}".format("PASS" if not failures else
                        "FAIL -- {} facet(s)".format(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
