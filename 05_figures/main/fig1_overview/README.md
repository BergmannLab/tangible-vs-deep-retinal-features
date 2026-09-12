# Figure 1 — graphical overview of the study

Author: Dennis Bontempi (github.com/denbonte)

The one main-text figure with nothing to compute. It is hand-drawn artwork, so the asset
*is* the source: no analysis regenerates it, and the file here is the drawing itself rather
than something a script plotted. `../fig1.py` renders it to PDF and PNG, which is what makes
Figure 1 reproducible on the same footing as the other five.

It is in the repository with the author's permission. Rights in the artwork stay with him
and with the Article — the repository's GPL-3.0 covers the code, not this figure.

## The file

`fig1_overview.svg` is the drawing as self-contained vector: 22 embedded PNGs (the retina
drawings, icons and cohort logos) and 226 paths, openable in Inkscape or Illustrator.

**It has no live text.** The export converts every glyph to an outline, so the file carries
zero `<text>` elements. It renders identically to the published figure, but it is not the
file to edit if you need to change a label.

The page is 521 × 293 mm, larger than a journal display item; the figure is scaled at layout.
