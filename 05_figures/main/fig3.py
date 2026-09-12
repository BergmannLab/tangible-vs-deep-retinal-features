#!/usr/bin/env python
"""Assemble the four Figure 3 strips into the single PDF the journal wants.

    pixi run python 05_figures/main/fig3.py

Authors: a, d  David Presby (github.com/presbyd), Michael Beyeler (github.com/mjbeyeler)
         rest  Michael Beyeler (github.com/mjbeyeler)

Nature asks for each main figure as one separate vector file. Figure 3 is built as four
horizontal strips -- a-c (SNP), d-f (gene), g-j (heritability, polygenicity, pleiotropy and
pathway overlap) and k (the pathway map, full width) -- matching the row structure of the
manuscript, and because a strip can be rebuilt on its own. This stacks them, losslessly.

Run the four strip scripts first; this only composes what they wrote.

  pixi run -e r Rscript 05_figures/main/fig3_snp.R
  pixi run -e r Rscript 05_figures/main/fig3_gene.R
  pixi run python      05_figures/main/fig3_polygenicity.py
  pixi run python      05_figures/main/fig3_pathway.py
  pixi run python      05_figures/main/fig3.py

WHY pypdf. The composition has to preserve vector content and embedded fonts, which rules
out every raster route (magick, pdftoppm). pdftk, pdfjam and inkscape are not installed
here and pdftools cannot compose pages; qpdf can rewrite a PDF but not n-up one. pypdf
merges page content streams directly, so the output is the inputs' own vector objects at a
translated origin -- and qpdf then re-deflates the result, see compress().
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.display_item import figure_dir  # noqa: E402 -- config: output.figures_dir
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.plot_style import load_plot_style  # noqa: E402

MM_PER_PT = 25.4 / 72.0
PT_PER_MM = 72.0 / 25.4

# Top to bottom, the order the panels are lettered in.
STRIPS = [
    ("snp", "fig3_panel_1_snp.pdf", "a b c"),
    ("gene", "fig3_panel_2_gene.pdf", "d e f"),
    ("polygenicity", "fig3_panel_3_ghij.pdf", "g h i j"),
    ("pathway", "fig3_panel_4_k.pdf", "k"),
]

# cairo_pdf and matplotlib both round the page box to a fraction of a point, so an exact
# width comparison would fail on a figure that is in fact correct.
WIDTH_TOLERANCE_PT = 1.0


def main(argv=None) -> int:
    # Rendering the 300 dpi companion means rasterising a page of several million vector
    # points, which is pure cost while a detail is being iterated on. The PDF is the
    # deliverable; the PNG is for drafts, so it is opt-out.
    parser = argparse.ArgumentParser(description="Assemble Figure 3 from its four strips")
    parser.add_argument("--no-png", dest="png", action="store_false",
                        help="skip the 300 dpi PNG companion")
    args = parser.parse_args(argv)
    want_png = args.png

    strips = list(STRIPS)

    config_path = REPO_ROOT / "config.yaml"
    with open(str(config_path), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    style = load_plot_style(str(config_path))
    fig3 = config["plot_styles"]["figures"]["fig3"]

    gap_mm = float(fig3["strip_gap_mm"])
    width_mm = style.display_item_width_mm
    width_pt = width_mm * PT_PER_MM
    max_height_mm = style.display_item_max_height_mm

    figures_dir = figure_dir("main_figures")
    readers, heights_mm = [], []
    for key, filename, panels in strips:
        path = figures_dir / filename
        if not path.is_file():
            raise SystemExit(
                "missing strip {} ({}): {}\n  Run the strip scripts first -- see this "
                "file's docstring.".format(key, panels, path))
        reader = PdfReader(str(path))
        page = reader.pages[0]
        w_pt, h_pt = float(page.mediabox.width), float(page.mediabox.height)
        if abs(w_pt - width_pt) > WIDTH_TOLERANCE_PT:
            raise SystemExit(
                "strip {} is {:.2f} mm wide, expected {:.2f} mm. Every strip must be "
                "authored at the same width or the figure will not line up.".format(
                    key, w_pt * MM_PER_PT, width_mm))
        readers.append((key, panels, reader, w_pt, h_pt))
        heights_mm.append(h_pt * MM_PER_PT)

    # One gap per boundary, top to bottom: strip_gap_mm everywhere, plus whatever
    # strip_gap_extra_mm asks for above a named strip.
    extra = fig3.get("strip_gap_extra_mm") or {}
    gaps_mm = [gap_mm + float(extra.get(readers[i + 1][0], 0.0))
               for i in range(len(readers) - 1)]
    total_mm = sum(heights_mm) + sum(gaps_mm)
    print("Figure 3 strips:")
    for index, ((key, panels, _, _, h_pt), h_mm) in enumerate(zip(readers, heights_mm)):
        above = "" if index == 0 else "  (+{:.1f} mm above)".format(gaps_mm[index - 1])
        print("  {:14s} {:6s} {:7.2f} mm{}".format(key, panels, h_mm, above))
    print("  {:14s} {:6s} {:7.2f} mm  (+{:g} mm gaps)".format(
        "TOTAL", "", total_mm, sum(gaps_mm)))

    if total_mm > max_height_mm:
        raise SystemExit(
            "the assembled figure is {:.2f} mm tall, over the {:.0f} mm display-item cap. "
            "Reduce a strip height in config.yaml -> plot_styles.figures.fig3."
            .format(total_mm, max_height_mm))

    total_pt = total_mm * PT_PER_MM
    t_merge = time.time()
    writer = PdfWriter()
    page = writer.add_blank_page(width=width_pt, height=total_pt)
    place_as_xobjects(writer, page, readers, width_pt,
                      [g * PT_PER_MM for g in gaps_mm])

    out_dir = figures_dir
    # The glow build writes beside the real one; it must never overwrite fig3.pdf, which is
    # what gets staged to publication_figures.
    out_pdf = out_dir / "fig3.pdf"
    with open(str(out_pdf), "wb") as handle:
        writer.write(handle)
    # Where the minutes go. pypdf merges and writes the strips' content streams in pure
    # Python, and the Manhattan point clouds make that stream tens of megabytes; the
    # C++ tools that follow are comparatively instant. Printed so a slow build can be
    # diagnosed instead of guessed at.
    print("  merge+write  : {:.1f} s".format(time.time() - t_merge))

    t = time.time()
    total_mm = trim_to_ink(out_pdf, width_mm)
    print("  trim (gs)    : {:.1f} s".format(time.time() - t))
    t = time.time()
    compress(out_pdf)
    print("  compress     : {:.1f} s".format(time.time() - t))
    verify(out_pdf)
    size_mb = out_pdf.stat().st_size / 1e6
    print("  file size    : {:.1f} MB {}".format(
        size_mb, "PASS" if size_mb <= 30 else "WARNING -- over Nature's ~30 MB guidance"))

    # A raster companion at the same physical size, for drafts and for pasting into
    # documents. The PDF is the deliverable.
    if want_png:
        t = time.time()
        # Named off the PDF, not hardcoded: the --glow-k build was writing its preview to
        # fig3.png and silently replacing the real figure's companion (7 Sep 2026).
        stem = out_pdf.with_suffix("")
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile",
                        str(out_pdf), str(stem)], check=False)
        print("  png preview  : {:.1f} s".format(time.time() - t))
        check_left_alignment(stem.with_suffix(".png"), width_mm,
                             float(fig3["axis_align_tol_mm"]))

    print("\nWrote {}  ({:.1f} x {:.1f} mm)".format(out_pdf, width_mm, total_mm))
    return 0


def place_as_xobjects(writer, page, readers, width_pt, gaps_pt):
    """Draw each strip onto the page as a Form XObject, bottom-up.

    NOT `merge_transformed_page`, which is what this used to do. That inlines: it decodes
    every strip's content stream, concatenates them into one, and pypdf then writes the
    result uncompressed -- 35 MB of raw path data for a figure whose four inputs total
    4 MB. With the Manhattan point clouds it took 121 s of pure Python, 80 % of the whole
    assembly (measured 4 Sep 2026), and it is why iterating on a detail cost minutes.

    A Form XObject instead REFERENCES each strip's stream exactly as it already is,
    compressed, and the page's own content is one `Do` per strip. Nothing is decoded, so
    the cost stops scaling with how much ink the figure carries. The strip's placement
    goes in the form's /Matrix; PDF's origin is bottom-left, so the strips are walked from
    the bottom of the page upward while the list is iterated from the last panel to the
    first.
    """
    from pypdf.generic import ArrayObject, FloatObject, NameObject

    xobjects = DictionaryObject()
    draw = []
    y_pt = 0.0
    for index, (key, panels, reader, w_pt, h_pt) in enumerate(reversed(readers)):
        source = reader.pages[0]
        form = source["/Contents"].get_object()
        if isinstance(form, ArrayObject):        # a page may split its stream in pieces
            joined = b"\n".join(part.get_object().get_data() for part in form)
            form = DecodedStreamObject()
            form.set_data(joined)
        form = form.clone(writer)                # brings the stream and its refs across
        form[NameObject("/Type")] = NameObject("/XObject")
        form[NameObject("/Subtype")] = NameObject("/Form")
        form[NameObject("/BBox")] = ArrayObject(
            [FloatObject(0), FloatObject(0), FloatObject(w_pt), FloatObject(h_pt)])
        form[NameObject("/Resources")] = source["/Resources"].get_object().clone(writer)
        form[NameObject("/Matrix")] = ArrayObject([
            FloatObject(1), FloatObject(0), FloatObject(0), FloatObject(1),
            FloatObject((width_pt - w_pt) / 2.0),   # centre any sub-point width residual
            FloatObject(y_pt)])
        name = "/S{}".format(index)
        xobjects[NameObject(name)] = writer._add_object(form)
        draw.append("q {} Do Q".format(name))
        # gaps_pt is top-to-bottom; walking upward from the bottom consumes it in reverse
        y_pt += h_pt + (gaps_pt[len(readers) - 2 - index] if index < len(gaps_pt) else 0.0)

    resources = page[NameObject("/Resources")].get_object()
    resources[NameObject("/XObject")] = xobjects
    content = DecodedStreamObject()
    content.set_data("\n".join(draw).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)


def trim_to_ink(pdf: Path, width_mm: float) -> float:
    """Shrink the page box to the drawn content, vertically. Returns the new height in mm.

    A display item must FILL its box: white padding round the artwork makes the journal
    place the figure smaller than it should be, and src/display_item.py rejects any margin
    over 0.25 mm. Each strip carries a little slack above and below its own ink, so the
    stack inherits the top strip's top slack and the bottom strip's bottom slack.

    This is a page-geometry change, not a crop of the artwork and not a rescale: nothing
    moves relative to anything else and no type changes size. The WIDTH is deliberately
    left alone -- it is already exactly the display-item width, and the row that spans it
    (k) is what guarantees that. If the ink does not reach both side edges, that is a
    layout problem in the strips and the caller is told so rather than having the page
    quietly narrowed.
    """
    from src.display_item import _ink_box_mm, _page_size_mm, figure_dir

    page_w, page_h = _page_size_mm(str(pdf))
    x0, y0, x1, y1 = _ink_box_mm(str(pdf))
    side = max(x0, page_w - x1)
    if side > 0.25:
        print("  trimmed      : SKIPPED -- ink stops {:.2f} mm short of a side edge; widen "
              "a strip rather than narrowing the page".format(side))
        return page_h

    reader = PdfReader(str(pdf))
    writer = PdfWriter()
    page = reader.pages[0]
    page.mediabox.lower_left = (0, y0 * PT_PER_MM)
    page.mediabox.upper_right = (page_w * PT_PER_MM, y1 * PT_PER_MM)
    writer.add_page(page)
    with open(str(pdf), "wb") as handle:
        writer.write(handle)

    new_h = y1 - y0
    print("  trimmed      : {:.2f} -> {:.2f} mm tall ({:.2f} mm top, {:.2f} mm bottom)".format(
        page_h, new_h, page_h - y1, y0))
    return new_h


def compress(pdf: Path) -> None:
    """Re-deflate the merged content stream, in place.

    pypdf composes the strips correctly but writes the merged content stream
    UNCOMPRESSED: 42 MB of raw path data in a 55 MB file whose four inputs total 4 MB.
    Neither `compress_content_streams()` nor replacing /Contents with a flate-encoded
    stream survives its writer -- the serialised page comes back with no /Filter either
    way. `qpdf --stream-data=compress` fixes it losslessly in well under a second, and
    takes the file to about 4 MB, comfortably inside Nature's ~30 MB guidance.
    (Ghostscript was the alternative and is worse twice over: five minutes, and it
    reported the page as damaged.)
    """
    before = pdf.stat().st_size
    tmp = pdf.with_suffix(".qpdf.tmp")
    result = subprocess.run(
        ["qpdf", "--stream-data=compress", "--object-streams=generate",
         str(pdf), str(tmp)],
        capture_output=True, text=True,
    )
    # qpdf exits 3 on warnings while still writing a valid file; only a hard failure
    # should cost us the compression, and never the figure.
    if result.returncode in (0, 3) and tmp.is_file():
        tmp.replace(pdf)
        print("  compressed   : {:.1f} MB -> {:.1f} MB".format(
            before / 1e6, pdf.stat().st_size / 1e6))
        if result.returncode == 3:
            print("    qpdf warning: {}".format(result.stderr.strip().splitlines()[:1]))
    else:
        tmp.unlink(missing_ok=True)
        print("  compressed   : SKIPPED -- qpdf failed ({}): {}".format(
            result.returncode, result.stderr.strip()[:200]))


def check_left_alignment(png: Path, width_mm: float, tol_mm: float) -> None:
    """Every panel whose plot area starts in the left margin must start at the same x.

    The strips are drawn by two different engines -- the Manhattans by ggplot, the g-j row
    by matplotlib's constrained layout -- and neither knows where the other put its axes,
    so g's left spine sat 1.35 mm inside a's until 5 Sep 2026. It is now placed there by
    hand (plot_styles.figures.fig3.ghi_left_mm), which means it can silently drift again:
    the Manhattans' panel edge moves whenever their y tick labels change width. This reads
    the spines back off the rendered composite, so the build fails instead of shipping a
    ragged column.

    A spine is a column of the raster carrying a long unbroken dark run. Only the leftmost
    ones matter -- panels a, d and g; the QQ and Venn column starts far to the right -- so
    the search stops at LEFT_BAND_MM.
    """
    from PIL import Image

    LEFT_BAND_MM = 25.0
    # A spine has to be a LONG dark run, not merely a dark one. At 8 mm this also caught
    # panel k's winner glow, a saturated magenta band running down a cell edge, and reported
    # a fourth spine that was not a panel at all. Measured on the built figure: the real
    # spines run 25.81 mm unbroken, the glow band 15.40 mm, so 20 mm separates them with
    # margin on both sides and is still shorter than the shortest of the three boxes.
    MIN_RUN_MM = 20.0

    image = np.array(Image.open(str(png)).convert("L"))
    px_per_mm = image.shape[1] / width_mm
    dark = image < 128
    min_run = int(MIN_RUN_MM * px_per_mm)

    spines = []
    for x in range(int(LEFT_BAND_MM * px_per_mm)):
        column = dark[:, x]
        if not column.any():
            continue
        # Longest unbroken run of dark pixels in this column.
        run = best = 0
        for value in column:
            run = run + 1 if value else 0
            best = max(best, run)
        if best >= min_run:
            spines.append(x)

    if not spines:
        print("  left align   : SKIPPED -- no spine found in the left {:g} mm"
              .format(LEFT_BAND_MM))
        return

    # Adjacent columns are one 2-px line, not two panels.
    groups = [[spines[0]]]
    for x in spines[1:]:
        if x - groups[-1][-1] <= 2:
            groups[-1].append(x)
        else:
            groups.append([x])
    centres = sorted(sum(g) / len(g) / px_per_mm for g in groups)
    spread = centres[-1] - centres[0]
    print("  left align   : {} at {} mm{}".format(
        "PASS" if spread <= tol_mm else "FAIL",
        ", ".join("{:.2f}".format(c) for c in centres),
        "" if spread <= tol_mm else "  -- spread {:.2f} mm over the {:.2f} mm tolerance; "
        "set plot_styles.figures.fig3.ghi_left_mm to the Manhattans' edge".format(
            spread, tol_mm)))


def verify(pdf: Path) -> None:
    """The checks the conventions in 05_figures/README.md requires, run on the deliverable."""
    images = subprocess.run(["pdfimages", "-list", str(pdf)],
                            capture_output=True, text=True).stdout
    raster_rows = [ln for ln in images.strip().splitlines()[2:] if ln.strip()]
    print("\nverification")
    print("  rasters      : {}".format(
        "PASS (none)" if not raster_rows else
        "FAIL -- {} raster(s), the figure must be vector throughout".format(
            len(raster_rows))))

    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    rows = [ln.split() for ln in fonts.strip().splitlines()[2:] if ln.strip()]
    names = [r[0].split("+")[-1] for r in rows]
    embedded = all("yes" in r for r in rows)
    families = sorted({n.split("-")[0] for n in names})
    print("  fonts        : {} ({})".format(
        "PASS" if embedded else "FAIL -- not all embedded", ", ".join(sorted(set(names)))))
    print("  one family   : {}".format(
        "PASS" if len(families) == 1 else "FAIL -- {}".format(families)))


if __name__ == "__main__":
    raise SystemExit(main())
