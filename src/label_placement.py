"""Shared point-label placement for multi-panel figures.

Figures annotate points through it with IDENTICAL behaviour and styling: a small
white-backdrop label per point, placed by adjustText then a deterministic refinement that
keeps every box inside the axes and off the data markers / obstacle boxes, with a dotted
leader arrow from each point to its label. The leader arrow's colour, linewidth and head
size are parameters.
"""
from __future__ import annotations

import matplotlib
import numpy as np
from adjustText import adjust_text


def _ann_arrowprops(arrow_color, arrow_lw, arrow_head, arrow_style="->"):
    """Leader line from point to label (shared by all panels). ``arrow_style`` is any
    matplotlib arrowstyle -- "->" for the dotted-head leader, "-" for a plain headless
    line (``arrow_head`` is then irrelevant)."""
    return dict(arrowstyle=arrow_style, color=arrow_color, lw=arrow_lw,
                linestyle="-", mutation_scale=arrow_head, shrinkA=0)


def _box_extent(t, r):
    """Rendered window extent of a label's backdrop box (falls back to the text)."""
    p = t.get_bbox_patch()
    return (p if p is not None else t).get_window_extent(r)


def _artist_extent(o, r):
    """Window extent of an obstacle artist -- its backdrop box if it has one (info box),
    else its own extent (legend)."""
    getp = getattr(o, "get_bbox_patch", None)
    p = getp() if getp is not None else None
    return (p if p is not None else o).get_window_extent(r)


def _refine_labels(ax, texts, obstacles, pad_px, point_xy=None, point_half_px=1.0,
                   max_iter=600, gap_px=1.0):
    """Deterministic post-placement cleanup, in display pixels. Each iteration first
    pushes every label box inside the axes minus ``pad_px`` (a hard margin the labels may
    not cross), then -- as the LAST step -- resolves overlaps with obstacles, the data
    markers (``point_xy``, so no box ever covers a dot), and other labels by the smallest
    translation. Loops until nothing moves or ``max_iter``."""
    fig = ax.figure
    inv = ax.transData.inverted()
    # One draw to settle geometry, then reuse the renderer: the axes box, obstacle boxes
    # and data-marker boxes are all STATIC after the layout freeze, and a label's text
    # extent is current as soon as set_position() is called -- so the iteration needs no
    # further full-figure draws (the per-iteration redraws were the dominant cost on a
    # many-panel figure). We measure label TEXT extents (the white backdrop pad is ~0).
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.patch.get_window_extent(r)
    lo_x, hi_x = ab.x0 + pad_px, ab.x1 - pad_px
    lo_y, hi_y = ab.y0 + pad_px, ab.y1 - pad_px
    # data markers -> small fixed no-go boxes (computed once; axes geometry is frozen)
    pt_boxes = []
    pt_centers = []
    cluster_cx = cluster_cy = 0.0
    if point_xy is not None and len(point_xy[0]):
        disp = ax.transData.transform(np.column_stack([point_xy[0], point_xy[1]]))
        pt_boxes = [matplotlib.transforms.Bbox.from_extents(
            px - point_half_px, py - point_half_px, px + point_half_px, py + point_half_px)
            for px, py in disp]
        pt_centers = [(px, py) for px, py in disp]
        cluster_cx = float(np.mean([p[0] for p in pt_centers]))
        cluster_cy = float(np.mean([p[1] for p in pt_centers]))
    hard = [_artist_extent(o, r) for o in obstacles] + pt_boxes  # never-cover statics
    for _ in range(max_iter):
        boxes = [t.get_window_extent(r) for t in texts]   # current after set_position
        shift = [[0.0, 0.0] for _ in texts]

        # (1) padded-bounds containment
        for i, b in enumerate(boxes):
            if b.width < (hi_x - lo_x):
                if b.x0 < lo_x:
                    shift[i][0] += lo_x - b.x0
                elif b.x1 > hi_x:
                    shift[i][0] += hi_x - b.x1
            if b.height < (hi_y - lo_y):
                if b.y0 < lo_y:
                    shift[i][1] += lo_y - b.y0
                elif b.y1 > hi_y:
                    shift[i][1] += hi_y - b.y1

        # (2) box-box overlaps: relaxable comfort term, damped to converge
        for i in range(len(texts)):
            bi = boxes[i]
            cix, ciy = 0.5 * (bi.x0 + bi.x1), 0.5 * (bi.y0 + bi.y1)
            for j in range(len(texts)):
                if j == i:
                    continue
                bk = boxes[j]
                ox = min(bi.x1, bk.x1) - max(bi.x0, bk.x0) + gap_px
                oy = min(bi.y1, bk.y1) - max(bi.y0, bk.y0) + gap_px
                if ox <= 0 or oy <= 0:
                    continue
                if ox < oy:
                    shift[i][0] += ox if cix >= 0.5 * (bk.x0 + bk.x1) else -ox
                else:
                    shift[i][1] += oy if ciy >= 0.5 * (bk.y0 + bk.y1) else -oy

        moved = False
        for i, t in enumerate(texts):
            dx, dy = 0.5 * shift[i][0], 0.5 * shift[i][1]   # damping
            if abs(dx) > 0.05 or abs(dy) > 0.05:
                moved = True
                xd, yd = ax.transData.transform(t.get_position())
                t.set_position(inv.transform((xd + dx, yd + dy)))

        # (3) HARD clearance from obstacles + data dots -- the LAST step, full strength,
        # so a box never covers a marker (overrides the relaxable box-box term above)
        if hard:
            boxes = [t.get_window_extent(r) for t in texts]
            for i, t in enumerate(texts):
                bi = boxes[i]
                cix, ciy = 0.5 * (bi.x0 + bi.x1), 0.5 * (bi.y0 + bi.y1)
                px_, py_ = 0.0, 0.0
                for bk in hard:
                    ox = min(bi.x1, bk.x1) - max(bi.x0, bk.x0) + gap_px
                    oy = min(bi.y1, bk.y1) - max(bi.y0, bk.y0) + gap_px
                    if ox <= 0 or oy <= 0:
                        continue
                    if ox < oy:
                        px_ += ox if cix >= 0.5 * (bk.x0 + bk.x1) else -ox
                    else:
                        py_ += oy if ciy >= 0.5 * (bk.y0 + bk.y1) else -oy
                # if the box still covers a dot, drive it RADIALLY away from the cluster
                # centroid (both axes) so it walks out into empty space rather than
                # sliding along the dense diagonal -- repeated until it clears
                covers = any(bi.x0 <= cx <= bi.x1 and bi.y0 <= cy <= bi.y1
                             for cx, cy in pt_centers)
                if covers:
                    px_ += 4.0 if cix >= cluster_cx else -4.0
                    py_ += 4.0 if ciy >= cluster_cy else -4.0
                if abs(px_) > 0.05 or abs(py_) > 0.05:
                    moved = True
                    xd, yd = ax.transData.transform(t.get_position())
                    t.set_position(inv.transform((xd + px_, yd + py_)))

        if not moved:
            break
    fig.canvas.draw()


def _draw_leaders(ax, tx, ty, texts, arrow_color, arrow_lw, arrow_head, arrow_style="->"):
    """Draw a dotted leader from each data point to its (now relocated) label box.
    The arrow sits below the box (lower zorder) so its tail is hidden under the backdrop;
    skipped when the point still lies inside the box (label did not move)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for x, y, t in zip(tx, ty, texts):
        bb = _box_extent(t, r)
        px, py = ax.transData.transform((x, y))
        if bb.x0 - 1 <= px <= bb.x1 + 1 and bb.y0 - 1 <= py <= bb.y1 + 1:
            continue
        # patchA clips the tail at the label's box boundary -> arrow is drawn only OUTSIDE
        arrowprops = dict(_ann_arrowprops(arrow_color, arrow_lw, arrow_head, arrow_style),
                          patchA=t.get_bbox_patch())
        ax.annotate("", xy=(x, y), xytext=t.get_position(),
                    xycoords="data", textcoords="data",
                    arrowprops=arrowprops, zorder=3, annotation_clip=False)


def _report_label_violations(ax, texts, label_strs, name, point_xy=None, tol=0.5):
    """Print whether any rendered label box overlaps another, exits the axes, or covers a
    data marker (px tol)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.patch.get_window_extent(r)
    boxes = [_box_extent(t, r) for t in texts]
    pts = []
    if point_xy is not None and point_xy[0] is not None:
        pts = ax.transData.transform(np.column_stack([point_xy[0], point_xy[1]]))
    oob, ov, cov = set(), set(), set()
    for i, b in enumerate(boxes):
        if b.x0 < ab.x0 - tol or b.x1 > ab.x1 + tol or b.y0 < ab.y0 - tol or b.y1 > ab.y1 + tol:
            oob.add(i)
        for (px, py) in pts:
            if b.x0 - tol <= px <= b.x1 + tol and b.y0 - tol <= py <= b.y1 + tol:
                cov.add(i); break
        for j in range(i + 1, len(boxes)):
            c = boxes[j]
            if not (b.x1 <= c.x0 + tol or c.x1 <= b.x0 + tol or
                    b.y1 <= c.y0 + tol or c.y1 <= b.y0 + tol):
                ov.add(i); ov.add(j)
    if oob or ov or cov:
        names = sorted({label_strs[i] for i in (oob | ov | cov)})
        print(f"  WARNING {name}: {len(ov)} overlapping, {len(oob)} out-of-bounds, "
              f"{len(cov)} cover-a-dot -> {names}")
    else:
        print(f"  {name} labels: PASS ({len(texts)} placed, 0 overlap / 0 oob / 0 cover)")


def place_labels_proximity(ax, tx, ty, label_strs, *, fontsize, color,
                           obstacles, name="panel",
                           avoid_pts_x=None, avoid_pts_y=None, bound_pad_frac=0.02,
                           nudge=None, prevent_crossings=False,
                           arrow_color="green", arrow_lw=0.5, arrow_head=6, arrow_style="->",
                           expand=(1.5, 1.9), force_text=(0.4, 0.6),
                           force_static=(0.6, 1.0), force_pull=(0.45, 0.2), iter_lim=None):
    """Place point labels in white backdrop boxes near their points.

    Two stages: (1) adjustText gives a good initial 2D layout that repels labels from the
    markers, each other, and the obstacle artists (info box / legend); (2) a deterministic
    refinement (`_refine_labels`) then guarantees every box stays inside the axes minus a
    ``bound_pad_frac`` margin and ends each iteration by resolving overlaps. Leader arrows
    are drawn last so they always point from the final box to its data point.

    ``avoid_pts_x``/``avoid_pts_y`` are the full set of plotted marker coordinates.
    ``nudge`` optionally biases specific labels' starting positions: a dict mapping label
    string -> (dx, dy) in axes-fraction, applied before solving so the clean-layout
    guarantees still hold around the nudged spot. Call AFTER the figure layout is final &
    frozen. Returns the Text artists.
    """
    if len(tx) == 0:
        return []
    # readability backdrop: white, slightly translucent fill, no border
    label_bbox = dict(boxstyle="square,pad=0.1", facecolor=(1.0, 1.0, 1.0, 0.8),
                      edgecolor="none", linewidth=0.0)
    texts = [ax.text(x, y, s, fontsize=fontsize, color=color,
                     ha="center", va="center", zorder=5, bbox=label_bbox)
             for x, y, s in zip(tx, ty, label_strs)]
    fig = ax.figure
    fig.canvas.draw()                       # adjustText needs a live renderer
    if nudge:                               # bias chosen labels' start (axes-fraction dx,dy)
        ab0 = ax.patch.get_window_extent(fig.canvas.get_renderer())
        for t, s in zip(texts, label_strs):
            if s in nudge:
                dxf, dyf = nudge[s]
                xd, yd = ax.transData.transform(t.get_position())
                t.set_position(ax.transData.inverted().transform(
                    (xd + dxf * ab0.width, yd + dyf * ab0.height)))
    pt_kw = {}
    if avoid_pts_x is not None and avoid_pts_y is not None:
        pt_kw = dict(x=np.asarray(avoid_pts_x, dtype=float),
                     y=np.asarray(avoid_pts_y, dtype=float))  # repel labels off markers
    adjust_text(
        texts,
        ax=ax,
        objects=obstacles,
        expand=expand,             # bbox padding for overlap detection (whitespace around labels)
        force_text=force_text,     # label<->label repulsion (spreads the cluster)
        force_static=force_static, # push away from data points/objects (fans off the cluster)
        force_pull=force_pull,     # tether back to own point (higher -> shorter leaders)
        ensure_inside_axes=True,
        prevent_crossings=prevent_crossings,  # block leader crossings (refinement fixes bounds after)
        iter_lim=iter_lim,         # fixed iteration count -> deterministic (else adjustText
                                   # defaults to a 1 s wall-clock stop, jittering run-to-run)
        **pt_kw,                            # no arrowprops: leaders drawn after refinement
    )
    # deterministic cleanup: hard padded bounds + overlap resolution as the last step
    r = fig.canvas.get_renderer()
    ab = ax.patch.get_window_extent(r)
    pad_px = bound_pad_frac * min(ab.width, ab.height)
    point_xy = pt_kw.get("x"), pt_kw.get("y")
    _refine_labels(ax, texts, obstacles, pad_px,
                   point_xy=point_xy if point_xy[0] is not None else None)
    _draw_leaders(ax, tx, ty, texts, arrow_color, arrow_lw, arrow_head, arrow_style)
    _report_label_violations(ax, texts, label_strs, name,
                             point_xy=point_xy if point_xy[0] is not None else None)
    return texts
