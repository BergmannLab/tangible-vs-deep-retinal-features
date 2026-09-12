"""Reproduce the 3-panel "feature combinations" figure:

  a) R^2 for risk factors (DBP, SBP, HbA1c, BMI)         -- elastic-net linear regression
  b) AUC for ocular diseases (Presbyopia, Amblyopia,
     Glaucoma, Cataract)                                  -- elastic-net logistic regression
  c) AUC for general diseases (Hypertension, Heart attack,
     Diabetes, Death)                                     -- elastic-net logistic regression

Nothing is refit; this only re-aggregates already-computed 5-fold CV results.

Sources
-------
David's script, reading its two inputs from the Zenodo deposit by concept DOI through
src.fetch_data, so a public checkout can rebuild the figure.

  a)   Fig6_01_a_linear_r2_by_feature_set.csv
       (per-fold test R^2 in columns r2_test_fold_1..5; bar = mean, whisker = + t_{0.975,4} * SEM, upper only)
  b/c) Fig6_02_bc_logistic_auc_by_feature_set.csv
       (per-fold held-out test AUC; bar = mean, whisker = + t_{0.975,n-1} * SEM, upper only)

Both tables carry the individual fold values, so every bar reproduces exactly.

Outputs
-------
PNG at 300 dpi alongside the vector PDF, into the results directory:

  fig6_feature_combinations.{png,pdf}                     a | b | c + legend
  fig6_feature_combinations_folds.{png,pdf}               the same, fold values overlaid
  fig6_feature_combinations_panel_{a,b,c}.{png,pdf}       single panels
  fig6_feature_combinations_legend.{png,pdf}              stand-alone legend
  fig6_feature_combinations_values.csv                    the numbers behind the bars

    pixi run python 05_figures/main/fig6_feature_combinations.py

Set $MVP_INTERMEDIATES_DIR to build from local intermediates before they are deposited.

Authors
-------
David Presby   github.com/presbyd     (the script, and everything it draws)
Michael Beyeler github.com/mjbeyeler   (the deposit wiring only)
"""

import colorsys
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Embed TrueType fonts so text in the PDFs stays editable/selectable (true vector output).
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as t_dist

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.display_item import figure_dir                             # noqa: E402
from src.fetch_data import fetch                                    # noqa: E402
from src.plot_style import (fold_marker_spec, load_plot_style,      # noqa: E402
                            pop_color, swarm_diameter, swarm_offsets)

CONFIG_PATH = REPO_ROOT / "config.yaml"
CONFIG_PATH_STR = str(CONFIG_PATH)

# The paper's typeface, from the one place it is defined. apply_font() sets the sans face
# (Nimbus Sans, Helvetica metrics), routes mathtext through it so the $R^2$ labels match the
# body text instead of falling back to DejaVu, and keeps pdf.fonttype 42 so the glyphs stay
# embedded and selectable rather than becoming outlines.
STYLE = load_plot_style(str(CONFIG_PATH))
STYLE.apply_font()

# Panel letters: the paper-wide value, not a scaled one. 8 pt bold lowercase is Nature's
# house style and config.yaml is where it is set for every figure.
PANEL_LETTER_PT = STYLE.panel_label_fontsize
PANEL_LETTER_WEIGHT = STYLE.panel_label_fontweight

# Individual fold markers. Nature's statistical guidance: "If using a bar chart, please
# note that individual data points must be overlaid on the bars" -- stated with no
# sample-size threshold. Geometry and colour treatment come from config and from
# src/plot_style, shared with Figure 2b: the two figures draw the same mark on the same
# page, so there is one spec and one implementation rather than a copy in each script.
# Absolute points -- these are NOT scaled with the canvas; the diameter is set from the
# bar's own thickness so a two-row swarm exactly fills the bar and never leaves it.
FOLD = fold_marker_spec(CONFIG_PATH_STR)
FOLD_EDGE_LW = float(FOLD["edge_linewidth_pt"])
FOLD_ALPHA = float(FOLD["alpha"])


def pop(color):
    """The bar's colour, deepened until a dot of it reads against the bar it sits on."""
    return pop_color(color, FOLD["saturation"], FOLD["lightness"], FOLD["gray_lightness"])


# David's x-axis ticks, pinned rather than left to the auto-locator. His 11 pt labels on a
# 462 mm canvas gave panel a every 0.05; at 5 pt on 179 mm matplotlib thinned that to every
# 0.10 on its own, which silently changed the axis. Keyed by the panel's xlim, with his
# decimal places.
XTICKS = {
    (0.0, 0.35): ([round(0.05 * k, 2) for k in range(8)], "{:.2f}"),
    (0.4, 1.0): ([round(0.4 + 0.1 * k, 1) for k in range(7)], "{:.1f}"),
}

def BAR_EDGE_LW():
    """The bar outline weight, and now the whisker's too, so the two match at any scale."""
    return S(0.8)


SHOW_FOLDS = False        # set per output by make_variant
_SWARMS = []              # (line, axes, x values, bar centre, bar height), laid out later


def lay_out_swarms(fig):
    """Apply the beeswarm once the axes are placed, so points-per-data-unit is known."""
    for collection, ax, bars in _SWARMS:
        if ax.figure is not fig:
            continue
        pos = ax.get_position()
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        pt_per_x = pos.width * fig.get_figwidth() * 72.0 / (x_hi - x_lo)
        pt_per_y = pos.height * fig.get_figheight() * 72.0 / (y_hi - y_lo)
        # The bar's thickness in points is only known now, and it is what sets the dot: a
        # half-bar, so two rows fill the bar exactly. Demanding no overlap AND containment
        # would cap the dot near a fifth of that, which is invisible at A4.
        bar_pt = bars[0][2] * pt_per_y
        diameter = swarm_diameter(bar_pt, FOLD_EDGE_LW, float(FOLD["diameter_pt"]))
        collection.set_sizes([diameter ** 2])
        collection.set_linewidths(FOLD_EDGE_LW)
        xs, ys = [], []
        for fold_x, centre, bar_h, _face in bars:
            room = max(0.0, (bar_h * pt_per_y - diameter - FOLD_EDGE_LW) / 2.0)
            offsets = swarm_offsets(fold_x * pt_per_x, diameter, room_pt=room)
            xs.append(fold_x)
            ys.append(centre + offsets / pt_per_y)
        collection.set_offsets(np.column_stack([np.concatenate(xs), np.concatenate(ys)]))
        print("panels: bar {:.2f} pt thick; fold dots {:.2f} pt "
              "= {:.0f}% of the bar".format(bar_pt, diameter,
                                            100 * (diameter + FOLD_EDGE_LW) / bar_pt))
FIG_DIR = str(figure_dir("main_figures"))
DATA_DIR = FIG_DIR
os.makedirs(FIG_DIR, exist_ok=True)

# The two deposit tables that stand in for the CSVs on David's filesystem.
LINREG_CSV = "Fig6_01_a_linear_r2_by_feature_set.csv"
LOGREG_FOLD_CSV = "Fig6_02_bc_logistic_auc_by_feature_set.csv"

_TABLES = {}


def deposit(name):
    """The deposited table `name`, resolved through the Zenodo concept DOI and cached."""
    if name not in _TABLES:
        path = fetch("figure_intermediates", files=[name])[name]
        _TABLES[name] = pd.read_csv(str(path))
    return _TABLES[name].copy()


class TrainFoldsUnavailable(RuntimeError):
    """Raised when a variant needs train-fold values the deposit does not carry."""

# One durable stem, reused for every output and for the staging entry.
STEM = "fig6_feature_combinations"
DPI = 300

# Uniform scale. Everything below is written at David's original sizes; SCALE multiplies
# the canvas AND every point-valued quantity (type, rules, caps, pads) by one factor, so
# the drawing is the same picture at a different size -- no reflow, no relative change
# between any two elements. Solved at run time by calibrate() so the combined figure comes
# out at the display-item width. Added 7 Sep 2026: the figure was 462 mm wide, and the
# page geometry was the one display-item rule that a pure resize could fix without
# touching the design.
TARGET_MM = STYLE.display_item_width_mm     # the ink; save() adds the border around it
MM_PER_IN = 25.4
SCALE = 1.0

# The x tick labels and the legend, at fixed sizes rather than scaled ones. Under the
# uniform scale they landed at 4.45 pt (David drew both at 11 pt against 14-15 pt
# neighbours, and that ratio survives a resize), were set to an even 4 pt, and are now at
# the Nature Genetics 5 pt floor: Nimbus Sans is Helvetica-metric and far narrower than the
# DejaVu Sans this figure used until the typeface was switched, so 5 pt fits where it did
# not before. With these two at 5 pt no text in the figure is below the floor.
TICK_PT = 5.0
LEGEND_PT = 5.0


def S(points):
    """A point-valued size at the current scale."""
    return points * SCALE


def apply_scale():
    """Scale the sizes matplotlib takes from rcParams rather than from our calls."""
    for key, base in (("axes.linewidth", 0.8), ("grid.linewidth", 0.8),
                      ("xtick.major.width", 0.8), ("ytick.major.width", 0.8),
                      ("xtick.major.size", 3.5), ("ytick.major.size", 3.5),
                      ("xtick.major.pad", 3.5), ("ytick.major.pad", 3.5),
                      ("axes.labelpad", 4.0), ("axes.titlepad", 6.0),
                      ("patch.linewidth", 1.0)):
        matplotlib.rcParams[key] = base * SCALE
    # NOT the legend metrics. legend.handlelength / borderpad / labelspacing /
    # handletextpad / borderaxespad are in FONT-SIZE units, so they already follow the type
    # down and scaling them a second time is wrong: it shortened the key handles to 0.81
    # font-widths and turned David's rectangles into squares. Points-valued things get
    # scaled; font-relative things are left alone.

# Display order of the feature combinations (legend order, top to bottom).
COV_ORDER = [
    "Covar",
    "Covar + mTIFs",
    "Covar + dTIFs",
    "Covar + LVs",
    "Covar + mTIFs + dTIFs",
    "Covar + mTIFs + LVs",
    "Covar + dTIFs + LVs",
    "Covar + mTIFs + dTIFs + LVs",
]

with open(str(CONFIG_PATH), "r", encoding="utf-8") as f:
    color_dict = yaml.safe_load(f)["colors"]

COV_COLORS = {
    "Covar": "#696969",
    "Covar + mTIFs": color_dict.get("mtifs", "#4DB8FF"),
    "Covar + dTIFs": color_dict.get("dtifs", "#FF7F00"),
    "Covar + LVs": color_dict.get("lvs", "#4DAF4A"),
    "Covar + mTIFs + dTIFs": "#FF00FF",
    "Covar + mTIFs + LVs": "#00FFFF",
    "Covar + dTIFs + LVs": "#FFFF00",
    "Covar + mTIFs + dTIFs + LVs": "#D3D3D3",
}

# Outcomes per panel, listed top -> bottom as they appear in the figure.
PANEL_A_OUTCOMES = {"DBP_both": "DBP", "SBP_both": "SBP", "HbA1c_both": "HbA1c", "BMI_both": "BMI"}
PANEL_B_OUTCOMES = {
    "eye_presbyopia_both": "Presbyopia",
    "eye_amblyopia_both": "Amblyopia",
    "age_glaucoma_both": "Glaucoma",
    "age_cataract_both": "Cataract",
}
PANEL_C_OUTCOMES = {
    "age_high_BP_both": "Hypertension",
    "age_heartattack_both": "Heart attack",
    "age_diabetes_both": "Diabetes",
    "age_death": "Death",
}


def normalize_cov(raw: str) -> str:
    """Map raw covariate strings (any case, '_' or ' ' separated) to display labels."""
    key = str(raw).strip().lower().replace("_", " ")
    key = " ".join(key.split())
    mapping = {
        "covars": "Covar",
        "covars + mtifs": "Covar + mTIFs",
        "covars + dtifs": "Covar + dTIFs",
        "covars + lvs": "Covar + LVs",
        "covars + mtifs + dtifs": "Covar + mTIFs + dTIFs",
        "covars + mtifs + lvs": "Covar + mTIFs + LVs",
        "covars + dtifs + lvs": "Covar + dTIFs + LVs",
        "covars + mtifs + dtifs + lvs": "Covar + mTIFs + dTIFs + LVs",
    }
    return mapping.get(key, str(raw))


# ----------------------------------------------------------------------------
# Aggregation -> tidy table: outcome | covariates | mean | err_lo | err_hi | n
# ----------------------------------------------------------------------------
def build_panel_a() -> pd.DataFrame:
    """Test R^2: mean of 5 folds; whisker = t(0.975, df=4) * SEM (95% CI half-width), drawn upper-only."""
    df = deposit(LINREG_CSV)
    df = df[df["disease"].isin(PANEL_A_OUTCOMES)].copy()
    fold_cols = sorted(c for c in df.columns if c.startswith("r2_test_fold_"))
    rows = []
    for _, r in df.iterrows():
        vals = r[fold_cols].to_numpy(dtype=float)
        n = len(vals)
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n)
        half = t_dist.ppf(0.975, df=n - 1) * sem
        rows.append(dict(
            outcome=PANEL_A_OUTCOMES[r["disease"]],
            covariates=normalize_cov(r["covariates"]),
            mean=mean, sem=sem, err_lo=0.0, err_hi=half, n=n, folds=list(vals),
        ))
    return pd.DataFrame(rows)


def build_auc_panel(outcomes: dict, pool_train_and_test: bool) -> pd.DataFrame:
    """AUC: mean over folds; whisker = t(0.975, df=n-1) * SEM (95% CI half-width), upper only.

    The same interval as panel a: at n = 5 a 1-SEM bar is under 40% of the interval that
    covers 95%. Both are optimistic -- the five folds share 3/4 of their training data
    pairwise, so the across-fold SEM underestimates the variance of the CV estimate.

    pool_train_and_test=True  -> train and test folds together (10 AUCs per bar).
    pool_train_and_test=False -> held-out test folds only (5 AUCs per bar), the figure.
    """
    df = deposit(LOGREG_FOLD_CSV)
    df = df[df["disease"].isin(outcomes)].copy()
    # The deposit table is one row per disease x feature set with the folds in columns;
    # the CSV this replaced was one row per fold, with a `set` column. Melt back to the
    # long form so the aggregation below is the same code operating on the same numbers:
    # the five test folds, or all ten, exactly as the `set == "test"` filter used to decide.
    prefixes = ("auc_test_fold_", "auc_train_fold_") if pool_train_and_test else ("auc_test_fold_",)
    fold_cols = sorted(c for c in df.columns if c.startswith(prefixes))
    if pool_train_and_test and not any(c.startswith("auc_train_fold_") for c in fold_cols):
        raise TrainFoldsUnavailable(
            "{} carries no auc_train_fold_* columns, so the train and test folds cannot be "
            "pooled.".format(LOGREG_FOLD_CSV))
    df = df.melt(id_vars=["disease", "covariates"], value_vars=fold_cols,
                 var_name="fold", value_name="auc")
    df["covariates"] = df["covariates"].map(normalize_cov)
    rows = []
    for (dz, cov), sub in df.groupby(["disease", "covariates"]):
        vals = sub["auc"].to_numpy(dtype=float)
        n = len(vals)
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        half = t_dist.ppf(0.975, df=n - 1) * sem if n > 1 else 0.0
        rows.append(dict(outcome=outcomes[dz], covariates=cov, mean=mean, sem=sem,
                         err_lo=0.0, err_hi=half, n=n, folds=list(vals)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
BAR_HEIGHT = 0.6


def draw_panel(ax, df: pd.DataFrame, outcomes_top_to_bottom, xlabel, ylabel, title, xlim,
               panel_label=None):
    """Grouped horizontal bars, one group per outcome, one bar per feature combination."""
    df = df.copy()
    df["covariates"] = pd.Categorical(df["covariates"], categories=COV_ORDER, ordered=True)
    # matplotlib puts y=0 at the bottom, so reverse to get the requested top->bottom order.
    outcomes_bottom_to_top = list(outcomes_top_to_bottom)[::-1]
    covs_bottom_to_top = COV_ORDER[::-1]
    offset = BAR_HEIGHT / (len(covs_bottom_to_top) / 1.3)

    # One pass over the frame instead of a boolean mask per (outcome, feature set): the
    # inner pair of loops asks for 32 cells per panel and each mask was a full scan.
    cells = {(rec["outcome"], rec["covariates"]): rec
             for rec in df.to_dict("records")}
    bars = []          # (fold values, bar centre, bar thickness, face colour)
    for i, outcome in enumerate(outcomes_bottom_to_top):
        for j, cov in enumerate(covs_bottom_to_top):
            row = cells.get((outcome, cov))
            if row is None:
                continue
            y = i - BAR_HEIGHT / 2 + j * offset
            m = float(row["mean"])
            lo = float(row["err_lo"])
            hi = float(row["err_hi"])
            # The whisker is drawn on BOTH versions. Nature's checklist asks only that the
            # individual points be overlaid ("always for n <= 10"); nothing in it asks for
            # the error bar to come off, and the legend defines the 95% CI explicitly, so
            # dropping it would leave figure and legend disagreeing.
            #
            # It is stroked at BAR_EDGE_LW, the bar's own outline weight, so the whisker and
            # the outline it grows out of read as one piece of furniture rather than two
            # weights. It sits BEHIND the dots at zorder 3: drawing it over them was tried
            # on 7 Sep 2026 and looked worse, so the dots are semi-transparent instead
            # (plot_styles.fold_markers.alpha) and the whisker reads through them.
            ax.barh(y=y, width=m, height=offset, edgecolor="black", linewidth=BAR_EDGE_LW(),
                    zorder=2, color=COV_COLORS[cov], label=cov if i == 0 else None)
            ax.errorbar(x=m, y=y, xerr=[[lo], [hi]], fmt="none", ecolor="black",
                        elinewidth=BAR_EDGE_LW(), capsize=S(3), capthick=BAR_EDGE_LW(),
                        alpha=0.9, zorder=3)
            if SHOW_FOLDS:
                bars.append((np.asarray(row["folds"], dtype=float), y, offset,
                             pop(COV_COLORS[cov])))

    if bars:
        # All 160 fold points of a panel as ONE collection. Drawn as a Line2D per bar they
        # were 96 artists per panel and dominated the profile (8,808 Line2D.draw calls
        # across a run); a single PathCollection draws the same marks in one pass and
        # writes a smaller PDF.
        xs = np.concatenate([b[0] for b in bars])
        ys = np.concatenate([np.full(len(b[0]), b[1]) for b in bars])
        faces = [b[3] for b in bars for _ in range(len(b[0]))]
        # size and linewidth are set in lay_out_swarms, once the bar's thickness in points
        # is known; seeded here only so the artist exists
        collection = ax.scatter(xs, ys, s=1.0, facecolors=faces, edgecolors="black",
                                linewidths=FOLD_EDGE_LW, alpha=FOLD_ALPHA, zorder=4)
        _SWARMS.append((collection, ax, bars))

    ax.set_xlabel(xlabel, fontsize=S(14))
    ax.set_ylabel(ylabel, fontsize=S(14))
    ax.set_title(title, fontsize=S(15))
    ax.set_yticks(range(len(outcomes_bottom_to_top)))
    ax.set_yticklabels(outcomes_bottom_to_top, fontsize=S(13), rotation=45)
    ax.tick_params(axis="x", labelsize=TICK_PT)
    ax.set_ylim(-0.5, len(outcomes_bottom_to_top) - 0.5)
    ax.set_xlim(*xlim)
    if tuple(xlim) in XTICKS:
        ticks, fmt = XTICKS[tuple(xlim)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt.format(t) for t in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    if panel_label:
        ax.text(-0.22, 1.02, panel_label, transform=ax.transAxes, fontsize=PANEL_LETTER_PT,
                fontweight=PANEL_LETTER_WEIGHT, va="bottom", ha="left")


def legend_handles():
    from matplotlib.patches import Patch
    
    return [Patch(facecolor=COV_COLORS[c], edgecolor="black", linewidth=S(1), label=c)
            for c in COV_ORDER]


def ink_bbox(fig, probe_path):
    """The figure's true ink box, in inches.

    matplotlib's tight bbox is not the ink: it is built from font metrics and from the
    bounding rectangle of rotated text, whose empty corners leave a fraction of a
    millimetre of white -- here 0.7 mm below the x-axis label, enough to fail the
    zero-margin rule. So save once, ask ghostscript what it can actually see, and trim by
    the difference. Same save-measure-trim as 05_figures/main/fig4_per_tif.py.

    The tight box stops at the legend frame's path, not its stroke, so the probe is padded
    first and trimmed back to what ghostscript sees; save() then adds the border.
    """
    from src.display_item import ink_margins_mm                     # noqa: E402
    PAD_IN = 2.0 / 72.0
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    box = Bbox.from_extents(box.x0 - PAD_IN, box.y0 - PAD_IN,
                            box.x1 + PAD_IN, box.y1 + PAD_IN)
    fig.savefig(probe_path, bbox_inches=box, pad_inches=0, facecolor="white")
    t = lambda mm: max(0.0, mm) / MM_PER_IN
    left, bottom, right, top = ink_margins_mm(probe_path)
    return Bbox.from_extents(box.x0 + t(left), box.y0 + t(bottom),
                             box.x1 - t(right), box.y1 - t(top))


def save(fig, basename, trim=False):
    """Write PDF + PNG at the same physical size.

    `trim` buys an exact zero-margin crop with a probe render and a ghostscript call, so it
    is spent only on the combined figure -- the one that is staged as submission artwork
    and has to satisfy the display-item geometry. The single panels and the legend are
    components: matplotlib's own tight box is close enough and costs neither.
    """
    # pad_inches=0 either way: matplotlib's default 0.1 in would leave ~2.5 mm of white on
    # every side, which is not part of the drawing.
    box = ink_bbox(fig, f"{basename}.pdf") if trim else "tight"
    if trim:
        # the paper-wide white border around the ink (plot_styles.natgen)
        pad = STYLE.display_item_border_mm / MM_PER_IN
        box = Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
    for ext in ("png", "pdf"):
        fig.savefig(f"{basename}.{ext}", dpi=DPI, bbox_inches=box, pad_inches=0,
                    facecolor="white")
    plt.close(fig)
    print(f"Saved: {basename}.png / .pdf")


def panel_spec(pool_train_and_test: bool):
    """(letter, frame, outcomes, xlabel, ylabel, title, xlim) for each of the three panels."""
    # "R squared" as mathtext, the project convention: an italic R with the 2 raised clear
    # of the cap line, at mathtext's own 0.7 of the base. It was set as an italic R plus
    # the SUPERSCRIPT TWO glyph (U+00B2) until 8 Sep 2026, to hold the 2 at the label's
    # nominal type size; but Nimbus Sans draws that glyph small and low, so it read as an
    # inline 2 rather than an exponent. Nature Genetics' 5 pt floor is a floor on TYPE, and
    # an exponent set smaller than the type it belongs to is what a superscript is -- so it
    # does not bind here, and forcing the 2 up to 5 pt only made it too big.
    R2 = "$R^{2}$"
    auc_label = "Mean AUC across Folds" if pool_train_and_test else "Mean test AUC across Folds"
    r2_label = f"Mean {R2} across Folds" if pool_train_and_test else f"Mean test {R2} across Folds"
    return [
        ("a", build_panel_a(), list(PANEL_A_OUTCOMES.values()), r2_label, "Risk Factors",
         f"{R2} Scores for Risk Factors", (0.0, 0.35)),
        ("b", build_auc_panel(PANEL_B_OUTCOMES, pool_train_and_test),
         list(PANEL_B_OUTCOMES.values()), auc_label, "Diseases",
         "AUC Scores for Ocular Diseases", (0.4, 1.0)),
        ("c", build_auc_panel(PANEL_C_OUTCOMES, pool_train_and_test),
         list(PANEL_C_OUTCOMES.values()), auc_label, "Diseases",
         "AUC Scores for General Diseases", (0.4, 1.0)),
    ]


def combined_figure(panels):
    """a | b | c with the legend on the right -- the submitted display item."""
    fig, axes = plt.subplots(1, 3, figsize=(17 * SCALE, 7.5 * SCALE))
    for ax, (label, df, outcomes, xlabel, ylabel, title, xlim) in zip(axes, panels):
        draw_panel(ax, df, outcomes, xlabel, ylabel, title, xlim, panel_label=label)
    fig.legend(handles=legend_handles(), title="Feature Combinations", loc="center left",
               bbox_to_anchor=(0.90, 0.80), fontsize=LEGEND_PT, title_fontsize=LEGEND_PT,
               frameon=True)
    fig.tight_layout(rect=(0, 0, 0.90, 1))
    lay_out_swarms(fig)
    return fig


def calibrate(panels):
    """Set SCALE so the combined figure saves at exactly TARGET_MM wide.

    The layout is uniform in SCALE, so page width is linear in it and one trial render
    pins the factor down; a second pass corrects the last rounding.
    """
    global SCALE
    sys.path.insert(0, str(REPO_ROOT))
    from src.display_item import _page_size_mm                      # noqa: E402
    tmp = os.path.join(FIG_DIR, "_calibration")
    def width_at(scale):
        global SCALE
        SCALE = scale
        apply_scale()
        fig = combined_figure(panels)
        # ink_bbox has already measured the page it will produce, so take the width from
        # the box rather than writing the file a second time and asking pdfinfo -- one
        # savefig and one subprocess per pass instead of two of each.
        box = ink_bbox(fig, tmp + ".pdf")
        plt.close(fig)
        return box.width * MM_PER_IN

    # Width is AFFINE in the scale, not proportional: the tick labels and the panel letters
    # are pinned to fixed point sizes, so they keep their width while everything else
    # shrinks. A multiplicative update assumes a zero intercept and oscillates; a secant
    # step solves an affine relation outright.
    s0 = SCALE
    w0 = width_at(s0)
    s1 = s0 * TARGET_MM / w0
    w1 = width_at(s1)
    for _ in range(8):
        if abs(w1 - TARGET_MM) <= 0.05:
            break
        if w1 == w0:
            raise SystemExit("calibration stalled at {:.2f} mm".format(w1))
        s0, w0, s1 = s1, w1, s1 + (TARGET_MM - w1) * (s1 - s0) / (w1 - w0)
        w1 = width_at(s1)
    else:
        raise SystemExit("calibration did not converge: {:.2f} mm".format(w1))
    width = w1
    os.remove(tmp + ".pdf")
    print(f"calibrated: scale {SCALE:.5f}, combined page {width:.2f} mm "
          f"(target {TARGET_MM:.0f})")


def make_variant(variant: str, pool_train_and_test: bool, show_folds: bool = False,
                 components: bool = True):
    global SHOW_FOLDS
    SHOW_FOLDS = show_folds
    _SWARMS.clear()
    # the test-fold variant is the unsuffixed default
    tag = "" if variant == "test_only" else f"_{variant}"
    if show_folds:
        tag += "_folds"
    panel_a = build_panel_a()
    panel_b = build_auc_panel(PANEL_B_OUTCOMES, pool_train_and_test)
    panel_c = build_auc_panel(PANEL_C_OUTCOMES, pool_train_and_test)

    # Numbers behind the bars -> test_output/ (alongside the other model outputs)
    auc_metric = "AUC_train+test_pooled" if pool_train_and_test else "AUC_test"
    # Both versions now draw the whisker; the folds version adds the five points on top.
    whisker = ("upper 95% CI (t*SEM); individual folds also plotted" if show_folds
               else "upper 95% CI (t*SEM)")
    values = pd.concat([
        panel_a.assign(panel="a", metric="R2_test", whisker=whisker, source=LINREG_CSV),
        panel_b.assign(panel="b", metric=auc_metric, whisker=whisker,
                       source=LOGREG_FOLD_CSV),
        panel_c.assign(panel="c", metric=auc_metric, whisker=whisker,
                       source=LOGREG_FOLD_CSV),
    ], ignore_index=True)
    values["covariates"] = pd.Categorical(values["covariates"], categories=COV_ORDER, ordered=True)
    values = values.sort_values(["panel", "outcome", "covariates"])
    values = values[["panel", "metric", "outcome", "covariates", "mean", "sem", "err_lo", "err_hi",
                     "n", "whisker", "source"]]
    values_csv = os.path.join(DATA_DIR, f"{STEM}{tag}_values.csv")
    values.to_csv(values_csv, index=False)
    print(f"\n=== {variant} ===")
    print(values[["panel", "outcome", "covariates", "mean", "sem", "n"]].round(4).to_string(index=False))
    print(f"Saved data: {values_csv}")

    base = os.path.join(FIG_DIR, f"{STEM}{tag}")

    panels = [(letter, frame, outcomes, xlabel, ylabel, title, xlim)
              for (letter, _, outcomes, xlabel, ylabel, title, xlim), frame
              in zip(panel_spec(pool_train_and_test), (panel_a, panel_b, panel_c))]

    # --- individual panels (no legend; use the stand-alone legend file) ---
    for label, df, outcomes, xlabel, ylabel, title, xlim in (panels if components else []):
        fig, ax = plt.subplots(figsize=(5.5 * SCALE, 8.5 * SCALE))
        draw_panel(ax, df, outcomes, xlabel, ylabel, title, xlim)
        fig.tight_layout()
        lay_out_swarms(fig)
        save(fig, f"{base}_panel_{label}")

    # --- stand-alone legend ---
    if not components:
        save(combined_figure(panels), base, trim=True)
        return
    fig = plt.figure(figsize=(3.2 * SCALE, 2.6 * SCALE))
    fig.legend(handles=legend_handles(), title="Feature Combinations", loc="center",
               fontsize=LEGEND_PT, title_fontsize=LEGEND_PT, frameon=True)
    save(fig, f"{base}_legend")

    # --- combined a | b | c with legend on the right ---
    save(combined_figure(panels), base, trim=True)


if __name__ == "__main__":
    # One scale for every output, solved on the figure that has to hit the width. The
    # overlaid dots sit inside the bars, so they do not move the ink box; calibrating with
    # them on keeps the submitted version the one the width was solved for.
    apply_scale()
    SHOW_FOLDS = True
    _SWARMS.clear()
    calibrate(panel_spec(False))

    # Two versions side by side:
    #   (1) plain bars;
    #   (2) the same bars with the five individual fold values overlaid, in Figure 2b's
    #       style, which is what Nature's statistical guidance asks of a bar chart.
    make_variant("test_only", pool_train_and_test=False, show_folds=False)
    make_variant("test_only", pool_train_and_test=False, show_folds=True)
    print("\nDone.")
