# Figure 3, SNP panel: Manhattan (a) + QQ (b) + feature-set Venn (c).
#
# Authors: a     David Presby (github.com/presbyd), Michael Beyeler (github.com/mjbeyeler)
#          b, c  Michael Beyeler (github.com/mjbeyeler)
#
# Self-contained: fetches its inputs from the public data deposit by DOI (downloading
# them on first run, reusing the local cache afterwards) and builds every panel from
# those files. No access-controlled data, no cluster, no GPU.
#
# Needs Python as well as R: the deposit fetch and panel c both shell out to it (set
# MVP_PYTHON if `python` is not the interpreter carrying this project's dependencies).
#
#   Rscript 05_figures/main/fig3_snp.R
#
# The Manhattan, the QQ and the Venn are built in one process from the deposited inputs.
# The QQ's rank column is looked up by name: "unthinned_rank", or "X1" where a thinning
# output left it unnamed. The deposited thinned files are the MAF > 0.01 / INFO > 0.8
# high-quality variant set that the reported association results use.
#
# The figure is vector throughout. Panel c is drawn from the Venn's seven deposited region
# counts by 05_figures/lib/render_venn3.py and imported as paths. PDFs are written with
# cairo_pdf alongside the PNGs, so the deliverable meets the journal's vector requirement
# while the PNGs stay available for drafts and for pasting into documents.

suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(scales)
    library(yaml)
    library(cowplot)
    library(rsvg)
    library(grImport2)
})

# ---------------------------------------------------------------------------
# Locate the repo, fetch inputs
# ---------------------------------------------------------------------------
script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", grep("^--file=", script_args, value = TRUE)[1])
lib_dir <- if (!is.na(script_path)) {
    file.path(dirname(normalizePath(script_path)), "..", "lib")
} else {
    "05_figures/lib"
}

source(file.path(lib_dir, "fetch_data.R"))
source(file.path(lib_dir, "fast_qq_double_log.R"))
source(file.path(lib_dir, "natgen_style.R"))

repo_root <- find_repo_root()
config <- read_config(file.path(repo_root, "config.yaml"))
color_codes <- config$colors
# Alternate chromosomes are the same colour mixed towards white, so the pair can never
# drift apart -- see colors.alternate_chromosome_tint.
tint <- function(hex, towards_white) {
    rgb <- grDevices::col2rgb(hex)[, 1] / 255
    grDevices::rgb(t(rgb + towards_white * (1 - rgb)))
}
pale <- lapply(color_codes[c("mtifs", "dtifs", "lvs")], tint,
               towards_white = config$colors$alternate_chromosome_tint)
style <- natgen_style(config)
burden <- gwas_burden(config)
register_font_family(style)

message("Fetching Figure 3 inputs from the data deposit ...")
inputs <- fetch_data("figure_intermediates", files = c(
    "Fig3_01_a_snp_manhattan_dtif_thinned.csv",
    "Fig3_02_a_snp_manhattan_mtif_thinned.csv",
    "Fig3_03_a_snp_manhattan_lv_thinned.csv",
    "Fig3_04_c_snp_featureset_venn_counts.csv"
))

out_dir <- figure_dir("main_figures")   # config: output.figures_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Backgrounds are stated explicitly rather than inherited, because the default moved
# between ggplot2 versions: theme_minimal() left plot.background blank under 3.5.1 and
# paints it opaque white under 4.0.3. The published panels are transparent, and the
# difference is invisible on a white page but not on any other, so relying on the
# default silently changes the figure when the environment is rebuilt.
transparent_bg <- theme(plot.background = element_blank())

save_panel <- function(file, plot, ...) {
    ggsave(file.path(out_dir, file), plot, bg = "transparent", ...)
}

# cairo_pdf with the family pinned. Without `family=`, cairo picks the system default sans
# for anything the theme did not explicitly claim -- which is how DejaVu Sans ended up
# embedded next to the document face in the previous PDFs. Nature wants one sans face,
# Helvetica or Arial; the family comes from config (Nimbus Sans, Helvetica-metric).
cairo_pdf_natgen <- function(filename, width, height, ...) {
    grDevices::cairo_pdf(filename, width = width, height = height,
                         family = style$family, ...)
}

# ---------------------------------------------------------------------------
# Layout options
# ---------------------------------------------------------------------------
# Geometry and type both come from config.yaml -- see 05_figures/lib/natgen_style.R for
# the two Nature documents behind the numbers. The width used to be a hardcoded 190 mm,
# which is over both the 180 mm 2-column width and the 179 mm display-item cap.
total_width_mm  <- style$width_mm
total_height_mm <- style$strip_height_mm$snp

left_width_weight  <- style$left_weight    # Manhattan
right_width_weight <- style$right_weight   # QQ + Venn column

hi_dpi   <- style$hi_dpi
docs_dpi <- style$docs_dpi

labels_snp <- list(left = "a", qq = "b", venn = "c")

# Every size this script sets, checked against the band before anything is drawn. The
# panel letter is deliberately absent -- it is an 8 pt bold exception, see natgen_style.R.
assert_text_sizes(style, c(
    axis_title = style$axis_label_pt,
    tick_label = style$tick_label_pt,
    venn_count = style$venn_label_pt
))

# ---------------------------------------------------------------------------
# Read the thinned SNP sets
# ---------------------------------------------------------------------------
# Only chr/pos/pp are plotted; the rank column is needed for the QQ's expected
# quantiles and is looked up by name (see header note on X1 / unthinned_rank).
read_thinned <- function(path) {
    df <- read.csv(path)
    missing <- setdiff(c("chr", "pos", "pp"), names(df))
    if (length(missing)) {
        stop(sprintf("%s lacks column(s): %s", basename(path), paste(missing, collapse = ", ")))
    }
    rank_col <- intersect(c("unthinned_rank", "X1"), names(df))
    if (!length(rank_col)) {
        stop(sprintf("%s has no rank column (expected 'unthinned_rank' or 'X1')", basename(path)))
    }
    df$rank <- df[[rank_col[1]]]
    keep <- c("chr", "pos", "pp", "rank")
    # original_index is the row index into the pooled pre-thinning HIGH-QUALITY
    # variant list; its maximum sizes the QQ null (see the n_variants block below).
    if ("original_index" %in% names(df)) keep <- c(keep, "original_index")
    df[, keep]
}

msnp  <- read_thinned(inputs[["Fig3_02_a_snp_manhattan_mtif_thinned.csv"]])
dsnp  <- read_thinned(inputs[["Fig3_01_a_snp_manhattan_dtif_thinned.csv"]])
lvsnp <- read_thinned(inputs[["Fig3_03_a_snp_manhattan_lv_thinned.csv"]])

message(sprintf("  mTIF %d, dTIF %d, LV %d thinned points",
                nrow(msnp), nrow(dsnp), nrow(lvsnp)))

# ---------------------------------------------------------------------------
# Panel a -- Manhattan
# ---------------------------------------------------------------------------
manhattan <- rbind(
    transform(msnp[, c("chr", "pos", "pp")],  condition = "measured"),
    transform(dsnp[, c("chr", "pos", "pp")],  condition = "deep"),
    transform(lvsnp[, c("chr", "pos", "pp")], condition = "lvs")
)

# GRCh38 chromosome lengths, used only for x-axis layout
chromosome_lengths <- data.frame(
    chr = 1:22,
    length = c(248956422, 242193529, 198295559, 190214555, 181538259, 170805979,
               159345973, 145138636, 138394717, 133797422, 135086622, 133275309,
               114364328, 107043718, 101991189, 90338345, 83257441, 80373285,
               58617616, 64444167, 46709983, 50818468)
) %>%
    mutate(cumulative_start = c(0, cumsum(length)[-n()]),
           midpoint = cumulative_start + length / 2)

# Cap the y-axis at 300 (values above are plotted at the ceiling)
manhattan[manhattan$pp > 300, "pp"] <- 300

manhattan <- manhattan %>%
    left_join(chromosome_lengths, by = "chr") %>%
    mutate(adjusted_pos = pos + cumulative_start)

thin_gray_boxes <- chromosome_lengths %>%
    filter(chr %% 2 == 1) %>%
    mutate(xmin = cumulative_start, xmax = cumulative_start + length)

# Label every chromosome except 19 and 21 (too narrow to read)
label_chromosomes <- chromosome_lengths %>% filter(chr != 19 & chr != 21)

# Saturated on odd chromosomes, pale on even, per feature set
manhattan <- manhattan %>%
    mutate(color = case_when(
        condition == "deep"     & chr %% 2 == 1 ~ color_codes$dtifs,
        condition == "deep"     & chr %% 2 == 0 ~ pale$dtifs,
        condition == "measured" & chr %% 2 == 1 ~ color_codes$mtifs,
        condition == "measured" & chr %% 2 == 0 ~ pale$mtifs,
        condition == "lvs"      & chr %% 2 == 1 ~ color_codes$lvs,
        condition == "lvs"      & chr %% 2 == 0 ~ pale$lvs
    ))

facet_levels <- c("Measured tangible image features (mTIFs)",
                  "Deep tangible image features (dTIFs)",
                  "Latent variables (LVs)")

manhattan <- manhattan %>%
    mutate(condition = recode(condition,
                              measured = facet_levels[1],
                              deep     = facet_levels[2],
                              lvs      = facet_levels[3]))
manhattan$condition <- factor(manhattan$condition, levels = facet_levels)

# Study-wide significance: the genome-wide threshold Bonferroni-corrected for the number of
# INDEPENDENT tests in each feature set -- 17 TIFs, and 177 effective LVs rather than the
# 1024 raw ones (the LVs are collinear; 177 explains 95 % of their variance). Read from
# config.yaml -> gwas so this panel, the Venn and the gene panels cannot drift apart.

line_data <- data.frame(
    condition = factor(facet_levels, levels = facet_levels),
    yintercept = -log10(burden$genome_wide_p /
                        c(burden$n_traits, burden$n_traits, burden$n_effective_lvs))
)

# The bottom of the data range is also the baseline the panel sits on, so it is a break:
# the black line at the foot of every facet now carries a number like every other tick.
Y_FLOOR <- 2
# 4, 32, 256: the axis is log2, so equal spacing means a constant
# ratio -- x8, x8 -- and the earlier 2, 16, 256 (x8, x16) was visibly uneven. The floor
# of the window stays 2; it is the baseline, not a break.
manhattan_y_breaks <- c(4, 32, 256)
Y_CEILING <- 300
manhattan_ylim <- padded_ylim(Y_FLOOR, Y_CEILING, style$manhattan_y_pad_frac,
                             style$manhattan_y_pad_frac_top)
manhattan_y_limit_upper <- max(manhattan_y_breaks) + 20
chromosomes_to_hide <- c(17)

manhattan_plot <- ggplot(data = manhattan) +
    # ymin = 0, NOT -Inf: the y scale is log2, and log2(-Inf) is NaN, which ggplot
    # silently drops, and with it the chromosome bands the legend
    # describes. log2(0) is -Inf, which the scale keeps (censor() only drops finite
    # out-of-range values) and the coord squishes to the panel edge; Inf likewise at the
    # top. A finite extent does not work either: anything above the scale's upper limit
    # is censored, and the padded window reaches past it.
    geom_rect(data = thin_gray_boxes,
              aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = Inf),
              fill = "darkgrey", alpha = 0.1) +
    geom_point(aes(x = adjusted_pos, y = pp, color = color),
               size = style$manhattan_point_pt) +
    # The significance line goes OVER the points, as published and as GWAS convention has
    # it: its one job is to be swept across the panel to see which peaks clear it, and
    # that needs it unbroken exactly where the cloud is densest -- at the threshold.
    # Dashed at half the published weight (config manhattan_threshold_linewidth) and at
    # full strength: it was the 0.4 mm width, not the dashes, that made the old line read
    # as a rule cutting the peaks off, and the alpha = 0.6 that used to compensate only
    # made it muddy.
    geom_hline(data = line_data, aes(yintercept = yintercept),
               color = "red", linetype = style$threshold_lty,
               linewidth = style$threshold_lw) +
    scale_color_identity() +
    scale_x_continuous(
        # A tick mark on EVERY chromosome; only the labels of the narrow ones (17, 19,
        # 21) are blank, so the marks still show where each chromosome sits.
        breaks = chromosome_lengths$midpoint,
        labels = ifelse(chromosome_lengths$chr %in% c(chromosomes_to_hide, 19, 21), "",
                        chromosome_lengths$chr),
        expand = c(0, 0)
    ) +
    scale_y_continuous(
        trans = scales::log_trans(base = 2),
        breaks = manhattan_y_breaks,
        limits = c(NA, manhattan_y_limit_upper),
        expand = c(0, 0)
    ) +
    labs(x = "Chromosome", y = axis_label_neglog10p()) +
    natgen_theme(style) +
    theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(),
          axis.ticks.x = element_line(color = "black", linewidth = style$axis_lw_mm),
          axis.ticks.y = element_line(color = "black", linewidth = style$axis_lw_mm),
          axis.line.x  = element_line(color = "black", linewidth = style$axis_lw_mm),
          axis.line.y  = element_line(color = "black", linewidth = style$axis_lw_mm),
          legend.position = "none",
          # the Manhattan draws on the page, not on a white card -- unlike the QQ,
          # whose white panel with a border is part of its design
          panel.background = element_blank()) +
    transparent_bg +
    coord_cartesian(ylim = manhattan_ylim) +
    # `axes = "all_x"` makes ggplot draw its OWN x axis -- the line at the lower y limit,
    # and the ticks -- on every facet, instead of only on the bottom one. That is what the
    # baseline should be: the axis, from the tool that draws the plot, not an hline at an
    # arbitrary y. Drawing it by hand is what put TWO lines in the bottom facet, the
    # hand-drawn one at y = 2 and ggplot's own a little below it at the padded limit.
    # `axis.labels = "margins"` keeps the chromosome numbers on the bottom facet only.
    facet_wrap(~condition, nrow = 3, axes = "all_x", axis.labels = "margins")

# ---------------------------------------------------------------------------
# Panel b -- QQ
# ---------------------------------------------------------------------------
# Expected quantiles come from each point's pre-thinning rank over the full variant
# count, which is why the rank column has to travel with the thinned points.
# The denominator is the number of tests ACTUALLY RUN: the high-quality variant set
# (MAF > 0.01, INFO > 0.8) each feature set was thinned from, times its traits. It is
# derived per file from original_index -- the row index into that pre-thinning variant
# list -- exactly as panel e takes its denominator from the gene tables' own
# unthinned_total, so neither panel carries a literal that can drift from the deposit.
#
# It was 15600563, the RAW pre-QC variant count: 1.99x the high-quality set, which lifted
# every expected quantile by ~0.3. At the thinning cut (observed = 2.00) the null came out
# at 2.17-2.24 -- ABOVE the data, i.e. deflation at P = 0.01, impossible in a GWAS that is
# inflated at the top. Derived, it lands at 1.89 / 1.87 / 1.94, below the cut.
#
# max(original_index) + 1 is a LOWER BOUND on that count (thinning need not retain the
# last variant), but the three sets agree to within ~900 of 7.83 million.
n_variants  <- function(df) max(df$original_index) + 1
n_snps_mtif <- n_variants(msnp) * burden$n_traits
n_snps_dtif <- n_variants(dsnp) * burden$n_traits
# 1024 LVs: a count of tests, not a significance threshold, so n_lvs not n_effective_lvs.
n_snps_lv   <- n_variants(lvsnp) * burden$n_lvs

# Three ticks per axis with the top y tick ON the axis limit: the
# y window ends at 300 and every series is capped there (the LVs always were; the TIFs
# never reach it), so nothing is dropped by the scale's limits.
QQ_Y_TOP <- 300
observed_pp <- lapply(list(msnp$pp, dsnp$pp, lvsnp$pp), function(v) pmin(v, QQ_Y_TOP))

expected_pp <- list(
    -log10(msnp$rank  / (n_snps_mtif + 1)),
    -log10(dsnp$rank  / (n_snps_dtif + 1)),
    -log10(lvsnp$rank / (n_snps_lv   + 1))
)

qq_colors <- c(color_codes$mtifs, color_codes$dtifs, color_codes$lvs)

# Axis treatment, from config (plot_styles.figures.fig3.qq_double_log). The linear branch
# is the published panel. The log2 branch shares the Manhattan's y ticks and needs a floor
# above zero, which log2 cannot reach: the points are thinned at observed -log10 P >= 2 and
# their expected quantiles start at 1.87, so window and cut both sit at 1.8.
if (style$qq_double_log) {
    qq_ticks <- list(c(2, 4, 8), c(4, 32, 256))
    qq_lims  <- list(c(1.8, 10.4), c(2, QQ_Y_TOP))
    qq_lower <- 1.8
} else {
    qq_ticks <- list(c(2, 6, 10), c(0, 150, QQ_Y_TOP))
    qq_lims  <- list(c(2, 10.4), c(0, QQ_Y_TOP))
    qq_lower <- 1.5
}

qq_snps <- fast_qq_double_log(
    observed_pp,
    expected = expected_pp,
    colors = qq_colors,
    ticks = qq_ticks,
    group_labels = c("measTIFs", "deepTIFs", "LVs"),
    show_legend = FALSE,
    axis_title_font_size = style$axis_label_pt,
    font_size = style$tick_label_pt,
    font_family = style$family,
    axis_linewidth_mm = style$axis_lw_mm,
    axis_tick_length_pt = style$axis_tick_len_pt,
    subsampling = FALSE,
    ax_lims = qq_lims,
    log10p_lower_limit = qq_lower,
    double_log_scale = style$qq_double_log
)

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
# Panel letters sit in two columns that run down the whole display item: a/d/g/k at
# panel_letter_x_mm from the left edge, b/c/e/f/j the same distance into the right column.
# The helper takes a fraction of the plot it is drawn on, so the
# millimetres are divided by that plot's own width.
add_label <- function(plot, label, x) add_panel_letter(plot, label, style, x = x)

mm_to_px <- function(mm, dpi) mm * dpi / 25.4

left_prop  <- left_width_weight / (left_width_weight + right_width_weight)
right_prop <- right_width_weight / (left_width_weight + right_width_weight)

left_width_mm  <- total_width_mm * left_prop
right_width_mm <- total_width_mm * right_prop

# The QQ box is a fixed height from config (plot_styles.figures.fig3.qq_height_mm) and the
# Venn takes the rest of the strip. The QQ panel stays
# square inside its box (aspect.ratio = 1 below), so it is the box height that sets its
# size, and the two knobs are coupled: qq_height_mm is set as large as the Venn's own
# height at venn_scale_mm_per_unit leaves room for, which the check below enforces.
right_top_height_mm    <- style$qq_height_mm
right_bottom_height_mm <- total_height_mm - right_top_height_mm
if (right_bottom_height_mm < 0) {
    stop("total_height_mm too small for plot_styles.figures.fig3.qq_height_mm")
}

qq_plot <- qq_snps$Plot +
    theme(plot.margin = margin(t = 25, r = 5, b = 5, l = 5), aspect.ratio = 1) +
    transparent_bg +
    # clip = "off" so the dots that sit on the 300 ceiling are drawn whole, not halved
    # by the panel edge (see fast_qq_double_log.R).
    coord_cartesian(xlim = qq_lims[[1]], ylim = qq_lims[[2]], clip = "off")

# Panel c: draw the Venn from its deposited region counts, then import it as vector
# paths and letterbox it into the panel box.
#
# The deposit ships the seven region counts, not a picture: matplotlib_venn derives the
# whole area-proportional layout from those counts, so they are a complete description of
# the panel and a deposited SVG would be a rendering nobody could check. The drawing code
# is 05_figures/lib/render_venn3.py, in this repository, and it is the same module the
# pipeline script uses -- see its docstring.
#
# grImport2 reads only Cairo-flavoured SVG, so the matplotlib file is passed through
# rsvg::rsvg_svg() first. That conversion DESTROYS TEXT -- verified: a matplotlib SVG
# written with svg.fonttype = "none" goes in with seven <text> elements and comes out with
# zero, every glyph converted to a path. Nature requires the text in a vector figure to
# remain editable, so the region counts are not in the SVG at all: render_venn3.py blanks
# them and writes their positions to a sidecar CSV, and they are drawn below as real text
# in the document font at a size we set in points.
#
# The box is wider than the diagram is tall relative to its width, so height binds and
# the drawing is centred horizontally with the remaining width left blank. This is the
# same letterboxing the old raster path did with image_extent(gravity = "center"), done
# in device coordinates instead of pixels.
svg_path <- file.path(tempdir(), "venn_panel_c.svg")
venn_labels_path <- file.path(tempdir(), "venn_panel_c_labels.csv")
venn_geometry_path <- file.path(tempdir(), "venn_panel_c_geometry.csv")
venn_status <- system2(
    mvp_python(repo_root),
    c(shQuote(file.path(lib_dir, "render_venn3.py")),
      shQuote(inputs[["Fig3_04_c_snp_featureset_venn_counts.csv"]]),
      shQuote(svg_path),
      "--labels-out", shQuote(venn_labels_path),
      "--geometry-out", shQuote(venn_geometry_path),
      "--config", shQuote(file.path(repo_root, "config.yaml")))
)
if (venn_status != 0) {
    stop(sprintf(paste0(
        "drawing panel c from its deposited counts failed (exit %d).\n",
        "  Set MVP_PYTHON if 'python' is not the interpreter with this project's deps."),
        venn_status))
}

right_width_px         <- as.integer(round(mm_to_px(right_width_mm, hi_dpi)))
right_bottom_height_px <- as.integer(round(mm_to_px(right_bottom_height_mm, hi_dpi)))

venn_cairo_svg <- file.path(tempdir(), "venn_panel_c_cairo.svg")
rsvg::rsvg_svg(svg_path, file = venn_cairo_svg)
venn_picture <- grImport2::readPicture(venn_cairo_svg)

# Place the diagram at the SHARED scale, not at whatever fills this box. All three of
# Figure 3's Venns (c, f, j) are drawn at plot_styles.figures.fig3.venn_scale_mm_per_unit
# millimetres per data unit, and matplotlib_venn normalises every diagram's total circle
# area to one square data unit, so the three panels carry the same physical area of ink.
# Fitting to the box instead made each panel a different size (see the config comment).
venn_geometry <- read.csv(venn_geometry_path)
venn_target_w_mm <- style$venn_scale_mm * venn_geometry$box_width_units
venn_target_h_mm <- style$venn_scale_mm * venn_geometry$box_height_units
if (venn_target_w_mm > right_width_mm || venn_target_h_mm > right_bottom_height_mm) {
    stop(sprintf(paste0(
        "the Venn does not fit its box at %.2f mm per unit: it needs %.1f x %.1f mm, ",
        "the box is %.1f x %.1f mm. Lower plot_styles.figures.fig3.venn_scale_mm_per_unit ",
        "(it is shared by panels c, f and j)."),
        style$venn_scale_mm, venn_target_w_mm, venn_target_h_mm,
        right_width_mm, right_bottom_height_mm))
}
venn_rel_width  <- venn_target_w_mm / right_width_mm
venn_rel_height <- venn_target_h_mm / right_bottom_height_mm
cat(sprintf("  panel c Venn: %.1f x %.1f mm at %.2f mm/unit (%.0f mm2 of circles)\n",
            venn_target_w_mm, venn_target_h_mm, style$venn_scale_mm,
            style$venn_scale_mm^2 * venn_geometry$total_circle_area_units2))

# expansion = 0 because the default 5% inset would shrink the diagram relative to the
# raster version. distort = TRUE looks alarming but is the correct choice here: the
# viewport passed to draw_grob() already has the diagram's own aspect ratio, so there is
# nothing to distort, and it makes the placement ours rather than grImport2's. Letting
# grImport2 do the fitting (distort = FALSE at full npc) does NOT work inside cowplot --
# its internal viewport is sized in square-npc units against the wrong reference and the
# diagram comes out several times too large, cropped by the panel.
venn_grob <- grImport2::pictureGrob(venn_picture, expansion = 0, distort = TRUE)

venn_labels <- read.csv(venn_labels_path, colClasses = c(text = "character"))
venn_x0 <- (1 - venn_rel_width) / 2
venn_y0 <- (1 - venn_rel_height) / 2

letter_x_left  <- style$letter_x_mm / left_width_mm
letter_x_right <- style$letter_x_mm / right_width_mm

manhattan_labeled <- add_label(manhattan_plot, labels_snp$left, letter_x_left)
qq_labeled        <- add_label(qq_plot, labels_snp$qq, letter_x_right)
venn_labeled      <- ggdraw() +
    draw_grob(venn_grob,
              x = venn_x0, y = venn_y0,
              width = venn_rel_width, height = venn_rel_height)
# The counts sit in the same npc box the diagram was placed in, so the fractions
# render_venn3.py recorded map straight onto it.
venn_labeled <- draw_venn_labels(venn_labeled, venn_labels, style,
                                 venn_x0, venn_y0, venn_rel_width, venn_rel_height)
venn_labeled <- venn_labeled +
    draw_label(labels_snp$venn, x = letter_x_right, y = 0.98, hjust = 0, vjust = 1,
               fontface = style$panel_label_face, fontfamily = style$family,
               size = style$panel_label_pt)

right_column <- plot_grid(qq_labeled, venn_labeled, nrow = 2, ncol = 1,
                          rel_heights = c(right_top_height_mm, right_bottom_height_mm))
full_panel <- plot_grid(manhattan_labeled, right_column, ncol = 2,
                        rel_widths = c(left_prop, right_prop))

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
px_to_in_hi <- function(px) px / hi_dpi

total_height_px_hi        <- as.integer(round(mm_to_px(total_height_mm, hi_dpi)))
right_top_height_px_hi    <- right_width_px
right_bottom_height_px_hi <- total_height_px_hi - right_top_height_px_hi

save_panel("fig3_panel_1_snp.png", full_panel,
       width = total_width_mm, height = total_height_mm, units = "mm", dpi = hi_dpi)
save_panel("fig3_a.png", manhattan_labeled,
       width = left_width_mm, height = total_height_mm, units = "mm", dpi = hi_dpi)
save_panel("fig3_b.png", qq_labeled,
       width = px_to_in_hi(right_width_px), height = px_to_in_hi(right_top_height_px_hi),
       dpi = hi_dpi)
save_panel("fig3_c.png", venn_labeled,
       width = px_to_in_hi(right_width_px), height = px_to_in_hi(right_bottom_height_px_hi),
       dpi = hi_dpi)

# Vector exports -- these are the submission files; the PNGs above are for drafts.
#
# cairo_pdf rather than the base pdf() device: it embeds the fonts it uses as subsets and
# supports the partial transparency of the alternating chromosome bands, which the base
# device would silently drop. Sizes are given in the same physical units as the PNGs, so
# the PDF and the PNG are the same figure at the same size, not two layouts.
save_panel("fig3_panel_1_snp.pdf", full_panel, device = cairo_pdf_natgen,
       width = total_width_mm, height = total_height_mm, units = "mm")
save_panel("fig3_a.pdf", manhattan_labeled, device = cairo_pdf_natgen,
       width = left_width_mm, height = total_height_mm, units = "mm")
save_panel("fig3_b.pdf", qq_labeled, device = cairo_pdf_natgen,
       width = right_width_mm, height = right_top_height_mm, units = "mm")
save_panel("fig3_c.pdf", venn_labeled, device = cairo_pdf_natgen,
       width = right_width_mm, height = right_bottom_height_mm, units = "mm")

# Google-Docs-friendly exports: same physical size, 96 dpi, so they land at the
# intended on-paper size when inserted at "Original size".
mm_to_px96 <- function(mm) mm_to_px(mm, docs_dpi)
px_to_in96 <- function(px) px / docs_dpi

total_px_96               <- as.integer(round(mm_to_px96(total_width_mm)))
left_px_96                <- as.integer(round(total_px_96 * left_prop))
right_px_96               <- total_px_96 - left_px_96
total_height_px_96        <- as.integer(round(mm_to_px96(total_height_mm)))
right_top_height_px_96    <- right_px_96
right_bottom_height_px_96 <- total_height_px_96 - right_top_height_px_96

save_panel("fig3_panel_1_snp__docs.png", full_panel,
       width = px_to_in96(total_px_96), height = px_to_in96(total_height_px_96), dpi = docs_dpi)
save_panel("fig3_a__docs.png", manhattan_labeled,
       width = px_to_in96(left_px_96), height = px_to_in96(total_height_px_96), dpi = docs_dpi)
save_panel("fig3_b__docs.png", qq_labeled,
       width = px_to_in96(right_px_96), height = px_to_in96(right_top_height_px_96), dpi = docs_dpi)
save_panel("fig3_c__docs.png", venn_labeled,
       width = px_to_in96(right_px_96), height = px_to_in96(right_bottom_height_px_96), dpi = docs_dpi)

message(sprintf("Wrote Figure 3 SNP panel and subpanels (pdf + png) to %s", out_dir))
