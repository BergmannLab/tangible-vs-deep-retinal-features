# Figure 3, gene panel: gene-score Manhattan (d) + QQ (e) + feature-set Venn (f).
#
# Authors: d     David Presby (github.com/presbyd), Michael Beyeler (github.com/mjbeyeler)
#          e, f  Michael Beyeler (github.com/mjbeyeler)
#
# The gene-level twin of 05_figures/main/fig3_snp.R, with the same geometry, the same
# deposit-driven inputs and the same vector export. Run it the same way:
#
#   Rscript 05_figures/main/fig3_gene.R
#
# Panel d is built from the deposited points. Panel f is drawn from the deposited region
# counts and imported as vector paths. The QQ's points are thinned upstream and carry the
# pre-thinning rank its expected quantiles need.

suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(ggrepel)
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

# --double-log: draw the QQ panel on log2-log2 axes instead of linear ones.
# RECOMMENDED but not the default, because the default reproduces the published panel and
# changing a display convention is an editorial call. These QQs span -log10 P from ~0 to
# 300; on linear axes the null-tracking body of the distribution is crushed into the
# bottom-left corner while a few extreme points own the panel. On log2 axes the ticks are
# a constant ratio -- the same 4, 32, 256 the Manhattan in the same strip already uses --
# so the bulk is legible and both panel types share one y scale. y = x stays straight
# because BOTH axes carry the transform; transforming one bends it (see fast_qq_double_log.R).
# commandArgs(trailingOnly = FALSE) is already read above to locate --file=; the user's own
# arguments follow --args in that same vector, so no further parsing is needed.
qq_double_log <- "--double-log" %in% script_args
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

message("Fetching Figure 3 gene inputs from the data deposit ...")
inputs <- fetch_data("figure_intermediates", files = c(
    "Fig3_06_de_gene_mtif_thinned.csv",
    "Fig3_07_de_gene_dtif_thinned.csv",
    "Fig3_08_de_gene_lv_thinned.csv",
    "Fig3_09_d_gene_manhattan_labels.csv",
    "Fig3_10_f_gene_featureset_venn_counts.csv"
))

out_dir <- figure_dir("main_figures")   # config: output.figures_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

transparent_bg <- theme(plot.background = element_blank())

save_panel <- function(file, plot, ...) {
    ggsave(file.path(out_dir, file), plot, bg = "transparent", ...)
}

cairo_pdf_natgen <- function(filename, width, height, ...) {
    grDevices::cairo_pdf(filename, width = width, height = height,
                         family = style$family, ...)
}

# ---------------------------------------------------------------------------
# Layout -- identical to the SNP strip so the two stack cleanly
# ---------------------------------------------------------------------------
total_width_mm  <- style$width_mm
total_height_mm <- style$strip_height_mm$gene

left_width_weight  <- style$left_weight
right_width_weight <- style$right_weight

hi_dpi   <- style$hi_dpi
docs_dpi <- style$docs_dpi

labels_gene <- list(left = "d", qq = "e", venn = "f")

assert_text_sizes(style, c(
    axis_title = style$axis_label_pt,
    tick_label = style$tick_label_pt,
    venn_count = style$venn_label_pt,
    gene_label = style$gene_label_pt
))

# ggplot's geom_text size is in MILLIMETRES, not points. .pt is ggplot2's own conversion
# constant (72.27/25.4), so dividing by it turns a point size into the mm ggplot wants --
# which is how the 5 pt in config.yaml reaches the page as 5 pt.
gene_label_mm <- style$gene_label_pt / .pt
gene_marker_size <- style$gene_marker_size

# ---------------------------------------------------------------------------
# Read the thinned gene sets
# ---------------------------------------------------------------------------
read_gene_points <- function(path) {
    df <- read.csv(path)
    required <- c("gene", "chr", "pos", "log10p", "unthinned_rank", "unthinned_total")
    missing <- setdiff(required, names(df))
    if (length(missing)) {
        stop(sprintf("%s lacks column(s): %s", basename(path), paste(missing, collapse = ", ")))
    }
    df[, required]
}

mgene  <- read_gene_points(inputs[["Fig3_06_de_gene_mtif_thinned.csv"]])
dgene  <- read_gene_points(inputs[["Fig3_07_de_gene_dtif_thinned.csv"]])
lvgene <- read_gene_points(inputs[["Fig3_08_de_gene_lv_thinned.csv"]])
gene_labels <- read.csv(inputs[["Fig3_09_d_gene_manhattan_labels.csv"]])

message(sprintf("  mTIF %d, dTIF %d, LV %d thinned gene-trait points",
                nrow(mgene), nrow(dgene), nrow(lvgene)))

# ---------------------------------------------------------------------------
# Panel d -- gene-score Manhattan
# ---------------------------------------------------------------------------
manhattan <- rbind(
    transform(mgene[, c("gene", "chr", "pos", "log10p")],  condition = "measured"),
    transform(dgene[, c("gene", "chr", "pos", "log10p")],  condition = "deep"),
    transform(lvgene[, c("gene", "chr", "pos", "log10p")], condition = "lvs")
)

# GRCh38 chromosome lengths, used only for x-axis layout. Same table as the SNP panel.
chromosome_lengths <- data.frame(
    chr = 1:22,
    length = c(248956422, 242193529, 198295559, 190214555, 181538259, 170805979,
               159345973, 145138636, 138394717, 133797422, 135086622, 133275309,
               114364328, 107043718, 101991189, 90338345, 83257441, 80373285,
               58617616, 64444167, 46709983, 50818468)
) %>%
    mutate(cumulative_start = c(0, cumsum(length)[-n()]),
           midpoint = cumulative_start + length / 2)

Y_MAX <- 400
# The bottom of the data range is also the baseline the panel sits on, so it is a break:
# the black line at the foot of every facet carries a number like every other tick.
Y_FLOOR <- 2
manhattan[manhattan$log10p > Y_MAX, "log10p"] <- Y_MAX


manhattan <- manhattan %>%
    left_join(chromosome_lengths, by = "chr") %>%
    mutate(adjusted_pos = pos + cumulative_start)

thin_gray_boxes <- chromosome_lengths %>%
    filter(chr %% 2 == 1) %>%
    mutate(xmin = cumulative_start, xmax = cumulative_start + length)

label_chromosomes <- chromosome_lengths %>% filter(chr != 19 & chr != 21)

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
recode_sets <- function(df, column) {
    df[[column]] <- recode(df[[column]],
                           measured = facet_levels[1], mtif = facet_levels[1],
                           deep     = facet_levels[2], dtif = facet_levels[2],
                           lvs      = facet_levels[3], lv   = facet_levels[3])
    df[[column]] <- factor(df[[column]], levels = facet_levels)
    df
}
manhattan <- recode_sets(manhattan, "condition")

gene_labels <- gene_labels %>%
    left_join(chromosome_lengths, by = "chr") %>%
    mutate(adjusted_pos = pos + cumulative_start,
           adjusted_label_x = label_x + cumulative_start,
           condition = feature_set)
gene_labels <- recode_sets(gene_labels, "condition")

# Study-wide GENE significance: 0.05 over the genes scored and the independent tests in the
# feature set.

gene_line_data <- data.frame(
    condition = factor(facet_levels, levels = facet_levels),
    yintercept = -log10(0.05 / burden$n_genes_scored /
                        c(burden$n_traits, burden$n_traits, burden$n_effective_lvs))
)

# 4, 32, 256: the axis is log2, so equal spacing means a constant
# ratio -- x8, x8 -- and the earlier 2, 16, 256 (x8, x16) was visibly uneven. The floor
# of the window stays 2; it is the baseline, not a break.
manhattan_y_breaks <- c(4, 32, 256)
manhattan_ylim <- padded_ylim(Y_FLOOR, Y_MAX, style$manhattan_y_pad_frac,
                             style$manhattan_y_pad_frac_top)

# ggrepel is a stochastic layout. Without a seed the gene labels land in different places
# on every run, so the figure is not reproducible and two builds cannot be diffed.
set.seed(0)

manhattan_plot <- ggplot(data = manhattan) +
    # ymin = 0, not -Inf: log2(-Inf) is NaN and the band would be dropped; log2(0) is
    # -Inf, which the scale keeps and the coord squishes to the edge (see fig3_snp.R).
    geom_rect(data = thin_gray_boxes,
              aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = Inf),
              fill = "darkgrey", alpha = 0.1) +
    geom_point(aes(x = adjusted_pos, y = log10p, color = color),
               size = style$manhattan_point_pt) +
    # Thin dashes, over the points, as in panel a (see fig3_snp.R for why).
    geom_hline(data = gene_line_data, aes(yintercept = yintercept),
               color = "red", linetype = style$threshold_lty,
               linewidth = style$threshold_lw) +
    scale_color_identity() +
    # The top-gene marker: a plain red dot, no black rim. The published marker was a
    # filled circle with a 0.5 pt black stroke at size 1.5, which at 179 mm is a ring
    # rather than a dot and buries the point it marks.
    geom_point(data = gene_labels, aes(x = adjusted_pos, y = log10p),
               color = "red", size = gene_marker_size, pch = 16) +
    # Gene symbols are italic -- Nature requires correct gene nomenclature in display items.
    # The labels repel FROM THEIR OWN POINT and ggrepel draws the leader, so every label
    # is connected to the gene it names. Until 4 Sep 2026 they were placed at the
    # deposit's label_x/label_y -- which are the point's own coordinates -- and the leader
    # was a separate zero-length segment, so a repelled label pointed at nothing.
    # `max.time = Inf` with a fixed `max.iter`: ggrepel's default stops after half a
    # second of wall clock, which makes the layout depend on how busy the machine is.
    # `min.segment.length` is the leader-line floor: ggrepel draws no segment shorter than
    # it, so a label that came to rest on its own marker gets none. It was 0 -- "always
    # draw one" -- which left stubs a millimetre long that read as smudges on the dot
    # rather than as lines. Passed as a grid unit in MILLIMETRES;
    # a bare number would be ggrepel's `lines`, which rides on the base font size. It is
    # now 0 -- draw one for every label -- and that is the ONLY thing here that changed
    # when the leaders were asked for: a nudge with the pull switched off did guarantee a
    # long leader on all 30, but it moved every label and the layout came out worse, so
    # the placement is back to repelling from the point with force_pull = 0.5.
    geom_text_repel(data = gene_labels,
                    aes(x = adjusted_pos, y = log10p, label = gene),
                    size = gene_label_mm, family = style$family, fontface = "italic",
                    box.padding = 0.25, point.padding = 0.15,
                    min.segment.length = unit(style$gene_label_min_segment_mm, "mm"),
                    segment.size = 0.15,
                    segment.color = "gray40",
                    force = 3, force_pull = 0.5,
                    max.overlaps = Inf, max.iter = 20000, max.time = Inf, seed = 0) +
    scale_x_continuous(
        # A tick mark on EVERY chromosome; only the labels of the narrow ones (17, 19,
        # 21) are blank, so the marks still show where each chromosome sits.
        breaks = chromosome_lengths$midpoint,
        labels = ifelse(chromosome_lengths$chr %in% c(17, 19, 21), "",
                        chromosome_lengths$chr),
        expand = c(0, 0)
    ) +
    scale_y_continuous(
        trans = scales::log_trans(base = 2),
        breaks = manhattan_y_breaks,
        limits = c(NA, Y_MAX),
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
# Panel e -- gene-score QQ
# ---------------------------------------------------------------------------
# Expected quantiles come from each point's rank BEFORE thinning, over the full number of
# gene-trait tests that set ran -- the same construction as the SNP QQ, and the reason the
# rank and the pre-thinning total travel with the deposited points.
observed_pp <- list(mgene$log10p, dgene$log10p, lvgene$log10p)
expected_pp <- list(
    -log10(mgene$unthinned_rank  / (mgene$unthinned_total[1]  + 1)),
    -log10(dgene$unthinned_rank  / (dgene$unthinned_total[1]  + 1)),
    -log10(lvgene$unthinned_rank / (lvgene$unthinned_total[1] + 1))
)

# 95 % null band from the largest set.
band_set <- which.max(c(mgene$unthinned_total[1], dgene$unthinned_total[1],
                        lvgene$unthinned_total[1]))
band_source <- list(mgene, dgene, lvgene)[[band_set]]
band_n <- band_source$unthinned_total[1]
band_idx <- band_source$unthinned_rank
MIN_PVAL <- 1e-300
confidence_band <- data.frame(
    Expected = expected_pp[[band_set]],
    Lower = -log10(pmax(qbeta(0.975, band_idx, band_n - band_idx + 1), MIN_PVAL)),
    Upper = -log10(pmax(qbeta(0.025, band_idx, band_n - band_idx + 1), MIN_PVAL))
)
confidence_band <- confidence_band[
    is.finite(confidence_band$Lower) & is.finite(confidence_band$Upper) &
        confidence_band$Expected >= 0.3 & confidence_band$Expected <= 8.01, ]

# Axis treatment, from --double-log (see the flag note at the top); linear is the
# published panel. Unlike panel b these tables reach -log10 P = 0, so the log2 branch
# floors at 0.1 -- as close to zero as a log axis goes -- and keeps the whole cloud.
if (qq_double_log) {
    qq_ticks <- list(c(2, 4, 8), c(4, 32, 256))
    qq_lims  <- list(c(0.1, 8), c(0.1, 300))
    qq_lower <- 0.1
} else {
    qq_ticks <- list(c(2, 5, 8), c(0, 150, 300))
    qq_lims  <- list(c(2, 8), c(0, 300))
    qq_lower <- 0.0
}

qq_genes <- fast_qq_double_log(
    observed_pp,
    expected = expected_pp,
    colors = c(color_codes$mtifs, color_codes$dtifs, color_codes$lvs),
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
    band_data = confidence_band,
    band_fill = "#b0b0b0",
    band_alpha = 0.2,
    double_log_scale = qq_double_log
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

qq_plot <- qq_genes$Plot +
    theme(plot.margin = margin(t = 25, r = 5, b = 5, l = 5), aspect.ratio = 1) +
    transparent_bg +
    # clip = "off" so the dots on the limits are drawn whole (see fig3_snp.R).
    coord_cartesian(xlim = qq_lims[[1]], ylim = qq_lims[[2]], clip = "off")

# Panel f: drawn from the deposited region counts, then imported as vector paths. The
# counts carry no text (rsvg::rsvg_svg would turn any glyph into a path), so they come back
# in a sidecar CSV and are drawn below as real text. See 05_figures/lib/natgen_style.R.
svg_path <- file.path(tempdir(), "venn_panel_f.svg")
venn_labels_path <- file.path(tempdir(), "venn_panel_f_labels.csv")
venn_geometry_path <- file.path(tempdir(), "venn_panel_f_geometry.csv")
venn_status <- system2(
    mvp_python(repo_root),
    c(shQuote(file.path(lib_dir, "render_venn3.py")),
      shQuote(inputs[["Fig3_10_f_gene_featureset_venn_counts.csv"]]),
      shQuote(svg_path),
      "--labels-out", shQuote(venn_labels_path),
      "--geometry-out", shQuote(venn_geometry_path),
      "--config", shQuote(file.path(repo_root, "config.yaml")))
)
if (venn_status != 0) {
    stop(sprintf(paste0(
        "drawing panel f from its deposited counts failed (exit %d).\n",
        "  Set MVP_PYTHON if 'python' is not the interpreter with this project's deps."),
        venn_status))
}

venn_cairo_svg <- file.path(tempdir(), "venn_panel_f_cairo.svg")
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
cat(sprintf("  panel f Venn: %.1f x %.1f mm at %.2f mm/unit (%.0f mm2 of circles)\n",
            venn_target_w_mm, venn_target_h_mm, style$venn_scale_mm,
            style$venn_scale_mm^2 * venn_geometry$total_circle_area_units2))

venn_grob <- grImport2::pictureGrob(venn_picture, expansion = 0, distort = TRUE)
venn_labels <- read.csv(venn_labels_path, colClasses = c(text = "character"))
venn_x0 <- (1 - venn_rel_width) / 2
venn_y0 <- (1 - venn_rel_height) / 2

letter_x_left  <- style$letter_x_mm / left_width_mm
letter_x_right <- style$letter_x_mm / right_width_mm

manhattan_labeled <- add_label(manhattan_plot, labels_gene$left, letter_x_left)
qq_labeled        <- add_label(qq_plot, labels_gene$qq, letter_x_right)
venn_labeled      <- ggdraw() +
    draw_grob(venn_grob, x = venn_x0, y = venn_y0,
              width = venn_rel_width, height = venn_rel_height)
venn_labeled <- draw_venn_labels(venn_labeled, venn_labels, style,
                                 venn_x0, venn_y0, venn_rel_width, venn_rel_height)
venn_labeled <- venn_labeled +
    draw_label(labels_gene$venn, x = letter_x_right, y = 0.98, hjust = 0, vjust = 1,
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
right_width_px            <- as.integer(round(mm_to_px(right_width_mm, hi_dpi)))
total_height_px_hi        <- as.integer(round(mm_to_px(total_height_mm, hi_dpi)))
right_top_height_px_hi    <- right_width_px
right_bottom_height_px_hi <- total_height_px_hi - right_top_height_px_hi

save_panel("fig3_panel_2_gene.png", full_panel,
       width = total_width_mm, height = total_height_mm, units = "mm", dpi = hi_dpi)
save_panel("fig3_d.png", manhattan_labeled,
       width = left_width_mm, height = total_height_mm, units = "mm", dpi = hi_dpi)
save_panel("fig3_e.png", qq_labeled,
       width = px_to_in_hi(right_width_px), height = px_to_in_hi(right_top_height_px_hi),
       dpi = hi_dpi)
save_panel("fig3_f.png", venn_labeled,
       width = px_to_in_hi(right_width_px), height = px_to_in_hi(right_bottom_height_px_hi),
       dpi = hi_dpi)

save_panel("fig3_panel_2_gene.pdf", full_panel, device = cairo_pdf_natgen,
       width = total_width_mm, height = total_height_mm, units = "mm")
save_panel("fig3_d.pdf", manhattan_labeled, device = cairo_pdf_natgen,
       width = left_width_mm, height = total_height_mm, units = "mm")
save_panel("fig3_e.pdf", qq_labeled, device = cairo_pdf_natgen,
       width = right_width_mm, height = right_top_height_mm, units = "mm")
save_panel("fig3_f.pdf", venn_labeled, device = cairo_pdf_natgen,
       width = right_width_mm, height = right_bottom_height_mm, units = "mm")

mm_to_px96 <- function(mm) mm_to_px(mm, docs_dpi)
px_to_in96 <- function(px) px / docs_dpi
total_px_96        <- as.integer(round(mm_to_px96(total_width_mm)))
total_height_px_96 <- as.integer(round(mm_to_px96(total_height_mm)))
save_panel("fig3_panel_2_gene__docs.png", full_panel,
       width = px_to_in96(total_px_96), height = px_to_in96(total_height_px_96),
       dpi = docs_dpi)

message(sprintf("Wrote Figure 3 gene panel and subpanels (pdf + png) to %s", out_dir))
