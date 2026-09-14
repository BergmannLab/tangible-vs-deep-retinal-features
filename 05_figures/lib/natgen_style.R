# Nature Genetics artwork rules, in one place, for the R figure scripts.
#
# Author: Michael Beyeler (github.com/mjbeyeler)
#
# The numbers live in config.yaml (plot_styles.natgen, plot_styles.fonts,
# plot_styles.panel_label, plot_styles.figures.fig3) so that R and Python read the same
# source of truth. This file is the R half of src/plot_style.py.
#
# The two governing documents, both quoted in config.yaml:
#   The journal's author guidance for this manuscript -- text not smaller than
#     6 pt, phrased "as a guideline"; a display item no larger than 260 x 179 mm.
#   https://www.nature.com/documents/aj-artworkguidelines.pdf -- sans-serif, preferably
#     Helvetica or Arial; maximum text size 7 pt, minimum 5 pt; RGB; vector with
#     editable text.
# The floors differ (5 vs 6). The artwork guide's 5 pt is the binding one and the docx's
# 6 pt is a guideline, so the usable band is 5-7 pt (widened from 6-7 on 15 Aug 2026).
# It is a floor, not a target: author at 6-7 wherever the layout permits.

# Multiple-testing burden, from config.yaml -> gwas. One reader, so no Figure 3 panel can
# quietly divide by a different number than its neighbours. See config.yaml for why the LV
# divisor is n_effective_lvs (177) and not n_lvs (1024).
gwas_burden <- function(config) {
    g <- config$gwas
    list(
        genome_wide_p   = as.numeric(g$genome_wide_p),
        n_traits        = as.integer(g$n_traits),
        n_lvs           = as.integer(g$n_lvs),
        n_effective_lvs = as.integer(g$n_effective_lvs),
        n_genes_scored  = as.integer(g$n_genes_scored)
    )
}

# grid and matplotlib disagree on the name of the regular weight: matplotlib says
# "normal", R's fontface says "plain" (and rejects "normal" outright). config.yaml is
# written in the matplotlib vocabulary because most of the figure scripts are Python, so
# the R side translates on the way in.
r_fontface <- function(weight) {
    switch(as.character(weight),
           normal = "plain",
           regular = "plain",
           bold = "bold",
           italic = "italic",
           as.character(weight))
}

# R's `lwd` unit, in points: 1 lwd = 1/96 inch = 0.75 pt. ggplot2's mm -> lwd conversion
# stops one step short of the page, so anything specified in real points goes through this.
R_LWD_PT <- 72 / 96

natgen_style <- function(config) {
    ps <- config$plot_styles
    ng <- ps$natgen
    fonts <- ps$fonts
    fig3 <- ps$figures$fig3
    page <- ps$figures$page

    derived_width_mm <- (page$width_cm - 2 * page$margin_cm) * 10
    list(
        # Nature's cap wins over the artwork guide's nominal 180 mm 2-column width; one
        # millimetre is invisible and the cap is the manuscript-specific instruction.
        # The ink width config.yaml documents: the page less the border on each side,
        # 178 - 2 x 1 = 176 mm. That is what src/plot_style.display_item_width_mm hands the
        # Python strips, and what fig3.py measures every strip against before it assembles
        # them. Deriving it from the page geometry instead left the R strips 1.8 mm too wide.
        width_mm         = ng$display_item_max_mm$width - 2 * ng$display_item_border_mm,
        max_height_mm    = ng$display_item_max_mm$height,
        text_min_pt      = ng$text_min_pt,
        text_max_pt      = ng$text_max_pt,
        family           = ng$font_family,
        # The config states points AS RENDERED, as Figure 2 does; getting there from a
        # ggplot2 `linewidth` takes both steps of the conversion. ggplot2 multiplies
        # linewidth (mm) by .pt to get R's `lwd`, and `lwd` is in 1/96 inch = 0.75 pt --
        # so dividing by .pt alone yields 0.375 pt on the page, not 0.5, and that is what
        # made panels a-f a quarter thinner than g-j (measured, 5 Sep 2026).
        axis_lw_mm       = ps$axis_linewidth_pt / .pt / R_LWD_PT,
        axis_tick_len_pt = ps$axis_tick_length_pt,
        title_pt         = fonts$title,
        axis_label_pt    = fonts$axis_label,
        tick_label_pt    = fonts$tick_label,
        panel_label_pt   = ps$panel_label$fontsize,
        panel_label_face = ps$panel_label$fontweight,
        venn_label_pt    = fig3$venn_label_fontsize,
        venn_scale_mm    = fig3$venn_scale_mm_per_unit,
        venn_label_face  = r_fontface(fig3$venn_label_fontweight),
        gene_label_pt    = fig3$gene_label_fontsize,
        gene_marker_size = fig3$gene_marker_size,
        gene_label_min_segment_mm = fig3$gene_label_min_segment_mm,
        manhattan_point_pt   = fig3$manhattan_point_size,
        threshold_lw         = fig3$manhattan_threshold_linewidth,
        threshold_lty        = fig3$manhattan_threshold_linetype,
        manhattan_y_pad_frac = fig3$manhattan_y_pad_frac,
        manhattan_y_pad_frac_top = fig3$manhattan_y_pad_frac_top,
        strip_gap_mm     = fig3$strip_gap_mm,
        strip_height_mm  = fig3$strip_height_mm,
        letter_x_mm      = fig3$panel_letter_x_mm,
        left_weight      = fig3$left_width_weight,
        right_weight     = fig3$right_width_weight,
        qq_height_mm     = fig3$qq_height_mm,
        # isTRUE so an absent key reads as FALSE rather than NULL.
        qq_double_log    = isTRUE(fig3$qq_double_log),
        hi_dpi           = fig3$hi_dpi,
        docs_dpi         = fig3$docs_dpi
    )
}

# Teach R's device-independent font databases about the family before anything measures
# text with it.
#
# grid computes text extents against the PostScript/PDF font database, which only knows
# the base-14 names. Asking for "Liberation Sans" there produces a "font family not found
# in PostScript font database" warning for every single text grob -- dozens per figure --
# and silently falls back for the measurement while cairo still renders the real face.
# Registering the family as an alias of the built-in sans metrics stops the warnings and
# is metrically correct: Nimbus Sans IS the URW Helvetica clone, and the built-in `sans`
# entry is Helvetica.
register_font_family <- function(style) {
    spec <- list(grDevices::postscriptFonts()$sans)
    names(spec) <- style$family
    do.call(grDevices::postscriptFonts, spec)
    spec <- list(grDevices::pdfFonts()$sans)
    names(spec) <- style$family
    do.call(grDevices::pdfFonts, spec)
    invisible(TRUE)
}

# Fail the build rather than the copy-edit: any size outside the band stops the script.
#
# Panel letters are NOT passed through here. They are a deliberate exception at 8 pt bold
# -- Nature's own style for a panel locator, which is a locator rather than body text.
# Excluding them by name keeps that a considered exception, not a silent leak.
assert_text_sizes <- function(style, sizes) {
    bad <- sizes[sizes < style$text_min_pt | sizes > style$text_max_pt]
    if (length(bad)) {
        stop(sprintf(
            "text size(s) outside Nature's %g-%g pt band: %s",
            style$text_min_pt, style$text_max_pt,
            paste(sprintf("%s = %g pt", names(bad), bad), collapse = ", ")
        ), call. = FALSE)
    }
    invisible(TRUE)
}

# theme_minimal()'s base_size of 11 pt is where the oversized axis text came from: nothing
# set a size, so everything inherited 11 pt titles and 8.8 pt ticks. Setting base_size and
# base_family here makes every text element land in the band and in the one accepted face.
natgen_theme <- function(style) {
    theme_minimal(base_size = style$axis_label_pt, base_family = style$family) +
        theme(
            axis.title  = element_text(size = style$axis_label_pt, family = style$family),
            axis.text   = element_text(size = style$tick_label_pt, family = style$family),
            strip.text  = element_text(size = style$axis_label_pt, family = style$family),
            legend.text = element_text(size = style$tick_label_pt, family = style$family),
            plot.title  = element_text(size = style$title_pt, family = style$family,
                                       face = "plain"),
            # theme_minimal()'s own default is base_size/4 -- 1.75 pt here -- which is a
            # consequence of the font size rather than a choice. One length for the whole
            # display item, from config.
            axis.ticks.length = unit(style$axis_tick_len_pt, "pt")
        )
}

# The plotted y-window, padded so a whole dot fits inside it.
#
# Doing this by hand rather than through scale expansion: coord_cartesian(ylim=) applies
# its limits as a hard window AFTER the scale has expanded, so the two mechanisms fight and
# which one wins depends on whether the scale's own limits happen to be narrower than the
# coord's. That is exactly how panel d stayed clipped while panel a came out right. One
# mechanism, computed in the transformed (log2) space the axis actually uses.
padded_ylim <- function(floor_y, ceiling_y, pad_frac, pad_frac_top = pad_frac) {
    span <- log2(ceiling_y) - log2(floor_y)
    # The floor is nudged a hair below floor_y so a break AT the floor (the 2 tick) is
    # inside the window rather than on its boundary, where floating point decides.
    c(2^(log2(floor_y) - pad_frac * span - 1e-3), 2^(log2(ceiling_y) + pad_frac_top * span))
}

# Nature sets p-values as an italic capital P. plotmath's own italic() gives us that
# without pulling a second font family, which the previous `expression("-" ~ log[10](p))`
# did -- its glyphs resolved to DejaVu Sans and the PDF ended up with two sans faces.
axis_label_neglog10p <- function() expression(-log[10](italic(P)))

# Panel letters: one definition, one size, read from config. Never hardcode.
add_panel_letter <- function(plot, letter, style, x = 0.02, y = 0.98) {
    ggdraw(plot) +
        draw_label(letter, x = x, y = y, hjust = 0, vjust = 1,
                   fontface = style$panel_label_face,
                   fontfamily = style$family,
                   size = style$panel_label_pt)
}

# Draw the Venn region counts as real text over an imported, text-free diagram.
#
# WHY THE TEXT IS NOT IN THE SVG. Nature requires the text in a vector figure to remain
# editable. matplotlib can emit real <text>, but rsvg::rsvg_svg() -- which we must go
# through, because grImport2 reads only the Cairo flavour of SVG -- converts every glyph
# to a path (verified: 7 <text> elements in, 0 out). So render_venn3.py blanks the counts
# and records their positions as fractions of the image box instead, and we draw them
# here. The text is then native to the PDF, in the document font, at a size we set in
# points rather than one that rides on the diagram's scale factor.
#
# `x0`, `y0`, `w`, `h` are the same npc box the diagram itself was drawn into, so the
# fractions map straight onto it. The y flip is because the recorded fraction is measured
# from the bottom of the image while cowplot's npc origin for this box is its lower edge
# -- both are bottom-up, so no flip is needed; the fraction is used as-is.
draw_venn_labels <- function(plot, labels, style, x0, y0, w, h) {
    for (i in seq_len(nrow(labels))) {
        plot <- plot + draw_label(
            as.character(labels$text[i]),
            x = x0 + labels$x_frac[i] * w,
            y = y0 + labels$y_frac[i] * h,
            hjust = 0.5, vjust = 0.5,
            fontface = style$venn_label_face,
            fontfamily = style$family,
            size = style$venn_label_pt
        )
    }
    plot
}
