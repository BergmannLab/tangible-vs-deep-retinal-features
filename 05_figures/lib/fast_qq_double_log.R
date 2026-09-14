# Double-log QQ plot for genome-wide p-value sets, subsampled to stay fast.
#
# Author: Michael Beyeler (github.com/mjbeyeler)

fast_qq_double_log <- function(pvalue_lists, colors, group_labels = NULL, 
                               legend_labels = NULL, expected = NULL, 
                               n_points_per_interval = 200, log2_interval = log2(3) - log2(2),
                               log10p_lower_limit = 2, log10p_upper_limit = NULL,
                               subsampling = TRUE,
                               ax_lims = NULL, ticks = NULL, draw_rectangle = FALSE, 
                               rectangle_coords = NULL, show_legend = TRUE, font_size = 10, 
                               axis_title_font_size = 12, hide_axis_titles = FALSE,
                               band_data = NULL, band_fill = "#b0b0b0", band_alpha = 0.25,
                               double_log_scale = TRUE, font_family = NULL,
                               show_grid = FALSE, axis_linewidth_mm = 0.3 / .pt,
                               axis_tick_length_pt = 2) {
    # Validate inputs
    if (length(colors) != length(pvalue_lists)) {
        stop("The number of colors must match the number of p-value lists.")
    }
    if (!is.null(group_labels) && length(group_labels) != length(pvalue_lists)) {
        stop("The number of group labels must match the number of p-value lists.")
    }
    if (!is.null(legend_labels) && length(legend_labels) != length(pvalue_lists)) {
        stop("The number of legend labels must match the number of p-value lists.")
    }
    if (!is.null(expected) && length(expected) != length(pvalue_lists)) {
        stop("The 'expected' list must match the number of p-value lists.")
    }
    if (draw_rectangle && is.null(rectangle_coords)) {
        stop("If draw_rectangle = TRUE, rectangle_coords must be provided as a list of x and y ranges.")
    }
    if (!is.null(ax_lims) && (!is.list(ax_lims) || length(ax_lims) != 2 || 
                              length(ax_lims[[1]]) != 2 || length(ax_lims[[2]]) != 2)) {
        stop("ax_lims must be a list of two vectors, each of length 2: list(c(x_lower, x_upper), c(y_lower, y_upper)).")
    }
    
    # Default group and legend labels if not provided
    if (is.null(group_labels)) {
        group_labels <- paste0("Group ", seq_along(pvalue_lists))
    }
    if (is.null(legend_labels)) {
        legend_labels <- group_labels
    }
    
    # Collect all plot data in a consistent way
    plot_data_list <- list()
    
    # Process each group using a for loop to preserve order
    for (i in seq_along(pvalue_lists)) {
        # Ensure p-values are numeric
        pvalues <- pvalue_lists[[i]]
        if (!is.numeric(pvalues)) {
            stop(paste("Element", i, "in pvalue_lists must be a numeric vector."))
        }
        
        # Step 1: Check if observed p-values are raw or log10
        if (all(pvalues >= 0 & pvalues <= 1)) {
            cat(paste("Group", group_labels[i], ": Raw observed p-values detected. Transforming to -log10 scale.\n"))
            observed <- -log10(pvalues)
        } else if (all(pvalues > 0)) {
            cat(paste("Group", group_labels[i], ": Observed values detected as already in -log10 scale.\n"))
            observed <- pvalues
        } else {
            stop(paste("Group", group_labels[i], ": Observed p-values are invalid or ambiguous. Ensure they are either raw (0 to 1) or in -log10 scale."))
        }
        
        # Step 2: Check if expected p-values are raw or log10 (if provided)
        if (!is.null(expected) && !is.null(expected[[i]])) {
            if (all(expected[[i]] >= 0 & expected[[i]] <= 1)) {
                cat(paste("Group", group_labels[i], ": Raw expected p-values detected. Transforming to -log10 scale.\n"))
                expected_values <- -log10(expected[[i]])
            } else if (all(expected[[i]] > 0)) {
                cat(paste("Group", group_labels[i], ": Expected values detected as already in -log10 scale.\n"))
                expected_values <- expected[[i]]
            } else {
                stop(paste("Group", group_labels[i], ": Expected p-values are invalid or ambiguous. Ensure they are either raw (0 to 1) or in -log10 scale."))
            }
        } else {
            n <- length(pvalues)
            expected_values <- -log10(ppoints(n))  # Generate expected distribution if not provided
        }
        
        # Step 3: Identify and discard elements where log10p = 0
        zero_log10_indices <- which(observed == 0)
        if (length(zero_log10_indices) > 0) {
            cat(paste("Group", group_labels[i], ":", length(zero_log10_indices), "elements with log10p = 0 detected. Discarding these elements.\n"))
            observed <- observed[-zero_log10_indices]
            expected_values <- expected_values[-zero_log10_indices]
        }
        
        # Ensure no zero values remain for log2 transformation
        if (any(observed == 0)) {
            stop(paste("Group", group_labels[i], ": log10p = 0 values still present after filtering. Check the input data."))
        }
        
        # Step 4: Restrict expected values to log10p_lower_limit and log10p_upper_limit
        if (!is.null(log10p_upper_limit)) {
            keep_indices <- which(expected_values >= log10p_lower_limit & expected_values <= log10p_upper_limit)
        } else {
            keep_indices <- which(expected_values >= log10p_lower_limit)
        }
        
        if (length(keep_indices) == 0) {
            stop(paste("Group", group_labels[i], ": No expected points remain after applying axis limits. Adjust log10p_lower_limit or log10p_upper_limit."))
        }
        
        # Filter both expected and corresponding observed values
        expected_values <- expected_values[keep_indices]
        observed <- observed[keep_indices]
        
        # Ensure observed and expected values are finite
        if (!all(is.finite(observed)) || !all(is.finite(expected_values))) {
            stop(paste("Group", group_labels[i], ": Non-finite values detected after filtering. Check input data or axis limits."))
        }
        
        # Step 5: Sort observed and reorder expected
        sorted_indices <- order(observed)
        sorted_observed <- observed[sorted_indices]
        sorted_expected <- expected_values[sorted_indices]
        
        if (!subsampling) {
            # No subsampling: use all points
            subsampled_observed <- sorted_observed
            subsampled_expected <- sorted_expected
        } else {
            # Step 6: Subsampling in log2 space
            min_log2 <- log2(min(sorted_expected))
            max_log2 <- log2(max(sorted_expected))
            
            # Ensure min_log2 and max_log2 are finite
            if (!is.finite(min_log2) || !is.finite(max_log2)) {
                stop(paste("Group", group_labels[i], ": Invalid range for log2 subsampling. Check input data or axis limits."))
            }
            
            # Generate intervals
            intervals <- seq(min_log2, max_log2, by = log2_interval)
            subsampled_observed <- c()
            subsampled_expected <- c()
            
            for (j in seq_along(intervals)[-length(intervals)]) {
                # Define the range of the current interval
                lower_bound <- intervals[j]
                upper_bound <- intervals[j + 1]
                
                # Identify points within the interval
                interval_indices <- which(log2(sorted_expected) >= lower_bound & log2(sorted_expected) < upper_bound)
                
                if (length(interval_indices) > 0) {
                    # Calculate equidistant points in log2 space
                    interval_log2 <- log2(sorted_expected[interval_indices])
                    target_log2 <- seq(lower_bound, upper_bound, length.out = min(n_points_per_interval, length(interval_indices)))
                    
                    # Match closest indices to equidistant log2 points
                    subsampled_indices <- sapply(target_log2, function(t) {
                        which.min(abs(interval_log2 - t))
                    })
                    
                    # Append subsampled points
                    subsampled_observed <- c(subsampled_observed, sorted_observed[interval_indices[subsampled_indices]])
                    subsampled_expected <- c(subsampled_expected, sorted_expected[interval_indices[subsampled_indices]])
                }
            }
        }
        
        # Print subsampled values for debugging
        cat("Group", group_labels[i], ": First few subsampled observed values:\n", head(subsampled_observed), "\n")
        cat("Group", group_labels[i], ": First few subsampled expected values:\n", head(subsampled_expected), "\n")
        
        # Append plotting data for each group
        plot_data_list[[i]] <- data.frame(
            Expected = subsampled_expected,
            Observed = subsampled_observed,
            Group = factor(group_labels[i], levels = group_labels)
        )
    }
    
    # Combine all plot data
    plot_data <- do.call(rbind, plot_data_list)
    
    # Collect all observed and expected values
    all_values <- setNames(lapply(plot_data_list, function(df) list(Observed = df$Observed, Expected = df$Expected)), group_labels)
    
    # Create QQ plot with log2-log2 scales
    p <- ggplot(plot_data, aes(x = Expected, y = Observed, color = Group))

    if (!is.null(band_data)) {
        p <- p + geom_ribbon(
            data = band_data,
            mapping = aes(x = Expected, ymin = Lower, ymax = Upper),
            inherit.aes = FALSE,
            fill = band_fill,
            alpha = band_alpha
        )
    }
    
    # Gridlines are OFF by default. Nature asks for "borders / shading / gridlines removed
    # from all figures unless they convey specific information that is defined in the
    # legend"; on a QQ the y = x null line already carries the reference, so a dashed grid
    # is decoration. `show_grid = TRUE` restores the ruled lines at the tick positions for
    # any caller that still wants them.
    has_custom_grid <- show_grid && !is.null(ticks) && length(ticks) == 2
    if (has_custom_grid) {
        grid_color <- "#c9c9c9"
        grid_linetype <- "dashed"
        grid_linewidth <- 0.3
        vline_df <- data.frame(x = ticks[[1]])
        hline_df <- data.frame(y = ticks[[2]])
        p <- p +
            geom_vline(
                data = vline_df,
                mapping = aes(xintercept = x),
                color = grid_color,
                linewidth = grid_linewidth,
                linetype = grid_linetype,
                show.legend = FALSE
            ) +
            geom_hline(
                data = hline_df,
                mapping = aes(yintercept = y),
                color = grid_color,
                linewidth = grid_linewidth,
                linetype = grid_linetype,
                show.legend = FALSE
            )
    }

    if (draw_rectangle && !is.null(rectangle_coords)) {
        rect_xmin <- rectangle_coords[[1]][1]
        rect_xmax <- rectangle_coords[[1]][2]
        rect_ymin <- rectangle_coords[[2]][1]
        rect_ymax <- rectangle_coords[[2]][2]
        epsilon_x <- max((rect_xmax - rect_xmin) * 1e-4, .Machine$double.eps)
        epsilon_y <- max((rect_ymax - rect_ymin) * 1e-4, .Machine$double.eps)
        rect_df <- data.frame(
            xmin = rect_xmin + epsilon_x,
            xmax = rect_xmax - epsilon_x,
            ymin = rect_ymin + epsilon_y,
            ymax = rect_ymax - epsilon_y
        )
        p <- p +
            geom_rect(
                data = rect_df,
                aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
                inherit.aes = FALSE,
                color = "darkgray",
                linewidth = 0.3,
                fill = NA,
                linetype = "dashed",
                show.legend = FALSE
            )
    }

    p <- p +
        geom_point(size = .3, alpha = 1.0)

    # The null, y = x, drawn as a SAMPLED line rather than geom_abline. geom_abline is
    # rendered as a straight segment between its two endpoints in the panel's
    # TRANSFORMED space, so it traces y = x only while both axes carry the same
    # transform -- transform y alone and the line silently stops meaning y = x while
    # still looking like a reference (the 95 % band, which is transformed correctly,
    # parts company with it and gives the error away). Sampling the curve is correct
    # under any combination of scales, including the log2-log2 view below.
    null_x_range <- if (!is.null(ax_lims)) ax_lims[[1]] else range(plot_data$Expected)
    null_line <- data.frame(x = seq(null_x_range[1], null_x_range[2], length.out = 512))
    null_line$y <- null_line$x
    p <- p + geom_line(data = null_line, mapping = aes(x = x, y = y),
                       inherit.aes = FALSE, color = "black", linewidth = 0.3)

    if (double_log_scale) {
        # Optional log2 scaling for both axes when the double-log view is desired
        p <- p +
            scale_x_continuous(breaks = ticks[[1]], trans = scales::log2_trans()) +
            scale_y_continuous(breaks = ticks[[2]], trans = scales::log2_trans())
    } else {
        scale_x_args <- list(breaks = ticks[[1]])
        scale_y_args <- list(breaks = ticks[[2]])
        if (!is.null(ax_lims)) {
            scale_x_args$limits <- ax_lims[[1]]
            scale_y_args$limits <- ax_lims[[2]]
            scale_x_args$expand <- expansion(mult = 0)
            scale_y_args$expand <- expansion(mult = 0)
        }
        p <- p +
            do.call(scale_x_continuous, scale_x_args) +
            do.call(scale_y_continuous, scale_y_args)
    }

    p <- p +
        scale_color_manual(values = colors, labels = group_labels) +
        labs(
            # Nature sets p-values as an italic capital P. plotmath's italic() supplies it
            # from the same family, unlike the previous "-"~log[10](p), whose glyphs
            # resolved to a second sans face (DejaVu) in the exported PDF.
            x = if (!hide_axis_titles) expression(Expected ~ -log[10](italic(P))) else NULL,
            y = if (!hide_axis_titles) expression(Observed ~ -log[10](italic(P))) else NULL,
            color = "Groups"
        ) +
        theme_minimal(base_size = font_size, base_family = font_family) +
        theme(
            # Axis and panel titles are NOT bold -- house convention, and Nature does not
            # bold them either. Only panel letters are bold.
            axis.title = element_text(size = axis_title_font_size, family = font_family),
            axis.text = element_text(size = font_size, family = font_family),
            plot.title = element_text(hjust = 0.5, size = axis_title_font_size,
                                      family = font_family),
            legend.position = "top",  # Legend at the top
            legend.text = element_text(size = font_size, family = font_family),
            legend.title = element_blank(),
            legend.background = element_rect(color = "black", fill = "white", linewidth = 0.5),  # Black box around legend
            # No gridlines. Nature asks for "borders / shading / gridlines removed from
            # all figures unless they convey specific information that is defined in the
            # legend", and on a QQ the null line already carries the reference.
            panel.grid.minor = element_blank(),
            panel.grid.major = element_blank(),
            panel.background = element_rect(fill = "white", color = NA),
            # No box: an open left/bottom axis in the same weight as panels g-i,
            # replacing the published bordered panel.
            panel.border = element_blank(),
            axis.line = element_line(color = "black", linewidth = axis_linewidth_mm),
            axis.ticks = element_line(color = "black", linewidth = axis_linewidth_mm),
            # Caller-set, so the QQ's ticks are the same length as every other panel's in
            # the display item it is assembled into.
            axis.ticks.length = unit(axis_tick_length_pt, "pt")
        ) +
        guides(color = if (show_legend) guide_legend(ncol = 1) else "none")
    
    # add legend to top right corner "Measured TIFs", "Deep TIFs', "LVs"
    p <- p + theme(legend.position = "top")
    
    # name labels
    
    
    # Set axis limits if ax_lims is provided
    if (!is.null(ax_lims)) {
    # clip = "off": a marker sitting exactly on a limit is otherwise sliced in half by
    # the panel edge -- visible on the LV curve, which runs into the ceiling at 300.
    # Nothing escapes the panel as a result: the scales carry the
    # same limits, and scale limits DROP out-of-range rows rather than just hiding them,
    # so only points that belong inside are left to draw.
        p <- p + coord_cartesian(xlim = ax_lims[[1]], ylim = ax_lims[[2]], clip = "off")
    }
    
    # Return both the plot and all values
    return(list(Plot = p, Values = all_values))
}
