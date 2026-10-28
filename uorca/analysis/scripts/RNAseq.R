
suppressPackageStartupMessages(library(ComplexHeatmap))
suppressPackageStartupMessages(library(edgeR))
suppressPackageStartupMessages(library(limma))
suppressPackageStartupMessages(library(tximport))
suppressPackageStartupMessages(library(gplots))

create_plot_output_dirs <- function(output_dir) {
  # Create main analysis directory (RNAseqAnalysis)
  analysis_dir <- file.path(output_dir, "RNAseqAnalysis")
  dir.create(analysis_dir, showWarnings = FALSE, recursive = TRUE)

  # Return the analysis directory path
  return(analysis_dir)
}

# Folder name of a contrast. The rule is unchanged, so existing result folders
# keep their names; check_contrast_folders() makes sure no two contrasts share
# a folder.
contrast_folder <- function(contrast_name) {
    folder <- gsub("[^a-zA-Z0-9]", "_", contrast_name)
    if (!grepl("^[A-Za-z0-9_]+$", folder)) {
        stop("Contrast name '", contrast_name, "' gives an unusable folder name '", folder, "'.",
             call. = FALSE)
    }
    folder
}

# Stop before any result is written if two contrasts map to one folder
# (KO.vs.WT and KO_vs_WT, or KO_vs_WT and ko_vs_wt on a case-insensitive file
# system). Warn about folders with results that this run will not write.
check_contrast_folders <- function(output_dir, contrast_names) {
    folders <- vapply(contrast_names, contrast_folder, character(1))
    key <- tolower(folders)
    clash <- unique(key[duplicated(key)])
    if (length(clash) > 0) {
        groups <- vapply(clash, function(k) paste0("'", contrast_names[key == k], "'", collapse = " and "),
                         character(1))
        stop("Contrasts would write to the same folder: ", paste(groups, collapse = "; "),
             ". Give each contrast a name that differs in more than case or punctuation.",
             call. = FALSE)
    }
    analysis_dir <- file.path(output_dir, "RNAseqAnalysis")
    existing <- list.dirs(analysis_dir, full.names = FALSE, recursive = FALSE)
    existing <- existing[file.exists(file.path(analysis_dir, existing, "DEG.csv"))]
    stale <- setdiff(existing, folders)
    if (length(stale) > 0) {
        warning("RNAseqAnalysis has results from an earlier run that this run does not ",
                "replace: ", paste(stale, collapse = ", "), call. = FALSE, immediate. = TRUE)
    }
    reused <- intersect(existing, folders)
    if (length(reused) > 0) {
        cat("Overwriting results from an earlier run in:", paste(reused, collapse = ", "), "\n")
    }
    invisible(folders)
}

# Folders whose DEG.csv this run has written; a second write would mean two
# contrasts share a folder.
written_contrast_folders <- new.env()

# Function to create a contrast-specific directory
create_contrast_dir <- function(output_dir, contrast_name) {
    clean_contrast <- contrast_folder(contrast_name)

    # Create contrast-specific directory
    contrast_dir <- file.path(output_dir, "RNAseqAnalysis", clean_contrast)
    dir.create(contrast_dir, showWarnings = FALSE, recursive = TRUE)

    return(contrast_dir)
}

# One row per contrast written; saved as RNAseqAnalysis/contrast_manifest.csv.
contrast_manifest <- data.frame(contrast_name = character(0), folder = character(0),
                                n_genes = integer(0), n_samples = integer(0),
                                stringsAsFactors = FALSE)

# Write DEG.csv for one contrast, refusing to overwrite one from this run.
# n_samples: samples in the groups with a non-zero weight in the contrast.
write_deg_table <- function(top_results, output_dir, contrast_name, n_samples) {
    contrast_dir <- create_contrast_dir(output_dir, contrast_name)
    folder <- basename(contrast_dir)
    if (exists(tolower(folder), envir = written_contrast_folders, inherits = FALSE)) {
        stop("Contrast '", contrast_name, "' would overwrite results written earlier in this ",
             "run to folder '", folder, "'.", call. = FALSE)
    }
    assign(tolower(folder), contrast_name, envir = written_contrast_folders)
    result_file <- file.path(contrast_dir, "DEG.csv")
    write.csv(top_results, file = result_file, row.names = FALSE)
    cat("Saved results to:", result_file, "\n")
    contrast_manifest[nrow(contrast_manifest) + 1, ] <<- list(
        contrast_name, folder, nrow(top_results), as.integer(n_samples)
    )
    result_file
}

# Number of samples whose group has a non-zero weight in a contrast column.
samples_in_contrast <- function(weights, groups) {
    sum(groups %in% rownames(weights)[weights[, 1] != 0])
}

# --- Contrast checks ----------------------------------------------------------
# Contrast names and expressions come from an LLM. They are checked here and
# never pasted into R code: only level names, numbers, + - * / and parentheses
# may appear in an expression, and the names are passed as list names.

CONTRAST_WEIGHT_TOLERANCE <- 1e-8

# stop() without the call, so the message is on the 'Error:' line.
fail <- function(...) stop(..., call. = FALSE)

# R names used in an expression (numbers are not names).
expression_identifiers <- function(expr) {
    m <- gregexpr("(?<![A-Za-z0-9._])(?:[A-Za-z]|\\.(?![0-9]))[A-Za-z0-9._]*", expr, perl = TRUE)
    unique(regmatches(expr, m)[[1]])
}

# Legacy input only: replace raw level text with its make.names form in one
# left-to-right pass, longest level first, matching whole tokens only.
# Replaced text is never scanned again, so a short level cannot corrupt a
# longer one (8h_GM inside X8h_GMTGW).
rewrite_legacy_expression <- function(expr, mapping) {
    changed <- mapping[names(mapping) != unname(mapping)]
    if (length(changed) == 0) return(expr)
    raws <- names(changed)[order(-nchar(names(changed)))]
    is_name_char <- function(ch) nzchar(ch) && grepl("[A-Za-z0-9._]", ch)
    out <- character(0)
    i <- 1L
    n <- nchar(expr)
    while (i <= n) {
        hit <- NA_character_
        if (i == 1L || !is_name_char(substr(expr, i - 1L, i - 1L))) {
            for (r in raws) {
                end <- i + nchar(r) - 1L
                if (end <= n && substr(expr, i, end) == r &&
                    !is_name_char(substr(expr, end + 1L, end + 1L))) {
                    hit <- r
                    break
                }
            }
        }
        if (!is.na(hit)) {
            out <- c(out, changed[[hit]])
            i <- i + nchar(hit)
        } else {
            out <- c(out, substr(expr, i, i))
            i <- i + 1L
        }
    }
    paste(out, collapse = "")
}

# Check every contrast and return the contrast matrix (levels x contrasts).
# Stops with a message naming the contrast on any problem.
build_contrast_matrix <- function(contrast_names, expressions, design_levels, mapping) {
    bad <- contrast_names[is.na(contrast_names) | make.names(contrast_names) != contrast_names |
                          contrast_names %in% c("levels", "contrasts")]
    if (length(bad) > 0) {
        fail("Contrast name(s) not a valid R name: ", paste0("'", bad, "'", collapse = ", "),
             ". Use letters, digits and underscores, starting with a letter.")
    }
    checked <- character(length(expressions))
    for (i in seq_along(expressions)) {
        label <- paste0("Contrast '", contrast_names[i], "' ('", expressions[i], "')")
        expr <- gsub("`([^`]+)`", "\\1", expressions[i])
        ids <- expression_identifiers(expr)
        if (length(ids) == 0 || !all(ids %in% design_levels)) {
            rewritten <- rewrite_legacy_expression(expr, mapping)
            if (rewritten != expr) cat("  Legacy rewrite:", expr, " -> ", rewritten, "\n")
            expr <- rewritten
        }
        if (!grepl("^[A-Za-z0-9._ +*/()-]+$", expr)) {
            fail(label, ": only level names, numbers, + - * / and parentheses are allowed.")
        }
        if (grepl("[A-Za-z0-9._][[:space:]]*\\(", expr)) {
            fail(label, ": function calls are not allowed.")
        }
        ids <- expression_identifiers(expr)
        if (length(ids) == 0) fail(label, ": uses no group level.")
        unknown <- setdiff(ids, design_levels)
        if (length(unknown) > 0) {
            fail(label, ": ", paste0("'", unknown, "'", collapse = ", "),
                 " not a group level. Levels: ", paste(design_levels, collapse = ", "))
        }
        parsed <- tryCatch(parse(text = expr, keep.source = FALSE),
                           error = function(e) fail(label, ": cannot be parsed: ",
                                                    conditionMessage(e)))
        if (length(parsed) != 1) fail(label, ": must be a single expression.")
        checked[i] <- expr
    }
    contrast_matrix <- do.call(
        makeContrasts,
        c(as.list(setNames(checked, contrast_names)), list(levels = design_levels))
    )
    nonfinite <- colnames(contrast_matrix)[colSums(!is.finite(contrast_matrix)) > 0]
    if (length(nonfinite) > 0) {
        fail("Contrast(s) ", paste0("'", nonfinite, "'", collapse = ", "),
             " give non-finite weights. A level whose name is a number can capture a ",
             "divisor such as '/2'; use valid level names.")
    }
    sums <- colSums(contrast_matrix)
    not_zero_sum <- colnames(contrast_matrix)[abs(sums) > CONTRAST_WEIGHT_TOLERANCE]
    if (length(not_zero_sum) > 0) {
        fail("Contrast(s) ", paste0("'", not_zero_sum, "'", collapse = ", "),
             " have weights that do not sum to 0, so they are not comparisons.")
    }
    empty <- colnames(contrast_matrix)[colSums(abs(contrast_matrix)) == 0]
    if (length(empty) > 0) {
        fail("Contrast(s) ", paste0("'", empty, "'", collapse = ", "), " have all weights 0.")
    }
    contrast_matrix
}

# Function to export normalized CPM values
export_cpm <- function(dge, output_dir) {
  # Calculate log-CPM values
  lcpm <- cpm(dge, log = TRUE)

  # Create a data frame with gene names as first column
  cpm_df <- as.data.frame(lcpm)
  cpm_df$Gene <- rownames(cpm_df)

  # Reorder columns to have Gene as first column
  cpm_df <- cpm_df[, c("Gene", setdiff(names(cpm_df), "Gene"))]

  # Save to file
  cpm_file <- file.path(output_dir, "RNAseqAnalysis", "CPM.csv")
  write.csv(cpm_df, file = cpm_file, row.names = FALSE)

  cat("Saved normalized log-CPM values to:", cpm_file, "\n")

  return(cpm_file)
}

args <- commandArgs(trailingOnly = TRUE)

metadata_file <- args[1]
merged_group <- args[2]
output_dir <- args[3]
tx2gene_file <- args[4]
contrasts_file <- if(length(args) > 4) args[5] else NULL

cat("Received command arguments:\n")
cat("  metadata_file =", metadata_file, "\n")
cat("  merged_group =", merged_group, "\n")
cat("  output_dir =", output_dir, "\n")
cat("  tx2gene_file =", tx2gene_file, "\n")
cat("  contrasts_file =", contrasts_file, "\n\n")

# The argument block above is kept verbatim: the download bundle replaces it
# with fixed values (uorca/gui/components/helpers/__init__.py).
usage <- "Usage: Rscript RNAseq.R <sample_table.csv> <group_column> <output_dir> <tx2gene.txt|NA> [contrasts.csv]"
if (length(metadata_file) != 1 || is.na(metadata_file) || !file.exists(metadata_file)) {
    stop("Sample table not found: '", metadata_file, "'. ", usage)
}
if (length(merged_group) != 1 || is.na(merged_group) || !nzchar(merged_group)) {
    stop("No group column given. ", usage)
}
if (length(output_dir) != 1 || is.na(output_dir) || !nzchar(output_dir)) {
    stop("No output directory given. ", usage)
}
if (length(tx2gene_file) != 1 || is.na(tx2gene_file)) {
    stop("No tx2gene file given (pass NA for none). ", usage)
}
if (tx2gene_file != "NA" && !file.exists(tx2gene_file)) {
    stop("tx2gene file not found: '", tx2gene_file, "'. Pass NA to run without one.")
}


# 1. MDS Plot
plot_mds <- function(dge, output_dir, groups = group_data, labels_column = "geo_accession",
                     save_png = TRUE, width = 800, height = 600, res = 100) {
    # Create log-transformed CPM values
    lcpm <- cpm(dge, log = TRUE)

    # Get labels for samples
    if (!labels_column %in% colnames(dge$samples)) {
        warning(paste0("Column '", labels_column, "' not found in dge$samples. Using row names as labels."))
        labels <- rownames(dge$samples)
    } else {
        labels <- dge$samples[[labels_column]]
    }

    # Set up colors using rainbow palette
    unique_groups <- unique(groups)
    colors <- rainbow(length(unique_groups))
    group_colors <- colors[as.factor(groups)]

    # Create directory if it doesn't exist
    analysis_dir <- file.path(output_dir, "RNAseqAnalysis")
    dir.create(analysis_dir, recursive = TRUE, showWarnings = FALSE)

    # Start PNG device if saving
    if (save_png) {
        png(file.path(analysis_dir, "MDS.png"), width = width, height = height, res = res)
    }

    # Calculate margin adjustments for labels
    max_label_width <- max(strwidth(labels, units = "inches"))
    buffer <- 0.1
    current_mai <- par("mai")
    new_right_margin <- max(current_mai[4], max_label_width + buffer)

    # Set margins and allow plotting outside plot region
    par(mai = c(current_mai[1], current_mai[2], current_mai[3], new_right_margin))
    par(xpd = NA)

    # Create the MDS plot
    plotMDS(lcpm,
        col = group_colors,
        labels = labels,
        main = "MDS Plot"
    )

    # Add legend
    legend("topright",
        legend = levels(as.factor(groups)),
        col = colors,
        pch = 16
    )

    # Close PNG device if saving
    if (save_png) {
        dev.off()
        return(paste0("MDS plot saved to ", file.path(analysis_dir, "MDS.png")))
    } else {
        return("MDS plot displayed")
    }
}

# 2. Filtering Plots (Density plots before/after filtering)
plot_filtering <- function(lcpm_pre, lcpm_post, output_dir, width = 1000, height = 600, res = 100) {
    analysis_dir <- file.path(output_dir, "RNAseqAnalysis")
    png(file.path(analysis_dir, "filtering_density.png"), width = width, height = height, res = res)
    par(mfrow = c(1, 2))
    nsamples <- ncol(lcpm_pre)
    col <- rainbow(nsamples)
    plot(density(lcpm_pre[, 1]),
        col = col[1], lwd = 2, ylim = c(0, 0.26),
        las = 2, main = "Before Filtering", xlab = "Log2-CPM"
    )
    for (i in 2:nsamples) {
        lines(density(lcpm_pre[, i]), col = col[i], lwd = 2)
    }
    plot(density(lcpm_post[, 1]),
        col = col[1], lwd = 2, ylim = c(0, 0.26),
        las = 2, main = "After Filtering", xlab = "Log2-CPM"
    )
    for (i in 2:nsamples) {
        lines(density(lcpm_post[, i]), col = col[i], lwd = 2)
    }
    if (nsamples <= 12) {
        legend("topright", colnames(lcpm_post), col = col, lwd = 2, cex = 0.6)
    }
    dev.off()
    return(paste0("Filtering plots saved to ", analysis_dir))
}

# 3. Normalization Boxplots
plot_normalization <- function(DGE_filtered, dge_norm, output_dir, width = 1000, height = 600, res = 100) {
    analysis_dir <- file.path(output_dir, "RNAseqAnalysis")
    lcpm_raw <- cpm(DGE_filtered, log = TRUE)
    lcpm_norm <- cpm(dge_norm, log = TRUE)
    png(file.path(analysis_dir, "normalization_boxplots.png"), width = width, height = height, res = res)
    par(mfrow = c(1, 2))
    ylim <- range(c(lcpm_raw, lcpm_norm))
    boxplot(lcpm_raw,
        las = 2, col = rainbow(ncol(lcpm_raw)),
        main = "Before Normalization", ylab = "Log2-CPM", ylim = ylim
    )
    boxplot(lcpm_norm,
        las = 2, col = rainbow(ncol(lcpm_norm)),
        main = "After Normalization", ylab = "Log2-CPM", ylim = ylim
    )
    dev.off()
    return(paste0("Normalization plots saved to ", analysis_dir))
}

# 4. Voom Plot
save_voom_plot <- function(dge, fit, design, output_dir, width = 800, height = 600, res = 100) {
    analysis_dir <- file.path(output_dir, "RNAseqAnalysis")

    # Save the voom plot
    png(file.path(analysis_dir, "voom_mean_variance.png"), width = width, height = height, res = res)
    v <- voom(dge, design, plot = TRUE)
    dev.off()

    # Save the SA plot
    png(file.path(analysis_dir, "sa_plot.png"), width = width, height = height, res = res)
    saplot <- plotSA(fit, main = "Final model: Mean-Variance trend", xlab = "Mean log2-counts")
    dev.off()

    return(paste0("Voom and SA plots saved to ", analysis_dir))
}

# 5. MA Plot
plot_ma <- function(fit_object, output_dir, coef = 1, status = NULL, highlight = 10, width = 800, height = 600, res = 100) {
    contrast_name <- colnames(fit_object)[coef]
    contrast_dir <- create_contrast_dir(output_dir, contrast_name)

    # Create PNG file
    png(file.path(contrast_dir, "ma_plot.png"), width = width, height = height, res = res)

    # Use limma's built-in plotMA function
    if (!is.null(status)) {
        plotMA(fit_object,
            coef = coef, status = status,
            main = paste0("MA Plot: ", contrast_name),
            xlab = "Average log2-expression",
            ylab = "log2-fold-change"
        )
    } else {
        plotMA(fit_object,
            coef = coef,
            main = paste0("MA Plot: ", contrast_name),
            xlab = "Average log2-expression",
            ylab = "log2-fold-change"
        )
    }

    dev.off()
    return(paste0("MA plot for ", contrast_name, " saved to ", contrast_dir))
}

# 6. Volcano Plot
plot_volcano <- function(fit_object, output_dir, coef = 1, highlight = 10, width = 800, height = 600, res = 100) {
    contrast_name <- colnames(fit_object)[coef]
    contrast_dir <- create_contrast_dir(output_dir, contrast_name)

    # Create PNG file
    png(file.path(contrast_dir, "volcano_plot.png"), width = width, height = height, res = res)

    # Use limma's built-in volcanoplot function
    volcanoplot(fit_object,
        coef = coef, highlight = highlight,
        main = paste0("Volcano Plot: ", contrast_name),
        names = rownames(fit_object)
    )

    dev.off()
    return(paste0("Volcano plot for ", contrast_name, " saved to ", contrast_dir))
}

# 7. Heatmap of Top DEGs
plot_deg_heatmap <- function(dge, fit_object, output_dir, group_data = NULL,
                                  coef = 1, n_genes = 100, clustering_method = "complete",
                                  width = 1000, height = 800, res = 100) {
    # Load required packages if not already loaded
    if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
        library(ComplexHeatmap)
    }
    if (!requireNamespace("circlize", quietly = TRUE)) {
        library(circlize)
    }

    # Determine the contrast name from the fit_object
    contrast_name <- colnames(fit_object)[coef]

    # Create contrast-specific directory
    contrast_dir <- create_contrast_dir(output_dir, contrast_name)

    # Identify top genes based on available p-values or table data
    if (!is.null(fit_object$p.value) && !is.null(fit_object$lods)) {
        o <- order(fit_object$p.value[, coef])
    } else if (!is.null(fit_object$table)) {
        o <- order(fit_object$table$PValue)
    } else {
        stop("Cannot determine top genes from the fit object")
    }
    top_genes_idx <- head(o, n_genes)

    # Calculate log-CPM from the DGE object
    lcpm <- cpm(dge, log = TRUE)
    lcpm_top <- lcpm[top_genes_idx, ]

    # Retrieve gene symbols (using either fit_object$genes or dge$genes)
    if (!is.null(fit_object$genes)) {
        gene_symbols <- fit_object$genes$SYMBOL[top_genes_idx]
        if (is.null(gene_symbols)) {
            gene_symbols <- rownames(lcpm_top)
        }
    } else if (!is.null(dge$genes)) {
        gene_symbols <- dge$genes$SYMBOL[top_genes_idx]
        if (is.null(gene_symbols)) {
            gene_symbols <- rownames(lcpm_top)
        }
    } else {
        gene_symbols <- rownames(lcpm_top)
    }
    rownames(lcpm_top) <- gene_symbols

    # Define sample (column) labels
    if (!is.null(dge$samples$geo_accession)) {
        col_labels <- dge$samples$geo_accession
    } else {
        col_labels <- colnames(lcpm_top)
    }
    colnames(lcpm_top) <- col_labels

    # Z-score normalization (row-wise)
    lcpm_scaled <- t(scale(t(lcpm_top)))

    # Create color palette for heatmap (blue-white-red)
    col_fun <- circlize::colorRamp2(c(min(lcpm_scaled), 0, max(lcpm_scaled)), c("blue", "white", "red"))

    # Prepare column annotation based on group_data if provided
    if (!is.null(group_data)) {
        if (length(group_data) != ncol(lcpm_top)) {
            stop("Length of group_data must equal number of samples (columns) in lcpm_top")
        }

        # Create a color palette for the groups
        unique_groups <- unique(group_data)
        group_colors <- setNames(rainbow(length(unique_groups)), unique_groups)

        # Create column annotation
        column_ha <- ComplexHeatmap::HeatmapAnnotation(
            Group = group_data,
            col = list(Group = group_colors),
            name = "Sample Groups",
            annotation_legend_param = list(
                Group = list(
                    title = "Sample Group",
                    at = unique_groups
                )
            )
        )
    } else {
        column_ha <- NULL
    }

    # Set up clustering parameters
    if (clustering_method == "complete") {
        clustering_distance_rows <- "euclidean"
        clustering_distance_columns <- "euclidean"
        clustering_method_rows <- "complete"
        clustering_method_columns <- "complete"
    } else {
        # Other clustering methods can be added here
        clustering_distance_rows <- "euclidean"
        clustering_distance_columns <- "euclidean"
        clustering_method_rows <- clustering_method
        clustering_method_columns <- clustering_method
    }

    # Adjust height based on number of genes
    height_adjusted <- max(height, n_genes * 8)

    # Save the ComplexHeatmap to a PNG file
    png_file <- file.path(contrast_dir, paste0("heatmap_top", n_genes, ".png"))

    png(png_file, width = width, height = height_adjusted, res = res)

    # Create the heatmap
    ht <- ComplexHeatmap::Heatmap(
        lcpm_scaled,
        name = "Z-score\n(log2-CPM)",  # Legend title
        col = col_fun,

        # Clustering parameters
        cluster_rows = TRUE,
        clustering_distance_rows = clustering_distance_rows,
        clustering_method_rows = clustering_method_rows,
        cluster_columns = TRUE,
        clustering_distance_columns = clustering_distance_columns,
        clustering_method_columns = clustering_method_columns,

        # Labels
        row_labels = gene_symbols,
        column_labels = col_labels,

        # Top annotation with group colors
        top_annotation = column_ha,

        # Layout
        row_names_gp = grid::gpar(fontsize = 8),
        column_names_gp = grid::gpar(fontsize = 8),
        column_names_rot = 45,

        # Title
        column_title = paste0("Top ", n_genes, " DE Genes: ", contrast_name),
        column_title_gp = grid::gpar(fontsize = 12, fontface = "bold")
    )

    # Draw the heatmap
    ComplexHeatmap::draw(ht, padding = grid::unit(c(2, 2, 2, 2), "cm"))

    dev.off()

    return(paste0(
        "ComplexHeatmap of top ", n_genes, " DEGs for ", contrast_name,
        " saved to ", png_file
    ))
}

analysis_dir <- create_plot_output_dirs(output_dir)

# Create DEG subfolder within the output directory is no longer needed
# as we'll store DEG files in contrast-specific directories
cat("Created analysis directory at:", analysis_dir, "\n")

cat("Step 1: Loading metadata from file...\n")
cat("Loading metadata from:", metadata_file, "\n")
# check.names = FALSE: headers are used exactly as written (no make.names).
# na.strings = character(0): a level spelled "NA" stays the string "NA".
metadata <- read.csv(metadata_file, stringsAsFactors = FALSE,
                     check.names = FALSE, na.strings = character(0))
if (nrow(metadata) == 0) {
    stop("Metadata is empty!")
}

if (!"abundance_file" %in% colnames(metadata)) {
    stop("Metadata must contain an 'abundance_file' column!")
}

# Group levels must reach the design matrix as valid, distinct R names.
# The UORCA pipeline writes such names (column uorca_group), so nothing is
# renamed. Older inputs (for example a download bundle of an earlier run) carry
# raw levels; those are renamed with make.names, but only when no two levels
# get the same name, which would silently merge two groups.
if (!merged_group %in% colnames(metadata)) {
    stop("Group column '", merged_group, "' is not in the sample table. Columns: ",
         paste(colnames(metadata), collapse = ", "))
}
raw_groups <- as.character(metadata[[merged_group]])
bad_rows <- which(is.na(raw_groups) | raw_groups == "")
if (length(bad_rows) > 0) {
    stop("Group column '", merged_group, "' is empty in ", length(bad_rows),
         " sample(s) (rows ", paste(head(bad_rows, 10), collapse = ", "),
         "). Every sample needs a group.")
}
original_groups <- unique(raw_groups)
sanitized_groups <- make.names(original_groups)
collided <- sanitized_groups[duplicated(sanitized_groups)]
if (length(collided) > 0) {
    clashes <- vapply(unique(collided), function(r) {
        paste0("'", r, "' <- ",
               paste0("'", original_groups[sanitized_groups == r], "'", collapse = " and "))
    }, character(1))
    stop("Different group levels get the same R name, so they would be merged ",
         "into one group: ", paste(clashes, collapse = "; "),
         ". Give each level a distinct valid R name (the UORCA pipeline writes ",
         "them to the uorca_group column).")
}
group_name_mapping <- setNames(sanitized_groups, original_groups)
if (identical(sanitized_groups, original_groups)) {
    cat("Group levels are valid R names; nothing renamed.\n")
} else {
    cat("Legacy input: renaming group levels with make.names:\n")
    for (i in seq_along(original_groups)) {
        if (original_groups[i] != sanitized_groups[i]) {
            cat("  ", original_groups[i], " -> ", sanitized_groups[i], "\n")
        }
    }
    metadata[[merged_group]] <- unname(group_name_mapping[raw_groups])
}

# Load tx2gene mapping if provided
if (tx2gene_file != "NA" && file.exists(tx2gene_file)) {
    cat("Step 2: Checking for and loading tx2gene mapping if provided...\n")
    cat("Loading tx2gene mapping from:", tx2gene_file, "\n")
    tx2gene <- read.delim(tx2gene_file, header = FALSE, stringsAsFactors = FALSE)
    # Select columns 1 and 3 (like dplyr::select(1,3))
    tx2gene <- tx2gene[, c(1, 3)]
    # Rename columns (like setNames)
    names(tx2gene) <- c("TXNAME", "GENEID")
    # Filter rows where GENEID is not NA and not empty (like dplyr::filter)
    tx2gene <- tx2gene[!is.na(tx2gene$GENEID) & tx2gene$GENEID != "", ]
    use_tx2gene <- TRUE
} else {
    cat("No valid tx2gene file provided. Proceeding without tx2gene mapping.\n")
    use_tx2gene <- FALSE
}

cat("Step 3: Importing Kallisto quantification data using tximport...\n")
cat("Importing Kallisto quantification data...\n")
if (use_tx2gene) {
    kallisto <- tximport(metadata$abundance_file,
        type = "kallisto",
        tx2gene = tx2gene, countsFromAbundance = "lengthScaledTPM",
        ignoreAfterBar = TRUE
    )
} else {
    kallisto <- tximport(metadata$abundance_file,
        type = "kallisto",
        txOut = TRUE, ignoreAfterBar = TRUE
    )
}

cat("Step 4: Creating DGEList with the imported counts...\n")
cat("Creating DGEList...\n")
DGE <- DGEList(counts = kallisto$counts)
lcpm_pre <- cpm(DGE, log = TRUE)

# DGEList() always populates samples$group with a single-level factor of 1s.
# If merged_group is also named "group", cbind keeps both columns and
# DGE$samples[["group"]] resolves to the default factor (single level),
# making model.matrix fail. Drop the default before merging metadata.
DGE$samples$group <- NULL
DGE$samples <- cbind(DGE$samples, metadata)


group_data <- as.character(DGE$samples[[merged_group]])
# The levels were checked (and, for legacy input, renamed) above.
if (!identical(make.names(group_data), group_data)) {
    stop("Internal error: group levels are not valid R names after the check: ",
         paste(unique(group_data[make.names(group_data) != group_data]), collapse = ", "))
}

# Filter genes with low expression
keep.exprs <- filterByExpr(DGE, group = group_data)
DGE_filtered <- DGE[keep.exprs, keep.lib.sizes = FALSE]

# Calculate post-filtering log-CPM values
lcpm_post <- cpm(DGE_filtered, log = TRUE)
plot_filtering(lcpm_pre, lcpm_post, output_dir)

cat("Number of genes before filtering:", nrow(DGE), "\n")
cat("Number of genes after filtering:", nrow(DGE_filtered), "\n")

cat("Step 5: Normalizing the DGEList using calcNormFactors...\n")
cat("Normalizing DGEList...\n")
DGE.norm <- calcNormFactors(DGE_filtered)

plot_normalization(DGE_filtered, DGE.norm, output_dir)

cat("Step 6: Generating design matrix using grouping column:", merged_group, "\n")
cat("Creating design matrix using grouping column:", merged_group, "\n")
# Build the factor directly: no formula is pasted from the column name and no
# regex strips it from the column names. factor() orders levels as
# model.matrix would for a character column, so the design is unchanged.
group_factor <- factor(as.character(DGE.norm$samples[[merged_group]]))
if (nlevels(group_factor) < 2) {
    stop("The group column '", merged_group, "' has ", nlevels(group_factor),
         " level(s); differential expression needs at least 2.")
}
design <- model.matrix(~0 + group_factor)
colnames(design) <- levels(group_factor)

plot_mds(DGE.norm, output_dir, labels_column = "geo_accession")

cat("Step 7: Performing voom transformation...\n")
cat("Performing voom transformation...\n")
v <- voom(DGE.norm,
    design = design,
    plot = FALSE
)

cat("Step 8: Fitting linear model with limma...\n")
cat("Fitting linear model...\n")

# Read the contrasts file, if one was given
custom_contrast_matrix <- NULL
if (!is.null(contrasts_file)) {
    if (!file.exists(contrasts_file)) {
        stop("Contrasts file not found: ", contrasts_file)
    }
    cat("Loading custom contrasts from CSV file:", contrasts_file, "\n")
    contrasts_df <- read.csv(contrasts_file, stringsAsFactors = FALSE, check.names = FALSE,
                             na.strings = character(0), colClasses = "character")
    if (!all(c("name", "expression") %in% colnames(contrasts_df)) || nrow(contrasts_df) == 0) {
        stop("Contrasts file ", contrasts_file, " needs 'name' and 'expression' columns and ",
             "at least one row; found columns: ", paste(colnames(contrasts_df), collapse = ", "))
    }
    cat("Loaded", nrow(contrasts_df), "contrasts from CSV file\n")
    custom_contrast_matrix <- build_contrast_matrix(
        contrasts_df$name, contrasts_df$expression, colnames(design), group_name_mapping
    )
    cat("Custom contrast matrix:\n")
    print(custom_contrast_matrix)
    check_contrast_folders(output_dir, colnames(custom_contrast_matrix))
}

if (!is.null(custom_contrast_matrix)) {
    cat("Using custom contrasts defined by the metadata agent...\n")
    vfit <- lmFit(v, design)
    vfit <- contrasts.fit(vfit, custom_contrast_matrix)
    efit <- eBayes(vfit)

    # Generate results for each contrast
    for (i in 1:ncol(custom_contrast_matrix)) {
        contrast_name <- colnames(custom_contrast_matrix)[i]
        cat("Analyzing contrast:", contrast_name, "\n")

        dt <- decideTests(efit, p.value = 0.05)[, i]
        save_voom_plot(DGE.norm, efit, design, output_dir)
        plot_ma(efit, output_dir, coef = i, status = dt)
        plot_volcano(efit, output_dir, coef = i, highlight = 10)
        plot_deg_heatmap(DGE.norm, efit, output_dir, group_data, coef = i, n_genes = 50)

        top_results <- topTable(efit, coef = i, number = Inf)
        top_results$Gene <- rownames(top_results)
        top_results <- top_results[, c("Gene", setdiff(names(top_results), "Gene"))]

        # Save results in the contrast-specific directory
        write_deg_table(top_results, output_dir, contrast_name,
                        samples_in_contrast(custom_contrast_matrix[, i, drop = FALSE], group_data))
    }
} else {
    # No contrasts file. The UORCA pipeline always writes one; this branch is
    # for a download bundle of an older run without contrasts. Only a
    # two-group design has an unambiguous comparison. The former multi-group
    # branch tested each group mean against zero, which is not a comparison,
    # so it now stops.
    if (ncol(design) != 2) {
        stop("No contrasts file was given and the design has ", ncol(design),
             " groups (", paste(colnames(design), collapse = ", "), "). ",
             "Provide a contrasts file with 'name' and 'expression' columns.")
    }
    contrast_name <- paste0(colnames(design)[2], "_vs_", colnames(design)[1])
    cat("Exactly two groups and no contrasts file. Contrast:", contrast_name, "\n")
    contrast <- build_contrast_matrix(
        contrast_name, paste(colnames(design)[2], "-", colnames(design)[1]),
        colnames(design), group_name_mapping
    )
    check_contrast_folders(output_dir, contrast_name)
    print(contrast)
    vfit <- lmFit(v, design)
    vfit <- contrasts.fit(vfit, contrast)
    efit <- eBayes(vfit)
    dt <- decideTests(efit, p.value = 0.05)
    save_voom_plot(DGE.norm, efit, design, output_dir)
    plot_ma(efit, output_dir, coef = 1, status = dt)
    plot_volcano(efit, output_dir, coef = 1, highlight = 10)
    plot_deg_heatmap(DGE.norm, efit, output_dir, group_data, coef = 1, n_genes = 50)

    top_results <- topTable(efit, number = Inf)
    top_results$Gene <- rownames(top_results)
    top_results <- top_results[, c("Gene", setdiff(names(top_results), "Gene"))]
    print(head(top_results))
    write_deg_table(top_results, output_dir, contrast_name,
                    samples_in_contrast(contrast, group_data))
}

cat("Step 9: Saving DEG results, normalized DGEList, and CPM values...\n")
cat("Saving normalized DGEList object...\n")
saveRDS(DGE.norm, file = file.path(analysis_dir, "DGE_norm.RDS"))

# Export CPM values
export_cpm(DGE.norm, output_dir)

# Record what was written and with which packages
manifest_file <- file.path(analysis_dir, "contrast_manifest.csv")
write.csv(contrast_manifest, file = manifest_file, row.names = FALSE)
cat("Saved contrast manifest to:", manifest_file, "\n")
writeLines(capture.output(sessionInfo()), file.path(analysis_dir, "sessionInfo.txt"))

cat("=== R Script: edgeR/limma Analysis Completed ===\n")
