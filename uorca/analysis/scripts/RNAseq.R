
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

# Function to create a contrast-specific directory
create_contrast_dir <- function(output_dir, contrast_name) {
  # Clean contrast name for file system use
  clean_contrast <- gsub("[^a-zA-Z0-9]", "_", contrast_name)

  # Create contrast-specific directory
  contrast_dir <- file.path(output_dir, "RNAseqAnalysis", clean_contrast)
  dir.create(contrast_dir, showWarnings = FALSE, recursive = TRUE)

  return(contrast_dir)
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
metadata <- read.csv(metadata_file, stringsAsFactors = FALSE)
if (nrow(metadata) == 0) {
    stop("Metadata is empty!")
}

if (!"abundance_file" %in% colnames(metadata)) {
    stop("Metadata must contain an 'abundance_file' column!")
}

# Sanitize group column values to be valid R names (handles /, spaces, etc.)
# Store mapping for contrast expression translation
if (merged_group %in% colnames(metadata)) {
    original_groups <- unique(metadata[[merged_group]])
    sanitized_groups <- make.names(original_groups)
    group_name_mapping <- setNames(sanitized_groups, original_groups)

    # Apply sanitization to the grouping column
    metadata[[merged_group]] <- make.names(metadata[[merged_group]])

    cat("Sanitized group names for R compatibility:\n")
    for (i in seq_along(original_groups)) {
        if (original_groups[i] != sanitized_groups[i]) {
            cat("  ", original_groups[i], " -> ", sanitized_groups[i], "\n")
        }
    }
} else {
    group_name_mapping <- NULL
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


group_data <- DGE$samples[[merged_group]]

# Sanitize group names to valid R names (handles SARS-CoV-2 -> SARS.CoV.2, etc.)
group_data <- make.names(group_data)
DGE$samples[[merged_group]] <- group_data

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
design <- model.matrix(as.formula(paste("~0 +", merged_group)), data = DGE.norm$samples)
colnames(design) <- sub(merged_group, "", colnames(design))

plot_mds(DGE.norm, output_dir, labels_column = "geo_accession")

cat("Step 7: Performing voom transformation...\n")
cat("Performing voom transformation...\n")
v <- voom(DGE.norm,
    design = design,
    plot = FALSE
)

cat("Step 8: Fitting linear model with limma...\n")
cat("Fitting linear model...\n")

# Check if we have a contrasts file to use
custom_contrast_matrix <- NULL
if (!is.null(contrasts_file) && file.exists(contrasts_file)) {
    cat("Loading custom contrasts from CSV file:", contrasts_file, "\n")

    # Read the CSV file containing contrasts
    contrasts_df <- try(read.csv(contrasts_file, stringsAsFactors = FALSE))

    # Sanitize group names in contrast expressions using the mapping we created earlier
    # This ensures contrast expressions match the sanitized column names in the design matrix
    if (class(contrasts_df) != "try-error" && "expression" %in% colnames(contrasts_df) && !is.null(group_name_mapping)) {
        cat("Sanitizing contrast expressions using group name mapping...\n")

        for (i in seq_len(nrow(contrasts_df))) {
            expr <- contrasts_df$expression[i]
            original_expr <- expr

            # Remove backticks first
            expr <- gsub("`([^`]+)`", "\\1", expr)

            # Replace original group names with sanitized versions
            # Sort by length (longest first) to avoid partial replacements
            sorted_originals <- names(group_name_mapping)[order(-nchar(names(group_name_mapping)))]
            for (orig in sorted_originals) {
                sanitized <- group_name_mapping[[orig]]
                if (orig != sanitized) {
                    # Literal (fixed) replacement: group names contain spaces,
                    # parens, slashes etc. that are unsafe to feed to a regex.
                    expr <- gsub(orig, sanitized, expr, fixed = TRUE)
                }
            }

            contrasts_df$expression[i] <- expr
            if (original_expr != expr) {
                cat("  Contrast", i, ":", original_expr, " -> ", expr, "\n")
            }
        }
        cat("Sanitized contrast expressions for R compatibility\n")
    }

    if (class(contrasts_df) != "try-error" && nrow(contrasts_df) > 0 &&
        all(c("name", "expression") %in% colnames(contrasts_df))) {

        cat("Loaded", nrow(contrasts_df), "contrasts from CSV file\n")

        # Extract unique factor levels from expressions
        levels_df <- unique(unlist(strsplit(contrasts_df$expression, " - ")))

        # Generate the makeContrasts call
        contrast_parts <- paste0(contrasts_df$name, " = \"", contrasts_df$expression, "\"")
        contrasts_str <- paste(contrast_parts, collapse = ", ")
        levels_str <- paste(paste0("\"", levels_df, "\""), collapse = ", ")
        makeContrasts_str <- paste0("makeContrasts(", contrasts_str, ", levels = colnames(design))")

        cat("Generated makeContrasts call:\n", makeContrasts_str, "\n\n")

        # Create the contrast matrix
        custom_contrast_matrix <- eval(parse(text = makeContrasts_str))

        cat("Custom contrast matrix:\n")
        print(custom_contrast_matrix)
    } else {
        warning("Failed to read contrasts from CSV file or invalid format")
    }
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
        contrast_name <- colnames(custom_contrast_matrix)[i]
        contrast_dir <- create_contrast_dir(output_dir, contrast_name)
        result_file <- file.path(contrast_dir, "DEG.csv")
        write.csv(top_results, file = result_file, row.names = FALSE)
        cat("Saved results to:", result_file, "\n")
    }
} else {
    # Fall back to default behavior
    if (ncol(design) == 2) {
        cat("Exactly two groups detected. Calculating contrast (group2 - group1)...\n")
        contrast_name <- paste(colnames(design)[2], "-", colnames(design)[1])
        contrast <- makeContrasts(diff = contrast_name, levels = design)
        cat("Contrast matrix:")
        print(contrast)
        vfit <- lmFit(v, design)
        vfit <- contrasts.fit(vfit, contrast)
        efit <- eBayes(vfit)
        dt <- decideTests(efit, p.value = 0.05)
        save_voom_plot(DGE.norm, efit, design, output_dir)
        plot_ma(efit, output_dir, coef = 1, status = dt)
        plot_volcano(efit, output_dir, coef = 1, highlight = 10)
        plot_deg_heatmap(DGE.norm, efit, output_dir, group_data, coef = 1, n_genes = 50)

        cat("Top differential expression results for contrast:\n")
        top_results <- topTable(efit, number = Inf)
        print(head(top_results))
        top_results$Gene <- rownames(top_results)
        top_results <- top_results[, c("Gene", setdiff(names(top_results), "Gene"))]
        print(head(top_results))

        # Save in contrast-specific directory
        contrast_name <- colnames(design)[2] + "_vs_" + colnames(design)[1]
        contrast_dir <- create_contrast_dir(output_dir, contrast_name)
        write.csv(top_results, file = file.path(contrast_dir, "DEG.csv"), row.names = FALSE)
    } else {
        cat("Multiple groups detected. Generating top results for each coefficient...\n")
        for (i in 1:ncol(design)) {
            vfit <- lmFit(v, design)
            efit <- eBayes(vfit)
            coef_name <- colnames(design)[i]
            dt <- decideTests(efit, p.value = 0.05)[, i]
            save_voom_plot(DGE.norm, efit, design, output_dir)
            plot_ma(efit, output_dir, coef = 1, status = dt)
            plot_volcano(efit, output_dir, coef = 1, highlight = 10)
            plot_deg_heatmap(DGE.norm, efit, output_dir, group_data, coef = 1, n_genes = 50)
            top_results <- topTable(efit, coef = i, number = Inf)

            top_results$Gene <- rownames(top_results)
            top_results <- top_results[, c("Gene", setdiff(names(top_results), "Gene"))]
            print(head(top_results))

            # Save in contrast-specific directory
            contrast_dir <- create_contrast_dir(output_dir, coef_name)
            write.csv(top_results, file = file.path(contrast_dir, "DEG.csv"), row.names = FALSE)
        }
    }
}

cat("Step 9: Saving DEG results, normalized DGEList, and CPM values...\n")
cat("Saving normalized DGEList object...\n")
saveRDS(DGE.norm, file = file.path(analysis_dir, "DGE_norm.RDS"))

# Export CPM values
export_cpm(DGE.norm, output_dir)

cat("=== R Script: edgeR/limma Analysis Completed ===\n")
