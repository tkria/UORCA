import os
import json
import pandas as pd
import time
import signal
import logging
from datetime import datetime
from functools import wraps
from mcp.server.fastmcp import FastMCP
from uorca.gui.results_integration import ResultsIntegrator, get_adjusted_p_column
from uorca.gui.components.helpers.ai_agent_tool_logger import log_ai_agent_tool

# Setup logging for MCP server
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout decorator for tool functions
def with_timeout(timeout_seconds=30):
    """Decorator to add timeout protection to tool functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"MCP TOOL: Starting {func.__name__} with {timeout_seconds}s timeout")

            def timeout_handler(signum, frame):
                elapsed = time.time() - start_time
                logger.error(f"MCP TOOL: {func.__name__} timed out after {elapsed:.2f}s")
                raise TimeoutError(f"Tool {func.__name__} timed out after {timeout_seconds} seconds")

            # Set timeout alarm (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

                result = func(*args, **kwargs)

                elapsed = time.time() - start_time
                logger.info(f"MCP TOOL: {func.__name__} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"MCP TOOL: {func.__name__} failed after {elapsed:.2f}s: {str(e)}")
                raise
            finally:
                signal.alarm(0)  # Clear the alarm
                try:
                    signal.signal(signal.SIGALRM, old_handler)
                except:
                    pass
        return wrapper
    return decorator


server = FastMCP("UORCA-Tools")

def load_long_table(results_dir: str) -> pd.DataFrame:
    """
    Build a DataFrame with one row per (analysis_id, contrast_id, gene)
    with columns: analysis_id, contrast_id, Gene, logFC, ``pvalue`` (adjusted), AveExpr.

    ``pvalue`` holds the Benjamini-Hochberg adjusted p-value (limma
    ``adj.P.Val``). A DEG file without ``adj.P.Val`` raises; the raw
    ``P.Value`` column is never substituted.
    """
    ri = ResultsIntegrator(results_dir)
    ri.load_data()
    pieces = []
    for aid, contrasts in ri.deg_data.items():
        for cid, df in contrasts.items():
            # The 'pvalue' field is the adjusted p-value (BH); raises if adj.P.Val is missing
            adj_col = get_adjusted_p_column(df, ri.deg_source(aid, cid))
            df2 = df.drop(columns=['pvalue'], errors='ignore').rename(columns={adj_col: 'pvalue'})

            # Ensure we have the required columns
            if 'Gene' not in df2.columns or 'logFC' not in df2.columns or 'pvalue' not in df2.columns:
                continue

            # Include AveExpr if available, otherwise set to NaN
            required_cols = ['Gene','logFC','pvalue']
            if 'AveExpr' in df2.columns:
                required_cols.append('AveExpr')
                df_sub = df2[required_cols].copy()
            else:
                df_sub = df2[['Gene','logFC','pvalue']].copy()
                df_sub['AveExpr'] = float('nan')

            # Map original contrast name to consistent name (with duplicate resolution)
            consistent_contrast_id = cid  # Default to original name
            for contrast_key, contrast_data in ri.contrast_info.items():
                if (contrast_data.get('original_name') == cid and
                    contrast_data.get('analysis_id') == aid):
                    consistent_contrast_id = contrast_data.get('name', cid)
                    break

            df_sub['analysis_id'] = aid
            df_sub['contrast_id'] = consistent_contrast_id
            pieces.append(df_sub[['analysis_id','contrast_id','Gene','logFC','pvalue','AveExpr']])

    if not pieces:
        # Return empty DataFrame with correct structure if no data
            return pd.DataFrame(columns=['analysis_id','contrast_id','Gene','logFC','pvalue','AveExpr'])

    return pd.concat(pieces, ignore_index=True)

# Global variables for data loading
CURRENT_RESULTS_DIR = None
FULL_DF = pd.DataFrame(columns=['analysis_id','contrast_id','Gene','logFC','pvalue','AveExpr'])

@with_timeout(60)  # Data loading can be slow, allow 60 seconds
def _ensure_data_loaded():
    """Ensure data is loaded with current RESULTS_DIR, reload if directory changed."""
    global CURRENT_RESULTS_DIR, FULL_DF

    results_dir = os.getenv('RESULTS_DIR', '/UORCA_results')

    # Check if we need to reload data
    if CURRENT_RESULTS_DIR != results_dir or FULL_DF.empty:
        try:
            logger.info(f"MCP DATA: Loading data from {results_dir}")
            start_time = time.time()
            FULL_DF = load_long_table(results_dir)
            CURRENT_RESULTS_DIR = results_dir
            elapsed = time.time() - start_time
            logger.info(f"MCP DATA: Loaded full dataframe: {len(FULL_DF)} rows in {elapsed:.2f}s")

            # Validate data structure
            if FULL_DF.empty:
                logger.warning("MCP DATA: Loaded dataframe is empty")
            else:
                unique_analyses = FULL_DF['analysis_id'].nunique()
                unique_contrasts = FULL_DF['contrast_id'].nunique()
                unique_genes = FULL_DF['Gene'].nunique()
                logger.info(f"MCP DATA: Data contains {unique_analyses} analyses, {unique_contrasts} contrasts, {unique_genes} genes")

        except Exception as e:
            logger.error(f"MCP DATA: Could not load long table from {results_dir}: {e}", exc_info=True)
            # Create empty DataFrame with correct structure as fallback
            FULL_DF = pd.DataFrame(columns=['analysis_id','contrast_id','Gene','logFC','pvalue','AveExpr'])
            CURRENT_RESULTS_DIR = results_dir



@with_timeout(30)
def get_filtered_dataframe() -> pd.DataFrame:
    """
    Get the dataframe filtered to selected contrasts, checking environment variable dynamically.
    """
    global FULL_DF

    # Ensure data is loaded with current RESULTS_DIR
    _ensure_data_loaded()

    # Check for selected contrasts environment variable
    selected_contrasts_json = os.getenv('SELECTED_CONTRASTS_FOR_AI')
    if selected_contrasts_json:
        try:
            selected_contrasts = json.loads(selected_contrasts_json)
            logger.info(f"MCP FILTER: Filtering to {len(selected_contrasts)} selected contrasts")

            # Create filter condition
            contrast_tuples = [(sc['analysis_id'], sc['contrast_id']) for sc in selected_contrasts]
            mask = FULL_DF.apply(lambda row: (row['analysis_id'], row['contrast_id']) in contrast_tuples, axis=1)
            filtered_df = FULL_DF[mask].copy()

            logger.info(f"MCP FILTER: Filtered dataframe: {len(filtered_df)} rows from {len(FULL_DF)} total rows")
            return filtered_df
        except json.JSONDecodeError:
            logger.warning("MCP FILTER: Could not parse SELECTED_CONTRASTS_FOR_AI, using full dataframe")
            return FULL_DF
        except Exception as e:
            logger.error(f"MCP FILTER: Error filtering contrasts: {e}")
            return FULL_DF
    else:
        logger.info("MCP FILTER: No selected contrasts specified, using full dataframe")
        return FULL_DF



# 1) Top recurring DEGs
@server.tool()
@log_ai_agent_tool
@with_timeout(30)
async def get_most_common_genes(lfc_thresh: float, p_thresh: float, top_n: int) -> str:
    """
    Find the most commonly differentially expressed genes across contrasts to identify robust, recurring differential expression patterns. This tool filters all available contrast data using your specified thresholds and ranks genes by how frequently they appear as significantly differentially expressed. Use this as your first step to get a broad overview of the most consistent differential expression signals in your dataset. Genes with higher occurrence counts represent more robust findings that are less likely to be experimental artifacts and more likely to represent core biological processes. The results help you prioritize which genes deserve deeper investigation and can reveal pathway components that are consistently activated across multiple experimental conditions. Consider using moderate thresholds initially (lfc_thresh=1.0, p_thresh=0.05) to capture a good range of candidates, then potentially rerun with stricter thresholds to focus on the strongest signals.

    Args:
        lfc_thresh: Minimum absolute log2 fold change threshold
        p_thresh: Maximum adjusted p-value (Benjamini-Hochberg, BH) threshold
        top_n: Number of top genes to return

    Returns:
        JSON string with list of genes and their occurrence counts
    """
    start_time = time.time()
    logger.info(f"MCP TOOL: get_most_common_genes called with lfc_thresh={lfc_thresh}, p_thresh={p_thresh}, top_n={top_n}")

    df = get_filtered_dataframe()
    logger.info(f"MCP TOOL: Working with {len(df)} total rows")

    df = df[(df.logFC.abs() >= lfc_thresh) & (df.pvalue < p_thresh)]
    logger.info(f"MCP TOOL: After filtering: {len(df)} rows meet criteria")

    counts = df.groupby("Gene").size().nlargest(top_n)
    result = json.dumps([{"gene": g, "count": int(c)} for g, c in counts.items()])

    elapsed = time.time() - start_time
    logger.info(f"MCP TOOL: get_most_common_genes completed in {elapsed:.2f}s, returning {len(counts)} genes")
    return result



# 2) Per-gene + contrast stats
@server.tool()
@log_ai_agent_tool
@with_timeout(45)  # Allow extra time for multiple gene lookups
async def get_gene_contrast_stats(genes: list, contrast_ids: list = None) -> str:
    """
    Get detailed differential expression statistics for multiple genes across experimental contrasts to understand their context-dependent regulation patterns. This tool allows you to drill down into multiple genes that you've identified as interesting from other analyses. When contrast_ids is not specified, you'll get comprehensive data showing how the genes behave across all available experimental conditions, which is invaluable for understanding whether genes show consistent directional changes or context-specific responses. When contrast_ids is specified, you get focused results for those particular experimental conditions. Use this tool to validate findings from broader analyses, investigate literature-supported candidate genes, or understand the experimental contexts where specific genes show the strongest differential expression. The logFC values tell you the magnitude and direction of change, while the pvalue field (the adjusted p-value, Benjamini-Hochberg, BH) indicates statistical confidence.

    Args:
        genes: List of gene symbols to look up
        contrast_ids: Optional list of specific contrasts to filter to

    Returns:
        JSON string with gene statistics across contrasts (long format with gene and contrast columns;
        the ``pvalue`` field is the adjusted p-value, BH)
    """
    df = get_filtered_dataframe()
    df = df[df.Gene.isin(genes)]
    if contrast_ids:
        df = df[df.contrast_id.isin(contrast_ids)]
        logger.info(f"MCP TOOL: After contrast filtering: {len(df)} rows for {len(contrast_ids)} specified contrasts")
    result = json.dumps(df[["Gene", "contrast_id", "logFC", "pvalue"]].to_dict("records"))

    return result

# 3) Filter by set A vs set B
@server.tool()
@log_ai_agent_tool
@with_timeout(60)  # Allow extra time for complex filtering operations
async def filter_genes_by_contrast_sets(set_a: list, set_b: list, lfc_thresh: float, p_thresh: float) -> str:
    """
    Identify genes that show significant differential expression in ALL contrasts of set A but not in any contrasts of set B, enabling discovery of highly robust condition-specific gene signatures. This powerful comparative tool helps you understand which genes respond consistently and selectively to particular experimental conditions. For example, you could compare treatment vs control contrasts to find genes that are consistently treatment-responsive across multiple experiments, or compare different cell types to identify genes that are robustly cell-type-specific. The tool requires genes to be significant in ALL contrasts of set A (ensuring consistency) while being non-significant in set B (ensuring specificity). This stringent approach is particularly valuable for identifying the most reliable biomarkers, understanding core mechanisms that are consistently activated, or finding genes that represent fundamental biological differences between conditions. The results include both the gene list and summary statistics showing the scope of differential expression in each set.

    Args:
        set_a: List of contrast IDs for set A
        set_b: List of contrast IDs for set B
        lfc_thresh: Minimum absolute log2 fold change threshold
        p_thresh: Maximum adjusted p-value (Benjamini-Hochberg, BH) threshold

    Returns:
        JSON string with genes unique to set A and summary counts
    """
    start_time = time.time()
    logger.info(f"MCP TOOL: filter_genes_by_contrast_sets called with set_a={len(set_a)}, set_b={len(set_b)}, lfc_thresh={lfc_thresh}, p_thresh={p_thresh}")

    df = get_filtered_dataframe()
    logger.info(f"MCP TOOL: Working with {len(df)} total rows")

    # --- SET A: genes that are significant in *every* contrast in set A ------------
    n_a = len(set_a)
    if n_a == 0:
        genesA = set()                         # nothing to compare
    else:
        dfA = df[
            df.contrast_id.isin(set_a)
            & (df.logFC.abs() >= lfc_thresh)
            & (df.pvalue < p_thresh)
        ]

        # How many distinct A-contrasts did each gene pass in?
        countsA = (
            dfA.groupby("Gene")
                .contrast_id.nunique()
        )
        # Keep those that hit the threshold in *all* A-contrasts
        genesA = set(countsA[countsA == n_a].index)

    # --- SET B: genes significant in *any* contrast in set B (unchanged) -----------
    if set_b:
        dfB = df[
            df.contrast_id.isin(set_b)
            & (df.logFC.abs() >= lfc_thresh)
            & (df.pvalue < p_thresh)
        ]
        genesB = set(dfB.Gene)
    else:
        genesB = set()

    # --- final result --------------------------------------------------------------
    result_genes = sorted(genesA - genesB)
    return json.dumps(
        {
            "genes": result_genes,
            "counts": {
                "in_A": len(result_genes),
                "total_A": len(genesA),
                "total_B": len(genesB),
            },
        }
    )

    return result

# 4) Contrast mini-summary
@server.tool()
@log_ai_agent_tool
@with_timeout(30)
async def summarise_contrast(contrast_id: str, lfc_thresh: float, p_thresh: float, max_genes: int = 10) -> str:
    """
    Generate a comprehensive summary of differential expression within a specific experimental contrast, providing both quantitative overview and identification of the most significantly changed genes. This tool gives you a rapid assessment of the scope and magnitude of differential expression in any single experimental condition. The summary includes the total count of significantly differentially expressed genes (which indicates the overall transcriptional response magnitude), the top most significantly changed genes ranked by absolute log fold change (which identifies the strongest individual responses), and summary statistics like mean and median log fold changes (which characterize the overall direction and magnitude of expression changes). Use this tool when you want to quickly assess whether a particular contrast shows strong differential expression signals worth investigating further, or to get context about the biological magnitude of responses in specific experimental conditions.

    Args:
        contrast_id: Contrast to summarize
        lfc_thresh: Minimum absolute log2 fold change threshold
        p_thresh: Maximum adjusted p-value (Benjamini-Hochberg, BH) threshold
        max_genes: Maximum number of top genes to include

    Returns:
        JSON string with contrast summary and top genes
    """
    start_time = time.time()
    logger.info(f"MCP TOOL: summarise_contrast called for contrast_id='{contrast_id}', lfc_thresh={lfc_thresh}, p_thresh={p_thresh}, max_genes={max_genes}")

    df = get_filtered_dataframe()
    logger.info(f"MCP TOOL: Working with {len(df)} total rows")
    df = df[df.contrast_id == contrast_id]
    df = df[(df.logFC.abs() >= lfc_thresh) & (df.pvalue < p_thresh)]

    if df.empty:
        result = json.dumps({
            "total_DEGs": 0,
            "top_genes": [],
            "mean_logFC": None,
            "median_logFC": None
        })
    else:
        top = df.reindex(df.logFC.abs().sort_values(ascending=False).index).head(max_genes)
        result = json.dumps({
            "total_DEGs": int(len(df)),
            "top_genes": [{"gene": r.Gene, "logFC": float(r.logFC)} for _, r in top.iterrows()],
            "mean_logFC": float(df.logFC.mean()),
            "median_logFC": float(df.logFC.median())
        })

    return result



# 5) Gene correlation analysis
@server.tool()
@log_ai_agent_tool
@with_timeout(45)  # Correlation calculations can be expensive
async def calculate_gene_correlation(genes: list) -> str:
    """
    Calculate Spearman's rank correlation coefficients between genes based on their average expression levels (AveExpr) to identify co-expression relationships and potential functional modules. This tool uses the AveExpr values from differential expression analyses to compute pairwise correlations between your genes of interest, which reveals genes that tend to be expressed at similar levels across experimental conditions. Strong positive correlations (close to +1) suggest genes that are co-activated and may participate in the same biological pathways or regulatory networks, while strong negative correlations (close to -1) may indicate opposing regulatory relationships or mutually exclusive expression patterns. Moderate correlations (0.3-0.7) can still indicate meaningful biological relationships. Use this tool after identifying candidate genes from other analyses to understand their functional relationships and to group genes into potential co-regulated modules. This is particularly valuable for pathway analysis and understanding regulatory networks.

    Args:
        genes: List of gene symbols to analyze for correlation

    Returns:
        JSON string with pairwise correlation matrix and statistics
    """
    start_time = time.time()
    logger.info(f"MCP TOOL: calculate_gene_correlation called with {len(genes)} genes: {genes}")

    df = get_filtered_dataframe()
    logger.info(f"MCP TOOL: Working with {len(df)} total rows")

    # Filter to specified genes and remove duplicates
    gene_data = df[df['Gene'].isin(genes)].copy()
    if gene_data.empty:
        return json.dumps({"error": "No data found for specified genes", "genes_requested": genes})

    # Check for missing AveExpr data
    if gene_data['AveExpr'].isna().all():
        return json.dumps({"error": "No AveExpr data available for correlation analysis", "genes_requested": genes})

    # Create pivot table with genes as columns and (analysis_id, contrast_id) as rows
    pivot_data = gene_data.pivot_table(
        index=['analysis_id', 'contrast_id'],
        columns='Gene',
        values='AveExpr',
        aggfunc='mean'  # In case of duplicates
    )

    # Remove rows/columns with too much missing data
    pivot_data = pivot_data.dropna(thresh=len(pivot_data.columns)*0.5)  # Keep rows with at least 50% data
    pivot_data = pivot_data.dropna(axis=1, thresh=len(pivot_data)*0.5)  # Keep columns with at least 50% data

    if pivot_data.empty or len(pivot_data.columns) < 2:
        return json.dumps({"error": "Insufficient data for correlation analysis after filtering", "genes_available": list(pivot_data.columns)})

    # Calculate Spearman correlation
    try:
        correlation_matrix = pivot_data.corr(method='spearman')

        # Convert to dictionary format for JSON
        correlation_dict = {}
        for gene1 in correlation_matrix.columns:
            correlation_dict[gene1] = {}
            for gene2 in correlation_matrix.columns:
                corr_val = correlation_matrix.loc[gene1, gene2]
                correlation_dict[gene1][gene2] = None if pd.isna(corr_val) else round(float(corr_val), 4)

        # Extract strong correlations (> 0.5 or < -0.5) excluding self-correlations
        strong_correlations = []
        for gene1 in correlation_matrix.columns:
            for gene2 in correlation_matrix.columns:
                if gene1 != gene2:
                    corr_val = correlation_matrix.loc[gene1, gene2]
                    if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                        strong_correlations.append({
                            "gene1": gene1,
                            "gene2": gene2,
                            "correlation": round(float(corr_val), 4)
                        })

        result = {
            "correlation_matrix": correlation_dict,
            "strong_correlations": strong_correlations,
            "genes_analyzed": list(correlation_matrix.columns),
            "sample_size": len(pivot_data)
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Correlation calculation failed: {str(e)}", "genes_requested": genes})


# 6) Expression variability analysis
@server.tool()
@log_ai_agent_tool
@with_timeout(45)
async def calculate_expression_variability(genes: list, contrasts: list = None) -> str:
    """
    Calculate the standard deviation of log fold changes for specified genes across experimental contrasts to assess expression consistency and identify reliably regulated genes. This tool measures how consistently genes show differential expression across different experimental conditions by computing the standard deviation of their logFC values. Genes with low standard deviation represent consistent, reliable differential expression patterns that may be core biological responses or better biomarker candidates. Genes with high standard deviation show variable responses that may be context-dependent or condition-specific. If contrasts parameter is specified, analysis is restricted to those experimental conditions; if omitted, all available contrasts are used. Use this tool to prioritize genes based on reliability - consistent genes (low SD) may represent more fundamental biological processes, while variable genes (high SD) may indicate context-specific regulation that could be important for understanding condition-specific mechanisms.

    Args:
        genes: List of gene symbols to analyze
        contrasts: Optional list of contrast IDs to restrict analysis to

    Returns:
        JSON string with variability statistics for each gene
    """
    start_time = time.time()
    logger.info(f"MCP TOOL: calculate_expression_variability called with {len(genes)} genes, contrasts filter: {'Yes' if contrasts else 'No'}")

    df = get_filtered_dataframe()
    logger.info(f"MCP TOOL: Working with {len(df)} total rows")

    # Filter to specified genes
    gene_data = df[df['Gene'].isin(genes)].copy()
    if gene_data.empty:
        return json.dumps({"error": "No data found for specified genes", "genes_requested": genes})

    # Filter to specified contrasts if provided
    if contrasts:
        gene_data = gene_data[gene_data['contrast_id'].isin(contrasts)]
        if gene_data.empty:
            return json.dumps({"error": "No data found for specified genes in specified contrasts",
                             "genes_requested": genes, "contrasts_requested": contrasts})

    # Calculate variability statistics for each gene
    variability_stats = []
    for gene in genes:
        gene_subset = gene_data[gene_data['Gene'] == gene]
        if len(gene_subset) == 0:
            variability_stats.append({
                "gene": gene,
                "std_dev": None,
                "mean_logFC": None,
                "median_logFC": None,
                "min_logFC": None,
                "max_logFC": None,
                "contrast_count": 0,
                "error": "No data found"
            })
        elif len(gene_subset) == 1:
            # Only one data point - cannot calculate std dev
            logfc = float(gene_subset['logFC'].iloc[0])
            variability_stats.append({
                "gene": gene,
                "std_dev": 0.0,  # By definition
                "mean_logFC": round(logfc, 4),
                "median_logFC": round(logfc, 4),
                "min_logFC": round(logfc, 4),
                "max_logFC": round(logfc, 4),
                "contrast_count": 1,
                "note": "Single data point - std_dev is 0"
            })
        else:
            # Multiple data points - calculate statistics
            logfc_values = gene_subset['logFC']
            variability_stats.append({
                "gene": gene,
                "std_dev": round(float(logfc_values.std()), 4),
                "mean_logFC": round(float(logfc_values.mean()), 4),
                "median_logFC": round(float(logfc_values.median()), 4),
                "min_logFC": round(float(logfc_values.min()), 4),
                "max_logFC": round(float(logfc_values.max()), 4),
                "contrast_count": len(gene_subset)
            })

    # Sort by standard deviation (most consistent first)
    variability_stats_valid = [stat for stat in variability_stats if stat.get('std_dev') is not None]
    variability_stats_valid.sort(key=lambda x: x['std_dev'])

    # Combine sorted valid stats with any error entries
    variability_stats_errors = [stat for stat in variability_stats if stat.get('std_dev') is None]
    final_stats = variability_stats_valid + variability_stats_errors

    result = {
        "variability_stats": final_stats,
        "summary": {
            "total_genes_requested": len(genes),
            "genes_with_data": len(variability_stats_valid),
            "most_consistent_gene": variability_stats_valid[0]['gene'] if variability_stats_valid else None,
            "most_variable_gene": variability_stats_valid[-1]['gene'] if variability_stats_valid else None,
            "contrasts_analyzed": list(gene_data['contrast_id'].unique()) if contrasts else "all_available"
        }
    }

    return json.dumps(result)


if __name__ == "__main__":
    server.run()
