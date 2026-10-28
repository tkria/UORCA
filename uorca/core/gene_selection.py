"""Gene selection and identification utilities."""
import pandas as pd
from typing import Dict, List, Tuple, Set
import logging

from uorca.core.pvalues import get_adjusted_p_column

logger = logging.getLogger(__name__)


def identify_frequent_degs(
    deg_data: Dict[str, Dict[str, pd.DataFrame]],
    contrasts: List[Tuple[str, str]],
    top_n: int,
    p_thresh: float,
    lfc_thresh: float
) -> List[str]:
    """
    Identify genes frequently differentially expressed across contrasts.

    Args:
        deg_data: DEG data dict (analysis_id -> contrast_id -> DataFrame)
        contrasts: List of (analysis_id, contrast_id) tuples to analyze
        top_n: Number of top genes to return
        p_thresh: Adjusted p-value (BH) threshold for significance
        lfc_thresh: Log fold-change threshold

    Returns:
        List of gene names sorted by frequency
    """
    gene_counts = {}
    contrasts_processed = 0

    for analysis_id, contrast_id in contrasts:
        if analysis_id in deg_data and contrast_id in deg_data[analysis_id]:
            deg_df = deg_data[analysis_id][contrast_id]

            # Adjusted p-value (BH) only; raises if the table lacks adj.P.Val
            p_col = get_adjusted_p_column(deg_df, f"DEG table for {analysis_id}/{contrast_id}")

            # Filter for significant genes
            if 'logFC' in deg_df.columns and 'Gene' in deg_df.columns:
                significant_genes = deg_df[
                    (deg_df[p_col] < p_thresh) &
                    (abs(deg_df['logFC']) > lfc_thresh)
                ]['Gene'].tolist()

                contrasts_processed += 1

                # Count occurrences
                for gene in significant_genes:
                    gene_counts[gene] = gene_counts.get(gene, 0) + 1

    # Sort by frequency (descending) then by gene name (ascending)
    sorted_genes = sorted(gene_counts.items(), key=lambda x: (-x[1], x[0]))

    logger.info(f"Processed {contrasts_processed} contrasts, found {len(gene_counts)} unique genes")

    # Return top N genes
    return [gene for gene, count in sorted_genes[:top_n]]


def get_all_genes_from_cpm(cpm_data: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Extract all unique genes from CPM data.

    Args:
        cpm_data: CPM data dict (analysis_id -> DataFrame)

    Returns:
        Sorted list of unique gene names
    """
    all_genes = set()
    for cpm_df in cpm_data.values():
        if 'Gene' in cpm_df.columns:
            all_genes.update(cpm_df['Gene'].tolist())
    return sorted(all_genes)


def get_available_genes_for_contrasts(
    deg_data: Dict[str, Dict[str, pd.DataFrame]],
    contrasts: List[Tuple[str, str]]
) -> Set[str]:
    """
    Get all genes available in selected contrasts.

    Args:
        deg_data: DEG data dict
        contrasts: List of (analysis_id, contrast_id) tuples

    Returns:
        Set of gene names present in at least one contrast
    """
    available_genes = set()

    for analysis_id, contrast_id in contrasts:
        if analysis_id in deg_data and contrast_id in deg_data[analysis_id]:
            deg_df = deg_data[analysis_id][contrast_id]
            if 'Gene' in deg_df.columns:
                available_genes.update(deg_df['Gene'].tolist())

    return available_genes


def validate_custom_genes(
    custom_genes: List[str],
    available_genes: Set[str]
) -> Tuple[List[str], List[str]]:
    """
    Validate custom gene list against available genes.

    Args:
        custom_genes: User-provided gene list
        available_genes: Genes available in the data

    Returns:
        Tuple of (found_genes, missing_genes)
    """
    found = [g for g in custom_genes if g in available_genes]
    missing = [g for g in custom_genes if g not in available_genes]
    return found, missing