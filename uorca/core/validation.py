"""Form validation utilities for heatmap parameters."""
from typing import Tuple, List


def validate_threshold_values(
    lfc_thresh: str,
    pvalue_thresh: str
) -> Tuple[float, float, bool]:
    """
    Validate log fold-change and adjusted p-value (BH) threshold inputs.

    Args:
        lfc_thresh: Log fold-change threshold as string
        pvalue_thresh: Adjusted p-value (BH) threshold as string

    Returns:
        Tuple of (lfc_value, pvalue_value, has_error)
        If validation fails, returns default values (1.0, 0.05) with has_error=True
    """
    try:
        lfc_val = float(lfc_thresh)
        pval_val = float(pvalue_thresh)
        return lfc_val, pval_val, False
    except ValueError:
        # Return defaults with error flag
        return 1.0, 0.05, True


def validate_gene_count(gene_count_input: str) -> Tuple[int, bool]:
    """
    Validate gene count input is a positive integer.

    Args:
        gene_count_input: Gene count as string

    Returns:
        Tuple of (gene_count, has_error)
        If validation fails, returns default value (50) with has_error=True
    """
    try:
        gene_count = int(gene_count_input)
        if gene_count <= 0:
            return 50, True
        return gene_count, False
    except ValueError:
        return 50, True


def validate_custom_gene_list(custom_genes: List[str]) -> bool:
    """
    Validate custom gene list is not empty.

    Args:
        custom_genes: List of gene names

    Returns:
        True if validation passes (list is not empty), False otherwise
    """
    return len(custom_genes) > 0


def validate_contrasts_selected(contrasts: List) -> bool:
    """
    Validate that at least one contrast is selected.

    Args:
        contrasts: List of selected contrasts

    Returns:
        True if validation passes (at least one contrast), False otherwise
    """
    return len(contrasts) > 0