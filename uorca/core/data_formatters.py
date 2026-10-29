"""Data formatting utilities for UI display."""
import re
import pandas as pd
from typing import List, Dict, Any

from uorca.core.pvalues import get_adjusted_p_column


def sort_by_geo_accession(df: pd.DataFrame, accession_col: str = "Accession") -> pd.DataFrame:
    """
    Sort DataFrame by GEO accession number.

    Args:
        df: DataFrame to sort
        accession_col: Name of accession column

    Returns:
        Sorted DataFrame
    """
    df = df.copy()
    try:
        df["_AccessionNum"] = (
            df[accession_col]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(int)
        )
    except Exception:
        df["_AccessionNum"] = float("inf")

    return df.sort_values(
        ["_AccessionNum", accession_col],
        ascending=[True, True]
    ).drop(columns=["_AccessionNum"], errors="ignore")


def create_contrast_table_data(
    analysis_info: Dict[str, Dict],
    deg_data: Dict[str, Dict[str, pd.DataFrame]],
    contrast_info: Dict[str, Dict],
    selected_datasets: List[str]
) -> List[Dict[str, Any]]:
    """
    Create contrast table data using standardized validation logic.

    Args:
        analysis_info: Analysis metadata
        deg_data: DEG data
        contrast_info: Contrast metadata
        selected_datasets: List of dataset IDs to include

    Returns:
        List of dicts for contrast table display
    """
    valid_contrasts = []

    for analysis_id in selected_datasets:
        # Check if analysis was successful
        if analysis_id not in analysis_info:
            continue
        info = analysis_info[analysis_id]
        if not info.get("analysis_success", False):
            continue

        # Get contrasts from analysis_info
        contrasts = info.get("contrasts", [])
        accession = info.get("accession", analysis_id)

        for contrast in contrasts:
            original_name = contrast["name"]

            # Validate contrast has DEG data
            if (analysis_id in deg_data and
                original_name in deg_data[analysis_id] and
                not deg_data[analysis_id][original_name].empty):

                # Find consistent name
                consistent_name = original_name
                for contrast_key, contrast_data in contrast_info.items():
                    if (contrast_data.get('original_name') == original_name and
                        contrast_data.get('analysis_id') == analysis_id):
                        consistent_name = contrast_data.get('name', original_name)
                        break

                valid_contrasts.append({
                    "Select": False,
                    "Accession": accession,
                    "Contrast": consistent_name,
                    "Description": contrast.get("description", "")
                })

    # Sort by accession
    if not valid_contrasts:
        return []

    df = pd.DataFrame(valid_contrasts)
    df = sort_by_geo_accession(df, "Accession")
    return df.to_dict('records')


def create_dataset_info_table(
    analysis_info: Dict[str, Dict],
    dataset_info: Dict[str, Dict],
    deg_data: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Create dataset information table.

    Args:
        analysis_info: Analysis metadata
        dataset_info: Dataset metadata (titles, descriptions)
        deg_data: DEG data for counting contrasts

    Returns:
        DataFrame with dataset information
    """
    rows = []

    for analysis_id, info in analysis_info.items():
        # Only include successful analyses
        if not info.get("analysis_success", False):
            continue

        accession = info.get("accession", analysis_id)
        organism = info.get("organism", "Unknown")

        # Count samples
        unique_groups = info.get("unique_groups", [])
        sample_count = len(unique_groups) if unique_groups else 0

        # Count contrasts
        contrast_count = len(deg_data.get(analysis_id, {}))

        # Get title/summary
        title = ""
        summary = ""
        if analysis_id in dataset_info:
            title = dataset_info[analysis_id].get("title", "")
            if title.startswith("Title:"):
                title = title[6:].strip()
            summary = dataset_info[analysis_id].get("summary", "")
            if summary.startswith("Summary:"):
                summary = summary[8:].strip()

        rows.append({
            "Accession": accession,
            "Organism": organism,
            "Samples": sample_count,
            "Contrasts": contrast_count,
            "Title": title,
            "Description": summary
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = sort_by_geo_accession(df, "Accession")
    return df


def create_contrast_info_table(
    analysis_info: Dict[str, Dict],
    deg_data: Dict[str, Dict[str, pd.DataFrame]],
    pvalue_thresh: float,
    lfc_thresh: float
) -> pd.DataFrame:
    """
    Create contrast information table with DEG counts.

    Args:
        analysis_info: Analysis metadata
        deg_data: DEG data
        pvalue_thresh: Adjusted p-value (BH) threshold for significance
        lfc_thresh: Log fold-change threshold

    Returns:
        DataFrame with contrast information
    """
    rows = []

    for analysis_id, contrasts_dict in deg_data.items():
        if analysis_id not in analysis_info:
            continue

        info = analysis_info[analysis_id]
        accession = info.get("accession", analysis_id)

        for contrast_id, deg_df in contrasts_dict.items():
            # Adjusted p-value (BH) only; raises if the table lacks adj.P.Val
            p_col = get_adjusted_p_column(deg_df, f"DEG table for {accession}/{contrast_id}")

            # Count significant DEGs
            if 'logFC' in deg_df.columns:
                sig_df = deg_df[
                    (deg_df[p_col] < pvalue_thresh) &
                    (abs(deg_df['logFC']) > lfc_thresh)
                ]
                n_sig = len(sig_df)
                n_total = len(deg_df)
            else:
                n_sig = 0
                n_total = len(deg_df)

            # Get description from analysis_info
            description = ""
            for contrast in info.get("contrasts", []):
                if contrast.get("name") == contrast_id:
                    description = contrast.get("description", "")
                    break

            rows.append({
                "Accession": accession,
                "Contrast": contrast_id,
                "Description": description,
                "Significant DEGs": n_sig,
                "Total Genes": n_total
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = sort_by_geo_accession(df, "Accession")
    return df