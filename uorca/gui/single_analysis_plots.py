import os
from typing import Optional, Dict

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from uorca.gui.sample_names import (
    find_sample_table,
    read_sample_table,
    sample_groups,
    sample_titles,
)
from uorca.gui.results_integration import (
    ADJUSTED_P_COLUMN,
    ADJUSTED_P_LABEL,
    RAW_P_COLUMN,
    RAW_P_LABEL,
)

logger = logging.getLogger(__name__)


def p_value_label(p_col: str) -> str:
    """Display name for a limma raw or adjusted p-value column."""
    if p_col == ADJUSTED_P_COLUMN:
        return ADJUSTED_P_LABEL
    if p_col == RAW_P_COLUMN:
        return RAW_P_LABEL
    raise ValueError(f"'{p_col}' is neither the adjusted nor the raw p-value column; expected '{ADJUSTED_P_COLUMN}' or '{RAW_P_COLUMN}'")




def load_deg(path: str) -> pd.DataFrame:
    """Load differential expression results from a CSV file."""
    return pd.read_csv(path)


def load_cpm(path: str) -> pd.DataFrame:
    """Load CPM expression values and set 'Gene' as index."""
    df = pd.read_csv(path)
    if "Gene" in df.columns:
        df = df.set_index("Gene")
    return df


def load_sample_groups(results_dir: str, analysis_id: str, cpm_df: pd.DataFrame, analysis_info: Optional[Dict] = None) -> Optional[pd.Series]:
    """Group of each CPM sample column, joined by sample ID (see uorca.gui.sample_names).

    Falls back to each sample as its own group when the sample table or the
    group column is missing, as before.
    """
    sample_cols = [c for c in cpm_df.columns if c != "Gene"]
    identity = pd.Series({c: c for c in sample_cols})
    sample_file = find_sample_table(results_dir, analysis_id)
    group_col = (analysis_info or {}).get("merged_column")
    if not sample_file or not group_col:
        return identity
    sample_df = read_sample_table(sample_file)
    if group_col not in sample_df.columns:
        logger.warning(f"Group column {group_col!r} not in {sample_file}")
        return identity
    groups = sample_groups(sample_df, group_col)
    present = set(sample_cols)
    matched = groups.reindex([s for s in groups.index if s in present])
    if matched.empty:
        logger.warning(f"No CPM sample column of {analysis_id} matches a sample ID in {sample_file}")
        return identity
    return matched


def load_sample_titles(results_dir: str, analysis_id: str) -> Optional[pd.Series]:
    """GEO sample title of each sample, keyed by sample ID; None without a sample table."""
    sample_file = find_sample_table(results_dir, analysis_id)
    if not sample_file:
        return None
    return sample_titles(read_sample_table(sample_file))


def create_volcano_plot(
    deg: pd.DataFrame,
    lfc_col: str = "logFC",
    p_col: str = "adj.P.Val",
    lfc_thresh: float = 1.0,
    p_thresh: float = 0.05,
) -> go.Figure:
    """Create an interactive volcano plot.

    The y-axis title names the p-value kind: adjusted (BH) for ``adj.P.Val``,
    raw for ``P.Value``.
    """
    p_label = p_value_label(p_col)
    df = deg.dropna(subset=[lfc_col, p_col]).copy()
    df["minus_log10_p"] = -np.log10(df[p_col])

    # Categorize genes by significance and direction of change
    conditions = [
        (df[p_col] < p_thresh) & (df[lfc_col] > lfc_thresh),
        (df[p_col] < p_thresh) & (df[lfc_col] < -lfc_thresh),
    ]
    choices = ["Up", "Down"]
    df["category"] = np.select(conditions, choices, default="Not significant")

    fig = px.scatter(
        df,
        x=lfc_col,
        y="minus_log10_p",
        color="category",
        hover_name="Gene",
        hover_data={p_col: True, "minus_log10_p": False},
        labels={p_col: p_label},
        color_discrete_map={
            "Up": "red",
            "Down": "blue",
            "Not significant": "gray",
        },
    )
    if p_col == ADJUSTED_P_COLUMN:
        y_title = "-log10(adjusted p-value, BH)"
    else:
        y_title = "-log10(raw p-value)"
    fig.update_layout(
        xaxis_title="log2 fold change",
        yaxis_title=y_title,
        legend_title=f"Gene category<br>{p_label}<br>< {p_thresh}, |log2FC| > {lfc_thresh}",
    )
    return fig


def create_ma_plot(
    deg: pd.DataFrame,
    avg_col: str = "AveExpr",
    lfc_col: str = "logFC",
    p_col: str = "adj.P.Val",
    lfc_thresh: float = 1.0,
    p_thresh: float = 0.05,
) -> go.Figure:
    """Create an interactive MA plot."""
    df = deg.dropna(subset=[avg_col, lfc_col, p_col]).copy()

    conditions = [
        (df[p_col] < p_thresh) & (df[lfc_col] > lfc_thresh),
        (df[p_col] < p_thresh) & (df[lfc_col] < -lfc_thresh),
    ]
    choices = ["Up", "Down"]
    df["category"] = np.select(conditions, choices, default="Not significant")

    fig = px.scatter(
        df,
        x=avg_col,
        y=lfc_col,
        color="category",
        hover_name="Gene",
        color_discrete_map={"Up": "red", "Down": "blue", "Not significant": "gray"},
    )
    fig.update_layout(
        xaxis_title="Average log2 expression",
        yaxis_title="log2 fold change",
        legend_title=f"Gene category<br>{p_value_label(p_col)}<br>< {p_thresh}, |log2FC| > {lfc_thresh}",
    )
    return fig


