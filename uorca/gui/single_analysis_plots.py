import os
from typing import Optional, Dict

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import zscore

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


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def load_sample_groups(results_dir: str, analysis_id: str, cpm_df: pd.DataFrame, analysis_info: Optional[Dict] = None) -> Optional[pd.Series]:
    """Return a mapping of sample names to group labels if available."""
    analysis_dir = os.path.join(results_dir, analysis_id)
    sample_file_locations = [
        os.path.join(analysis_dir, "metadata", "edger_analysis_samples.csv"),
        os.path.join(analysis_dir, "edger_analysis_samples.csv"),
        os.path.join(results_dir, "edger_analysis_samples.csv"),
        os.path.join(os.path.dirname(results_dir), "edger_analysis_samples.csv"),
    ]
    sample_file = _first_existing(sample_file_locations)
    sample_cols = [c for c in cpm_df.columns if c != "Gene"]
    if not sample_file:
        return pd.Series({c: c for c in sample_cols})
    try:
        tmp = pd.read_csv(sample_file, nrows=0)
        first_col = tmp.columns[0]
        has_index = first_col == "" or first_col.startswith("Unnamed")
        sample_df = pd.read_csv(sample_file, index_col=0 if has_index else None)
        group_col = None
        if analysis_info and "merged_column" in analysis_info:
            group_col = analysis_info["merged_column"]
        if group_col and group_col in sample_df.columns:
            sample_pattern = [c for c in sample_cols if c.startswith("Sample")]
            mapping = {}
            if sample_pattern:
                for i in range(min(len(sample_pattern), len(sample_df))):
                    mapping[sample_pattern[i]] = sample_df.iloc[i][group_col]
            else:
                for i in range(min(len(sample_cols), len(sample_df))):
                    mapping[sample_cols[i]] = sample_df.iloc[i][group_col]
            return pd.Series(mapping)
    except Exception:
        pass
    return pd.Series({c: c for c in sample_cols})


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


def create_deg_heatmap(cpm: pd.DataFrame,
                      deg: pd.DataFrame,
                      group_labels: Optional[pd.Series] = None,
                      p_col: str = "adj.P.Val",
                      top_n: int = 50,
                      font_size: int = 12,
                      gene_height_px: int = 22,
                      sample_width_px: int = 80) -> Optional[go.Figure]:
    """
    Create an interactive clustered heatmap of CPM expression for the top differentially expressed genes.

    Parameters
    ----------
    cpm : pd.DataFrame
        CPM expression DataFrame, with genes as index and samples as columns
    deg : pd.DataFrame
        DEG results DataFrame with at least columns 'Gene' and an adjusted p-value column (e.g., 'adj.P.Val')
    group_labels : Optional[pd.Series]
        Mapping from sample names to biological group labels
    p_col : str
        Column name in deg for adjusted p-value
    top_n : int
        Number of top DE genes to include
    font_size : int
        Font size for axis labels
    gene_height_px : int
        Pixel height per gene row for figure height calculation
    sample_width_px : int
        Pixel width per sample column for figure width calculation

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive heatmap figure
    """

    # Filter deg for top N genes by smallest p-value
    if p_col not in deg.columns:
        raise ValueError(
            f"DEG table has no '{p_col}' column ({p_value_label(p_col)}); "
            f"cannot rank genes for the heatmap. Columns present: {list(deg.columns)}"
        )

    deg_filtered = deg.dropna(subset=['Gene', p_col])
    if deg_filtered.empty:
        logger.warning(f"No DEG rows after dropping rows with a missing gene or {p_col} (adjusted/raw p-value) value")
        return None

    top_genes = deg_filtered.nsmallest(top_n, p_col)['Gene'].tolist()
    if not top_genes:
        logger.warning("No top genes selected for heatmap")
        return None

    # Ensure CPM gene index
    if 'Gene' in cpm.columns:
        cpm = cpm.set_index('Gene')

    # Subset CPM to top genes, only keep genes present in both datasets
    genes_in_cpm = cpm.index.intersection(top_genes)
    heat_df = cpm.loc[genes_in_cpm]

    if heat_df.empty:
        logger.warning("No overlap between top DE genes and CPM data for heatmap")
        return None

    # Fill missing values if any
    heat_df = heat_df.fillna(0)

    heat_df = pd.DataFrame(
        zscore(heat_df, axis=1, nan_policy="omit"),   # row-wise
        index=heat_df.index,
        columns=heat_df.columns
    ).fillna(0)

    # Hierarchical clustering of genes (rows) and samples (columns)
    if heat_df.shape[0] > 1:
        try:
            linkage_genes = linkage(pdist(heat_df.values), method='complete')
            order_genes = leaves_list(linkage_genes)
            heat_df = heat_df.iloc[order_genes, :]
        except Exception as e:
            logger.warning(f"Gene clustering failed, using original order: {e}")

    if heat_df.shape[1] > 1:
        try:
            linkage_samples = linkage(pdist(heat_df.values.T), method='complete')
            order_samples = leaves_list(linkage_samples)
            heat_df = heat_df.iloc[:, order_samples]
        except Exception as e:
            logger.warning(f"Sample clustering failed, using original order: {e}")

    # Prepare hovertemplate with sample group info
    hover_text = []
    for gene in heat_df.index:
        row_hover = []
        for sample in heat_df.columns:
            expr_val = heat_df.at[gene, sample]
            group = group_labels[sample] if group_labels is not None and sample in group_labels.index else sample
            row_hover.append(f"Gene: {gene}<br>Sample: {sample}<br>Group: {group}<br>Expression (logCPM): {expr_val:.2f}")
        hover_text.append(row_hover)

    # Determine figure size dynamically
    width = max(600, min(1500, heat_df.shape[1] * sample_width_px))
    height = max(600, min(2000, heat_df.shape[0] * gene_height_px))

    # Create heatmap using go.Heatmap for better control
    limit = np.nanmax(np.abs(heat_df.values))
    fig = go.Figure(
        data=go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns,
            y=heat_df.index,
            colorscale="RdBu_r",                 # diverging palette
            zmin=-limit,
            zmax=limit,
            zmid=0,                              # centre at 0, like ComplexHeatmap
            hoverinfo="text",
            text=hover_text,
            colorbar=dict(
                title=dict(text="Z-score<br>(log\u2082 CPM)", side="right")
            )
        )
    )

    # Add color bar annotation for sample groups
    if group_labels is not None:
        # Create a unique color for each group
        unique_groups = pd.unique(group_labels.loc[heat_df.columns].dropna())
        import plotly.colors as pc
        colors = pc.qualitative.Plotly
        group_color_map = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}

        # Create color bar annotations as rectangles above heatmap
        annotation_shapes = []
        for i, sample in enumerate(heat_df.columns):
            if sample in group_labels.index:
                group = group_labels[sample]
                color = group_color_map.get(group, 'grey')

                # Add rectangle annotation
                annotation_shapes.append(
                    dict(type='rect',
                         xref='x', yref='paper',
                         x0=i - 0.5, x1=i + 0.5,
                         y0=1.02, y1=1.06,
                         fillcolor=color,
                         line=dict(width=0))
                )

        if annotation_shapes:
            fig.update_layout(shapes=annotation_shapes)

            # Add legend for group colors
            legend_items = []
            for grp, colr in group_color_map.items():
                legend_items.append(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=10, color=colr),
                        legendgroup=grp,
                        showlegend=True,
                        name=str(grp)
                    )
                )
            fig.add_traces(legend_items)

            # Position legend
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.12,
                    xanchor="right",
                    x=1,
                    title="Sample Groups"
                )
            )

    # Update layout for labels and formatting
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=160, r=40, t=100, b=160),
        font=dict(family="Arial", size=font_size),
        title=dict(
            text=f"Top {len(heat_df)} Differentially Expressed Genes",
            x=0.5,
            font=dict(size=font_size + 2)
        )
    )

    # Update axes
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=font_size * 0.8),
        side='bottom',
        title="Samples"
    )
    fig.update_yaxes(
        tickfont=dict(size=font_size * 0.8),
        title="Genes"
    )

    return fig


def create_pca_plot(cpm: pd.DataFrame, group_labels: Optional[pd.Series] = None,
                    n_components: int = 2) -> go.Figure:
    """Perform PCA on logCPM expression values."""
    if "Gene" in cpm.columns:
        cpm = cpm.set_index("Gene")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cpm.T)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)
    pc_df = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(n_components)])
    pc_df.index = cpm.columns
    if group_labels is not None:
        pc_df["group"] = pc_df.index.map(group_labels).fillna(pc_df.index)
        fig = px.scatter(pc_df, x="PC1", y="PC2", color="group", hover_name=pc_df.index)
    else:
        fig = px.scatter(pc_df, x="PC1", y="PC2", hover_name=pc_df.index)
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return fig
