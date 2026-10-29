"""Unsupervised views of one dataset's samples: PCA and a clustering heatmap.

Both use the same gene set: the genes with the highest logCPM variance across
samples (default 500), as limma plotMDS(gene.selection="common") and DESeq2
plotPCA do. PCA centres each gene and does not scale it, so noisy low-count
genes do not get the same weight as strong signals.

Group names in UORCA are often long and share a prefix (for example
"Placenta_CT27_Human trophoblast stem cells_..."). group_display() removes the
shared prefix and, when names are still long, gives each group a short code,
so that legends stay small and plots keep their width.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

DEFAULT_TOP_GENES = 500
MAX_LEGEND_LABEL = 40
MAX_LABELLED_SAMPLES = 40
Z_LIMIT = 3.0
_SEPARATORS = "_ -."
_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "x", "cross", "star", "hexagon")
_NO_GROUP = "(no group)"
_EMPTY_GROUP = "(empty)"


@dataclass
class GeneSelection:
    matrix: pd.DataFrame  # genes x samples, logCPM
    n_requested: int
    n_available: int      # genes with no missing value, from which the top n are chosen
    n_missing: int = 0    # genes left out because they have a missing value

    @property
    def n_used(self) -> int:
        return self.matrix.shape[0]


def top_variable_genes(cpm: pd.DataFrame, n: int = DEFAULT_TOP_GENES) -> GeneSelection:
    """The n genes with the highest logCPM variance across samples (all genes if fewer).

    Ties break by gene name, so the selection is deterministic.
    """
    mat = cpm.set_index("Gene") if "Gene" in cpm.columns else cpm
    mat = mat.astype(float)
    complete = np.asarray(mat.notna().to_numpy().all(axis=1))
    n_missing = int((~complete).sum())
    mat = mat[complete]
    if mat.shape[1] < 2:
        raise ValueError(f"Need at least 2 samples for sample structure plots, found {mat.shape[1]}")
    var = np.asarray(mat.to_numpy().var(axis=1, ddof=1))
    names = mat.index.astype(str).to_numpy()
    order = np.lexsort((names, -var))
    keep = order[: min(n, len(order))]
    return GeneSelection(matrix=mat.iloc[keep], n_requested=n, n_available=mat.shape[0], n_missing=n_missing)


@dataclass
class PCAResult:
    coords: pd.DataFrame  # samples x PC1..PCk
    variance_fraction: List[float]


def run_pca(matrix: pd.DataFrame, n_components: int = 3) -> PCAResult:
    """PCA of samples on a genes x samples matrix; genes centred, not scaled."""
    x = matrix.to_numpy(dtype=float).T
    x = x - x.mean(axis=0)
    k = min(n_components, x.shape[0], x.shape[1])
    if k < 2:
        raise ValueError(f"PCA needs at least 2 samples and 2 genes; found {x.shape[0]} samples and {x.shape[1]} genes")
    if not np.any(x):
        raise ValueError("All selected genes are constant across samples, so there is no variance for PCA")
    pca = PCA(n_components=k, svd_solver="full")
    coords = pca.fit_transform(x)
    cols = [f"PC{i + 1}" for i in range(k)]
    return PCAResult(
        coords=pd.DataFrame(coords, index=matrix.columns, columns=pd.Index(cols)),
        variance_fraction=[float(v) for v in pca.explained_variance_ratio_],
    )


@dataclass
class GroupDisplay:
    prefix: str                 # removed from every name ("" if none)
    labels: Dict[str, str]      # full name -> legend label
    codes: Dict[str, str]       # full name -> G1..Gk
    order: List[str]            # full names in display order
    uses_codes: bool

    def key_table(self, sample_groups: pd.Series) -> pd.DataFrame:
        counts = sample_groups.value_counts()
        return pd.DataFrame(
            {
                "Code": [self.codes[g] for g in self.order],
                "Group": [g for g in self.order],
                "Samples": [int(counts[g]) if g in counts.index else 0 for g in self.order],
            }
        )


def _shared_prefix(names: Sequence[str]) -> str:
    """Longest prefix shared by all names that ends at a separator and leaves
    every name non-empty."""
    if len(names) < 2:
        return ""
    prefix = names[0]
    for n in names[1:]:
        i = 0
        while i < min(len(prefix), len(n)) and prefix[i] == n[i]:
            i += 1
        prefix = prefix[:i]
    cut = max((i + 1 for i, ch in enumerate(prefix) if ch in _SEPARATORS), default=0)
    prefix = prefix[:cut]
    if any(len(n) <= len(prefix) for n in names):
        return ""
    return prefix


def known_groups(groups: Optional[pd.Series]) -> Optional[pd.Series]:
    """None when the groups are not known.

    load_sample_groups() returns each sample as its own group when the sample
    table or the group column is missing; that is not a grouping to colour by.
    """
    if groups is None or len(groups) == 0:
        return None
    if all(str(k) == str(v) for k, v in groups.items()):
        return None
    return groups


def group_display(groups: Sequence[str], max_label: int = MAX_LEGEND_LABEL) -> GroupDisplay:
    order = sorted({str(g) if str(g) != "" else _EMPTY_GROUP for g in groups})
    named = [g for g in order if g not in (_NO_GROUP, _EMPTY_GROUP)]
    prefix = _shared_prefix(named)
    shown = {g: (g[len(prefix):] if g in named else g) for g in order}
    if prefix and any(not any(ch.isalnum() for ch in shown[g]) for g in named):
        # Removing the prefix would leave a name with no letter or digit.
        prefix = ""
        shown = {g: g for g in order}
    codes = {g: f"G{i + 1}" for i, g in enumerate(order)}
    uses_codes = any(len(s) > max_label for s in shown.values())
    labels = {}
    for g in order:
        if uses_codes:
            text = f"{codes[g]} · {shown[g]}"
            labels[g] = text if len(text) <= max_label else text[: max_label - 1].rstrip() + "…"
        else:
            labels[g] = shown[g]
    return GroupDisplay(prefix=prefix, labels=labels, codes=codes, order=order, uses_codes=uses_codes)


def group_styles(n: int) -> List[Dict[str, str]]:
    """Distinct colour (and, above 24 groups, marker symbol) for each of n groups."""
    palette = pc.qualitative.Dark24
    return [
        {"color": palette[i % len(palette)], "symbol": _SYMBOLS[(i // len(palette)) % len(_SYMBOLS)]}
        for i in range(n)
    ]


def sample_group_series(samples: Sequence[str], group_labels: Optional[pd.Series]) -> pd.Series:
    labels = group_labels if group_labels is not None else pd.Series(dtype=object)
    values = [str(labels.get(s, _NO_GROUP)) for s in samples]
    return pd.Series([v if v != "" else _EMPTY_GROUP for v in values], index=list(samples))


def _unique_labels(names: List[str]) -> List[str]:
    """Plotly merges equal category labels into one row, so repeated gene names
    get a numbered suffix."""
    seen: Dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def _legend_below(fig: go.Figure, n_groups: int, plot_height: int, gap_px: int) -> None:
    """Horizontal legend below the plot, gap_px below the plot area, so that a
    long legend adds height instead of taking width from the plot."""
    rows = int(np.ceil(n_groups / 3))
    fig.update_layout(
        height=plot_height + 30 + 24 * rows,
        legend=dict(orientation="h", yanchor="top", y=-gap_px / plot_height, xanchor="left", x=0, title_text=""),
        margin=dict(l=60, r=20, t=40, b=40 + 24 * rows),
    )


def create_pca_figure(
    pca: PCAResult,
    group_labels: Optional[pd.Series],
    display: GroupDisplay,
    sample_titles: Optional[pd.Series] = None,
    x_pc: str = "PC1",
    y_pc: str = "PC2",
) -> go.Figure:
    coords = pca.coords
    groups = sample_group_series([str(s) for s in coords.index], group_labels)
    styles = dict(zip(display.order, group_styles(len(display.order))))
    pct = {f"PC{i + 1}": 100 * v for i, v in enumerate(pca.variance_fraction)}
    fig = go.Figure()
    for g in display.order:
        ids = [s for s, sg in groups.items() if sg == g]
        if len(ids) == 0:
            continue
        titles = [str(sample_titles.get(s, "")) if sample_titles is not None else "" for s in ids]
        fig.add_trace(
            go.Scatter(
                x=coords.loc[ids, x_pc],
                y=coords.loc[ids, y_pc],
                mode="markers",
                name=display.labels[g],
                marker=dict(size=10, color=styles[g]["color"], symbol=styles[g]["symbol"],
                            line=dict(width=0.5, color="white")),
                customdata=[[str(i), t, g] for i, t in zip(ids, titles)],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> %{customdata[1]}<br>Group: %{customdata[2]}"
                    f"<br>{x_pc}: %{{x:.2f}}<br>{y_pc}: %{{y:.2f}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        xaxis_title=f"{x_pc} ({pct[x_pc]:.1f}% of variance)",
        yaxis_title=f"{y_pc} ({pct[y_pc]:.1f}% of variance)",
        template="plotly_white",
    )
    _legend_below(fig, len(display.order), plot_height=520, gap_px=60)
    return fig


def create_clustering_heatmap(
    selection: GeneSelection,
    group_labels: Optional[pd.Series],
    display: GroupDisplay,
    sample_titles: Optional[pd.Series] = None,
) -> go.Figure:
    """Z-scored logCPM of the selected genes; genes and samples clustered
    (complete linkage, Euclidean distance on the z-scores)."""
    mat = selection.matrix
    sd = pd.Series(mat.to_numpy().std(axis=1, ddof=1), index=mat.index).replace(0, np.nan)
    z = mat.sub(mat.mean(axis=1), axis=0).div(sd, axis=0).fillna(0.0)
    if z.shape[0] > 1:
        z = z.iloc[leaves_list(linkage(pdist(z.to_numpy()), method="complete"))]
    if z.shape[1] > 1:
        z = z.iloc[:, leaves_list(linkage(pdist(z.to_numpy().T), method="complete"))]
    samples = list(z.columns)
    groups = sample_group_series(samples, group_labels)
    styles = dict(zip(display.order, group_styles(len(display.order))))
    index = {g: i for i, g in enumerate(display.order)}
    k = len(display.order)
    # Discrete colourscale: one flat band per group.
    scale = []
    for i, g in enumerate(display.order):
        lo, hi = i / k, (i + 1) / k
        scale += [[lo, styles[g]["color"]], [hi, styles[g]["color"]]]
    titles = [str(sample_titles.get(s, "")) if sample_titles is not None else "" for s in samples]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.04, 0.96], vertical_spacing=0.01)
    fig.add_trace(
        go.Heatmap(
            z=[[index[g] for g in groups]],
            x=samples,
            y=["Group"],
            zmin=-0.5,
            zmax=k - 0.5,
            colorscale=scale if k > 1 else [[0, styles[display.order[0]]["color"]], [1, styles[display.order[0]]["color"]]],
            showscale=False,
            customdata=[[[t, g] for t, g in zip(titles, groups)]],
            hovertemplate="<b>%{x}</b> %{customdata[0]}<br>Group: %{customdata[1]}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=z.to_numpy(),
            x=samples,
            y=_unique_labels([str(g) for g in z.index]),
            zmin=-Z_LIMIT,
            zmax=Z_LIMIT,
            zmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title=dict(text="Z-score<br>(logCPM)", side="right"), len=0.8, y=0.45),
            hovertemplate="Gene: %{y}<br>Sample: %{x}<br>Z-score: %{z:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    # Legend entries for the group colour bar.
    for g in display.order:
        if (groups == g).any():
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="markers", name=display.labels[g],
                           marker=dict(size=10, color=styles[g]["color"], symbol="square"),
                           hoverinfo="skip"),
                row=2, col=1,
            )
    show_labels = len(samples) <= MAX_LABELLED_SAMPLES
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=show_labels, tickangle=-45, row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(showticklabels=True, row=1, col=1)
    fig.update_layout(template="plotly_white", plot_bgcolor="white")
    _legend_below(fig, len(display.order), plot_height=720, gap_px=110 if show_labels else 20)
    return fig


def caption(selection: GeneSelection, n_samples: int, display: GroupDisplay, what: str) -> str:
    if selection.n_available > selection.n_requested:
        genes = f"the {selection.n_used} genes with the highest logCPM variance"
    elif selection.n_available == selection.n_requested:
        genes = f"all {selection.n_used} genes"
    else:
        genes = f"all {selection.n_used} genes (fewer than {selection.n_requested} passed filtering)"
    text = f"{what} {n_samples} samples, {genes}."
    if selection.n_missing:
        text += f" {selection.n_missing} genes with a missing value were left out."
    if display.prefix:
        text += f" Group names share the prefix “{display.prefix}”; it is not shown."
    return text
