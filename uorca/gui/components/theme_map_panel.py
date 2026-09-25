"""The theme-map panel of the Identify page (spec section 9).

This module **reads artifacts and draws**. It never calls the OpenAI client and never
runs the numeric pipeline (spec section 8). An API call inside a Streamlit rerun is the
``BrokenPipeError`` hazard of ``CLAUDE.md`` Issue 3, so the build is dispatched to
``TaskManager`` as a background task in the pattern ``_run_identification`` already uses,
and the panel itself only ever touches ``theme_map/``.

The panel has exactly three states:

1. ``theme_map/FAILED.txt`` present — ``st.error`` with the recorded message, plus a
   rebuild button. Loud, never silent.
2. No ``theme_map/`` — the build button, the valid-dataset count and the estimated cost
   (spec 9.7). The button confirms before it spends.
3. A complete artifact set — the plot, then the theme table (spec 9.4, 9.6).

Every piece of data shaping is a plain function so it can be unit tested without a
Streamlit session; ``render_theme_map_panel`` is a thin shell over those functions.

Nothing here degrades quietly. A mismatch between ``theme_map.csv`` and the results CSV
raises (spec section 7), rather than drawing a map of the wrong run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

from uorca.core import TaskManager, TaskStatus
from uorca.identification.theme_map.artifacts import (
    ThemeMapArtifactError,
    ThemeMapBuildFailed,
    artifacts_exist,
    read_artifacts,
    read_failure,
    write_failure,
)
from uorca.identification.theme_map.types import DatasetPoint, Theme, ThemeMapResult

__all__ = [
    "render_theme_map_panel",
    "ThemeMapPanelError",
    "RELEVANCE_COLORSCALE",
    "THEME_COLOR_SEQUENCE",
    "COLOR_BY_OPTIONS",
    "THEME_TABLE_COLUMNS",
    "MarkerStyle",
    "SCORED_MARKER",
    "UNSCORED_MARKER",
    "DIMMED_COLOR",
    "UNTHEMED_COLOR",
    "DEFAULT_HIGHLIGHT_THRESHOLD",
    "LOW_COVERAGE_PERCENT",
    "hidden_caption",
    "opening_threshold",
    "low_coverage_caption",
    "HIGHLIGHT_THRESHOLD_MIN",
    "HIGHLIGHT_THRESHOLD_MAX",
    "HIGHLIGHT_THRESHOLD_STEP",
    "THEME_LABEL_FONT_MIN",
    "THEME_LABEL_FONT_MAX",
    "CoverageSummary",
    "BuildEstimate",
    "marker_style_for",
    "is_highlighted",
    "highlight_caption",
    "theme_medoid",
    "theme_label_font_size",
    "theme_label_annotations",
    "summarize_coverage",
    "coverage_caption",
    "population_caption",
    "build_theme_table",
    "build_theme_map_figure",
    "theme_dataset_table",
    "check_accessions",
    "valid_texts",
    "estimate_build",
]


class ThemeMapPanelError(RuntimeError):
    """The panel cannot draw what it was handed, and will not guess."""


# ---------------------------------------------------------------------------
# Encoding constants (spec 9.4, 9.5, 12.1)
# ---------------------------------------------------------------------------

#: The sequential ramp for the relevance view. Viridis is perceptually uniform and
#: colour-blind safe (it is monotone in lightness, so deuteranopes and protanopes read
#: it by luminance alone), and because it spans dark blue to bright yellow it keeps
#: contrast against both the light and the dark Streamlit background. Spec section 12
#: left the exact ramp open; this is the choice.
RELEVANCE_COLORSCALE = "Viridis"

#: Qualitative colours for the theme view. k is 20-40 per run (spec, decision 10), which
#: exceeds every single Plotly qualitative palette, so two mid-luminance palettes are
#: concatenated to 50 hues. Mid-luminance matters: pure-dark or pure-light sequences
#: vanish into one of the two Streamlit themes.
THEME_COLOR_SEQUENCE: tuple[str, ...] = tuple(qualitative.Dark24) + tuple(qualitative.Alphabet)

#: Unthemed points are grey and always drawn (spec, decision 11).
UNTHEMED_COLOR = "#9aa0a6"
UNTHEMED_LABEL = "Unthemed"

#: The colour of a point that carries a relevance score below the highlight threshold.
#: It is the same neutral mid grey as :data:`UNTHEMED_COLOR` — it reads against both the
#: light and the dark Streamlit background — but it is drawn at :data:`DIMMED_OPACITY`
#: so a backgrounded point sits visibly behind a highlighted one.
DIMMED_COLOR = "#9aa0a6"
DIMMED_OPACITY = 0.45

#: The highlight threshold controls **colour only**. Nothing is ever hidden: spec
#: decision 11 and spec 9.4 both require every valid dataset to stay on the plot, and a
#: point below the threshold is greyed rather than dropped.
DEFAULT_HIGHLIGHT_THRESHOLD = 7.0
#: Below this scored coverage a run opens with the threshold at 0.0 instead. An unscored
#: dataset can never clear a threshold, so on a run that scored only a tenth of its
#: corpus the 7.0 default draws the whole map grey in both views and hides the theme
#: structure behind a slider drag. Measured on the archived runs: sc_hipsc_npc* score
#: 36-38% and three scratch runs score 9.3-10.0%, while every run written by current code
#: scores 100%. No colouring rule changes -- only where the slider starts.
LOW_COVERAGE_PERCENT = 80.0
HIGHLIGHT_THRESHOLD_MIN = 0.0
HIGHLIGHT_THRESHOLD_MAX = 10.0
HIGHLIGHT_THRESHOLD_STEP = 0.5

#: Theme labels are drawn on the plot at the theme medoid. Font size carries the theme's
#: dataset count so the large themes read first, clamped so the smallest theme is still
#: legible and the largest does not swamp the map.
THEME_LABEL_FONT_MIN = 9
THEME_LABEL_FONT_MAX = 16
#: A neutral grey wash. Neither white nor black works in both Streamlit themes; a mid
#: grey does, and it carries dark text at a contrast ratio of roughly 7:1.
THEME_LABEL_BGCOLOR = "rgba(160,160,160,0.85)"
THEME_LABEL_TEXT_COLOR = "#111111"
THEME_LABEL_BORDER_COLOR = "rgba(70,70,70,0.9)"

COLOR_BY_OPTIONS: tuple[str, ...] = ("Relevance", "Theme")

THEME_TABLE_COLUMNS: tuple[str, ...] = (
    "ThemeId",
    "Label",
    "n",
    "Mean relevance",
    "Max relevance",
    "Keywords",
)

#: text-embedding-3-small list price, US dollars per million tokens.
EMBEDDING_USD_PER_MILLION_TOKENS = 0.02
#: Characters per token, the standard English approximation. The estimate is an upper
#: bound: it counts every valid row, while the build de-duplicates sub-series first.
CHARS_PER_TOKEN = 4.0

RESULTS_CSV_NAME = "Dataset_identification_result.csv"


@dataclass(frozen=True)
class MarkerStyle:
    """How one class of point is drawn. Size is uniform within a class.

    ``CONSTITUTION-analysis.md`` section 3 reserves size for set magnitude, so size does
    not encode relevance. The only size difference is the one spec 9.5 requires: an
    unscored dataset is a *small hollow* marker so it cannot be read as a low-scoring
    one.
    """

    size: float
    symbol: str
    line_width: float
    opacity: float
    hollow: bool


SCORED_MARKER = MarkerStyle(size=8.0, symbol="circle", line_width=0.0, opacity=0.85, hollow=False)
UNSCORED_MARKER = MarkerStyle(
    size=5.0, symbol="circle-open", line_width=1.2, opacity=0.65, hollow=True
)


def marker_style_for(point: DatasetPoint) -> MarkerStyle:
    """Hollow for a dataset the run never scored, filled for every scored one.

    A ``relevance_score`` of ``0.0`` is a score. Only ``None`` is unscored.
    """
    return UNSCORED_MARKER if point.relevance_score is None else SCORED_MARKER


# ---------------------------------------------------------------------------
# Captions (spec 9.3, 9.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageSummary:
    """How much of the mapped corpus carries a relevance score."""

    n_points: int
    n_scored: int

    @property
    def n_unscored(self) -> int:
        return self.n_points - self.n_scored

    @property
    def percent_scored(self) -> float:
        if self.n_points == 0:
            return 0.0
        return 100.0 * self.n_scored / self.n_points


def summarize_coverage(points: Sequence[DatasetPoint]) -> CoverageSummary:
    scored = sum(1 for point in points if point.relevance_score is not None)
    return CoverageSummary(n_points=len(points), n_scored=scored)


def hidden_caption(points: Sequence[DatasetPoint], threshold: float) -> str:
    """What the hide control removed, stated plainly.

    Spec decision 11 says nothing is hidden; decision 16 lets a user override that. An
    override must account for itself, so this names both the total hidden and the
    unscored share of it -- an unscored dataset disappears for a different reason from a
    low-scoring one, and a reader should not have to work that out.
    """
    hidden = [point for point in points if not is_highlighted(point, threshold)]
    unscored = sum(1 for point in hidden if point.relevance_score is None)
    if not hidden:
        return f"Nothing hidden: every dataset is at or above {threshold:.1f}."
    tail = f", of which {unscored:,} were never scored" if unscored else ""
    return (
        f"Hiding {len(hidden):,} of {len(points):,} datasets below {threshold:.1f}{tail}. "
        "The layout does not change, so the survivors sit where they always sat."
    )


def opening_threshold(points: Sequence[DatasetPoint]) -> float:
    """Where the highlight slider starts for this run.

    :data:`DEFAULT_HIGHLIGHT_THRESHOLD` on a fully scored run, 0.0 on one that scored less
    than :data:`LOW_COVERAGE_PERCENT` of its corpus. See that constant for the reason and
    the measurements. This changes no colouring rule: it only picks a starting position
    that shows the user something on the first render.
    """
    return (
        DEFAULT_HIGHLIGHT_THRESHOLD
        if summarize_coverage(points).percent_scored >= LOW_COVERAGE_PERCENT
        else HIGHLIGHT_THRESHOLD_MIN
    )


def low_coverage_caption(points: Sequence[DatasetPoint]) -> str | None:
    """Why the slider starts at 0.0, or ``None`` when it starts at the default."""
    coverage = summarize_coverage(points)
    if coverage.percent_scored >= LOW_COVERAGE_PERCENT:
        return None
    return (
        f"This run scored {coverage.n_scored:,} of {coverage.n_points:,} valid datasets "
        f"({coverage.percent_scored:.1f}%), so the threshold opens at "
        f"{HIGHLIGHT_THRESHOLD_MIN:.1f}. An unscored dataset is hollow and never clears a "
        "threshold, so at the usual default this map would draw entirely grey."
    )


def coverage_caption(points: Sequence[DatasetPoint]) -> str:
    """State relevance coverage as a count and a percent (spec 9.5).

    Three of the four runs on disk scored only 10-38% of valid rows, so an unscored
    point must be both visually and verbally distinguished from a low-scoring one.
    """
    summary = summarize_coverage(points)
    text = (
        f"Relevance scored for {summary.n_scored:,} of {summary.n_points:,} mapped "
        f"datasets ({summary.percent_scored:.1f}%)."
    )
    if summary.n_unscored:
        text += (
            f" The {summary.n_unscored:,} unscored datasets are drawn as small hollow "
            "markers — they are not low-scoring, they were never scored."
        )
    else:
        text += " No dataset is drawn as a hollow, unscored marker."
    return text


def population_caption(n_table_rows: int, n_points: int) -> str:
    """State both populations, so the gap reads as intended (spec 9.3)."""
    return (
        f"The results table above shows the {n_table_rows:,} datasets with a relevance "
        f"score above zero. The map below shows all {n_points:,} valid datasets in this "
        "run, scored or not — that difference is the point of the map, not a bug."
    )


# ---------------------------------------------------------------------------
# The highlight threshold (spec 9.4, decision 11)
# ---------------------------------------------------------------------------


def is_highlighted(point: DatasetPoint, threshold: float) -> bool:
    """Does this dataset clear the highlight threshold?

    Only a **scored** dataset can. ``relevance_score is None`` means the run never
    scored it, not that it scored zero (spec 9.5), so an unscored dataset is never
    highlighted — not even at a threshold of ``0.0``. Collapsing the two would recreate
    exactly the confusion spec 9.5 exists to prevent.

    The comparison is inclusive: a dataset scoring exactly the threshold is highlighted,
    which is what "at or above" says on the control.
    """
    score = point.relevance_score
    if score is None:
        return False
    return float(score) >= float(threshold)


def highlight_caption(points: Sequence[DatasetPoint], threshold: float) -> str:
    """How many of the mapped datasets the current threshold picks out.

    The total is every mapped dataset, scored or not, because that is what the plot
    draws. Nothing is filtered out of the map by the threshold.
    """
    n_highlighted = sum(1 for point in points if is_highlighted(point, threshold))
    text = (
        f"{n_highlighted:,} of {len(points):,} datasets at or above {threshold:.1f}. "
        "Everything below is drawn grey — nothing is hidden."
    )
    n_unscored = sum(1 for point in points if point.relevance_score is None)
    if n_unscored:
        text += (
            f" The {n_unscored:,} unscored datasets cannot pass any threshold and stay "
            "hollow grey."
        )
    return text


# ---------------------------------------------------------------------------
# Theme table (spec 9.6)
# ---------------------------------------------------------------------------


def _theme_label(themes: Sequence[Theme]) -> dict[int, str]:
    return {theme.id: theme.label for theme in themes}


def _unthemed(points: Iterable[DatasetPoint]) -> list[DatasetPoint]:
    return [point for point in points if point.theme_id < 0]


def build_theme_table(points: Sequence[DatasetPoint], themes: Sequence[Theme]) -> pd.DataFrame:
    """One row per theme, plus one ``Unthemed (n)`` row when anything is unthemed.

    Sorted by size, descending. Every column is sortable once ``st.dataframe`` renders
    it. The ``Max relevance`` cell of the unthemed row is the honest warning of spec 9.6:
    in the PPAR-gamma run the highest-scoring dataset of all was unthemed.
    """
    known = {theme.id for theme in themes}
    unknown = sorted({point.theme_id for point in points if point.theme_id >= 0} - known)
    if unknown:
        raise ThemeMapPanelError(
            f"theme_map.csv references theme ids {unknown} that themes.json does not "
            "define. The artifacts are written as a set, so this run's theme_map/ "
            "directory is inconsistent and must be rebuilt."
        )

    rows: list[dict[str, Any]] = [
        {
            "ThemeId": theme.id,
            "Label": theme.label,
            "n": theme.n,
            "Mean relevance": theme.mean_relevance,
            "Max relevance": theme.max_relevance,
            "Keywords": ", ".join(theme.keywords),
        }
        for theme in themes
    ]

    unthemed = _unthemed(points)
    if unthemed:
        scores = [p.relevance_score for p in unthemed if p.relevance_score is not None]
        rows.append(
            {
                "ThemeId": -1,
                "Label": f"{UNTHEMED_LABEL} ({len(unthemed)})",
                "n": len(unthemed),
                "Mean relevance": (sum(scores) / len(scores)) if scores else None,
                "Max relevance": max(scores) if scores else None,
                "Keywords": "",
            }
        )

    table = pd.DataFrame(rows, columns=pd.Index(THEME_TABLE_COLUMNS))
    # An all-``None`` relevance column would otherwise arrive as object dtype, which
    # ``st.column_config.NumberColumn`` cannot format. ``None`` becomes ``NaN`` and stays
    # an absent value — it is never coerced to 0.0.
    for column in ("Mean relevance", "Max relevance"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.sort_values("n", ascending=False, kind="stable").reset_index(drop=True)
    return table


def theme_dataset_table(
    points: Sequence[DatasetPoint],
    titles: Mapping[str, str],
    theme_id: int,
) -> pd.DataFrame:
    """The datasets of one theme, for the in-panel expansion of spec 9.6.

    This expands inside the panel only. It never filters the results table above, which
    spec section 2 keeps out of scope.
    """
    members = [point for point in points if point.theme_id == theme_id]
    rows = [
        {
            "Accession": point.geo_accession,
            "Title": titles.get(point.geo_accession, ""),
            "Relevance": point.relevance_score,
            "Theme probability": point.theme_prob,
            "QuerySim": point.query_sim,
            "QuerySim rank": point.query_sim_rank,
            "GEO": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={point.geo_accession}",
        }
        for point in members
    ]
    table = pd.DataFrame(
        rows,
        columns=pd.Index([
            "Accession",
            "Title",
            "Relevance",
            "Theme probability",
            "QuerySim",
            "QuerySim rank",
            "GEO",
        ]),
    )
    if not table.empty:
        table = table.sort_values(
            "Relevance", ascending=False, na_position="last", kind="stable"
        ).reset_index(drop=True)
    return table


# ---------------------------------------------------------------------------
# Figure (spec 9.4)
# ---------------------------------------------------------------------------

_HOVER_TEMPLATE = (
    "<b>%{customdata[0]}</b><br>"
    "%{customdata[1]}<br>"
    "Theme: %{customdata[2]}<br>"
    "Relevance: %{customdata[3]}"
    "<extra></extra>"
)


def _theme_color(theme_id: int) -> str:
    if theme_id < 0:
        return UNTHEMED_COLOR
    return THEME_COLOR_SEQUENCE[theme_id % len(THEME_COLOR_SEQUENCE)]


def _customdata(
    group: Sequence[DatasetPoint], titles: Mapping[str, str], labels: Mapping[int, str]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for point in group:
        title = titles.get(point.geo_accession)
        if title is None:
            raise ThemeMapPanelError(
                f"No title for {point.geo_accession} in the results CSV, so the map "
                "cannot label it on hover. The theme map and the results CSV disagree "
                "and the map must be rebuilt."
            )
        label = UNTHEMED_LABEL if point.theme_id < 0 else labels[point.theme_id]
        score = "not scored" if point.relevance_score is None else f"{point.relevance_score:.2f}"
        rows.append([point.geo_accession, str(title), label, score])
    return rows


# ---------------------------------------------------------------------------
# Theme labels drawn on the plot
# ---------------------------------------------------------------------------


def theme_medoid(points: Sequence[DatasetPoint], theme_id: int) -> tuple[float, float]:
    """The 2-D position of the theme member nearest that theme's centroid.

    The label goes on a real dataset rather than on the arithmetic centroid, because a
    t-SNE cluster is often crescent- or ring-shaped and its centroid can fall in empty
    space — or inside a neighbouring theme.

    A theme with no members is not an empty theme, it is a contradiction between
    ``themes.json`` and ``theme_map.csv``. Spec section 7 requires that to raise.
    """
    members = [point for point in points if point.theme_id == theme_id]
    if not members:
        raise ThemeMapPanelError(
            f"Theme {theme_id} is defined in themes.json but no row of theme_map.csv "
            "belongs to it, so it has no position to label. The artifacts are written "
            "as a set; this run's theme_map/ directory is inconsistent and must be "
            "rebuilt."
        )
    centroid_x = sum(point.x for point in members) / len(members)
    centroid_y = sum(point.y for point in members) / len(members)
    nearest = min(
        members,
        key=lambda point: (point.x - centroid_x) ** 2 + (point.y - centroid_y) ** 2,
    )
    return (float(nearest.x), float(nearest.y))


def theme_label_font_size(n: int, n_min: int, n_max: int) -> int:
    """Label point size for a theme of ``n`` datasets, given the run's smallest and largest.

    Interpolated on a square root between the SMALLEST theme present at
    :data:`THEME_LABEL_FONT_MIN` and the largest at :data:`THEME_LABEL_FONT_MAX`, clamped
    at both ends. Size is the only channel a label has, and k is 20-40 (spec,
    decision 10), so this is what makes the big themes read first in a crowded map.

    Both details were measured, not chosen. On the real 20-theme PPARg map the themes
    span 15 to 169 datasets, because ``hdbscan_min_cluster_size`` is 15 (spec 5.2), so no
    theme can be smaller than that. Interpolating from a notional one-dataset theme put
    14 of the 20 labels at 9-10 points and only one above 11 -- effectively uniform, which
    is not the encoding that was asked for. Anchoring at the smallest theme present uses
    the full range. The square root then spreads the middle: theme sizes are heavily
    skewed towards the floor, so a linear map still bunches most labels at the bottom.

    This is a *label* size, not a mark size: ``CONSTITUTION-analysis.md`` section 3
    reserves mark size for set magnitude, and a theme's dataset count is exactly that.
    """
    span = max(int(n_max) - int(n_min), 1)
    fraction = (int(n) - int(n_min)) / span
    fraction = min(max(fraction, 0.0), 1.0)
    size = THEME_LABEL_FONT_MIN + math.sqrt(fraction) * (
        THEME_LABEL_FONT_MAX - THEME_LABEL_FONT_MIN
    )
    return int(round(size))


def theme_label_annotations(
    points: Sequence[DatasetPoint], themes: Sequence[Theme]
) -> list[dict[str, Any]]:
    """One Plotly annotation per theme, at the theme medoid.

    ``Unthemed`` is deliberately absent. It is not a theme (spec 9.6) and its members are
    scattered across the whole map, so a single label for them would point at nothing.
    The unthemed count stays in the theme table, where it is honest.

    Labels will overlap at k = 20-40. That is accepted: the plot pans and zooms, and the
    checkbox turns them off.
    """
    members: dict[int, list[DatasetPoint]] = {}
    for point in points:
        if point.theme_id >= 0:
            members.setdefault(point.theme_id, []).append(point)

    # A theme with no member among ``points`` gets no label. This matters when the
    # caller hides the below-threshold datasets: a label floating over a theme whose
    # every member has just been hidden points at nothing.
    labelled = [theme for theme in themes if theme.id >= 0 and members.get(theme.id)]
    if not labelled:
        return []

    sizes = [len(members.get(theme.id, [])) for theme in labelled]
    n_min, n_max = min(sizes), max(sizes)
    annotations: list[dict[str, Any]] = []
    for theme in labelled:
        x, y = theme_medoid(points, theme.id)
        annotations.append(
            {
                "x": x,
                "y": y,
                "text": theme.label,
                "showarrow": False,
                "align": "center",
                "captureevents": False,
                "font": {
                    "size": theme_label_font_size(
                        len(members.get(theme.id, [])), n_min, n_max
                    ),
                    "color": THEME_LABEL_TEXT_COLOR,
                },
                "bgcolor": THEME_LABEL_BGCOLOR,
                "bordercolor": THEME_LABEL_BORDER_COLOR,
                "borderwidth": 1,
                "borderpad": 2,
            }
        )
    return annotations


def build_theme_map_figure(
    points: Sequence[DatasetPoint],
    titles: Mapping[str, str],
    themes: Sequence[Theme],
    color_by: str = "Relevance",
    threshold: float = DEFAULT_HIGHLIGHT_THRESHOLD,
    show_theme_names: bool = True,
    hide_below: bool = False,
) -> go.Figure:
    """The t-SNE scatter of spec 9.4.

    Position is the t-SNE coordinates and encodes local structure only. Colour is either
    relevance (the default view — spec decision 6 was reversed after the PPAR-gamma
    benchmark showed 44% of the best datasets drawn as identical grey dots under theme
    colour) or theme. Size is uniform. There is no legend: the theme table is the legend.

    ``threshold`` controls colour in **both** views. A dataset at or above it draws in
    colour — on the Viridis ramp in the relevance view, in its theme colour in the theme
    view — and one below it draws grey. Nothing is hidden (spec decision 11): a greyed
    point is still on the map, still hoverable, and still counted in every caption.

    An unscored dataset is grey and hollow in both views at every threshold. It was never
    scored, so it cannot clear a bar, and spec 9.5 forbids drawing it like a low-scoring
    one.

    ``show_theme_names`` draws the theme labels of :func:`theme_label_annotations`. They
    appear in the theme view only; in the relevance view they would label a colour
    encoding that has nothing to do with themes.

    ``hide_below`` removes the below-threshold datasets from the figure entirely rather
    than greying them (decision 16, 2026-09-22, at the user's request). It defaults to
    ``False``, so spec decision 11 remains what the map does unless a user asks otherwise.
    It also removes the **unscored** datasets: they can never clear a threshold, and
    keeping a dataset nobody scored while hiding one that scored 6.9 would be incoherent.
    A theme whose every member is hidden loses its label too.
    """
    if color_by not in COLOR_BY_OPTIONS:
        raise ThemeMapPanelError(
            f"Unknown colour mode {color_by!r}; expected one of {list(COLOR_BY_OPTIONS)}."
        )

    known = {theme.id for theme in themes}
    unknown = sorted({p.theme_id for p in points if p.theme_id >= 0} - known)
    if unknown:
        raise ThemeMapPanelError(
            f"theme_map.csv references theme ids {unknown} that themes.json does not define."
        )

    labels = _theme_label(themes)
    scored = [point for point in points if point.relevance_score is not None]
    unscored = [point for point in points if point.relevance_score is None]
    highlighted = [point for point in scored if is_highlighted(point, threshold)]
    dimmed = [point for point in scored if not is_highlighted(point, threshold)]
    if hide_below:
        dimmed = []
        unscored = []

    # The ramp is pinned to the full score range of the corpus and does not depend on
    # the threshold, so moving the slider re-colours points without rescaling the ramp
    # underneath them.
    ramp_max = max(
        [HIGHLIGHT_THRESHOLD_MAX] + [float(p.relevance_score or 0.0) for p in scored]
    )

    fig = go.Figure()

    if color_by == "Relevance":
        # A dedicated, empty trace owns the colourbar. Attaching it to the highlighted
        # trace instead would make the legend to the ramp appear and disappear as the
        # slider moves, which is precisely the rescaling this guards against.
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker={
                    "color": [],
                    "colorscale": RELEVANCE_COLORSCALE,
                    "cmin": HIGHLIGHT_THRESHOLD_MIN,
                    "cmax": ramp_max,
                    "showscale": True,
                    "colorbar": {"title": "Relevance", "thickness": 12},
                    "size": SCORED_MARKER.size,
                },
                customdata=[],
                hoverinfo="skip",
                showlegend=False,
                name="",
            )
        )

    def _marker_for(group: Sequence[DatasetPoint], kind: str) -> dict[str, Any]:
        if kind == "unscored":
            return {"color": UNTHEMED_COLOR}
        if kind == "dimmed":
            return {"color": DIMMED_COLOR}
        if color_by == "Relevance":
            return {
                "color": [float(p.relevance_score or 0.0) for p in group],
                "colorscale": RELEVANCE_COLORSCALE,
                "cmin": HIGHLIGHT_THRESHOLD_MIN,
                "cmax": ramp_max,
                "showscale": False,
            }
        return {"color": [_theme_color(p.theme_id) for p in group]}

    # Draw order is z order: the backgrounded points first, the highlighted ones last so
    # they are never hidden under the grey.
    groups: tuple[tuple[Sequence[DatasetPoint], MarkerStyle, str], ...] = (
        (dimmed, SCORED_MARKER, "dimmed"),
        (unscored, UNSCORED_MARKER, "unscored"),
        (highlighted, SCORED_MARKER, "highlighted"),
    )
    for group, style, kind in groups:
        if not group:
            continue
        marker = _marker_for(group, kind)
        marker.update(
            {
                "size": style.size,
                "symbol": style.symbol,
                "opacity": DIMMED_OPACITY if kind == "dimmed" else style.opacity,
                "line": {"width": style.line_width},
            }
        )
        fig.add_trace(
            go.Scattergl(
                x=[point.x for point in group],
                y=[point.y for point in group],
                mode="markers",
                marker=marker,
                customdata=_customdata(group, titles, labels),
                hovertemplate=_HOVER_TEMPLATE,
                showlegend=False,
                name="",
            )
        )

    # Labels are placed from the VISIBLE points, so a hidden theme loses its label and
    # a surviving one is labelled at the medoid of what is actually on screen.
    visible = highlighted + dimmed + unscored if hide_below else list(points)
    annotations = (
        theme_label_annotations(visible, themes)
        if (color_by == "Theme" and show_theme_names)
        else []
    )

    # No legend: the theme table below is the legend (spec 9.4).
    fig.update_layout(
        showlegend=False,
        height=620,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest",
        dragmode="pan",
        annotations=annotations,
    )
    return fig


# ---------------------------------------------------------------------------
# Reconciliation with the results CSV (spec section 7)
# ---------------------------------------------------------------------------


def _accession_column(df: pd.DataFrame) -> str:
    for candidate in ("GEO_Accession", "Accession"):
        if candidate in df.columns:
            return candidate
    raise ThemeMapPanelError(
        f"{RESULTS_CSV_NAME} has no GEO_Accession or Accession column; columns are "
        f"{list(df.columns)}."
    )


def check_accessions(points: Sequence[DatasetPoint], results: pd.DataFrame) -> None:
    """Every mapped accession must exist in the results CSV, or raise (spec section 7).

    The reverse containment is deliberately *not* checked: the build drops duplicate
    sub-series on the bracket-stripped title (spec 5.1), which is 9.3-10.5% of rows, so
    the map legitimately holds fewer accessions than the CSV.
    """
    column = _accession_column(results)
    known = set(results[column].astype(str))
    missing = sorted({point.geo_accession for point in points} - known)
    if missing:
        shown = ", ".join(missing[:10])
        raise ThemeMapPanelError(
            f"{len(missing)} accession(s) in theme_map.csv are absent from "
            f"{RESULTS_CSV_NAME}: {shown}"
            f"{'…' if len(missing) > 10 else ''}. The theme map belongs to a different "
            "run, or the results CSV was regenerated. Rebuild the theme map."
        )


def _titles(results: pd.DataFrame) -> dict[str, str]:
    column = _accession_column(results)
    if "Title" not in results.columns:
        raise ThemeMapPanelError(
            f"{RESULTS_CSV_NAME} has no Title column, so the map cannot label points on "
            "hover (spec 9.4)."
        )
    return {
        str(accession): "" if pd.isna(title) else str(title)
        for accession, title in zip(results[column], results["Title"])
    }


# ---------------------------------------------------------------------------
# Build estimate (spec 9.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildEstimate:
    """What a build of this run would cost, from a token count rather than a guess."""

    n_valid: int
    n_tokens: int
    cost_usd: float


def valid_texts(results: pd.DataFrame) -> list[str]:
    """``Title + ". " + Summary`` for every ``Valid == "Yes"`` row (spec 5.1).

    ``Species`` is excluded, as the spec requires. This is only used to size the
    estimate; the build itself re-derives the text, including the scaffolding strip and
    the sub-series de-duplication this function does not do.
    """
    if "Valid" not in results.columns:
        raise ThemeMapPanelError(
            f"{RESULTS_CSV_NAME} has no Valid column, so the valid-dataset population "
            "cannot be determined."
        )
    for column in ("Title", "Summary"):
        if column not in results.columns:
            raise ThemeMapPanelError(f"{RESULTS_CSV_NAME} has no {column} column.")

    valid = results[results["Valid"].astype(str) == "Yes"]
    texts: list[str] = []
    for title, summary in zip(valid["Title"], valid["Summary"]):
        title_text = "" if pd.isna(title) else str(title).strip()
        summary_text = "" if pd.isna(summary) else str(summary).strip()
        texts.append(f"{title_text}. {summary_text}".strip())
    return texts


def estimate_build(texts: Sequence[str]) -> BuildEstimate:
    """Estimated embedding spend for one build.

    Tokens are approximated at :data:`CHARS_PER_TOKEN` characters each, plus one call for
    the run query (spec 5.7). This is an upper bound on the embedding step and excludes
    the single naming call, which is charged by the configured chat provider rather than
    by the embedding model. Measured embedding spend was $0.0038 at n=793 and $0.0105 at
    n=2,065 (spec 9.7).
    """
    characters = sum(len(text) for text in texts)
    tokens = math.ceil(characters / CHARS_PER_TOKEN) + 1
    cost = tokens / 1_000_000 * EMBEDDING_USD_PER_MILLION_TOKENS
    return BuildEstimate(n_valid=len(texts), n_tokens=tokens, cost_usd=cost)


# ---------------------------------------------------------------------------
# Background build (spec 9.7) — never called during a rerun, only submitted
# ---------------------------------------------------------------------------


def build_theme_map_task(run_dir: str, progress_callback=None) -> str:
    """``TaskManager`` entry point for a theme-map build.

    This is the only place in this module that reaches the numeric pipeline, and it runs
    in a background thread, never inside a Streamlit rerun. A failure is recorded in
    ``FAILED.txt`` *and* re-raised, so the task is marked failed and the panel shows why.
    """
    from uorca.identification.theme_map import build_theme_map

    if progress_callback:
        progress_callback(0.02, "Starting theme map build…")
    try:
        build_theme_map(run_dir)
    except Exception as exc:
        write_failure(run_dir, f"{type(exc).__name__}: {exc}")
        raise
    if progress_callback:
        progress_callback(1.0, "Theme map complete")
    return run_dir


def _submit_build(run_dir: str, run_id: str) -> None:
    task_id = f"theme_map_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    TaskManager().submit_task(
        task_id=task_id,
        task_type="theme_map",
        task_func=build_theme_map_task,
        parameters={"run_dir": run_dir},
    )
    st.session_state[_task_key(run_id)] = task_id


def _task_key(run_id: str) -> str:
    return f"theme_map_task_{run_id}"


def _render_task_status(run_id: str) -> bool:
    """Show the status of an in-flight build. True while one is pending or running."""
    task_id = st.session_state.get(_task_key(run_id))
    if not task_id:
        return False

    status_info = TaskManager().get_task_status(task_id)
    if status_info is None:
        st.caption(f"Build task `{task_id}` is not known to the task manager.")
        return False

    status: TaskStatus = status_info["status"]
    if status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        st.progress(
            float(status_info.get("progress", 0.0) or 0.0),
            text=status_info.get("progress_message") or "Building theme map…",
        )
        if st.button("Refresh", key=f"theme_map_refresh_{run_id}"):
            st.rerun()
        return True

    if status == TaskStatus.FAILED:
        st.error("Theme map build failed.")
        with st.expander("Error details"):
            st.code(status_info.get("error") or "Unknown error")
    elif status == TaskStatus.CANCELLED:
        st.warning("Theme map build was cancelled.")
    elif status == TaskStatus.COMPLETED:
        st.success("Theme map build completed.")
    st.session_state.pop(_task_key(run_id), None)
    return False


# ---------------------------------------------------------------------------
# Standalone HTML download (spec 5.8)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=2)
def _standalone_html(
    _result: ThemeMapResult,
    _titles: Mapping[str, str],
    _results: pd.DataFrame,
    run_id: str,
    built_at: str,
    threshold: float,
    color_by: str,
    show_theme_names: bool,
    hide_below: bool,
) -> str:
    """The offline HTML export, cached on everything that can change its contents.

    Cached rather than built lazily: ``st.download_button`` needs the bytes at render
    time, so "build it only when the button is rendered" would mean rebuilding five
    megabytes on every rerun -- including every nudge of the threshold slider. Caching on
    the inputs instead keeps the download matched to what is on screen while a repeat
    rerun at the same settings costs nothing. The measured build is 0.39 s at n=1,067.

    ``_result``, ``_titles`` and ``_results`` are underscore-prefixed because Streamlit
    cannot hash a dataclass holding a numpy array, nor a DataFrame cheaply. ``built_at`` is the artifact timestamp and carries
    the identity of those two arguments into the cache key, so a rebuilt map is never
    served from a stale entry.

    The import is deferred: ``theme_map_export`` imports this module for the figure and
    the colour rules, so importing it at module scope would close a cycle.
    """
    from .theme_map_export import build_standalone_html, dataset_details

    return build_standalone_html(
        _result,
        _titles,
        run_id,
        threshold=threshold,
        color_by=color_by,
        show_theme_names=show_theme_names,
        hide_below=hide_below,
        details=dataset_details(_results),
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _run_dir(run: Any) -> str:
    output_path = getattr(run, "output_path", "")
    if not output_path:
        raise ThemeMapPanelError(
            f"Run {getattr(run, 'id', '?')!r} has no output_path, so its theme map "
            "directory cannot be located."
        )
    return str(output_path)


def _load_results_for_panel(run_dir: str) -> pd.DataFrame | None:
    """The run's results CSV, or ``None`` when the run has not written one yet.

    ``None`` here is an absence, not a swallowed error: a read failure propagates.
    """
    csv_path = Path(run_dir) / RESULTS_CSV_NAME
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _render_build_prompt(run: Any, run_dir: str, results: pd.DataFrame | None) -> None:
    """State 2 of spec 9.7: no map yet, so offer to build one and price it first."""
    run_id = str(getattr(run, "id", "run"))
    st.markdown("#### Theme map")
    st.caption(
        "No theme map has been built for this run. The map groups every valid dataset "
        "by theme, so you can see which kinds of study the query returned — including "
        "categories the scored table never surfaced."
    )

    if results is None:
        st.info(
            f"No `{RESULTS_CSV_NAME}` in `{run_dir}` yet, so there is nothing to map. "
            "The run may still be in progress."
        )
        return

    texts = valid_texts(results)
    estimate = estimate_build(texts)

    left, right = st.columns(2)
    with left:
        st.metric("Valid datasets", f"{estimate.n_valid:,}")
    with right:
        st.metric("Estimated cost", f"${estimate.cost_usd:.4f}")
    st.caption(
        f"Estimate from {estimate.n_tokens:,} tokens of title and summary text at "
        f"${EMBEDDING_USD_PER_MILLION_TOKENS:.2f} per million, plus one naming call "
        "through the configured chat provider. It is an upper bound on the embedding "
        "step: the build de-duplicates sub-series first."
    )

    if estimate.n_valid < 50:
        st.warning(
            f"This run has {estimate.n_valid:,} valid datasets. The theme map needs at "
            "least 50, and will refuse to build below that."
        )

    if _render_task_status(run_id):
        return

    confirm_key = f"theme_map_confirm_{run_id}"
    st.checkbox(
        f"I understand this spends about ${estimate.cost_usd:.4f} on the OpenAI API.",
        key=confirm_key,
    )
    if st.button(
        "Build theme map",
        type="primary",
        key=f"theme_map_build_{run_id}",
        disabled=not st.session_state.get(confirm_key, False),
    ):
        _submit_build(run_dir, run_id)
        st.rerun()


def _render_failure(run: Any, run_dir: str, message: str) -> None:
    """State 1: a recorded failure. Loud, with a way to try again."""
    run_id = str(getattr(run, "id", "run"))
    st.markdown("#### Theme map")
    st.error(f"The last theme map build for this run failed:\n\n```\n{message}\n```")

    if _render_task_status(run_id):
        return

    if st.button("Rebuild theme map", type="primary", key=f"theme_map_rebuild_{run_id}"):
        _submit_build(run_dir, run_id)
        st.rerun()


def _render_map(run: Any, result: ThemeMapResult, results: pd.DataFrame) -> None:
    """State 3: the plot, then the theme table (spec 9.2, 9.4, 9.6)."""
    run_id = str(getattr(run, "id", "run"))
    points = result.points

    check_accessions(points, results)
    titles = _titles(results)

    n_table_rows = 0
    if "RelevanceScore" in results.columns:
        scores = pd.Series(pd.to_numeric(results["RelevanceScore"], errors="coerce"))
        n_table_rows = int((scores.notna() & (scores > 0)).sum())

    st.markdown("#### Theme map")
    st.caption(population_caption(n_table_rows=n_table_rows, n_points=len(points)))
    st.caption(coverage_caption(points))
    low_coverage = low_coverage_caption(points)
    if low_coverage:
        st.caption(low_coverage)
    # Filled once the slider below has been read. The caption belongs with the other two,
    # above the controls, but its text depends on where the slider sits.
    highlight_slot = st.empty()

    # One row of controls, so they read as one set rather than four decisions.
    colour_col, threshold_col, hide_col, names_col = st.columns([1.1, 2.0, 1.0, 1.1])
    with colour_col:
        color_by = st.radio(
            "Colour by",
            COLOR_BY_OPTIONS,
            index=0,
            horizontal=True,
            key=f"theme_map_color_by_{run_id}",
            help=(
                "Relevance opens first: in the benchmark run, 44% of the best datasets "
                "were unthemed and would otherwise be drawn as identical grey dots."
            ),
        )
    with threshold_col:
        threshold = st.slider(
            "Highlight at or above",
            min_value=HIGHLIGHT_THRESHOLD_MIN,
            max_value=HIGHLIGHT_THRESHOLD_MAX,
            value=opening_threshold(points),
            step=HIGHLIGHT_THRESHOLD_STEP,
            key=f"theme_map_threshold_{run_id}",
            help=(
                "Colour only. Datasets below the threshold stay on the map in grey, and "
                "unscored datasets stay hollow grey at every threshold — they were never "
                "scored, so they cannot clear one."
            ),
        )
    with hide_col:
        hide_below = st.checkbox(
            "Hide below threshold",
            value=False,
            key=f"theme_map_hide_{run_id}",
            help=(
                "Off by default: the map normally keeps every dataset on screen and "
                "only changes colour. Ticking this removes the below-threshold datasets "
                "from the plot, and the unscored ones with them — they were never "
                "scored, so they can never clear a threshold. The layout does not move."
            ),
        )
    with names_col:
        if color_by == "Theme":
            show_theme_names = st.checkbox(
                "Show theme names",
                value=True,
                key=f"theme_map_names_{run_id}",
                help=(
                    "Each label sits on the theme's medoid dataset. With 20-40 themes "
                    "they crowd; the plot pans and zooms."
                ),
            )
        else:
            show_theme_names = True

    highlight_slot.caption(highlight_caption(points, threshold))
    if hide_below:
        st.caption(hidden_caption(points, threshold))

    figure = build_theme_map_figure(
        points,
        titles,
        result.themes,
        color_by=color_by,
        threshold=threshold,
        show_theme_names=show_theme_names,
        hide_below=hide_below,
    )
    st.plotly_chart(figure, use_container_width=True, key=f"theme_map_plot_{run_id}")

    # See :func:`_standalone_html` for why the import is deferred.
    from .theme_map_export import export_filename

    download_col, note_col = st.columns([1.0, 3.0])
    with download_col:
        st.download_button(
            "Download interactive HTML",
            data=_standalone_html(
                result,
                titles,
                results,
                run_id,
                result.provenance.timestamp,
                threshold,
                color_by,
                show_theme_names,
                hide_below,
            ),
            file_name=export_filename(run_id),
            mime="text/html",
            key=f"theme_map_download_{run_id}",
            help=(
                "One self-contained file, about 5 MB. plotly.js is inlined, so it opens "
                "with no internet connection, and it carries its own colour toggle, "
                "threshold slider, theme table, per-dataset table and provenance block."
            ),
        )
    with note_col:
        st.caption(
            "The file opens at the settings above, and its own controls work offline. "
            "Coordinates and labels are persisted artifacts, not regenerable from the "
            "seed (spec 5.8), so the download is the archival copy."
        )

    table = build_theme_table(points, result.themes)
    st.caption(
        f"{len(result.themes)} themes, {result.provenance.noise_fraction * 100:.1f}% of "
        "datasets unthemed. Click a row to list that theme's datasets. This does not "
        "filter the table above."
    )
    event = st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"theme_map_table_{run_id}",
        column_config={
            "ThemeId": st.column_config.NumberColumn("Theme", format="%d"),
            "Mean relevance": st.column_config.NumberColumn("Mean relevance", format="%.2f"),
            "Max relevance": st.column_config.NumberColumn("Max relevance", format="%.2f"),
        },
    )

    selected = event.selection.rows if event and event.selection else []
    if selected:
        row = table.iloc[selected[0]]
        theme_id = int(row["ThemeId"])
        st.markdown(f"**{row['Label']}** — {int(row['n']):,} datasets")
        if theme_id >= 0:
            theme = next(t for t in result.themes if t.id == theme_id)
            st.caption(theme.description)
            if theme.keywords:
                st.caption("Keywords: " + ", ".join(theme.keywords))
        else:
            st.caption(
                "These datasets did not join any theme. They are not lower quality — in "
                "the benchmark run the single highest-scoring dataset was unthemed."
            )
        st.dataframe(
            theme_dataset_table(points, titles, theme_id),
            use_container_width=True,
            hide_index=True,
            column_config={
                "GEO": st.column_config.LinkColumn("GEO", display_text="View"),
                "Relevance": st.column_config.NumberColumn("Relevance", format="%.2f"),
                "Theme probability": st.column_config.NumberColumn(
                    "Theme probability", format="%.2f"
                ),
                "QuerySim": st.column_config.NumberColumn("QuerySim", format="%.3f"),
            },
        )

    with st.expander("Provenance"):
        provenance = result.provenance
        st.caption(
            "Coordinates and labels are persisted artifacts. They are not regenerable "
            "from the seed alone: t-SNE output depends on the thread count at large n."
        )
        st.json(
            {
                "embedding_model": provenance.embedding_model,
                "seed": provenance.seed,
                "OMP_NUM_THREADS": provenance.omp_num_threads,
                "scikit-learn": provenance.sklearn_version,
                "numpy": provenance.numpy_version,
                "hdbscan": provenance.hdbscan_version,
                "n_datasets": provenance.n_datasets,
                "n_themes": provenance.n_themes,
                "noise_fraction": provenance.noise_fraction,
                "estimated_cost_usd": provenance.estimated_cost_usd,
                "timestamp": provenance.timestamp,
            }
        )


def render_theme_map_panel(run: Any) -> None:
    """Render the theme map for one identification run (spec section 9).

    ``run`` is the :class:`~uorca.gui.project.models.RunEntry` selected on the Identify
    page. The selector lives in ``page()`` and is passed in, so this panel always knows
    which run it is drawing and never disappears when the results table returns early
    (spec 9.1).

    Reads artifacts only. It never calls the OpenAI client and never runs the numeric
    pipeline (spec section 8).
    """
    run_dir = _run_dir(run)
    results = _load_results_for_panel(run_dir)

    failure = read_failure(run_dir)
    if failure and not artifacts_exist(run_dir):
        _render_failure(run, run_dir, failure)
        return

    if not artifacts_exist(run_dir):
        _render_build_prompt(run, run_dir, results)
        return

    try:
        result = read_artifacts(run_dir)
    except ThemeMapBuildFailed as exc:
        _render_failure(run, run_dir, exc.failure_message)
        return
    except ThemeMapArtifactError as exc:
        st.error(f"The theme map artifacts for this run cannot be read: {exc}")
        if st.button(
            "Rebuild theme map", type="primary", key=f"theme_map_repair_{getattr(run, 'id', 'run')}"
        ):
            _submit_build(run_dir, str(getattr(run, "id", "run")))
            st.rerun()
        return

    if results is None:
        raise ThemeMapPanelError(
            f"A theme map exists in {run_dir} but `{RESULTS_CSV_NAME}` does not. The two "
            "must be reconciled before the map can be drawn (spec section 7)."
        )

    _render_map(run, result, results)
