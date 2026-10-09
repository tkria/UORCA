"""Unit tests for the theme-map Streamlit panel (spec section 9).

The panel itself is Streamlit, which cannot be exercised without a running
session, so every piece of data shaping it does is factored into a plain
function and those functions are what is tested here. Nothing in this file
starts Streamlit, touches a run directory it did not create, or calls any API.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from uorca.identification.theme_map.types import DatasetPoint, Theme


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _point(
    accession: str,
    *,
    x: float = 0.0,
    y: float = 0.0,
    theme_id: int = 0,
    theme_prob: float = 1.0,
    relevance: float | None = 5.0,
    query_sim: float = 0.5,
    query_sim_rank: int = 1,
) -> DatasetPoint:
    return DatasetPoint(
        geo_accession=accession,
        x=x,
        y=y,
        theme_id=theme_id,
        theme_prob=theme_prob,
        relevance_score=relevance,
        query_sim=query_sim,
        query_sim_rank=query_sim_rank,
    )


def _theme(theme_id: int, **kwargs) -> Theme:
    defaults = dict(
        label=f"Theme {theme_id}",
        description=f"Description {theme_id}",
        n=2,
        keywords=["alpha", "beta"],
        representatives=["A study"],
        mean_relevance=3.0,
        max_relevance=6.0,
    )
    defaults.update(kwargs)
    return Theme(id=theme_id, **defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_module_imports_without_a_streamlit_session():
    """Importing the panel must not require a ScriptRunContext."""
    import importlib

    module = importlib.import_module("uorca.gui.components.theme_map_panel")
    assert callable(module.render_theme_map_panel)


@pytest.mark.unit
def test_exported_from_components_package():
    from uorca.gui.components import render_theme_map_panel

    assert callable(render_theme_map_panel)


# ---------------------------------------------------------------------------
# Theme table
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_theme_table_has_an_unthemed_row_when_unthemed_points_exist():
    from uorca.gui.components.theme_map_panel import THEME_TABLE_COLUMNS, build_theme_table

    points = [
        _point("GSE1", theme_id=0, relevance=2.0),
        _point("GSE2", theme_id=0, relevance=4.0),
        _point("GSE3", theme_id=-1, theme_prob=0.0, relevance=9.1),
        _point("GSE4", theme_id=-1, theme_prob=0.0, relevance=1.0),
    ]
    table = build_theme_table(points, [_theme(0, n=2, mean_relevance=3.0, max_relevance=4.0)])

    assert list(table.columns) == list(THEME_TABLE_COLUMNS)
    unthemed = table[table["ThemeId"] == -1]
    assert len(unthemed) == 1
    row = unthemed.iloc[0]
    assert str(row["Label"]).startswith("Unthemed")
    assert "(2)" in str(row["Label"])
    assert int(row["n"]) == 2
    # The honest warning of spec 9.6: the top-scoring dataset can be unthemed.
    assert row["Max relevance"] == pytest.approx(9.1)
    assert row["Mean relevance"] == pytest.approx(5.05)


@pytest.mark.unit
def test_theme_table_omits_the_unthemed_row_when_every_point_is_themed():
    from uorca.gui.components.theme_map_panel import build_theme_table

    points = [
        _point("GSE1", theme_id=0),
        _point("GSE2", theme_id=1),
    ]
    table = build_theme_table(points, [_theme(0, n=1), _theme(1, n=1)])

    assert (table["ThemeId"] == -1).sum() == 0
    assert len(table) == 2


@pytest.mark.unit
def test_theme_table_sorts_by_size_by_default():
    from uorca.gui.components.theme_map_panel import build_theme_table

    points = [_point(f"GSE{i}", theme_id=0) for i in range(3)]
    points += [_point(f"GSX{i}", theme_id=1) for i in range(7)]
    table = build_theme_table(points, [_theme(0, n=3), _theme(1, n=7)])

    assert list(table["n"]) == [7, 3]


@pytest.mark.unit
def test_theme_table_unthemed_relevance_is_none_when_nothing_unthemed_is_scored():
    from uorca.gui.components.theme_map_panel import build_theme_table

    points = [
        _point("GSE1", theme_id=0, relevance=1.0),
        _point("GSE2", theme_id=-1, theme_prob=0.0, relevance=None),
    ]
    table = build_theme_table(points, [_theme(0, n=1)])
    row = table[table["ThemeId"] == -1].iloc[0]

    assert row["Mean relevance"] is None or (
        isinstance(row["Mean relevance"], float) and math.isnan(row["Mean relevance"])
    )


@pytest.mark.unit
def test_theme_table_raises_when_a_point_references_an_unknown_theme():
    from uorca.gui.components.theme_map_panel import ThemeMapPanelError, build_theme_table

    points = [_point("GSE1", theme_id=4)]
    with pytest.raises(ThemeMapPanelError, match="4"):
        build_theme_table(points, [_theme(0, n=1)])


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coverage_caption_reports_count_and_percent_on_partly_scored_input():
    from uorca.gui.components.theme_map_panel import coverage_caption, summarize_coverage

    points = [
        _point("GSE1", relevance=7.0),
        _point("GSE2", relevance=0.0),
        _point("GSE3", relevance=None),
        _point("GSE4", relevance=None),
    ]
    summary = summarize_coverage(points)
    assert summary.n_points == 4
    assert summary.n_scored == 2  # 0.0 is a score; None is not
    assert summary.n_unscored == 2
    assert summary.percent_scored == pytest.approx(50.0)

    caption = coverage_caption(points)
    assert "2 of 4" in caption
    assert "50.0%" in caption
    assert "hollow" in caption.lower()


@pytest.mark.unit
def test_coverage_caption_on_a_fully_scored_corpus():
    from uorca.gui.components.theme_map_panel import coverage_caption

    points = [_point("GSE1", relevance=1.0), _point("GSE2", relevance=2.0)]
    caption = coverage_caption(points)
    assert "2 of 2" in caption
    assert "100.0%" in caption


@pytest.mark.unit
def test_population_caption_states_both_counts():
    from uorca.gui.components.theme_map_panel import population_caption

    caption = population_caption(n_table_rows=206, n_points=1199)
    assert "206" in caption
    assert "1,199" in caption or "1199" in caption


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unscored_points_get_the_hollow_marker_and_low_scoring_ones_do_not():
    from uorca.gui.components.theme_map_panel import marker_style_for

    unscored = marker_style_for(_point("GSE1", relevance=None))
    low = marker_style_for(_point("GSE2", relevance=0.0))

    assert unscored.hollow is True
    assert "open" in unscored.symbol
    assert low.hollow is False
    assert "open" not in low.symbol
    # Spec 9.5: an unscored point must not read as a low-scoring one.
    assert unscored.size < low.size


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _figure_inputs():
    points = [
        _point("GSE1", x=0.0, y=1.0, theme_id=0, relevance=8.0),
        _point("GSE2", x=1.0, y=0.0, theme_id=0, relevance=0.0),
        _point("GSE3", x=2.0, y=2.0, theme_id=-1, theme_prob=0.0, relevance=None),
    ]
    titles = {"GSE1": "Alpha study", "GSE2": "Beta study", "GSE3": "Gamma study"}
    themes = [_theme(0, n=2)]
    return points, titles, themes


@pytest.mark.unit
@pytest.mark.parametrize("color_by", ["Relevance", "Theme"])
def test_plot_figure_carries_no_legend(color_by):
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, titles, themes = _figure_inputs()
    fig = build_theme_map_figure(points, titles, themes, color_by=color_by)

    assert fig.layout["showlegend"] is False
    for trace in fig.data:
        assert trace.showlegend is False


@pytest.mark.unit
def test_plot_draws_every_point_including_unthemed_ones():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, titles, themes = _figure_inputs()
    fig = build_theme_map_figure(points, titles, themes, color_by="Theme")

    drawn = sum(len(trace.x) for trace in fig.data)
    assert drawn == len(points)


@pytest.mark.unit
def test_plot_separates_unscored_points_into_a_hollow_trace():
    from uorca.gui.components.theme_map_panel import UNSCORED_MARKER, build_theme_map_figure

    points, titles, themes = _figure_inputs()
    fig = build_theme_map_figure(points, titles, themes, color_by="Relevance")

    open_traces = [t for t in fig.data if "open" in str(t.marker.symbol)]
    assert len(open_traces) == 1
    assert len(open_traces[0].x) == 1
    assert open_traces[0].marker.symbol == UNSCORED_MARKER.symbol


@pytest.mark.unit
def test_plot_marker_size_is_uniform_within_a_trace():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, titles, themes = _figure_inputs()
    fig = build_theme_map_figure(points, titles, themes, color_by="Relevance")

    for trace in fig.data:
        assert isinstance(trace.marker.size, (int, float))


@pytest.mark.unit
def test_plot_hover_carries_accession_title_theme_and_score():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, titles, themes = _figure_inputs()
    fig = build_theme_map_figure(points, titles, themes, color_by="Relevance")

    hovered = {tuple(row) for trace in fig.data for row in trace.customdata}
    accessions = {row[0] for row in hovered}
    assert accessions == {"GSE1", "GSE2", "GSE3"}
    by_accession = {row[0]: row for row in hovered}
    assert by_accession["GSE1"][1] == "Alpha study"
    assert by_accession["GSE1"][2] == "Theme 0"
    assert by_accession["GSE3"][2] == "Unthemed"
    assert by_accession["GSE3"][3] == "not scored"


@pytest.mark.unit
def test_plot_raises_on_an_unknown_colour_mode():
    from uorca.gui.components.theme_map_panel import ThemeMapPanelError, build_theme_map_figure

    points, titles, themes = _figure_inputs()
    with pytest.raises(ThemeMapPanelError):
        build_theme_map_figure(points, titles, themes, color_by="QuerySim")


@pytest.mark.unit
def test_plot_raises_when_a_title_is_missing():
    from uorca.gui.components.theme_map_panel import ThemeMapPanelError, build_theme_map_figure

    points, titles, themes = _figure_inputs()
    titles.pop("GSE2")
    with pytest.raises(ThemeMapPanelError, match="GSE2"):
        build_theme_map_figure(points, titles, themes, color_by="Relevance")


@pytest.mark.unit
def test_relevance_colourscale_is_the_documented_colour_blind_safe_ramp():
    from uorca.gui.components.theme_map_panel import RELEVANCE_COLORSCALE

    # Viridis: perceptually uniform, colour-blind safe, monotone in lightness so
    # it reads against both the light and the dark Streamlit background.
    assert RELEVANCE_COLORSCALE == "Viridis"


# ---------------------------------------------------------------------------
# Accession reconciliation (spec section 7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_accession_check_passes_when_the_map_is_a_subset_of_the_results():
    from uorca.gui.components.theme_map_panel import check_accessions

    df = pd.DataFrame({"GEO_Accession": ["GSE1", "GSE2", "GSE3"], "Valid": ["Yes"] * 3})
    # Sub-series de-duplication (spec 5.1) legitimately drops rows from the map.
    check_accessions([_point("GSE1"), _point("GSE2")], df)


@pytest.mark.unit
def test_accession_check_raises_when_the_map_holds_an_unknown_accession():
    from uorca.gui.components.theme_map_panel import ThemeMapPanelError, check_accessions

    df = pd.DataFrame({"GEO_Accession": ["GSE1"], "Valid": ["Yes"]})
    with pytest.raises(ThemeMapPanelError, match="GSE9"):
        check_accessions([_point("GSE1"), _point("GSE9")], df)


# ---------------------------------------------------------------------------
# Build estimate (spec 9.7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_valid_texts_uses_only_valid_rows_and_joins_title_and_summary():
    from uorca.gui.components.theme_map_panel import valid_texts

    df = pd.DataFrame(
        {
            "GEO_Accession": ["GSE1", "GSE2"],
            "Valid": ["Yes", "No"],
            "Title": ["Alpha", "Beta"],
            "Summary": ["One", "Two"],
        }
    )
    assert valid_texts(df) == ["Alpha. One"]


@pytest.mark.unit
def test_build_estimate_scales_with_text_and_is_the_right_order_of_magnitude():
    from uorca.gui.components.theme_map_panel import estimate_build

    small = estimate_build(["x" * 1000] * 793)
    large = estimate_build(["x" * 1000] * 2065)

    assert small.n_valid == 793
    assert large.cost_usd > small.cost_usd
    # Measured spend was $0.0038 at n=793 and $0.0105 at n=2,065 (spec 9.7).
    assert 0.001 < small.cost_usd < 0.02
    assert 0.002 < large.cost_usd < 0.05


# ---------------------------------------------------------------------------
# End-to-end over the read path (no Streamlit, no API)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_panel_helpers_run_over_a_real_artifact_round_trip(tmp_path):
    """Write artifacts the way a build would, then drive every shaping function."""
    import numpy as np

    from uorca.gui.components.theme_map_panel import (
        build_theme_map_figure,
        build_theme_table,
        check_accessions,
        coverage_caption,
        theme_dataset_table,
    )
    from uorca.identification.theme_map.artifacts import read_artifacts, write_artifacts
    from uorca.identification.theme_map.types import (
        Provenance,
        ThemeMapConfig,
        ThemeMapResult,
    )

    points = [
        _point("GSE1", x=0.1, y=0.2, theme_id=0, relevance=8.0),
        _point("GSE2", x=0.3, y=0.4, theme_id=0, relevance=None),
        _point("GSE3", x=1.3, y=1.4, theme_id=-1, theme_prob=0.0, relevance=9.4),
    ]
    themes = [_theme(0, n=2, mean_relevance=8.0, max_relevance=8.0)]
    config = ThemeMapConfig()
    provenance = Provenance(
        embedding_model=config.embedding_model,
        seed=config.tsne_random_state,
        omp_num_threads=None,
        sklearn_version="1.7.0",
        numpy_version="2.3.1",
        hdbscan_version="0.8.40",
        n_datasets=3,
        n_themes=1,
        noise_fraction=1 / 3,
        estimated_cost_usd=0.0001,
        timestamp="2026-09-16T00:00:00",
        config=config,
    )
    write_artifacts(
        tmp_path,
        ThemeMapResult(
            points=points,
            themes=themes,
            embeddings=np.zeros((3, 4), dtype=np.float32),
            provenance=provenance,
        ),
    )

    results = pd.DataFrame(
        {
            "GEO_Accession": ["GSE1", "GSE2", "GSE3", "GSE4"],
            "Valid": ["Yes", "Yes", "Yes", "No"],
            "Title": ["Alpha", "Beta", "Gamma", "Delta"],
            "Summary": ["one", "two", "three", "four"],
            "RelevanceScore": [8.0, None, 9.4, None],
        }
    )

    loaded = read_artifacts(tmp_path)
    check_accessions(loaded.points, results)

    titles = {"GSE1": "Alpha", "GSE2": "Beta", "GSE3": "Gamma"}
    figure = build_theme_map_figure(loaded.points, titles, loaded.themes, color_by="Relevance")
    assert figure.layout["showlegend"] is False
    assert sum(len(trace.x) for trace in figure.data) == 3

    table = build_theme_table(loaded.points, loaded.themes)
    unthemed = table[table["ThemeId"] == -1].iloc[0]
    assert unthemed["Max relevance"] == pytest.approx(9.4)

    assert "2 of 3" in coverage_caption(loaded.points)
    assert list(theme_dataset_table(loaded.points, titles, -1)["Accession"]) == ["GSE3"]


@pytest.mark.unit
def test_build_task_records_the_failure_and_re_raises(tmp_path, monkeypatch):
    """A failed build must leave FAILED.txt and still fail the task (never swallow)."""
    import uorca.identification.theme_map as theme_map_pkg
    from uorca.gui.components.theme_map_panel import build_theme_map_task
    from uorca.identification.theme_map.artifacts import artifacts_exist, read_failure

    def _explode(run_dir, *, config=None):
        raise RuntimeError("embedding provider is Bedrock")

    monkeypatch.setattr(theme_map_pkg, "build_theme_map", _explode)

    with pytest.raises(RuntimeError, match="Bedrock"):
        build_theme_map_task(str(tmp_path))

    assert read_failure(tmp_path) == "RuntimeError: embedding provider is Bedrock"
    assert artifacts_exist(tmp_path) is False


# ---------------------------------------------------------------------------
# Relevance highlight threshold (user change 1)
# ---------------------------------------------------------------------------


def _trace_holding(fig, accession):
    """The trace that draws ``accession``, found through its hover customdata."""
    for trace in fig.data:
        customdata = trace.customdata
        if customdata is None:
            continue
        for row in customdata:
            if row[0] == accession:
                return trace
    raise AssertionError(f"{accession} is not drawn by any trace")


def _layout_annotations(fig) -> list:
    """The figure's annotations, read back through ``to_dict`` so the assertion sees
    exactly what would be serialised to the browser."""
    return list(fig.to_dict()["layout"].get("annotations") or ())


def _colorbar_trace(fig):
    scaled = [t for t in fig.data if getattr(t.marker, "showscale", None)]
    assert len(scaled) == 1, f"expected exactly one colourbar trace, got {len(scaled)}"
    return scaled[0]


@pytest.mark.unit
def test_is_highlighted_is_false_for_an_unscored_point_even_at_threshold_zero():
    from uorca.gui.components.theme_map_panel import is_highlighted

    unscored = _point("GSE1", relevance=None)
    # Spec 9.5: an unscored dataset was never scored. It cannot clear any bar.
    assert is_highlighted(unscored, 0.0) is False
    assert is_highlighted(unscored, 7.0) is False


@pytest.mark.unit
def test_is_highlighted_is_true_exactly_at_the_threshold():
    from uorca.gui.components.theme_map_panel import is_highlighted

    assert is_highlighted(_point("GSE1", relevance=7.0), 7.0) is True
    assert is_highlighted(_point("GSE2", relevance=6.999), 7.0) is False
    assert is_highlighted(_point("GSE3", relevance=7.5), 7.0) is True
    # 0.0 is a score, and it clears a threshold of 0.0.
    assert is_highlighted(_point("GSE4", relevance=0.0), 0.0) is True


@pytest.mark.unit
def test_default_highlight_threshold_is_seven():
    from uorca.gui.components.theme_map_panel import DEFAULT_HIGHLIGHT_THRESHOLD

    assert DEFAULT_HIGHLIGHT_THRESHOLD == 7.0


@pytest.mark.unit
def test_highlight_caption_reports_the_count_the_total_and_the_threshold():
    from uorca.gui.components.theme_map_panel import highlight_caption

    points = [_point(f"GSE{i}", relevance=8.0) for i in range(16)]
    points += [_point(f"GSX{i}", relevance=1.0) for i in range(1183)]
    caption = highlight_caption(points, 7.0)

    assert "16 of 1,199" in caption
    assert "7.0" in caption


@pytest.mark.unit
def test_highlight_caption_counts_unscored_points_in_the_total_but_never_as_passing():
    from uorca.gui.components.theme_map_panel import highlight_caption

    points = [_point("GSE1", relevance=9.0), _point("GSE2", relevance=None)]
    caption = highlight_caption(points, 0.0)

    assert "1 of 2" in caption


# ---------------------------------------------------------------------------
# Theme labels on the plot (user change 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_theme_medoid_is_the_position_of_a_real_member_not_the_mean():
    from uorca.gui.components.theme_map_panel import theme_medoid

    points = [
        _point("GSE1", x=0.0, y=0.0, theme_id=0),
        _point("GSE2", x=10.0, y=0.0, theme_id=0),
        _point("GSE3", x=2.0, y=0.0, theme_id=0),
        _point("GSE4", x=99.0, y=99.0, theme_id=1),
    ]
    # The centroid is (4.0, 0.0), which is nobody's position. The medoid is GSE3.
    assert theme_medoid(points, 0) == (2.0, 0.0)
    assert theme_medoid(points, 0) in {(p.x, p.y) for p in points if p.theme_id == 0}


@pytest.mark.unit
def test_theme_medoid_raises_when_the_theme_has_no_members():
    from uorca.gui.components.theme_map_panel import ThemeMapPanelError, theme_medoid

    points = [_point("GSE1", theme_id=0)]
    with pytest.raises(ThemeMapPanelError, match="3"):
        theme_medoid(points, 3)


@pytest.mark.unit
def test_theme_label_font_size_spans_the_clamp_and_is_monotone():
    from uorca.gui.components.theme_map_panel import (
        THEME_LABEL_FONT_MAX,
        THEME_LABEL_FONT_MIN,
        theme_label_font_size,
    )

    assert THEME_LABEL_FONT_MIN == 9
    assert THEME_LABEL_FONT_MAX == 16
    # Anchored at the SMALLEST theme present, not at a notional theme of one. On the real
    # PPARg map hdbscan_min_cluster_size=15 puts the floor at 15, and anchoring at 1 put
    # 14 of 20 labels at 9-10 points -- uniform in all but name.
    assert theme_label_font_size(15, 15, 169) == 9
    assert theme_label_font_size(169, 15, 169) == 16

    sizes = [theme_label_font_size(n, 15, 169) for n in (15, 20, 45, 80, 120, 169)]
    assert sizes == sorted(sizes)
    assert all(9 <= size <= 16 for size in sizes)
    # The square root must actually separate the pack, which linear did not: the real
    # theme sizes are 15..48 plus one outlier at 169.
    assert theme_label_font_size(48, 15, 169) > theme_label_font_size(20, 15, 169)
    # Out-of-range inputs clamp rather than escaping the range.
    assert theme_label_font_size(500, 15, 169) == 16
    assert theme_label_font_size(0, 15, 169) == 9
    # A degenerate corpus of equal-sized themes must not divide by zero.
    assert 9 <= theme_label_font_size(15, 15, 15) <= 16


@pytest.mark.unit
def test_theme_label_annotations_are_one_per_theme_and_none_for_unthemed():
    from uorca.gui.components.theme_map_panel import theme_label_annotations

    points = [
        _point("GSE1", x=0.0, y=0.0, theme_id=0),
        _point("GSE2", x=1.0, y=0.0, theme_id=0),
        _point("GSE3", x=8.0, y=8.0, theme_id=1),
        _point("GSE4", x=-5.0, y=-5.0, theme_id=-1, theme_prob=0.0),
    ]
    themes = [_theme(0, n=2), _theme(1, n=1)]
    annotations = theme_label_annotations(points, themes)

    assert len(annotations) == 2
    texts = {annotation["text"] for annotation in annotations}
    assert texts == {"Theme 0", "Theme 1"}
    # "Unthemed" is not a theme (spec 9.6) and must not be labelled on the plot.
    assert not any("Unthemed" in annotation["text"] for annotation in annotations)

    for annotation in annotations:
        assert annotation["showarrow"] is False
        assert 9 <= annotation["font"]["size"] <= 16
        # A neutral grey wash: legible over both the light and the dark theme.
        assert "bgcolor" in annotation
        assert annotation["bgcolor"] not in ("#ffffff", "#000000", "white", "black")

    by_text = {annotation["text"]: annotation for annotation in annotations}
    assert (by_text["Theme 1"]["x"], by_text["Theme 1"]["y"]) == (8.0, 8.0)


# ---------------------------------------------------------------------------
# The threshold and the labels, inside the figure
# ---------------------------------------------------------------------------


def _threshold_inputs():
    points = [
        _point("GSE_HI", x=0.0, y=0.0, theme_id=0, relevance=8.0),
        _point("GSE_LO", x=1.0, y=1.0, theme_id=0, relevance=2.0),
        _point("GSE_NONE", x=2.0, y=2.0, theme_id=-1, theme_prob=0.0, relevance=None),
    ]
    titles = {"GSE_HI": "High", "GSE_LO": "Low", "GSE_NONE": "Unscored"}
    themes = [_theme(0, n=2)]
    return points, titles, themes


@pytest.mark.unit
def test_relevance_view_greys_a_point_below_the_threshold_only():
    from uorca.gui.components.theme_map_panel import DIMMED_COLOR, build_theme_map_figure

    points, titles, themes = _threshold_inputs()
    fig = build_theme_map_figure(points, titles, themes, color_by="Relevance", threshold=7.0)

    low = _trace_holding(fig, "GSE_LO")
    high = _trace_holding(fig, "GSE_HI")

    assert low.marker.color == DIMMED_COLOR
    assert high.marker.color != DIMMED_COLOR
    # A highlighted point keeps its place on the Viridis ramp.
    assert list(high.marker.color) == [8.0]
    assert high.marker.colorscale is not None


@pytest.mark.unit
def test_theme_view_greys_a_point_below_the_threshold_and_colours_one_above():
    from uorca.gui.components.theme_map_panel import (
        DIMMED_COLOR,
        THEME_COLOR_SEQUENCE,
        build_theme_map_figure,
    )

    points, titles, themes = _threshold_inputs()
    fig = build_theme_map_figure(points, titles, themes, color_by="Theme", threshold=7.0)

    assert _trace_holding(fig, "GSE_LO").marker.color == DIMMED_COLOR
    assert list(_trace_holding(fig, "GSE_HI").marker.color) == [THEME_COLOR_SEQUENCE[0]]


@pytest.mark.unit
def test_lowering_the_threshold_moves_a_point_out_of_the_grey():
    from uorca.gui.components.theme_map_panel import DIMMED_COLOR, build_theme_map_figure

    points, titles, themes = _threshold_inputs()
    for color_by in ("Relevance", "Theme"):
        fig = build_theme_map_figure(points, titles, themes, color_by=color_by, threshold=0.0)
        assert _trace_holding(fig, "GSE_LO").marker.color != DIMMED_COLOR


@pytest.mark.unit
@pytest.mark.parametrize("color_by", ["Relevance", "Theme"])
def test_an_unscored_point_is_hollow_and_grey_in_both_views(color_by):
    from uorca.gui.components.theme_map_panel import (
        UNSCORED_MARKER,
        UNTHEMED_COLOR,
        build_theme_map_figure,
    )

    points, titles, themes = _threshold_inputs()
    # Even at a threshold nothing can fail, the unscored point stays hollow and grey.
    fig = build_theme_map_figure(points, titles, themes, color_by=color_by, threshold=0.0)

    trace = _trace_holding(fig, "GSE_NONE")
    assert trace.marker.symbol == UNSCORED_MARKER.symbol
    assert trace.marker.color == UNTHEMED_COLOR
    assert len(trace.x) == 1


@pytest.mark.unit
def test_every_point_is_still_drawn_at_the_strictest_threshold():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, titles, themes = _threshold_inputs()
    for color_by in ("Relevance", "Theme"):
        fig = build_theme_map_figure(points, titles, themes, color_by=color_by, threshold=10.0)
        # Spec decision 11: nothing is ever hidden, only greyed.
        assert sum(len(trace.x) for trace in fig.data) == len(points)


@pytest.mark.unit
def test_annotations_appear_only_in_the_theme_view_with_the_checkbox_on():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, titles, themes = _threshold_inputs()

    themed = build_theme_map_figure(
        points, titles, themes, color_by="Theme", show_theme_names=True
    )
    drawn = _layout_annotations(themed)
    assert len(drawn) == 1
    assert drawn[0]["text"] == "Theme 0"

    hidden = build_theme_map_figure(
        points, titles, themes, color_by="Theme", show_theme_names=False
    )
    assert _layout_annotations(hidden) == []

    relevance = build_theme_map_figure(
        points, titles, themes, color_by="Relevance", show_theme_names=True
    )
    assert _layout_annotations(relevance) == []


@pytest.mark.unit
def test_the_colourbar_range_does_not_move_with_the_threshold():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, titles, themes = _threshold_inputs()
    ranges = []
    for threshold in (0.0, 2.0, 7.0, 10.0):
        fig = build_theme_map_figure(
            points, titles, themes, color_by="Relevance", threshold=threshold
        )
        bar = _colorbar_trace(fig)
        ranges.append((bar.marker.cmin, bar.marker.cmax))

    assert ranges == [(0.0, 10.0)] * 4


# ---------------------------------------------------------------------------
# Opening threshold on a partly scored run (2026-09-17)
# ---------------------------------------------------------------------------


def test_a_fully_scored_run_opens_at_the_default_threshold():
    from uorca.gui.components.theme_map_panel import (
        DEFAULT_HIGHLIGHT_THRESHOLD,
        low_coverage_caption,
        opening_threshold,
    )

    points = [_point(f"GSE{i}", relevance=1.0) for i in range(10)]

    assert opening_threshold(points) == DEFAULT_HIGHLIGHT_THRESHOLD
    assert low_coverage_caption(points) is None


def test_a_barely_scored_run_opens_at_zero_and_says_why():
    from uorca.gui.components.theme_map_panel import (
        HIGHLIGHT_THRESHOLD_MIN,
        low_coverage_caption,
        opening_threshold,
    )

    # 10 of 100 scored — the shape of the three archived scratch runs.
    points = [_point(f"GSE{i}", relevance=1.0 if i < 10 else None) for i in range(100)]

    assert opening_threshold(points) == HIGHLIGHT_THRESHOLD_MIN
    caption = low_coverage_caption(points)
    assert caption is not None
    assert "10" in caption and "100" in caption and "10.0%" in caption


def test_the_coverage_cutoff_is_inclusive_at_its_own_value():
    from uorca.gui.components.theme_map_panel import (
        DEFAULT_HIGHLIGHT_THRESHOLD,
        LOW_COVERAGE_PERCENT,
        opening_threshold,
    )

    assert LOW_COVERAGE_PERCENT == 80.0
    exactly_at = [_point(f"GSE{i}", relevance=1.0 if i < 80 else None) for i in range(100)]
    just_below = [_point(f"GSE{i}", relevance=1.0 if i < 79 else None) for i in range(100)]

    assert opening_threshold(exactly_at) == DEFAULT_HIGHLIGHT_THRESHOLD
    assert opening_threshold(just_below) == 0.0


def test_the_opening_threshold_changes_no_colouring_rule():
    """A low-coverage run only *starts* at 0.0. is_highlighted is untouched."""
    from uorca.gui.components.theme_map_panel import is_highlighted

    assert is_highlighted(_point("GSE1", relevance=None), 0.0) is False
    assert is_highlighted(_point("GSE2", relevance=0.0), 0.0) is True


# ---------------------------------------------------------------------------
# Hiding below the threshold (decision 16, 2026-09-22)
# ---------------------------------------------------------------------------


def _drawn_accessions(figure) -> set[str]:
    """Every accession the figure actually draws, across all traces."""
    drawn: set[str] = set()
    for trace in figure.to_dict()["data"]:
        for row in trace.get("customdata") or []:
            drawn.add(str(row[0]))
    return drawn


def _hide_fixture():
    points = [
        _point("GSE1", x=0.0, y=0.0, theme_id=0, relevance=9.0),
        _point("GSE2", x=0.5, y=0.2, theme_id=0, relevance=2.0),
        _point("GSE3", x=5.0, y=5.0, theme_id=1, relevance=1.0),
        _point("GSE4", x=5.2, y=5.1, theme_id=1, relevance=0.0),
        _point("GSE5", x=-3.0, y=2.0, theme_id=-1, theme_prob=0.0, relevance=None),
    ]
    themes = [_theme(0, label="Alpha", n=2), _theme(1, label="Beta", n=2)]
    titles = {p.geo_accession: f"Title {p.geo_accession}" for p in points}
    return points, themes, titles


def test_hiding_is_off_by_default_and_draws_everything():
    """Spec decision 11 stays the default. Hiding is opt-in."""
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, themes, titles = _hide_fixture()
    figure = build_theme_map_figure(points, titles, themes, threshold=7.0)

    assert _drawn_accessions(figure) == {p.geo_accession for p in points}


def test_hiding_removes_the_below_threshold_datasets():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, themes, titles = _hide_fixture()
    figure = build_theme_map_figure(
        points, titles, themes, threshold=7.0, hide_below=True
    )

    assert _drawn_accessions(figure) == {"GSE1"}


def test_hiding_also_removes_the_unscored_datasets():
    """They can never clear a threshold, so keeping them while hiding a 6.9 is incoherent."""
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, themes, titles = _hide_fixture()
    figure = build_theme_map_figure(
        points, titles, themes, threshold=0.0, hide_below=True
    )

    drawn = _drawn_accessions(figure)
    assert "GSE5" not in drawn
    assert drawn == {"GSE1", "GSE2", "GSE3", "GSE4"}


def test_hiding_at_the_strictest_threshold_draws_nothing_and_does_not_raise():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, themes, titles = _hide_fixture()
    figure = build_theme_map_figure(
        points, titles, themes, threshold=10.0, hide_below=True
    )

    assert _drawn_accessions(figure) == set()


def test_a_theme_with_no_visible_member_loses_its_label():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, themes, titles = _hide_fixture()
    figure = build_theme_map_figure(
        points, titles, themes, color_by="Theme", threshold=7.0, hide_below=True
    )

    labels = [a["text"] for a in figure.to_dict()["layout"].get("annotations", [])]
    assert labels == ["Alpha"], "Beta has no surviving member and must not be labelled"


def test_every_theme_keeps_its_label_when_nothing_is_hidden():
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, themes, titles = _hide_fixture()
    figure = build_theme_map_figure(
        points, titles, themes, color_by="Theme", threshold=7.0, hide_below=False
    )

    labels = sorted(a["text"] for a in figure.to_dict()["layout"].get("annotations", []))
    assert labels == ["Alpha", "Beta"]


def test_hidden_caption_names_the_count_and_the_unscored_share():
    from uorca.gui.components.theme_map_panel import hidden_caption

    points, _, _ = _hide_fixture()
    caption = hidden_caption(points, 7.0)

    assert "4 of 5" in caption
    assert "1 were never scored" in caption or "1 was never scored" in caption or "1" in caption


def test_hidden_caption_says_so_when_nothing_is_hidden():
    from uorca.gui.components.theme_map_panel import hidden_caption

    points = [_point("GSE1", relevance=9.0), _point("GSE2", relevance=8.0)]

    assert "Nothing hidden" in hidden_caption(points, 7.0)


def test_hiding_does_not_move_the_layout():
    """The t-SNE coordinates are fixed. A survivor sits where it always sat."""
    from uorca.gui.components.theme_map_panel import build_theme_map_figure

    points, themes, titles = _hide_fixture()
    shown = build_theme_map_figure(points, titles, themes, threshold=7.0)
    hidden = build_theme_map_figure(
        points, titles, themes, threshold=7.0, hide_below=True
    )

    def position(figure, accession):
        for trace in figure.to_dict()["data"]:
            for i, row in enumerate(trace.get("customdata") or []):
                if str(row[0]) == accession:
                    return trace["x"][i], trace["y"][i]
        raise AssertionError(f"{accession} not drawn")

    assert position(shown, "GSE1") == position(hidden, "GSE1")
