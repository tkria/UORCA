"""Heatmap contrast labels: readable, unique, never silently cut.

Covers plan 2026-10-27-02 section 3A. The contrast names below are the 17
GSE295568 contrasts from the PPARg results, six of which are longer than the
25 characters the old label code kept.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from uorca.gui.results_integration import (
    ELLIPSIS,
    LABEL_LINE_CHARS,
    LABEL_MAX_LINES,
    ResultsIntegrator,
    build_contrast_labels,
    wrap_contrast_label,
    wrap_hover_text,
)

GSE295568_CONTRASTS = [
    "EVT_DMOG_vs_control",
    "EVT_SCH772984_vs_control",
    "EVT_T0070907_day0_vs_control",
    "EVT_T0070907_day1_vs_control",
    "EVT_T0070907_day2_vs_control",
    "EVT_T0070907_day3_vs_control",
    "EVT_T0070907_day4_vs_control",
    "EVT_T0070907_day6_vs_control",
    "PPARG_KD1_DLX6_overexpression_vs_control",
    "PPARG_KD1_EVT_vs_control",
    "PPARG_KD1_vs_control_TSC_self_renewal",
    "PPARG_rescue_12p5_vs_15_dox",
    "T0070907_0p3_vs_vehicle_TSC",
    "T0070907_1_vs_vehicle_TSC",
    "T0070907_10_vs_vehicle_TSC",
    "T0070907_3_vs_vehicle_TSC",
    "T0070907_all_doses_vs_vehicle_TSC",
]


def _flatten(display: str) -> str:
    """Join a display label's lines back into one string."""
    return display.replace("<br>", "")


def _assert_well_formed(display: str, full: str) -> None:
    lines = display.split("<br>")
    assert len(lines) <= LABEL_MAX_LINES, display
    for line in lines:
        assert len(line) <= LABEL_LINE_CHARS, (line, display)
    flat = _flatten(display)
    if flat.endswith(ELLIPSIS):
        kept = flat[: -len(ELLIPSIS)]
        assert full.startswith(kept), (display, full)
        assert len(kept) < len(full)
    else:
        assert flat == full, (display, full)


@pytest.mark.unit
def test_gse295568_labels_are_unique_and_at_most_three_lines():
    labels = build_contrast_labels([(None, name) for name in GSE295568_CONTRASTS])
    displays = [lab.display for lab in labels]
    assert len(set(displays)) == len(displays)
    for lab, name in zip(labels, GSE295568_CONTRASTS):
        assert lab.full == name
        _assert_well_formed(lab.display, name)


@pytest.mark.unit
def test_gse295568_long_names_are_not_truncated():
    """Every GSE295568 name fits in three 18-character lines, so none needs an ellipsis."""
    labels = build_contrast_labels([(None, name) for name in GSE295568_CONTRASTS])
    for lab in labels:
        assert ELLIPSIS not in lab.display
        assert _flatten(lab.display) == lab.full
    long_label = next(lab for lab in labels if lab.full.startswith("PPARG_KD1_DLX6"))
    assert "overexpression" in long_label.display


@pytest.mark.unit
def test_wrap_breaks_on_underscores_without_losing_characters():
    display = wrap_contrast_label("PPARG_KD1_DLX6_overexpression_vs_control")
    assert display.split("<br>")[0].endswith("_")
    assert _flatten(display) == "PPARG_KD1_DLX6_overexpression_vs_control"


@pytest.mark.unit
def test_overlong_name_gets_ellipsis_only_after_three_full_lines():
    name = "_".join(["segment"] * 20)
    display = wrap_contrast_label(name)
    _assert_well_formed(display, name)
    assert display.endswith(ELLIPSIS)
    assert len(display.split("<br>")) == LABEL_MAX_LINES


@pytest.mark.unit
def test_token_longer_than_a_line_is_split_not_dropped():
    name = "averyveryverylongtokenwithoutbreaks_vs_ctrl"
    display = wrap_contrast_label(name)
    _assert_well_formed(display, name)


@pytest.mark.unit
def test_accession_prefix_added_when_more_than_one_dataset():
    labels = build_contrast_labels([
        ("GSE295568", "PPARG_KD1_EVT_vs_control"),
        ("GSE298581", "PPARG_KD1_EVT_vs_control"),
    ])
    assert labels[0].full == "GSE295568: PPARG_KD1_EVT_vs_control"
    assert labels[1].full == "GSE298581: PPARG_KD1_EVT_vs_control"
    assert labels[0].display != labels[1].display
    for lab in labels:
        _assert_well_formed(lab.display, lab.full)
        assert lab.display.startswith("GSE")


@pytest.mark.unit
def test_no_accession_prefix_for_a_single_dataset():
    labels = build_contrast_labels([("GSE295568", name) for name in GSE295568_CONTRASTS])
    assert all(not lab.full.startswith("GSE295568") for lab in labels)


@pytest.mark.unit
def test_colliding_truncated_labels_are_numbered():
    stem = "_".join(["segment"] * 20)
    labels = build_contrast_labels([(None, stem + "_a"), (None, stem + "_b")])
    displays = [lab.display for lab in labels]
    assert len(set(displays)) == 2
    assert displays[1].endswith("(2)")


@pytest.mark.unit
def test_wrap_hover_text_wraps_and_caps():
    text = "word " * 200
    wrapped = wrap_hover_text(text, width=80, max_chars=300)
    lines = wrapped.split("<br>")
    assert all(len(line) <= 80 for line in lines)
    assert len(_flatten(wrapped)) <= 300 + len(ELLIPSIS) + len(lines)
    assert wrapped.endswith(ELLIPSIS)
    assert wrap_hover_text("short", width=80) == "short"


# ---------------------------------------------------------------------------
# End-to-end: build a tiny results tree and render the heatmap figure
# ---------------------------------------------------------------------------

def _write_dataset(root: Path, analysis_id: str, contrasts: dict) -> None:
    """contrasts: name -> (description, expression)."""
    ds = root / analysis_id
    (ds / "metadata").mkdir(parents=True)
    info = {
        "accession": analysis_id,
        "organism": "Homo sapiens",
        "analysis_success": True,
        "unique_groups": ["a", "b"],
        "contrasts": [
            {"name": n, "description": d, "expression": e} for n, (d, e) in contrasts.items()
        ],
    }
    (ds / "metadata" / "analysis_info.json").write_text(json.dumps(info))
    pd.DataFrame(
        [{"name": n, "description": d, "expression": e} for n, (d, e) in contrasts.items()]
    ).to_csv(ds / "metadata" / "contrasts.csv", index=False)
    for i, name in enumerate(contrasts):
        cdir = ds / "RNAseqAnalysis" / name
        cdir.mkdir(parents=True)
        pd.DataFrame({
            "Gene": ["G1", "G2", "G3"],
            "logFC": [2.0 + i, -2.5, 3.0 - i * 0.1],
            "AveExpr": [5.0, 6.0, 7.0],
            "P.Value": [1e-6, 1e-5, 1e-4],
            "adj.P.Val": [1e-4, 1e-3, 1e-2],
        }).to_csv(cdir / "DEG.csv", index=False)


@pytest.fixture
def two_dataset_results(tmp_path):
    long_formula = "(" + " + ".join(["groupA_condition_level"] * 30) + ")/30 - groupB"
    _write_dataset(tmp_path, "GSE1", {
        "shared_vs_control": ("Description for dataset one", "A - B"),
        "PPARG_KD1_DLX6_overexpression_vs_control": ("Long one", long_formula),
    })
    _write_dataset(tmp_path, "GSE2", {
        "shared_vs_control": ("Description for dataset two", "C - D"),
    })
    ri = ResultsIntegrator(str(tmp_path))
    ri.load_data()
    return ri


@pytest.mark.unit
def test_heatmap_hover_and_axes(two_dataset_results):
    ri = two_dataset_results
    contrasts = [
        ("GSE1", "shared_vs_control"),
        ("GSE1", "PPARG_KD1_DLX6_overexpression_vs_control"),
        ("GSE2", "shared_vs_control"),
    ]
    fig = ri.create_lfc_heatmap(genes=["G1", "G2", "G3"], contrasts=contrasts,
                                hide_empty_rows_cols=False)
    assert fig is not None
    assert fig.layout.xaxis.automargin is True
    assert fig.layout.yaxis.automargin is True

    ticktext = list(fig.layout.xaxis.ticktext)
    assert len(set(ticktext)) == 3
    assert all(t.startswith("GSE") for t in ticktext)

    custom = fig.data[0].customdata
    # Column order follows clustering; index each column by its full name.
    by_full = {custom[0][j][2]: custom[0][j] for j in range(len(custom[0]))}
    assert set(by_full) == {
        "GSE1: shared_vs_control",
        "GSE1: PPARG_KD1_DLX6_overexpression_vs_control",
        "GSE2: shared_vs_control",
    }
    # Descriptions are keyed by (analysis_id, contrast_id), not by name alone.
    assert by_full["GSE1: shared_vs_control"][0] == "Description for dataset one"
    assert by_full["GSE2: shared_vs_control"][0] == "Description for dataset two"
    # Long formula is wrapped at 80 characters and capped at 300.
    formula = by_full["GSE1: PPARG_KD1_DLX6_overexpression_vs_control"][3]
    assert all(len(line) <= 80 for line in formula.split("<br>"))
    assert formula.endswith(ELLIPSIS)
    assert len(formula.replace("<br>", "")) <= 300 + len(ELLIPSIS)

    # Display labels are stored with the context used for the download package.
    ctx = ri.last_heatmap_context
    label_map = dict(zip(ctx["contrast_labels"], ctx["simplified_labels"]))
    assert [label_map[c] for c in ctx["clustered_contrasts"]] == ticktext
    assert set(ctx["full_labels"]) == set(by_full)
