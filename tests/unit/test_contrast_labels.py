"""Heatmap contrast labels: short, one line, unique; full information kept elsewhere.

Covers plan 2026-10-27-02 section 3A and the owner's redesign: tick labels are
one angled line of at most 25 characters; the accession appears once per
dataset block; the full name and description are in the hover and the key.
The contrast names are the 17 GSE295568 contrasts from the PPARg results.
"""

import html
import json
from pathlib import Path

import pandas as pd
import pytest

from uorca.gui.results_integration import (
    ELLIPSIS,
    SHORT_LABEL_CHARS,
    ResultsIntegrator,
    build_contrast_labels,
    shorten_label,
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
    """Join a display label's lines back into one plain-text string."""
    return html.unescape(display.replace("<br>", ""))


def _assert_short_label(display: str, source: str) -> None:
    """One line, at most SHORT_LABEL_CHARS, and a prefix of ``source`` (plus "…" if cut)."""
    text = html.unescape(display)
    assert "<br>" not in display
    assert len(text) <= SHORT_LABEL_CHARS, text
    if text.endswith(ELLIPSIS):
        assert source.startswith(text[: -len(ELLIPSIS)]), (text, source)
    else:
        assert text == source, (text, source)


@pytest.mark.unit
def test_gse295568_labels_are_short_one_line_and_unique():
    labels = build_contrast_labels([("GSE295568", name) for name in GSE295568_CONTRASTS])
    displays = [lab.display for lab in labels]
    assert len(set(displays)) == len(displays)
    for lab, name in zip(labels, GSE295568_CONTRASTS):
        assert lab.full == name and lab.accession == "GSE295568"
        _assert_short_label(lab.display, name)


@pytest.mark.unit
def test_accession_is_not_repeated_in_labels():
    pairs = [("GSE295568", n) for n in GSE295568_CONTRASTS] + [
        ("GSE298581", "T0070907_vs_DMSO"), ("GSE298581", "FTX6746_vs_DMSO")]
    labels = build_contrast_labels(pairs)
    assert all("GSE" not in lab.display for lab in labels)
    assert [lab.accession for lab in labels] == [acc for acc, _ in pairs]
    assert len({lab.display for lab in labels}) == len(labels)


@pytest.mark.unit
def test_shared_leading_words_are_dropped_within_a_dataset():
    names = ["PPARG_KD1_DLX6_overexpression_vs_control", "PPARG_KD1_EVT_vs_control",
             "PPARG_KD1_vs_control_TSC_self_renewal"]
    labels = build_contrast_labels([("GSE1", n) for n in names] + [("GSE2", "PPARG_KD1_alone")])
    assert [lab.display for lab in labels[:3]] == [
        "DLX6_overexpression_vs_c" + ELLIPSIS, "EVT_vs_control", "vs_control_TSC_self_rene" + ELLIPSIS]
    # A dataset with a single contrast keeps its whole name
    assert labels[3].display == "PPARG_KD1_alone"
    for lab, name in zip(labels[:3], names):
        _assert_short_label(lab.display, name[len("PPARG_KD1_"):])


@pytest.mark.unit
def test_prefix_is_not_dropped_when_it_would_empty_a_name():
    labels = build_contrast_labels([("GSE1", "A_"), ("GSE1", "A_b")])
    assert [lab.display for lab in labels] == ["A_", "A_b"]


@pytest.mark.unit
def test_shorten_label():
    assert shorten_label("short") == "short"
    cut = shorten_label("x" * 40)
    assert len(cut) == SHORT_LABEL_CHARS and cut.endswith(ELLIPSIS)


@pytest.mark.unit
def test_numbering_never_creates_a_duplicate():
    labels = build_contrast_labels([(None, "a (2)"), (None, "a"), (None, "a")])
    displays = [lab.display for lab in labels]
    assert displays == ["a (2)", "a", "a (3)"]


@pytest.mark.unit
def test_same_name_in_two_datasets_gets_unique_labels():
    labels = build_contrast_labels([("GSE1", "FX909_vs_DMSO"), ("GSE2", "FX909_vs_DMSO")])
    assert [lab.display for lab in labels] == ["FX909_vs_DMSO", "FX909_vs_DMSO (2)"]


@pytest.mark.unit
def test_colliding_truncated_labels_are_numbered_within_the_length_limit():
    stem = "_".join(["segment"] * 20)
    labels = build_contrast_labels([("GSE1", stem + "_a"), ("GSE2", stem + "_b")])
    displays = [html.unescape(lab.display) for lab in labels]
    assert len(set(displays)) == 2
    assert displays[1].endswith("(2)")
    assert all(len(d) <= SHORT_LABEL_CHARS for d in displays)


@pytest.mark.unit
def test_html_in_names_and_hover_is_escaped():
    labels = build_contrast_labels([(None, "<b>x</b>_&_vs_ctrl")])
    assert "<b>" not in labels[0].display
    assert "&lt;b&gt;" in labels[0].display
    assert labels[0].full == "<b>x</b>_&_vs_ctrl"
    hover = wrap_hover_text("A <script>alert(1)</script> & B")
    assert "<script>" not in hover
    assert "&lt;script&gt;" in hover and "&amp;" in hover


@pytest.mark.unit
def test_group_label_wrapping_keeps_characters():
    """wrap_contrast_label is used for expression-plot group labels."""
    display = wrap_contrast_label("PPARG_KD1_DLX6_overexpression_vs_control")
    assert _flatten(display) == "PPARG_KD1_DLX6_overexpression_vs_control"


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
    assert all("GSE" not in t for t in ticktext)
    assert fig.layout.xaxis.tickangle == -45

    # Columns are grouped per dataset, and each accession is shown once
    meta = fig.layout.meta
    assert [b[0] for b in meta["dataset_blocks"]] == ["GSE1", "GSE2"]
    assert [a.text for a in fig.layout.annotations if "GSE" in a.text] == ["<b>GSE1</b>", "<b>GSE2</b>"]

    custom = fig.data[0].customdata
    # Hover: dataset (customdata[1]) and full contrast name (customdata[2])
    by_col = {(custom[0][j][1], custom[0][j][2]): custom[0][j] for j in range(len(custom[0]))}
    assert set(by_col) == {
        ("GSE1", "shared_vs_control"),
        ("GSE1", "PPARG_KD1_DLX6_overexpression_vs_control"),
        ("GSE2", "shared_vs_control"),
    }
    # Descriptions are keyed by (analysis_id, contrast_id), not by name alone.
    assert by_col[("GSE1", "shared_vs_control")][0] == "Description for dataset one"
    assert by_col[("GSE2", "shared_vs_control")][0] == "Description for dataset two"
    # Long formula is wrapped at 80 characters and capped at 300.
    formula = by_col[("GSE1", "PPARG_KD1_DLX6_overexpression_vs_control")][3]
    assert all(len(line) <= 80 for line in formula.split("<br>"))
    assert formula.endswith(ELLIPSIS)
    assert len(formula.replace("<br>", "")) <= 300 + len(ELLIPSIS)

    # Numbered key in column order: label on chart, dataset, full name, description
    key = meta["column_key"]
    assert [row["#"] for row in key] == [1, 2, 3]
    assert [html.escape(row["Label"], quote=False) for row in key] == ticktext
    assert {(row["Dataset"], row["Contrast"], row["Description"]) for row in key} == {
        ("GSE1", "shared_vs_control", "Description for dataset one"),
        ("GSE1", "PPARG_KD1_DLX6_overexpression_vs_control", "Long one"),
        ("GSE2", "shared_vs_control", "Description for dataset two"),
    }
    assert [row["Dataset"] for row in key] == ["GSE1", "GSE1", "GSE2"]

    # The download-package context matches the figure
    ctx = ri.last_heatmap_context
    label_map = dict(zip(ctx["contrast_labels"], ctx["simplified_labels"]))
    assert [label_map[c] for c in ctx["clustered_contrasts"]] == ticktext
    assert ctx["column_key"] == key


@pytest.mark.unit
def test_hover_says_when_a_gene_is_not_in_a_contrast(two_dataset_results):
    ri = two_dataset_results
    contrasts = [("GSE1", "shared_vs_control"), ("GSE2", "shared_vs_control")]
    fig = ri.create_lfc_heatmap(genes=["G1", "G3", "ABSENT"], contrasts=contrasts,
                                hide_empty_rows_cols=False, cluster_genes=False)
    assert fig is not None
    assert "%{customdata[4]}" in fig.data[0].hovertemplate
    assert "%{z:.2f}" not in fig.data[0].hovertemplate
    genes = list(fig.data[0].y)
    custom = fig.data[0].customdata
    absent_row = custom[genes.index("ABSENT")]
    assert all(cell[4] == "not in this contrast's results" for cell in absent_row)
    assert list(fig.data[0].z[genes.index("ABSENT")]) == [0, 0]  # neutral colour
    g1 = custom[genes.index("G1")]
    assert all(cell[4] in {"2.00", "3.00"} for cell in g1)
    # G3 has adj.P.Val 0.01 < 0.05 and |logFC| ~3: significant
    g3 = custom[genes.index("G3")]
    assert all(not cell[4].startswith("not in") for cell in g3)


@pytest.mark.unit
def test_hover_marks_non_significant_values(two_dataset_results):
    ri = two_dataset_results
    fig = ri.create_lfc_heatmap(genes=["G1", "G3"], contrasts=[("GSE1", "shared_vs_control")],
                                p_value_threshold=0.005, hide_empty_rows_cols=False, cluster_genes=False)
    genes = list(fig.data[0].y)
    g3 = fig.data[0].customdata[genes.index("G3")][0][4]
    assert "not significant" in g3 and "shown as 0" in g3
