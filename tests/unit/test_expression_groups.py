"""Expression plots keep every sample group separate, however long and similar the names."""

import json
from pathlib import Path

import pandas as pd
import pytest

from uorca.gui.results_integration import ResultsIntegrator, build_group_labels

PREFIX = "Placenta_CT27_Human trophoblast stem cells_"
GROUPS = [
    f"{PREFIX}Cells were treated with 1 uM T0070907 starting on day {d} of EVT differentiation and maintained until day 8."
    for d in range(11)
] + [
    f"{PREFIX}TSCs were treated with {dose} uM T0070907 and maintained for 72 hours under condition {k}."
    for k, dose in enumerate(["0.3", "1", "3", "10", "0.3", "1", "3", "10", "30", "100", "300"])
]


@pytest.mark.unit
def test_22_similar_long_group_names_get_22_unique_labels():
    assert len(GROUPS) == 22 and len(set(GROUPS)) == 22
    labels = build_group_labels([("GSE295568", g) for g in GROUPS])
    assert len(labels) == 22
    assert len(set(labels.values())) == 22
    for label in labels.values():
        assert not label.startswith("Placenta"), "shared prefix should be dropped"


@pytest.mark.unit
def test_same_group_name_in_two_datasets_stays_two_groups():
    labels = build_group_labels([("GSE1", "control"), ("GSE2", "control")])
    assert len(set(labels.values())) == 2


def _write_dataset(root: Path) -> None:
    ds = root / "GSE295568"
    (ds / "metadata").mkdir(parents=True)
    (ds / "metadata" / "analysis_info.json").write_text(json.dumps({
        "accession": "GSE295568", "organism": "Homo sapiens", "analysis_success": True,
        "merged_column": "experimental_group", "unique_groups": GROUPS, "contrasts": [],
    }))
    samples = [f"Sample{i + 1}" for i in range(44)]
    pd.DataFrame({"experimental_group": [g for g in GROUPS for _ in range(2)]}, index=pd.Index(samples)).to_csv(
        ds / "metadata" / "edger_analysis_samples.csv")
    rna = ds / "RNAseqAnalysis"
    (rna / "c1").mkdir(parents=True)
    cpm = pd.DataFrame([[float(i % 7) for i in range(44)]], columns=pd.Index(samples))
    cpm.insert(0, "Gene", ["PPARG"])
    cpm.to_csv(rna / "CPM.csv", index=False)
    pd.DataFrame({"Gene": ["PPARG"], "logFC": [1.0], "adj.P.Val": [0.01], "P.Value": [0.001]}).to_csv(
        rna / "c1" / "DEG.csv", index=False)


@pytest.mark.unit
def test_expression_plot_draws_22_boxes_with_two_samples_each(tmp_path):
    _write_dataset(tmp_path)
    ri = ResultsIntegrator(str(tmp_path))
    ri.load_data()
    fig = ri.create_expression_plots(genes=["PPARG"], analyses=["GSE295568"])
    assert fig is not None
    xs = [x for trace in fig.data for x in trace.x]
    assert len(xs) == 44
    counts = pd.Series(xs).value_counts()
    assert len(counts) == 22
    assert set(counts) == {2}
    assert fig.layout["xaxis"]["tickangle"] == -90
