"""A DEG table without adj.P.Val fails loudly; raw P.Value is never used in its place."""

import json
import re

import pandas as pd
import pytest

from uorca.gui.results_integration import (
    ADJUSTED_P_LABEL,
    MissingAdjustedPValueError,
    ResultsIntegrator,
    get_adjusted_p_column,
)
from uorca.gui.single_analysis_plots import create_volcano_plot


@pytest.mark.unit
def test_returns_adjusted_column_when_present():
    df = pd.DataFrame({"Gene": ["A"], "P.Value": [0.01], "adj.P.Val": [0.04]})
    assert get_adjusted_p_column(df, "x/DEG.csv") == "adj.P.Val"


@pytest.mark.unit
def test_raises_naming_the_source_when_only_raw_p_present():
    df = pd.DataFrame({"Gene": ["A"], "P.Value": [0.01], "logFC": [1.5]})
    with pytest.raises(MissingAdjustedPValueError) as exc:
        get_adjusted_p_column(df, "/data/GSE1/RNAseqAnalysis/c1/DEG.csv")
    msg = str(exc.value)
    assert "/data/GSE1/RNAseqAnalysis/c1/DEG.csv" in msg
    assert "adj.P.Val" in msg
    assert isinstance(exc.value, ValueError)


def _results_with_raw_only(tmp_path):
    ds = tmp_path / "GSE9"
    (ds / "metadata").mkdir(parents=True)
    (ds / "metadata" / "analysis_info.json").write_text(json.dumps({
        "accession": "GSE9", "organism": "Homo sapiens", "analysis_success": True,
        "unique_groups": ["a", "b"], "contrasts": [{"name": "c1", "description": "d"}],
    }))
    cdir = ds / "RNAseqAnalysis" / "c1"
    cdir.mkdir(parents=True)
    pd.DataFrame({"Gene": ["G1", "G2"], "logFC": [2.0, -3.0], "P.Value": [1e-5, 1e-6]}).to_csv(
        cdir / "DEG.csv", index=False)
    ri = ResultsIntegrator(str(tmp_path))
    ri.load_data()
    return ri, cdir / "DEG.csv"


@pytest.mark.unit
def test_identify_important_genes_raises_on_raw_only_file(tmp_path):
    ri, deg_path = _results_with_raw_only(tmp_path)
    with pytest.raises(MissingAdjustedPValueError, match=re.escape(str(deg_path))):
        ri.identify_important_genes()


@pytest.mark.unit
def test_heatmap_raises_on_raw_only_file(tmp_path):
    ri, deg_path = _results_with_raw_only(tmp_path)
    with pytest.raises(MissingAdjustedPValueError, match=re.escape(str(deg_path))):
        ri.create_lfc_heatmap(genes=["G1", "G2"], contrasts=[("GSE9", "c1")])


@pytest.mark.unit
def test_volcano_axis_names_the_p_value_kind():
    deg = pd.DataFrame({"Gene": ["A", "B"], "logFC": [2.0, -2.0],
                        "P.Value": [1e-4, 0.5], "adj.P.Val": [1e-3, 0.6]})
    fig = create_volcano_plot(deg)
    assert fig.layout["yaxis"]["title"]["text"] == "-log10(adjusted p-value, BH)"
    raw = create_volcano_plot(deg, p_col="P.Value")
    assert raw.layout["yaxis"]["title"]["text"] == "-log10(raw p-value)"
    assert ADJUSTED_P_LABEL == "Adjusted p-value (BH)"
