"""AI results table: adjusted p-values only, 0.0 kept, no raw P.Value fallback."""

import pandas as pd
import pytest

from uorca.gui.components.ai_assistant_tab import _build_expression_rows
from uorca.gui.results_integration import MissingAdjustedPValueError, ResultsIntegrator


def _integrator(tmp_path, df):
    ri = ResultsIntegrator(str(tmp_path))
    ri.deg_data = {"GSE1": {"c1": df}}
    ri.deg_file_paths = {"GSE1": {"c1": "/results/GSE1/RNAseqAnalysis/c1/DEG.csv"}}
    return ri


@pytest.mark.unit
def test_adjusted_p_of_zero_is_kept(tmp_path):
    df = pd.DataFrame({"Gene": ["A", "B"], "logFC": [3.0, -1.0],
                       "P.Value": [1e-300, 0.2], "adj.P.Val": [0.0, 0.4]})
    rows, missing = _build_expression_rows(_integrator(tmp_path, df), ["A", "B", "Z"], [("GSE1", "c1")])
    by_gene = {r["Gene"]: r for r in rows}
    assert by_gene["A"]["Adjusted p-value (BH)"] == 0.0
    assert by_gene["B"]["Adjusted p-value (BH)"] == 0.4
    assert missing == ["Z"]


@pytest.mark.unit
def test_missing_adjusted_p_raises_instead_of_using_raw(tmp_path):
    df = pd.DataFrame({"Gene": ["A"], "logFC": [3.0], "P.Value": [0.01]})
    with pytest.raises(MissingAdjustedPValueError, match="/results/GSE1/RNAseqAnalysis/c1/DEG.csv"):
        _build_expression_rows(_integrator(tmp_path, df), ["A"], [("GSE1", "c1")])
