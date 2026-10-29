"""View Dataset Analyses keeps the chosen contrast after "Prepare Download Package".

The button calls st.rerun() inside an st.fragment, which used to reset the
contrast selectbox to its first option: the page still showed the chosen
contrast while the description and plots came from the first one.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

CONTRASTS = {
    "KD_vs_control": "First contrast description",
    "rescue_vs_dox": "Second contrast description",
}


def _write_results(root: Path) -> None:
    ds = root / "GSE1"
    (ds / "metadata").mkdir(parents=True)
    (ds / "metadata" / "analysis_info.json").write_text(json.dumps({
        "accession": "GSE1", "organism": "Homo sapiens", "analysis_success": True,
        "unique_groups": ["a", "b"], "merged_column": "group",
        "contrasts": [{"name": n, "description": d, "expression": "a - b"} for n, d in CONTRASTS.items()],
    }))
    pd.DataFrame([{"name": n, "description": d, "expression": "a - b"} for n, d in CONTRASTS.items()]).to_csv(
        ds / "metadata" / "contrasts.csv", index=False)
    pd.DataFrame({"group": ["a", "a", "b", "b"]}, index=pd.Index(["S1", "S2", "S3", "S4"])).to_csv(
        ds / "metadata" / "edger_analysis_samples.csv")
    rna = ds / "RNAseqAnalysis"
    rna.mkdir()
    pd.DataFrame({"Gene": ["G1", "G2"], "Sample1": [1.0, 2.0], "Sample2": [1.5, 2.5],
                  "Sample3": [3.0, 0.5], "Sample4": [3.5, 0.2]}).to_csv(rna / "CPM.csv", index=False)
    for name in CONTRASTS:
        (rna / name).mkdir()
        pd.DataFrame({"Gene": ["G1", "G2"], "logFC": [2.0, -2.0], "AveExpr": [5.0, 6.0],
                      "P.Value": [1e-5, 1e-4], "adj.P.Val": [1e-3, 1e-2]}).to_csv(rna / name / "DEG.csv", index=False)


def _descriptions(at: AppTest) -> list:
    return [m.value for m in at.markdown if m.value.startswith("**Description:**")]


@pytest.mark.unit
def test_contrast_choice_survives_prepare_download(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "results"
    _write_results(root)
    script = f"""
from uorca.gui.results_integration import ResultsIntegrator
from uorca.gui.components.analysis_plots_tab import render_analysis_plots_tab
ri = ResultsIntegrator({str(root)!r}); ri.load_data()
render_analysis_plots_tab(ri, {str(root)!r})
"""
    at = AppTest.from_string(script, default_timeout=120).run()
    assert not at.exception

    contrast_box = next(s for s in at.selectbox if s.label == "Select a contrast:")
    second = next(o for o in contrast_box.options if o.startswith("rescue_vs_dox"))
    contrast_box.set_value(second).run()
    assert _descriptions(at) == ["**Description:** Second contrast description"]

    next(b for b in at.button if b.label == "Prepare Download Package").click().run()
    assert not at.exception

    contrast_box = next(s for s in at.selectbox if s.label == "Select a contrast:")
    assert contrast_box.value == second
    assert _descriptions(at) == ["**Description:** Second contrast description"]
