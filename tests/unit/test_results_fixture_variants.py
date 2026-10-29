"""The fixture variants in tests/_results_fixtures.py are what they claim to be.

Later plans build tests on them (long contrast names, a mouse dataset, a DEG file without
``adj.P.Val``), so each one is checked here against the importer and the results loader.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tests._results_fixtures import (
    LONG_CONTRASTS,
    make_long_contrast_dataset,
    make_mouse_dataset,
    make_no_adj_p_dataset,
    make_variants_folder,
)

pytestmark = pytest.mark.unit


def test_long_contrast_dataset(tmp_path):
    ds = make_long_contrast_dataset(tmp_path)
    names = sorted(p.name for p in (ds / "RNAseqAnalysis").iterdir() if p.is_dir())
    assert names == sorted(LONG_CONTRASTS)
    assert all(len(n) > 60 for n in names)
    info = json.loads((ds / "metadata" / "analysis_info.json").read_text())
    assert [c["name"] for c in info["contrasts"]] == list(LONG_CONTRASTS)


def test_mouse_dataset(tmp_path):
    ds = make_mouse_dataset(tmp_path)
    info = json.loads((ds / "metadata" / "analysis_info.json").read_text())
    assert info["organism"] == "Mus musculus"
    genes = pd.read_csv(ds / "RNAseqAnalysis" / "CPM.csv")["Gene"].tolist()
    assert genes[0] == "Fakegene1"


def test_no_adj_p_dataset(tmp_path):
    ds = make_no_adj_p_dataset(tmp_path)
    deg = pd.read_csv(ds / "RNAseqAnalysis" / "Treated_vs_Control" / "DEG.csv")
    assert "adj.P.Val" not in deg.columns
    assert {"Gene", "logFC", "P.Value"} <= set(deg.columns)


def test_variants_folder_imports_four_complete_datasets(tmp_path):
    from uorca.gui.project.importer import inspect_results
    from uorca.gui.results_integration import ResultsIntegrator

    root = make_variants_folder(tmp_path / "analysis")
    candidate = inspect_results(root)
    assert sorted(candidate.complete) == ["GSE900001", "GSE900010", "GSE900020", "GSE900030"]

    ri = ResultsIntegrator(results_dir=str(root))
    ri.load_data()
    assert sorted(ri.cpm_data) == ["GSE900001", "GSE900010", "GSE900020", "GSE900030"]
    assert sorted(ri.deg_data["GSE900010"]) == sorted(LONG_CONTRASTS)
    assert ri.analysis_info["GSE900020"]["organism"] == "Mus musculus"
