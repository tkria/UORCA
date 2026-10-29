"""Tests for script_generation module."""
import pytest
from uorca.core.script_generation import build_repro_script, build_readme_text


def test_build_repro_script_returns_string():
    """Test that build_repro_script returns a non-empty string."""
    script = build_repro_script()

    assert isinstance(script, str)
    assert len(script) > 0


def test_build_repro_script_has_shebang():
    """Test that script starts with shebang."""
    script = build_repro_script()

    assert script.startswith('#!/usr/bin/env python3')


def test_build_repro_script_has_main_function():
    """Test that script contains main function."""
    script = build_repro_script()

    assert 'def main():' in script
    assert "if __name__ == '__main__':" in script


def test_build_repro_script_has_required_imports():
    """Test that script has all required imports."""
    script = build_repro_script()

    required_imports = [
        'import argparse',
        'import json',
        'import pandas as pd',
        'import numpy as np',
        'import plotly.express as px',
        'import plotly.graph_objects as go'
    ]

    for required_import in required_imports:
        assert required_import in script


def test_build_repro_script_has_argparse_setup():
    """Test that script sets up argument parser."""
    script = build_repro_script()

    assert 'ArgumentParser' in script
    assert '--input' in script
    assert '--output' in script
    assert '--xlabel-mapping' in script


def test_build_repro_script_has_metadata_loading():
    """Test that script loads metadata.json."""
    script = build_repro_script()

    assert 'metadata.json' in script
    assert 'json.load' in script


def test_build_repro_script_has_plotting_code():
    """Test that script contains plotting logic."""
    script = build_repro_script()

    assert 'px.imshow' in script
    assert 'fig.update_layout' in script
    assert 'fig.update_xaxes' in script
    assert 'fig.write_image' in script


def test_build_repro_script_has_error_handling():
    """Test that script has error handling."""
    script = build_repro_script()

    assert 'try:' in script
    assert 'except Exception' in script
    assert 'traceback' in script


def test_build_repro_script_supports_multiple_formats():
    """Test that script supports PDF, PNG, and SVG formats."""
    script = build_repro_script()

    assert 'pdf' in script.lower()
    assert 'png' in script.lower()
    assert 'svg' in script.lower()


def test_build_repro_script_has_usage_docs():
    """Test that script includes usage documentation."""
    script = build_repro_script()

    assert 'Usage' in script
    assert 'uv run python' in script


def test_build_repro_script_ends_with_newline():
    """Test that script ends with newline."""
    script = build_repro_script()

    assert script.endswith('\n')


def test_build_readme_text_returns_string():
    """Test that build_readme_text returns a non-empty string."""
    readme = build_readme_text()

    assert isinstance(readme, str)
    assert len(readme) > 0


def test_build_readme_text_has_title():
    """Test that README has title."""
    readme = build_readme_text()

    assert 'UORCA Heatmap Reproducible Package' in readme


def test_build_readme_text_lists_files():
    """Test that README lists expected files."""
    readme = build_readme_text()

    expected_files = [
        'heatmap_data.csv',
        'metadata.json',
        'reproduce_heatmap.py',
        'README.txt'
    ]

    for filename in expected_files:
        assert filename in readme


def test_build_readme_text_has_quick_start():
    """Test that README includes quick start instructions."""
    readme = build_readme_text()

    assert 'Quick start' in readme
    assert 'Unzip' in readme
    assert 'uv run python' in readme


def test_build_readme_text_has_tips():
    """Test that README includes tips section."""
    readme = build_readme_text()

    assert 'Tips:' in readme


def test_build_readme_text_mentions_label_mapping():
    """Test that README mentions label mapping feature."""
    readme = build_readme_text()

    assert '--xlabel-mapping' in readme or 'label' in readme.lower()


def test_build_readme_text_mentions_formats():
    """Test that README mentions supported formats."""
    readme = build_readme_text()

    assert 'PDF' in readme
    assert 'PNG' in readme or 'SVG' in readme

def test_build_repro_script_matches_app_label_layout():
    """Angled one-line labels, a computed bottom margin, automargin and dataset blocks, like the app."""
    script = build_repro_script()

    assert "tickangle=-45" in script
    assert "dataset_blocks" in script
    assert "b=bottom_margin" in script
    assert "b=120" not in script
    assert "automargin=True" in script
    assert "fig.update_yaxes(automargin=True)" in script
    compile(script, "reproduce_heatmap.py", "exec")


@pytest.mark.unit
def test_repro_script_renders_app_labels_and_dataset_blocks(tmp_path, monkeypatch):
    """Run the generated script on app-style labels from two datasets."""
    import json
    import runpy
    import sys

    import pandas as pd
    import plotly.graph_objects as go

    from uorca.gui.results_integration import build_contrast_labels

    names = ["PPARG_KD1_DLX6_overexpression_vs_control", "PPARG_KD1_vs_control_TSC_self_renewal",
             "EVT_DMOG_vs_control"]
    labels = build_contrast_labels([("GSE1", names[0]), ("GSE1", names[1]), ("GSE2", names[2])])
    cols = [f"GSE1_{names[0]}", f"GSE1_{names[1]}", f"GSE2_{names[2]}"]
    pd.DataFrame({"Gene": ["A", "B"], cols[0]: [1.5, -2.0], cols[1]: [0.0, 3.0], cols[2]: [2.2, 0.0]}).to_csv(
        tmp_path / "heatmap_data.csv", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "output_format": "png",
        "x_axis_labels": {c: lab.display for c, lab in zip(cols, labels)},
        "clustered_contrasts": cols,
        "dataset_blocks": [["GSE1", 0, 1], ["GSE2", 2, 2]],
    }))
    script = tmp_path / "reproduce_heatmap.py"
    script.write_text(build_repro_script())

    captured = {}

    def fake_write_image(self, path, *args, **kwargs):
        captured["fig"] = self
        captured["path"] = path

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["reproduce_heatmap.py"])
    runpy.run_path(str(script), run_name="__main__")

    fig = captured["fig"]
    assert list(fig.layout.xaxis.ticktext) == [lab.display for lab in labels]
    assert fig.layout.xaxis.automargin is True
    assert fig.layout.yaxis.automargin is True
    assert fig.layout.xaxis.tickangle == -45
    assert [a.text for a in fig.layout.annotations] == ["<b>GSE1</b>", "<b>GSE2</b>"]
    assert fig.layout.margin.b > 100


@pytest.mark.unit
def test_repro_script_reads_its_own_folder_and_adds_subtitle(tmp_path, monkeypatch):
    """Run from another directory: metadata.json and the data next to the script are used."""
    import json
    import runpy
    import sys

    import pandas as pd
    import plotly.graph_objects as go

    pkg = tmp_path / "package"
    pkg.mkdir()
    pd.DataFrame({"Gene": ["A", "B"], "c1": [1.5, 0.0], "c2": [0.0, -2.0]}).to_csv(pkg / "heatmap_data.csv", index=False)
    (pkg / "metadata.json").write_text(json.dumps({
        "output_format": "png", "p_value_threshold": 0.01, "lfc_threshold": 1.5,
        "x_axis_labels": {"c1": "first", "c2": "second"}, "clustered_contrasts": ["c1", "c2"],
    }))
    script = pkg / "reproduce_heatmap.py"
    script.write_text(build_repro_script())

    captured = {}
    monkeypatch.setattr(go.Figure, "write_image",
                        lambda self, path, *a, **k: captured.update(fig=self, path=path))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sys, "argv", ["reproduce_heatmap.py"])
    runpy.run_path(str(script), run_name="__main__")

    fig = captured["fig"]
    assert list(fig.layout.xaxis.ticktext) == ["first", "second"]  # labels from metadata.json
    assert "Adjusted p-value (BH) < 0.01" in fig.layout.title.text
    assert "|Log2FC| > 1.5" in fig.layout.title.text
    assert captured["path"] == str(pkg / "heatmap.png")
