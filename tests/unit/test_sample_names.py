"""CPM sample columns are named by sample ID and joined to groups by name."""

import pandas as pd
import pytest

from uorca.gui.sample_names import (
    SampleTableMismatch,
    name_cpm_columns,
    sample_groups,
    sample_ids,
    sample_titles,
)


def _table(**cols):
    return pd.DataFrame(cols)


def _cpm(names):
    data: dict[str, list] = {"Gene": ["G1", "G2"]}
    for i, n in enumerate(names):
        data[n] = [float(i), float(i) + 0.5]
    return pd.DataFrame(data)


def test_default_names_are_renamed_by_row_position():
    table = _table(geo_accession=["GSM3", "GSM1", "GSM2"], group=["A", "B", "B"])
    out = name_cpm_columns(_cpm(["Sample1", "Sample2", "Sample3"]), table)
    assert list(out.columns) == ["Gene", "GSM3", "GSM1", "GSM2"]
    assert out["GSM3"].tolist() == [0.0, 0.5]


def test_columns_already_named_by_id_are_unchanged():
    table = _table(geo_accession=["GSM1", "GSM2"])
    cpm = _cpm(["GSM2", "GSM1"])
    assert name_cpm_columns(cpm, table) is cpm


def test_default_names_with_wrong_count_raise():
    table = _table(geo_accession=["GSM1", "GSM2", "GSM3"])
    with pytest.raises(SampleTableMismatch, match="2 sample columns.*3 rows"):
        name_cpm_columns(_cpm(["Sample1", "Sample2"]), table)


def test_unknown_column_names_raise():
    table = _table(geo_accession=["GSM1", "GSM2"])
    with pytest.raises(SampleTableMismatch, match="not in the sample table"):
        name_cpm_columns(_cpm(["GSM1", "X9"]), table)


def test_repeated_gsm_falls_back_to_run_accession():
    table = _table(geo_accession=["GSM1", "GSM1"], GSM=["GSM1", "GSM1"], SRR=["SRR1", "SRR2"])
    assert sample_ids(table) == ["SRR1", "SRR2"]


def test_empty_gsm_falls_back_to_run_accession():
    table = _table(geo_accession=["GSM1", None], SRR=["SRR1", "SRR2"])
    assert sample_ids(table) == ["SRR1", "SRR2"]


def test_no_id_column_keeps_default_names():
    table = _table(sample=["a", "b"], group=["A", "B"])
    assert sample_ids(table) == ["Sample1", "Sample2"]
    cpm = _cpm(["Sample1", "Sample2"])
    assert list(name_cpm_columns(cpm, table).columns) == ["Gene", "Sample1", "Sample2"]


def test_groups_and_titles_are_keyed_by_id():
    table = _table(geo_accession=["GSM1", "GSM2"], group=["A", "B"], title=["t1", None])
    assert sample_groups(table, "group").to_dict() == {"GSM1": "A", "GSM2": "B"}
    assert sample_titles(table).to_dict() == {"GSM1": "t1", "GSM2": ""}


def _write_dataset(root, cpm_names, table):
    rna = root / "GSE1" / "RNAseqAnalysis"
    meta = root / "GSE1" / "metadata"
    rna.mkdir(parents=True)
    meta.mkdir(parents=True)
    _cpm(cpm_names).to_csv(rna / "CPM.csv", index=False)
    table.to_csv(meta / "edger_analysis_samples.csv")


def test_load_sample_groups_joins_by_name_not_position(tmp_path):
    from uorca.gui.single_analysis_plots import load_sample_groups

    table = _table(geo_accession=["GSM1", "GSM2", "GSM3"], group=["A", "B", "C"])
    # A new run: CPM columns named by ID, in an order unlike the table's.
    cpm = _cpm(["GSM3", "GSM1", "GSM2"])
    _write_dataset(tmp_path, ["GSM3", "GSM1", "GSM2"], table)
    groups = load_sample_groups(str(tmp_path), "GSE1", cpm, {"merged_column": "group"})
    assert groups is not None
    assert groups.to_dict() == {"GSM1": "A", "GSM2": "B", "GSM3": "C"}


def test_integrator_renames_old_columns_and_records_mismatch(tmp_path):
    from uorca.gui.results_integration import ResultsIntegrator

    table = _table(geo_accession=["GSM1", "GSM2"], group=["A", "B"])
    _write_dataset(tmp_path, ["Sample1", "Sample2", "Sample3"], table)
    (tmp_path / "GSE1" / "metadata" / "analysis_info.json").write_text(
        '{"accession": "GSE1", "analysis_success": true, "merged_column": "group", "unique_groups": ["A", "B"]}'
    )
    ri = ResultsIntegrator(str(tmp_path))
    ri.load_data()
    assert "GSE1" in ri.sample_name_errors
    assert "3 sample columns" in ri.sample_name_errors["GSE1"]
    assert list(ri.cpm_data["GSE1"].columns) == ["Gene", "Sample1", "Sample2", "Sample3"]


def test_unreadable_sample_table_keeps_cpm_and_records_error(tmp_path):
    from uorca.gui.results_integration import ResultsIntegrator

    _write_dataset(tmp_path, ["Sample1", "Sample2"], _table(geo_accession=["GSM1", "GSM2"], group=["A", "B"]))
    (tmp_path / "GSE1" / "metadata" / "edger_analysis_samples.csv").write_text("")
    (tmp_path / "GSE1" / "metadata" / "analysis_info.json").write_text(
        '{"accession": "GSE1", "analysis_success": true, "merged_column": "group", "unique_groups": ["A", "B"]}'
    )
    ri = ResultsIntegrator(str(tmp_path))
    ri.load_data()
    assert "GSE1" in ri.cpm_data
    assert "GSE1" in ri.sample_name_errors


def test_literal_na_string_is_an_id_as_in_r(tmp_path):
    from uorca.gui.sample_names import read_sample_table

    path = tmp_path / "t.csv"
    _table(geo_accession=["GSM1", "NA"], SRR=["SRR1", "SRR2"]).to_csv(path)
    assert sample_ids(read_sample_table(str(path))) == ["GSM1", "NA"]


def test_pca_plot_renders_with_groups_and_titles():
    from uorca.gui.single_analysis_plots import create_pca_plot

    cpm = pd.DataFrame({"Gene": [f"G{i}" for i in range(5)],
                        "GSM1": [1.0, 2, 3, 4, 5], "GSM2": [2.0, 1, 4, 3, 6], "GSM3": [5.0, 4, 3, 2, 1]})
    groups = pd.Series({"GSM1": "A", "GSM2": "A"})  # GSM3 has no known group
    fig = create_pca_plot(cpm, groups, sample_titles=pd.Series({"GSM1": "t1", "GSM2": "t2", "GSM3": "t3"}))
    names = {t.name: list(t.hovertext) for t in fig.data}
    assert names == {"A": ["GSM1", "GSM2"], "GSM3": ["GSM3"]}
