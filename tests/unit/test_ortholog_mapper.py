"""The orthologue loader fails loudly when data/mammalian_orthologues.csv is missing."""

import pytest

import uorca.gui.ortholog_mapper as om


@pytest.fixture
def missing_csv(tmp_path, monkeypatch):
    path = tmp_path / "data" / "mammalian_orthologues.csv"
    monkeypatch.setattr(om, "ORTHOLOG_CSV_PATH", str(path))
    om.clear_ortholog_cache()
    yield path
    om.clear_ortholog_cache()


@pytest.mark.unit
def test_missing_file_raises_with_download_command(missing_csv):
    with pytest.raises(FileNotFoundError) as exc:
        om._load_ortholog_df()
    msg = str(exc.value)
    assert "./download_orthologues.sh" in msg
    assert "download_orthologues.ps1" in msg
    assert str(missing_csv) in msg


@pytest.mark.unit
def test_expansion_raises_instead_of_reporting_no_orthologues(missing_csv):
    with pytest.raises(FileNotFoundError):
        om.expand_genes_all_vs_all(["PPARG"], ["Homo sapiens", "Mus musculus"], return_mapping=True)


@pytest.mark.unit
def test_availability_check_reports_missing_file(missing_csv):
    error = om.ortholog_file_error()
    assert error is not None
    assert "./download_orthologues.sh" in error


@pytest.mark.unit
def test_failure_is_not_cached(missing_csv):
    with pytest.raises(FileNotFoundError):
        om._load_ortholog_df()
    missing_csv.parent.mkdir(parents=True)
    missing_csv.write_text("Human_Symbol,Mouse_Symbol\nPPARG,Pparg\n")
    df, species = om._load_ortholog_df()
    assert species == {"Human": "Human_Symbol", "Mouse": "Mouse_Symbol"}
    assert om.ortholog_file_error() is None
    genes, mapping = om.expand_genes_all_vs_all(["PPARG"], ["Homo sapiens", "Mus musculus"], return_mapping=True)
    assert set(genes) == {"PPARG", "Pparg"}
    assert mapping == {"PPARG": {"Mus musculus": ["Pparg"]}}
