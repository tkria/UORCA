"""Defect D3: the dataset download package must never contain files from outside the dataset.

With the R script component, a sample table with an empty or missing ``abundance_file`` took
``Path("")`` as the abundance file. Its parent is ``.``, the process's working directory (in the
app, the repository root with ``.env``), and that whole directory was walked into the zip. Read
errors were swallowed by ``except Exception: pass``.

Each test puts a fake ``.env`` holding MARKER in the working directory and asserts that MARKER
never appears in any package. Fake data only.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from tests._results_fixtures import make_analysis_folder, make_dataset
from uorca.gui.components import helpers
from uorca.gui.results_integration import ResultsIntegrator

pytestmark = pytest.mark.unit

MARKER = "SENTINEL_NOT_A_KEY_d3"
ALL = {"metadata": True, "abundance": True, "qc_plots": True, "deg_analysis": True, "r_script": True}
SAMPLES = ("Sample1", "Sample2", "Sample3", "Sample4")
GROUPS = ("Control", "Control", "Treated", "Treated")


@pytest.fixture
def server_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "server_cwd"
    cwd.mkdir()
    (cwd / ".env").write_text(f"{MARKER}=1\n")
    monkeypatch.chdir(cwd)
    return cwd


@pytest.fixture
def st_errors(monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(helpers.st, "error", lambda msg, *a, **k: shown.append(str(msg)))
    return shown


def _write_abundance(ds: Path, samples=SAMPLES) -> None:
    for s in samples:
        d = ds / "abundance" / s
        d.mkdir(parents=True, exist_ok=True)
        (d / "abundance.tsv").write_text(f"target_id\tlength\teff_length\test_counts\ttpm\nT1\t100\t90\t5\t1.0\n# {s}\n")
        (d / "run_info.json").write_text("{}")


def _write_samples(ds: Path, paths) -> None:
    lines = [",sample,group,abundance_file"]
    for i, (s, g, p) in enumerate(zip(SAMPLES, GROUPS, paths)):
        lines.append(f"{i},{s},{g},{p}")
    (ds / "metadata" / "edger_analysis_samples.csv").write_text("\n".join(lines) + "\n")


def _bundle(results: Path, dataset_id: str) -> tuple[zipfile.ZipFile, io.BytesIO]:
    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, "w")
    ri = ResultsIntegrator(str(results))
    helpers.add_rscript_bundle_to_zip(zf, ri, str(results), dataset_id, dataset_id, "pkg")
    return zf, buf


def _assert_no_marker(data: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            assert not name.endswith(".env"), name
            assert MARKER.encode() not in zf.read(name), name


def test_package_never_contains_files_from_the_working_directory(tmp_path, server_cwd, st_errors):
    """The crawler's reproduction: no abundance/ folder and no abundance_file column."""
    results = make_analysis_folder(tmp_path / "results", partial=())
    ri = ResultsIntegrator(str(results))
    ri.load_data()

    data = helpers.create_dataset_download_package(ri, str(results), "GSE900001", "GSE900001", dict(ALL))

    if data is not None:
        _assert_no_marker(data)
    # The R script cannot be built without abundance files: that must be shown, not skipped.
    assert data is None
    assert st_errors and "abundance_file" in st_errors[0], st_errors


@pytest.mark.parametrize("value", ["", "nan"])
def test_empty_abundance_path_raises_even_with_abundance_folder(tmp_path, server_cwd, value):
    ds = make_dataset(tmp_path, "GSE1")
    _write_abundance(ds)
    _write_samples(ds, [f"abundance/{s}/abundance.tsv" for s in SAMPLES[:3]] + [value])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        with pytest.raises(ValueError, match="Sample4.*empty abundance_file"):
            helpers.add_rscript_bundle_to_zip(zf, ResultsIntegrator(str(tmp_path)), str(tmp_path), "GSE1", "GSE1", "pkg")
    _assert_no_marker(buf.getvalue())


def test_empty_abundance_path_without_folder_does_not_walk_cwd(tmp_path, server_cwd):
    ds = make_dataset(tmp_path, "GSE1")
    _write_samples(ds, [""] * 4)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        with pytest.raises(ValueError, match="empty abundance_file"):
            helpers.add_rscript_bundle_to_zip(zf, ResultsIntegrator(str(tmp_path)), str(tmp_path), "GSE1", "GSE1", "pkg")
    _assert_no_marker(buf.getvalue())


def test_path_outside_dataset_is_rejected(tmp_path, server_cwd):
    ds = make_dataset(tmp_path, "GSE1")
    _write_abundance(ds, SAMPLES[:3])
    outside = tmp_path / "elsewhere" / "Sample4"
    outside.mkdir(parents=True)
    (outside / "abundance.tsv").write_text(MARKER)
    _write_samples(ds, [f"abundance/{s}/abundance.tsv" for s in SAMPLES[:3]] + [str(outside / "abundance.tsv")])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        with pytest.raises(FileNotFoundError, match="Sample4"):
            helpers.add_rscript_bundle_to_zip(zf, ResultsIntegrator(str(tmp_path)), str(tmp_path), "GSE1", "GSE1", "pkg")
    _assert_no_marker(buf.getvalue())


def test_traversal_and_symlink_out_of_dataset_are_rejected(tmp_path, server_cwd):
    ds = make_dataset(tmp_path, "GSE1")
    _write_abundance(ds, SAMPLES[:3])
    (ds / "abundance" / "Sample4").mkdir()
    os.symlink(server_cwd / ".env", ds / "abundance" / "Sample4" / "abundance.tsv")
    _write_samples(ds, [f"abundance/{s}/abundance.tsv" for s in SAMPLES[:3]] + ["abundance/Sample4/abundance.tsv"])
    with zipfile.ZipFile(io.BytesIO(), "w") as zf:
        with pytest.raises(FileNotFoundError, match="Sample4"):
            helpers.add_rscript_bundle_to_zip(zf, ResultsIntegrator(str(tmp_path)), str(tmp_path), "GSE1", "GSE1", "pkg")

    _write_samples(ds, [f"abundance/{s}/abundance.tsv" for s in SAMPLES[:3]] + ["../../server_cwd/abundance.tsv"])
    (server_cwd / "abundance.tsv").write_text(MARKER)
    with zipfile.ZipFile(io.BytesIO(), "w") as zf:
        with pytest.raises(FileNotFoundError, match="Sample4"):
            helpers.add_rscript_bundle_to_zip(zf, ResultsIntegrator(str(tmp_path)), str(tmp_path), "GSE1", "GSE1", "pkg")


def test_symlink_inside_abundance_folder_is_not_copied_by_the_full_package(tmp_path, server_cwd, st_errors):
    ds = make_dataset(tmp_path, "GSE1")
    _write_abundance(ds)
    os.symlink(server_cwd / ".env", ds / "abundance" / "Sample1" / "extra.txt")
    ri = ResultsIntegrator(str(tmp_path))
    ri.load_data()
    data = helpers.create_dataset_download_package(
        ri, str(tmp_path), "GSE1", "GSE1",
        {"metadata": False, "abundance": True, "qc_plots": False, "deg_analysis": False, "r_script": False})
    if data is not None:
        _assert_no_marker(data)
    assert data is None and st_errors and "outside" in st_errors[0], st_errors


def test_container_paths_map_to_the_datasets_own_abundance_files(tmp_path, server_cwd, st_errors):
    """Real tables store container paths: /UORCA_results/<GSE>/abundance/<run>/abundance.tsv."""
    ds = make_dataset(tmp_path, "GSE1")
    _write_abundance(ds)
    _write_samples(ds, [f"/UORCA_results/GSE1/abundance/{s}/abundance.tsv" for s in SAMPLES])
    ri = ResultsIntegrator(str(tmp_path))
    ri.load_data()

    data = helpers.create_dataset_download_package(ri, str(tmp_path), "GSE1", "GSE1", dict(ALL))

    assert data is not None, st_errors
    _assert_no_marker(data)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert len(names) == len(set(names)), "duplicate entries"
        for s in SAMPLES:
            assert f"GSE1_analysis_package/abundance/{s}/abundance.tsv" in names
        assert "GSE1_analysis_package/RNAseq.R" in names
        csv = zf.read("GSE1_analysis_package/metadata/edger_analysis_samples.csv").decode()
    for s in SAMPLES:
        assert f"abundance/{s}/abundance.tsv" in csv
    assert "UORCA_results" not in csv and "UNKNOWN" not in csv


def test_relative_paths_copy_only_the_abundance_files(tmp_path, server_cwd):
    ds = make_dataset(tmp_path, "GSE1")
    _write_abundance(ds)
    _write_samples(ds, [f"abundance/{s}/abundance.tsv" for s in SAMPLES])
    zf, buf = _bundle(tmp_path, "GSE1")
    zf.close()
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
        abund = sorted(n for n in z.namelist() if "/abundance/" in n)
    assert abund == [f"pkg/abundance/{s}/abundance.tsv" for s in SAMPLES]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_read_error_surfaces(tmp_path, server_cwd):
    ds = make_dataset(tmp_path, "GSE1")
    _write_abundance(ds)
    _write_samples(ds, [f"abundance/{s}/abundance.tsv" for s in SAMPLES])
    locked = ds / "abundance" / "Sample2" / "abundance.tsv"
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            _bundle(tmp_path, "GSE1")
    finally:
        locked.chmod(0o644)


@pytest.mark.parametrize("dataset_id", ["", ".", "..", "../server_cwd"])
def test_dataset_id_must_name_a_folder_inside_results(tmp_path, server_cwd, st_errors, dataset_id):
    results = tmp_path / "results"
    make_dataset(results, "GSE1")
    _write_abundance(results / "GSE1")
    ri = ResultsIntegrator(str(results))
    data = helpers.create_dataset_download_package(
        ri, str(results), dataset_id, "X",
        {"metadata": True, "abundance": True, "qc_plots": False, "deg_analysis": False, "r_script": False})
    assert data is None
    assert st_errors and "dataset" in st_errors[0].lower(), st_errors
