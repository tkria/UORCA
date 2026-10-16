"""`uorca explore <path>` validates the path with the importer's rules before starting."""

from __future__ import annotations

import pytest

from tests._results_fixtures import make_analysis_folder, make_arm_folder, make_dataset
from uorca.explore import check_results_dir

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("layout", ["analysis", "arm", "dataset"])
def test_accepts_every_importable_layout(tmp_path, capsys, layout):
    if layout == "analysis":
        path = make_analysis_folder(tmp_path / "res")
    elif layout == "arm":
        path = make_arm_folder(tmp_path / "arm")  # rejected by the old */metadata check
    else:
        path = make_dataset(tmp_path, "GSE900001")
    assert check_results_dir(path) == path.resolve()
    assert "complete" in capsys.readouterr().out


def test_rejects_folder_without_analysis(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as exc:
        check_results_dir(tmp_path / "empty")
    assert exc.value.code == 1
    assert "No UORCA analysis found" in capsys.readouterr().out


def test_rejects_folder_with_only_partial_datasets(tmp_path, capsys):
    root = make_analysis_folder(tmp_path / "res", complete=(), partial=("GSE900003",))
    with pytest.raises(SystemExit) as exc:
        check_results_dir(root)
    assert exc.value.code == 1
    assert "no complete dataset" in capsys.readouterr().out


def test_rejects_folder_whose_datasets_all_have_null_unique_groups(tmp_path, capsys):
    root = tmp_path / "res"
    make_dataset(root, "GSE900001", null_groups=True)
    with pytest.raises(SystemExit) as exc:
        check_results_dir(root)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "no complete dataset" in out
    assert "unique_groups null" in out
