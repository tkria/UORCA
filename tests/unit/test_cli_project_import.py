"""Tests for `uorca project import`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._results_fixtures import make_analysis_folder, make_arm_folder
from uorca.gui.project.manager import ProjectManager

pytestmark = pytest.mark.unit


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HOME so ProjectManager() never touches the real ~/.uorca."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    assert Path.home() == h
    return h


def _run_cli(monkeypatch, *argv: str) -> int:
    from uorca import cli

    monkeypatch.setattr(sys, "argv", ["uorca", *argv])
    try:
        cli.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_import_with_create(home, tmp_path, monkeypatch, capsys):
    arm = make_arm_folder(tmp_path / "arm_openai")
    code = _run_cli(monkeypatch, "project", "import", str(arm), "--project", "HPC copy", "--create",
                    "--note", "copied from HPC")
    out = capsys.readouterr().out
    assert code == 0, out
    assert "2 complete, 1 partial, 0 failed" in out
    mgr = ProjectManager()
    assert mgr.projects_dir == home / ".uorca" / "projects"
    project = mgr.load_project("hpc-copy")
    assert [r.type for r in project.runs] == ["identification", "pipeline"]
    pipe = project.runs[1]
    assert pipe.status == "completed"
    assert pipe.output_path == str((arm / "analysis").resolve())
    assert pipe.note == "copied from HPC"
    assert pipe.input_from == project.runs[0].id


def test_import_into_missing_project_without_create_fails(home, tmp_path, monkeypatch, capsys):
    root = make_analysis_folder(tmp_path / "res")
    code = _run_cli(monkeypatch, "project", "import", str(root), "--project", "Nope")
    out = capsys.readouterr().out
    assert code == 1
    assert "does not exist" in out
    assert not ProjectManager().project_exists("Nope")


def test_failed_create_import_leaves_no_project(home, tmp_path, monkeypatch, capsys):
    root = make_analysis_folder(tmp_path / "res", complete=(), partial=("GSE900003",))
    code = _run_cli(monkeypatch, "project", "import", str(root), "--project", "Empty", "--create")
    out = capsys.readouterr().out
    assert code == 1
    assert "No complete dataset" in out
    assert not ProjectManager().project_exists("Empty")


def test_import_into_existing_project(home, tmp_path, monkeypatch, capsys):
    ProjectManager().create_project("Existing")
    root = make_analysis_folder(tmp_path / "res")
    assert _run_cli(monkeypatch, "project", "import", str(root), "--project", "Existing") == 0
    assert len(ProjectManager().load_project("existing").runs) == 1


def test_duplicate_import_refused(home, tmp_path, monkeypatch, capsys):
    root = make_analysis_folder(tmp_path / "res")
    assert _run_cli(monkeypatch, "project", "import", str(root), "--project", "P", "--create") == 0
    capsys.readouterr()
    code = _run_cli(monkeypatch, "project", "import", str(root), "--project", "P")
    out = capsys.readouterr().out
    assert code == 1
    assert "already in project 'P'" in out
    assert len(ProjectManager().load_project("p").runs) == 1


def test_path_without_analysis_fails(home, tmp_path, monkeypatch, capsys):
    (tmp_path / "empty").mkdir()
    code = _run_cli(monkeypatch, "project", "import", str(tmp_path / "empty"), "--project", "P", "--create")
    assert code == 1
    assert "No UORCA analysis found" in capsys.readouterr().out
    assert not ProjectManager().project_exists("P")


def test_project_without_action_prints_help(home, monkeypatch, capsys):
    assert _run_cli(monkeypatch, "project") == 1
    assert "import" in capsys.readouterr().out
