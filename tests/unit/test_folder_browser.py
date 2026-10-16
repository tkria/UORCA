"""Unit tests for the in-app folder browser's path logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from uorca.gui.components.folder_browser import list_subfolders, start_folder

pytestmark = pytest.mark.unit


def test_list_subfolders_sorts_case_insensitively_and_skips_files_and_hidden(tmp_path):
    for name in ["beta", "Alpha", ".hidden", "gamma"]:
        (tmp_path / name).mkdir()
    (tmp_path / "notes.txt").write_text("x")

    assert [p.name for p in list_subfolders(tmp_path)] == ["Alpha", "beta", "gamma"]


def test_list_subfolders_raises_for_missing_folder(tmp_path):
    with pytest.raises(OSError):
        list_subfolders(tmp_path / "missing")


def test_start_folder_uses_typed_existing_folder(tmp_path):
    assert start_folder(f"  {tmp_path}  ") == tmp_path.resolve()


@pytest.mark.parametrize("typed", ["", "   ", "/no/such/folder"])
def test_start_folder_falls_back_to_home(typed):
    assert start_folder(typed) == Path.home()


def test_start_folder_falls_back_to_home_for_a_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    assert start_folder(str(f)) == Path.home()


# --- macOS Finder dialog (osascript is mocked; the dialog itself needs a person) ---

import subprocess

from uorca.gui.components import folder_browser
from uorca.gui.components.folder_browser import (
    FolderDialogError,
    choose_folder_with_finder,
    finder_dialog_available,
)


@pytest.mark.parametrize(
    ("platform", "ip", "expected"),
    [
        ("darwin", None, True),
        ("darwin", "127.0.0.1", True),
        ("darwin", "::1", True),
        ("darwin", "10.0.0.5", False),  # browser on another machine
        ("linux", None, False),
        ("win32", None, False),
    ],
)
def test_finder_dialog_available(platform, ip, expected):
    assert finder_dialog_available(platform, ip) is expected


def _fake_run(monkeypatch, returncode, stdout="", stderr=""):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    monkeypatch.setattr(folder_browser.subprocess, "run", run)
    return calls


def test_choose_folder_returns_the_chosen_path_and_passes_start_as_argument(monkeypatch, tmp_path):
    start = tmp_path / 'odd "name"'
    calls = _fake_run(monkeypatch, 0, stdout="/Users/me/results/run_1/\n")

    assert choose_folder_with_finder(start) == Path("/Users/me/results/run_1")
    args = calls[0]
    assert args[0] == "osascript"
    assert args[-1] == str(start)  # passed as argv, not spliced into the script
    assert not any(str(start) in a for a in args[:-1])


def test_choose_folder_returns_none_when_cancelled(monkeypatch, tmp_path):
    _fake_run(monkeypatch, 1, stderr="execution error: User canceled. (-128)")
    assert choose_folder_with_finder(tmp_path) is None


def test_choose_folder_raises_on_other_failures(monkeypatch, tmp_path):
    _fake_run(monkeypatch, 1, stderr="execution error: Not authorised. (-1743)")
    with pytest.raises(FolderDialogError, match="-1743"):
        choose_folder_with_finder(tmp_path)


def test_choose_folder_raises_on_empty_output(monkeypatch, tmp_path):
    _fake_run(monkeypatch, 0, stdout="\n")
    with pytest.raises(FolderDialogError):
        choose_folder_with_finder(tmp_path)


def test_choose_folder_raises_when_osascript_is_missing(monkeypatch, tmp_path):
    def run(*_a, **_k):
        raise FileNotFoundError("osascript")

    monkeypatch.setattr(folder_browser.subprocess, "run", run)
    with pytest.raises(FolderDialogError):
        choose_folder_with_finder(tmp_path)
