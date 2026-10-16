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
