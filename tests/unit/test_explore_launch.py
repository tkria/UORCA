"""Regression tests for ``uorca explore`` on a fresh install.

- ``--headless`` was inverted: it made Streamlit non-headless, which then stopped on
  its first-run email prompt (exit 255 without a terminal).
- ``uorca explore <results_dir>`` set ``UORCA_DEFAULT_RESULTS_DIR`` but no page read it,
  so the README's "explore a results folder" instructions showed an empty project page.
- The Identify page turned an unreadable results CSV into "no results".
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def test_streamlit_is_always_headless() -> None:
    from uorca.explore import build_streamlit_command

    cmd = build_streamlit_command(Path("app.py"), 8501, "127.0.0.1")
    assert cmd[cmd.index("--server.headless") + 1] == "true"


@pytest.mark.parametrize(("headless", "opens"), [(True, False), (False, True)])
def test_headless_flag_controls_the_browser(headless: bool, opens: bool) -> None:
    from uorca.explore import should_open_browser

    assert should_open_browser(headless) is opens


def test_explore_page_renders_the_command_line_results_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from uorca.gui.pages import explore

    monkeypatch.setenv(explore.RESULTS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(explore, "st", MagicMock())
    rendered: list[str] = []
    monkeypatch.setattr(explore, "_render_explorer", rendered.append)

    explore.page()

    assert rendered == [str(tmp_path)]


def test_explore_page_without_results_dir_needs_a_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uorca.gui.pages import explore

    monkeypatch.delenv(explore.RESULTS_DIR_ENV, raising=False)
    fake_st = MagicMock()
    fake_st.session_state = {}
    monkeypatch.setattr(explore, "st", fake_st)
    rendered: list[str] = []
    monkeypatch.setattr(explore, "_render_explorer", rendered.append)

    explore.page()

    assert rendered == []
    fake_st.warning.assert_called_once()


def test_unreadable_identification_csv_is_shown_as_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from uorca.gui.pages import identify

    (tmp_path / "Dataset_identification_result.csv").write_text("")
    fake_st = MagicMock()
    monkeypatch.setattr(identify, "st", fake_st)

    assert identify._load_results(str(tmp_path)) is None
    fake_st.error.assert_called_once()
