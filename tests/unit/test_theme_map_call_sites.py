"""The two call sites of ``build_theme_map`` that are not the terminal (spec section 8).

1. ``dataset_identification.py``, the last block of a finished identification run.
2. ``uorca/gui/pages/identify.py``, which had to be refactored first (spec 9.1).

The identification call site is the **one** place in this feature that does not re-raise.
Spec section 7 says every failure raises and decision 4 says the theme map must never
cost a completed 60-minute run; both cannot hold at the same call site. The approved
resolution is that this one logs the traceback at ERROR level and records ``FAILED.txt``
so the panel can show it loudly, while the run itself still exits clean. These tests pin
that behaviour so nobody "fixes" it in either direction.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Call site 1 — dataset_identification.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_identification_records_a_failed_theme_map_instead_of_failing_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from uorca.identification import theme_map
    from uorca.identification.dataset_identification import (
        build_theme_map_for_finished_run,
    )

    def explode(run_dir, **kwargs):
        raise RuntimeError("clustering was degenerate")

    monkeypatch.setattr(theme_map, "build_theme_map", explode)

    with caplog.at_level(logging.ERROR):
        built = build_theme_map_for_finished_run(tmp_path)

    assert built is False
    recorded = theme_map.read_failure(tmp_path)
    assert recorded is not None
    assert "clustering was degenerate" in recorded
    # Logged loudly, with the traceback, not swallowed.
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "theme map" in messages.lower()
    assert "Traceback" in messages


@pytest.mark.unit
def test_identification_builds_the_map_on_the_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from uorca.identification import theme_map
    from uorca.identification.dataset_identification import (
        build_theme_map_for_finished_run,
    )

    seen: list[Path] = []

    def fake_build(run_dir, **kwargs):
        seen.append(Path(run_dir))
        return None

    monkeypatch.setattr(theme_map, "build_theme_map", fake_build)

    assert build_theme_map_for_finished_run(tmp_path) is True
    assert seen == [tmp_path]
    assert theme_map.read_failure(tmp_path) is None


@pytest.mark.unit
def test_identification_main_calls_the_theme_map_last_inside_the_try() -> None:
    """The call is the last block of ``main``, so it can never cost a finished run."""
    from uorca.identification import dataset_identification

    source = inspect.getsource(dataset_identification.main)
    assert "build_theme_map_for_finished_run" in source
    metadata_write = source.index("Saved identification metadata")
    theme_call = source.index("build_theme_map_for_finished_run(")
    assert theme_call > metadata_write


# ---------------------------------------------------------------------------
# Call site 2 — the Identify page (spec 9.1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_results_table_takes_the_selected_run_as_an_argument() -> None:
    from uorca.gui.pages import identify

    parameters = list(inspect.signature(identify._render_results_table).parameters)
    assert parameters == ["project", "selected_run"]


@pytest.mark.unit
def test_the_run_selector_lives_outside_the_results_table() -> None:
    """Spec 9.1: the table has three early returns, so the selector must move up."""
    from uorca.gui.pages import identify

    assert hasattr(identify, "_render_run_selector")
    table_source = inspect.getsource(identify._render_results_table)
    assert "selectbox" not in table_source
    assert "selectbox" in inspect.getsource(identify._render_run_selector)


@pytest.mark.unit
def test_the_page_draws_the_theme_map_below_the_results_table() -> None:
    from uorca.gui.pages import identify

    source = inspect.getsource(identify.page)
    assert "_render_run_selector(" in source
    assert "render_theme_map_panel(" in source
    assert source.index("_render_results_table(") < source.index(
        "render_theme_map_panel("
    )
