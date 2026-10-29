"""GUI pages check the AI provider before they offer an action that needs it.

* Identify page: a missing key for the selected provider shows ``st.error`` naming the
  variable, and "Run Identification" is disabled.
* Theme-map panel: under any provider other than OpenAI, the page says the theme map
  needs OpenAI and offers no build (no spend checkbox, no Build or Rebuild button).
"""

from __future__ import annotations

import pytest

import uorca.ai_provider as ai_provider

pytestmark = pytest.mark.unit

_ENV = ("UORCA_AI_PROVIDER", "UORCA_OPENAI_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", tmp_path / "config.yaml")
    ai_provider._load_ai_config.cache_clear()
    yield
    ai_provider._load_ai_config.cache_clear()


# ---------------------------------------------------------------------------
# Identify page
# ---------------------------------------------------------------------------

def _identify_form() -> None:
    from uorca.gui.pages.identify import _render_left_panel
    from uorca.gui.project.models import Project

    _render_left_panel(Project(name="contract"), None)  # type: ignore[arg-type]


def _run_button(at):
    return next(b for b in at.button if b.label == "Run Identification")


def test_identify_missing_key_is_named_and_run_disabled(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("UORCA_AI_PROVIDER", "anthropic:claude-sonnet-4-5")
    at = AppTest.from_function(_identify_form, default_timeout=10).run()
    assert at.exception == []
    assert any("ANTHROPIC_API_KEY" in e.value for e in at.error)
    assert _run_button(at).disabled


def test_identify_invalid_setting_is_shown_and_run_disabled(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("UORCA_AI_PROVIDER", "typo")
    at = AppTest.from_function(_identify_form, default_timeout=10).run()
    assert any("Unknown AI provider 'typo'" in e.value for e in at.error)
    assert _run_button(at).disabled


def test_identify_with_key_has_no_error_and_run_enabled(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("UORCA_AI_PROVIDER", "anthropic:claude-sonnet-4-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    at = AppTest.from_function(_identify_form, default_timeout=10).run()
    assert not at.error
    assert not _run_button(at).disabled


# ---------------------------------------------------------------------------
# Theme-map panel
# ---------------------------------------------------------------------------

_RUN_DIR_KEY = "UORCA_TEST_RUN_DIR"


def _panel() -> None:
    import os
    from types import SimpleNamespace

    from uorca.gui.components.theme_map_panel import render_theme_map_panel

    render_theme_map_panel(SimpleNamespace(id="run1", output_path=os.environ["UORCA_TEST_RUN_DIR"]))


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    from uorca.gui.components.theme_map_panel import RESULTS_CSV_NAME

    run = tmp_path / "run1"
    run.mkdir()
    rows = ["GEO_Accession,Title,Summary,Valid"] + [
        f"GSE{i},Title {i},Summary {i},Yes" for i in range(60)
    ]
    (run / RESULTS_CSV_NAME).write_text("\n".join(rows) + "\n")
    monkeypatch.setenv(_RUN_DIR_KEY, str(run))
    return run


def _build_controls(at) -> list[str]:
    labels = [b.label for b in at.button] + [c.label for c in at.checkbox]
    return [l for l in labels if "Build" in l or "Rebuild" in l or "spends" in l]


@pytest.mark.parametrize(
    "setting", ["anthropic:claude-sonnet-4-5", "bedrock", "google-gla:gemini-2.5-flash"]
)
def test_theme_map_not_offered_under_another_provider(monkeypatch, run_dir, setting):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("UORCA_AI_PROVIDER", setting)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    at = AppTest.from_function(_panel, default_timeout=10).run()
    assert at.exception == []
    assert any("OpenAI" in w.value and "cannot be built" in w.value
               for w in at.warning)
    assert not at.code, "the reason must be wrapped text, not a code box"
    assert _build_controls(at) == []


def test_theme_map_rebuild_not_offered_after_a_failure(monkeypatch, run_dir):
    from streamlit.testing.v1 import AppTest

    import uorca.gui.components.theme_map_panel as panel

    monkeypatch.setattr(panel, "read_failure", lambda _d: "earlier failure")
    monkeypatch.setenv("UORCA_AI_PROVIDER", "anthropic:claude-sonnet-4-5")
    at = AppTest.from_function(_panel, default_timeout=10).run()
    assert any("cannot be built" in w.value for w in at.warning)
    assert _build_controls(at) == []


def test_theme_map_build_offered_under_openai(monkeypatch, run_dir):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    at = AppTest.from_function(_panel, default_timeout=10).run()
    assert at.exception == []
    assert not any("cannot be built" in w.value for w in at.warning)
    assert "Build theme map" in [b.label for b in at.button]
