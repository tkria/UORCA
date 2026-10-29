"""Tab isolation: one failed Explore tab (or page) must not stop the others.

``render_tab_safely`` imports a tab module lazily, calls its render function, and on failure
shows ``st.error`` with the exception type and message, the traceback in a collapsed
expander, and logs it. These tests run real Streamlit scripts in-process with ``AppTest``.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest

pytestmark = pytest.mark.unit

_TABS_SCRIPT = """
import streamlit as st
from uorca.gui.components.helpers.tab_guard import render_tab_safely

a, b, c = st.tabs(["Good one", "Broken", "Good two"])
with a:
    render_tab_safely("Good one", "fake_tabs.good_tab", "render_good_tab", text="first ok")
with b:
    render_tab_safely("Broken", {broken_module!r}, {broken_func!r})
with c:
    render_tab_safely("Good two", "fake_tabs.good_tab", "render_good_tab", text="second ok")
"""


@pytest.fixture
def fake_tabs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A package ``fake_tabs`` with a good tab, a tab that raises, and a module that fails to import."""
    pkg = tmp_path / "fake_tabs"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good_tab.py").write_text(textwrap.dedent("""
        import streamlit as st
        def render_good_tab(text):
            st.write(text)
    """))
    (pkg / "raising_tab.py").write_text(textwrap.dedent("""
        def render_raising_tab():
            raise ValueError("bad contrast table")
    """))
    (pkg / "unimportable_tab.py").write_text("import a_module_that_does_not_exist_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield pkg
    for name in [m for m in sys.modules if m == "fake_tabs" or m.startswith("fake_tabs.")]:
        del sys.modules[name]


def _run(script: str):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string(script, default_timeout=180).run()


def _texts(at) -> list[str]:
    return [m.value for m in at.markdown]


def _top_level_tabs(at) -> list:
    """The tabs of the first ``st.tabs`` container in the main area (``at.tabs`` also lists nested tabs)."""
    from streamlit.testing.v1.element_tree import Tab

    for node in at.main.children.values():
        kids = list(getattr(node, "children", {}).values())
        if kids and all(isinstance(k, Tab) for k in kids):
            return kids
    raise AssertionError("no st.tabs container in the main area")


@pytest.mark.parametrize(
    ("broken_module", "broken_func", "expected"),
    [
        ("fake_tabs.raising_tab", "render_raising_tab",
         "The Broken tab failed to load: ValueError: bad contrast table"),
        ("fake_tabs.unimportable_tab", "render_unimportable_tab",
         "The Broken tab failed to load: ModuleNotFoundError: No module named "
         "'a_module_that_does_not_exist_xyz'"),
    ],
    ids=["render-raises", "import-raises"],
)
def test_failed_tab_shows_error_and_other_tabs_render(
    fake_tabs, broken_module, broken_func, expected
):
    # A handler on the guard's own logger: uorca's Streamlit logging setup replaces the root
    # handlers (so pytest's caplog) the first time a tab helper logs.
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    guard_logger = logging.getLogger("uorca.gui.components.helpers.tab_guard")
    guard_logger.addHandler(handler)
    try:
        at = _run(_TABS_SCRIPT.format(broken_module=broken_module, broken_func=broken_func))
    finally:
        guard_logger.removeHandler(handler)

    assert at.exception == [], at.exception
    assert [e.value for e in at.error] == [expected]
    # The error sits inside the broken tab, not in the good ones.
    assert [e.value for e in at.tabs[1].error] == [expected]
    assert [m.value for m in at.tabs[0].markdown] == ["first ok"]
    assert [m.value for m in at.tabs[2].markdown] == ["second ok"]
    # The traceback is in a collapsed expander, and it was logged.
    expander = at.tabs[1].expander[0]
    assert expander.label == "Error details (traceback)"
    assert "Traceback (most recent call last)" in expander.code[0].value
    assert any("The Broken tab failed to load" in r.getMessage() and r.exc_info for r in records)


def test_break_switch_raises_import_error_for_named_tab_only(fake_tabs, monkeypatch):
    monkeypatch.setenv("UORCA_TEST_BREAK_TAB", "good")
    at = _run(_TABS_SCRIPT.format(broken_module="fake_tabs.raising_tab",
                                  broken_func="render_raising_tab"))
    errors = [e.value for e in at.error]
    assert len(errors) == 3  # both "good" tabs broken on purpose, plus the raising one
    assert errors[0].startswith("The Good one tab failed to load: ImportError: fake_tabs.good_tab "
                                "was broken on purpose by UORCA_TEST_BREAK_TAB")


def test_streamlit_control_flow_is_not_caught(fake_tabs):
    """st.stop() derives from BaseException and must pass through the guard."""
    (fake_tabs / "stopping_tab.py").write_text(
        "import streamlit as st\n"
        "def render_stopping_tab():\n"
        "    st.stop()\n"
    )
    at = _run(textwrap.dedent("""
        import streamlit as st
        from uorca.gui.components.helpers.tab_guard import render_tab_safely
        render_tab_safely("Stops", "fake_tabs.stopping_tab", "render_stopping_tab")
        st.write("after stop")
    """))
    assert at.error == []
    assert "after stop" not in _texts(at)


def test_tab_key():
    from uorca.gui.components.helpers.tab_guard import tab_key

    assert tab_key("uorca.gui.components.ai_assistant_tab") == "ai_assistant"
    assert tab_key("uorca.gui.pages.identify") == "identify"


def test_explore_ai_tab_broken_other_six_render(tmp_path, monkeypatch):
    """Plan 6, Plan 1 layer 2: the AI tab module fails to import; the other 6 tabs render."""
    from tests._results_fixtures import make_analysis_folder

    results = make_analysis_folder(tmp_path / "analysis")
    monkeypatch.setenv("UORCA_TEST_BREAK_TAB", "ai_assistant")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))  # Path.home() on Windows
    at = _run(textwrap.dedent(f"""
        from uorca.gui.pages.explore import _render_explorer
        _render_explorer({str(results)!r})
    """))

    assert at.exception == [], at.exception
    tabs = _top_level_tabs(at)
    names = [t.label for t in tabs]
    assert names == ["UORCA Summary", "AI Assistant", "Explore DEG Heatmap", "Plot Gene Expression",
                     "View Dataset Analyses", "View Dataset Info", "View Contrast Info"]
    ai = tabs[1]
    assert len(ai.error) == 1
    assert ai.error[0].value.startswith("The AI Assistant tab failed to load: ImportError: "
                                        "uorca.gui.components.ai_assistant_tab was broken on purpose")
    for i, tab in enumerate(tabs):
        if i == 1:
            continue
        assert tab.error == [], (tab.label, [e.value for e in tab.error])
        assert len(tab.children) > 0, f"{tab.label} rendered nothing"


def test_app_page_broken_shows_error_and_navigation_survives(tmp_path, monkeypatch):
    """uorca/gui/app.py: a page that fails to import shows its error; the navigation still works."""
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[2] / "uorca" / "gui" / "app.py"
    monkeypatch.setenv("HOME", str(tmp_path))  # never touch the real ~/.uorca
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    assert Path.home() == tmp_path
    monkeypatch.delenv("UORCA_DEFAULT_RESULTS_DIR", raising=False)
    monkeypatch.setenv("UORCA_TEST_BREAK_TAB", "home")
    at = AppTest.from_file(str(app), default_timeout=180).run()

    assert at.exception == [], at.exception
    assert [e.value for e in at.error] == [
        "The Home page failed to load: ImportError: uorca.gui.pages.home was broken on purpose "
        "by UORCA_TEST_BREAK_TAB='home' (test-only switch)"
    ]

    # Pages are imported lazily: a broken Identify page does not affect the Home page.
    monkeypatch.setenv("UORCA_TEST_BREAK_TAB", "identify")
    at = AppTest.from_file(str(app), default_timeout=180).run()
    assert at.exception == [], at.exception
    assert at.error == []
    assert any("Welcome to UORCA Explorer" in m.value for m in at.markdown)
