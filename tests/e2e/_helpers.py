"""Shared Playwright checks for the browser tests (plan 6, section 4.3).

Each plan adds its own expected content through ``expect_text``; the helper fails on any
Streamlit exception or ``st.error`` box in the tab it opened.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

EXCEPTION = '[data-testid="stException"]'
ERROR_ALERT = '[data-testid="stAlertContentError"]'


def wait_idle(page) -> None:
    """Wait for the current Streamlit rerun to finish, so the page text is complete."""
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden")
    page.wait_for_timeout(300)


def visible_tab_panel(page):
    """The panel of the selected tab in the first ``st.tabs`` container."""
    return page.locator('[data-testid="stTabs"] [role="tabpanel"]:visible').first


def open_tab(page, tab_name: str):
    """Click ``tab_name``, wait for the rerun, and return its panel."""
    page.get_by_role("tab", name=tab_name, exact=True).click()
    wait_idle(page)
    return visible_tab_panel(page)


def assert_tab_ok(
    page,
    tab_name: str,
    expect_text: Union[str, Iterable[str], None] = None,
    *,
    timeout_ms: int = 30_000,
) -> None:
    """Open ``tab_name``, wait for each ``expect_text``, and fail on an exception or error box.

    Text is waited for with ``state="attached"``: dataframe cells are drawn on a canvas, and
    their text is in the DOM but not "visible" to Playwright.
    """
    panel = open_tab(page, tab_name)
    texts = [expect_text] if isinstance(expect_text, str) else list(expect_text or [])
    for text in texts:
        panel.get_by_text(text, exact=False).first.wait_for(state="attached", timeout=timeout_ms)
    _assert_none(panel, EXCEPTION, tab_name, "a Streamlit exception")
    _assert_none(panel, ERROR_ALERT, tab_name, "an error box")


def assert_tab_error(page, tab_name: str, expect_text: str, *, timeout_ms: int = 30_000) -> str:
    """Open ``tab_name`` and wait for an error box that contains ``expect_text``; return its text."""
    panel = open_tab(page, tab_name)
    box = panel.locator(ERROR_ALERT).filter(has_text=expect_text).first
    box.wait_for(timeout=timeout_ms)
    return box.inner_text()


def _assert_none(panel, selector: str, tab_name: str, what: str) -> None:
    found = panel.locator(selector)
    count = found.count()
    detail: Optional[str] = found.first.inner_text() if count else None
    assert count == 0, f"tab {tab_name!r} shows {what}: {detail}"
