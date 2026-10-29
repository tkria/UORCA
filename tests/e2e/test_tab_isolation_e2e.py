"""Browser test: a broken Explore tab does not stop the other six (plan 1, plan 6 section 5).

``UORCA_TEST_BREAK_TAB=ai_assistant`` makes the tab guard raise ``ImportError`` before it
imports the AI Assistant tab (see uorca/gui/components/helpers/tab_guard.py). The app runs
with an isolated HOME and fake results.
"""

from __future__ import annotations

import pytest

from tests._results_fixtures import FAKE_QUESTION, make_arm_folder
from tests.e2e._helpers import assert_tab_error, assert_tab_ok

pytestmark = pytest.mark.e2e

# Each healthy tab and content that proves it rendered from the fixture.
HEALTHY_TABS = {
    "UORCA Summary": [FAKE_QUESTION, "Total datasets: 2"],
    "Explore DEG Heatmap": ["Explore DEG Heatmap", "Construct a heatmap of selected genes"],
    "Plot Gene Expression": ["Expression Plots"],
    "View Dataset Analyses": ["Differential Expression Results", "Fake contrast Treated_vs_Control"],
    "View Dataset Info": ["GSE900001"],
    "View Contrast Info": ["Treated_vs_Control"],
}


def test_broken_ai_tab_leaves_other_tabs_working(page, start_app, tmp_path, artifacts_dir):
    arm = make_arm_folder(tmp_path / "isolation_arm")
    app = start_app({"UORCA_DEFAULT_RESULTS_DIR": str(arm), "UORCA_TEST_BREAK_TAB": "ai_assistant"})
    page.goto(app.url + "/")
    page.get_by_text("Select pipeline run").wait_for(timeout=90_000)

    message = assert_tab_error(page, "AI Assistant",
                               "The AI Assistant tab failed to load: ImportError")
    assert "uorca.gui.components.ai_assistant_tab was broken on purpose" in message
    # The traceback is offered, collapsed.
    page.get_by_text("Error details (traceback)").first.wait_for()
    page.screenshot(path=str(artifacts_dir / "ai_tab_broken.png"), full_page=True)

    for tab, texts in HEALTHY_TABS.items():
        assert_tab_ok(page, tab, expect_text=texts)
    page.screenshot(path=str(artifacts_dir / "contrast_info_after_broken_ai.png"), full_page=True)
