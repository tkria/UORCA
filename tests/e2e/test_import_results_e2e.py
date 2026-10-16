"""Browser tests: import results produced outside the GUI, then explore them.

Every test runs the app with an isolated HOME (never the real ~/.uorca) and fake fixtures.
Navigation starts at "/" and uses the sidebar: on a fresh session, deep links such as
/explore resolve to the legacy pages/ directory and render blank (pre-existing behaviour).
"""

from __future__ import annotations

import sys

import pytest

from tests._results_fixtures import FAKE_QUESTION, make_arm_folder
from uorca.gui.project.manager import ProjectManager

pytestmark = pytest.mark.e2e

TABS = [
    "UORCA Summary", "AI Assistant", "Explore DEG Heatmap", "Plot Gene Expression",
    "View Dataset Analyses", "View Dataset Info", "View Contrast Info",
]


def _manager(home):
    return ProjectManager(projects_dir=home / ".uorca" / "projects",
                          config_path=home / ".uorca" / "config.yaml")


def _open(page, app, ready_text: str = "UORCA Explorer") -> None:
    page.goto(app.url + "/")
    page.get_by_text(ready_text, exact=False).first.wait_for(timeout=90_000)


def _nav(page, name: str) -> None:
    page.get_by_role("link", name=name).click()


def _assert_no_exception(page) -> None:
    exc = page.locator('[data-testid="stException"]')
    assert exc.count() == 0, exc.first.inner_text()


def _assert_no_error_alert(page) -> None:
    err = page.locator('[data-testid="stAlertContentError"]')
    assert err.count() == 0, err.first.inner_text()


def _tab_panel(page):
    return page.locator('[data-testid="stTabs"] [role="tabpanel"]:visible').first


def _assert_explorer_renders(page, project_name: str) -> None:
    """Every tab renders real content from the fixture (2 complete datasets, 1 contrast each)."""
    page.get_by_text("Select pipeline run").wait_for(timeout=60_000)
    page.get_by_text("— imported").first.wait_for()
    sidebar = page.locator('[data-testid="stSidebar"]')
    assert project_name in sidebar.inner_text()
    for tab in TABS:
        page.get_by_role("tab", name=tab).click()
        page.wait_for_timeout(1500)
        _assert_no_exception(page)
        _assert_no_error_alert(page)

    page.get_by_role("tab", name="UORCA Summary").click()
    page.get_by_text(FAKE_QUESTION).first.wait_for()
    # Only the 2 complete datasets count; the partial one (3 contrasts) must not.
    page.get_by_text("Total datasets: 2").first.wait_for()
    page.get_by_text("Total contrasts: 2").first.wait_for()
    page.get_by_text("Homo sapiens: 2").first.wait_for()

    page.get_by_role("tab", name="View Dataset Info").click()
    panel = _tab_panel(page)
    panel.get_by_text("GSE900001").first.wait_for(state="attached")  # dataframe cell (canvas grid)
    assert "No dataset" not in panel.inner_text()

    page.get_by_role("tab", name="View Contrast Info").click()
    panel = _tab_panel(page)
    panel.get_by_text("Treated_vs_Control").first.wait_for(state="attached")
    text = panel.inner_text()
    assert "No contrast information available" not in text
    assert "No contrasts match" not in text

    page.get_by_role("tab", name="View Dataset Analyses").click()
    panel = _tab_panel(page)
    panel.get_by_text("Differential Expression Results").first.wait_for()
    panel.get_by_text("Fake contrast Treated_vs_Control").first.wait_for(state="attached")
    assert "No contrast-specific plots found" not in panel.inner_text()


def _wait_idle(page) -> None:
    """Wait for the current rerun to finish, so the page text is complete."""
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden")
    page.wait_for_timeout(300)


def _type_path(page, path) -> None:
    box = page.get_by_label("Results folder")
    box.fill(str(path))
    box.press("Enter")
    page.get_by_text("Preview", exact=True).wait_for()
    _wait_idle(page)


def test_existing_pages_still_load(page, start_app):
    app = start_app()
    _open(page, app, "Welcome to UORCA Explorer")
    for name, marker in [("Project Setup", "Create New Project"),
                         ("Identify", "Identify Datasets"),
                         ("Run", "Run Pipeline"),
                         ("Explore", "Import existing results"),
                         ("Home", "Welcome to UORCA Explorer")]:
        _nav(page, name)
        page.get_by_text(marker).first.wait_for()
        _assert_no_exception(page)


def test_gui_import_into_new_project_from_explore(page, start_app, home, tmp_path):
    arm = make_arm_folder(tmp_path / "arm_fake")
    app = start_app()
    _open(page, app, "Welcome to UORCA Explorer")
    _nav(page, "Explore")
    page.get_by_text("Import existing results").first.wait_for()
    _type_path(page, arm)

    main = page.locator('[data-testid="stMain"]')
    text = main.inner_text()
    assert "Run folder (analysis/ and identify/)" in text
    assert "Datasets found\n\n3" in text or "Datasets found\n3" in text
    assert "Complete\n\n2" in text or "Complete\n2" in text
    assert "Partial\n\n1" in text or "Partial\n1" in text
    assert FAKE_QUESTION in text
    assert page.get_by_label("New project name").input_value() == "arm_fake"

    page.get_by_role("button", name="Import results").click()
    page.get_by_text("Imported 2 complete dataset(s)").wait_for(timeout=60_000)
    _assert_explorer_renders(page, "arm_fake")

    project = _manager(home).load_project("arm-fake")
    assert [r.type for r in project.runs] == ["identification", "pipeline"]
    assert project.runs[1].output_path == str((arm / "analysis").resolve())

    # The other pages still work with an imported project active.
    id_run, pipe_run = project.runs
    _nav(page, "Identify")
    page.get_by_text("Identify Datasets").first.wait_for()
    page.get_by_text("Fake title one").first.wait_for(state="attached")  # dataframe cell (canvas grid)
    _assert_no_exception(page)
    _nav(page, "Run")
    page.get_by_text("Pipeline Monitor").wait_for()
    page.get_by_text(pipe_run.id).first.wait_for()
    _assert_no_exception(page)
    _nav(page, "Project Setup")
    page.get_by_text("Create New Project").first.wait_for()
    _assert_no_exception(page)
    _nav(page, "Home")
    page.get_by_text("Recent Activity").wait_for()
    page.get_by_text(id_run.id).first.wait_for()
    _assert_no_exception(page)


def test_gui_import_into_active_project_from_project_setup(page, start_app, home, tmp_path):
    _manager(home).create_project("Existing study", description="made before import")
    arm = make_arm_folder(tmp_path / "copied_from_hpc", identify=False)
    app = start_app()
    _open(page, app, "Projects")
    _nav(page, "Project Setup")
    page.get_by_text("Import Existing Results").first.click()
    page.get_by_role("heading", name="Import Existing Results").wait_for()
    _type_path(page, arm)
    page.get_by_text("Active project (Existing study)").click()
    page.get_by_role("button", name="Import results").click()
    page.get_by_text("Imported 2 complete dataset(s)").wait_for(timeout=60_000)
    _assert_explorer_renders(page, "Existing study")

    mgr = _manager(home)
    project = mgr.load_project("existing-study")
    assert project.description == "made before import"
    assert [r.type for r in project.runs] == ["pipeline"]
    assert len(mgr.list_projects()) == 1


def test_duplicate_gui_import_is_refused(page, start_app, home, tmp_path):
    arm = make_arm_folder(tmp_path / "dup_arm")
    mgr = _manager(home)
    from uorca.gui.project.importer import import_into_named_project, inspect_results
    import_into_named_project(mgr, inspect_results(arm), "dup_arm", create=True)

    app = start_app()
    _open(page, app, "Projects")
    page.get_by_role("button", name="Select").first.click()
    page.wait_for_timeout(1500)
    _nav(page, "Explore")
    page.get_by_text("Select pipeline run").wait_for(timeout=60_000)
    page.get_by_text("Import existing results").first.click()
    _type_path(page, arm)
    page.get_by_text("Active project (dup_arm)").click()
    page.get_by_role("button", name="Import results").click()
    page.get_by_text("is already in project 'dup_arm'").wait_for()
    assert len(mgr.load_project("dup-arm").runs) == 2  # identification + pipeline, unchanged


def test_home_import_button_opens_project_setup_import(page, start_app):
    app = start_app()
    _open(page, app, "Welcome to UORCA Explorer")
    page.get_by_role("button", name="Import Existing Results").click()
    page.get_by_role("heading", name="Import Existing Results").wait_for()
    page.get_by_label("Results folder").wait_for()
    _assert_no_exception(page)


def test_explore_path_imports_once_and_selects_project(page, start_app, home, tmp_path):
    arm = make_arm_folder(tmp_path / "hpc_arm")
    env = {"UORCA_DEFAULT_RESULTS_DIR": str(arm)}

    app = start_app(env)
    _open(page, app, "Explore Results")
    page.get_by_text("Imported").first.wait_for(timeout=60_000)
    _assert_explorer_renders(page, "hpc_arm")
    app.stop()

    # A restart (new session) finds the run already recorded: no duplicate.
    app2 = start_app(env)
    _open(page, app2, "Explore Results")
    _assert_explorer_renders(page, "hpc_arm")
    mgr = _manager(home)
    assert len(mgr.list_projects()) == 1
    project = mgr.load_project("hpc-arm")
    assert [r.type for r in project.runs] == ["identification", "pipeline"]


def test_explore_path_error_is_shown(page, start_app, tmp_path):
    empty = tmp_path / "not_results"
    empty.mkdir()
    app = start_app({"UORCA_DEFAULT_RESULTS_DIR": str(empty)})
    _open(page, app, "Explore Results")  # a results dir makes Explore the start page
    page.get_by_text("No UORCA analysis found").wait_for()
    # The error is a real st.error alert (also proves the selector _assert_no_error_alert uses).
    assert page.locator('[data-testid="stAlertContentError"]').count() >= 1


def test_import_panel_stays_open_when_project_has_runs(page, start_app, home, tmp_path):
    """After one import, the 'Import existing results' expander must keep its preview visible.

    The first import leaves a flash message on the page; the next rerun drops it, which used to
    re-create the expander collapsed and hide the preview and the Import button.
    """
    first = make_arm_folder(tmp_path / "first_arm", identify=False)
    second = make_arm_folder(tmp_path / "second_arm", identify=False)
    mgr = _manager(home)
    mgr.create_project("Normal study")

    app = start_app()
    _open(page, app, "Projects")
    page.get_by_role("button", name="Select").first.click()
    page.wait_for_timeout(1500)
    _nav(page, "Explore")
    _type_path(page, first)
    page.get_by_text("Active project (Normal study)").click()
    page.get_by_role("button", name="Import results").click()
    page.get_by_text("Imported 2 complete dataset(s)").wait_for(timeout=60_000)
    page.get_by_text("Select pipeline run").wait_for(timeout=60_000)

    expander = page.locator('[data-testid="stExpander"]').filter(has_text="Import existing results")
    expander.get_by_text("Import existing results").first.click()
    box = expander.get_by_label("Results folder")
    box.fill(str(second))
    box.press("Enter")
    page.wait_for_timeout(4000)
    expander.get_by_text("Preview", exact=True).wait_for(state="visible", timeout=10_000)
    expander.get_by_text("Active project (Normal study)").click()
    expander.get_by_role("button", name="Import results").click()
    page.get_by_text("Imported 2 complete dataset(s)").wait_for(timeout=60_000)
    assert len(mgr.load_project("normal-study").get_runs_by_type("pipeline")) == 2


def test_browse_for_folder_fills_path_and_imports(page, start_app, home):
    """The folder browser starts at HOME, navigates down and up, and fills the path box."""
    arm = make_arm_folder(home / "Data" / "browsed_arm")
    (home / "Data" / ".hidden_dir").mkdir()
    app = start_app()
    _open(page, app, "Welcome to UORCA Explorer")
    _nav(page, "Explore")
    page.get_by_text("Import existing results").first.wait_for()

    if sys.platform == "darwin":  # the browser is local, so the Finder button is offered
        page.get_by_role("button", name="Choose in Finder…").wait_for()
    page.get_by_text("Browse for a folder").click()  # the toggle
    main = page.locator('[data-testid="stMain"]')
    main.get_by_text(f"Current folder: {home}").wait_for()

    # Click the visible box: clicking the hidden input (get_by_label) does not always open the list.
    picker = page.locator('[data-testid="stSelectbox"]').filter(has_text="Open a subfolder")

    def _open_list() -> None:
        # Wait for the rerun to finish first: a list opened mid-rerun shows the previous
        # folder's subfolders.
        _wait_idle(page)
        picker.click()
        page.get_by_role("option").first.wait_for()

    def _open_subfolder(name: str) -> None:
        _open_list()
        page.get_by_role("option", name=name).click()

    _open_subfolder("📁 Data")
    main.get_by_text(f"Current folder: {home / 'Data'}").wait_for()
    _open_list()
    options = page.get_by_role("option").all_inner_texts()
    assert options == ["📁 browsed_arm"], options  # hidden folder left out
    page.keyboard.press("Escape")

    _open_subfolder("📁 browsed_arm")
    main.get_by_text(f"Current folder: {arm}").wait_for()
    page.get_by_role("button", name="⬆ Up").click()
    main.get_by_text(f"Current folder: {home / 'Data'}").wait_for()
    _open_subfolder("📁 browsed_arm")
    main.get_by_text(f"Current folder: {arm}").wait_for()

    page.get_by_role("button", name="Use this folder").click()
    page.get_by_text("Preview", exact=True).wait_for()
    assert page.get_by_label("Results folder").input_value() == str(arm)
    assert page.get_by_label("New project name").input_value() == "browsed_arm"
    _assert_no_exception(page)

    page.get_by_role("button", name="Import results").click()
    page.get_by_text("Imported 2 complete dataset(s)").wait_for(timeout=60_000)
    _assert_explorer_renders(page, "browsed_arm")

    # With a run in the project, the import panel sits inside an expander; the browser must
    # still render there.
    expander = page.locator('[data-testid="stExpander"]').filter(has_text="Import existing results")
    expander.get_by_text("Import existing results").first.click()
    expander.get_by_text("Browse for a folder").click()
    expander.get_by_text("Current folder:").wait_for()
    _assert_no_exception(page)
