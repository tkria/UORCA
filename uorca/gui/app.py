import streamlit as st

st.set_page_config(
    page_title="UORCA Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Set by ``uorca explore <results_dir>``; the same name as uorca.gui.pages.explore.RESULTS_DIR_ENV.
DEFAULT_RESULTS_ENV = "UORCA_DEFAULT_RESULTS_DIR"


def _guarded_page(title: str, module_path: str):
    """A page callable that imports its module lazily and contains its failure.

    A page that fails to import or render shows the error (with traceback) on that page; the
    navigation and the other pages keep working. See ``render_tab_safely``.
    """
    from uorca.gui.components.helpers.tab_guard import render_tab_safely

    def run_page() -> None:
        render_tab_safely(title, module_path, "page", "page")

    run_page.__name__ = f"page_{module_path.rsplit('.', 1)[-1]}"
    return run_page


def main():
    import os

    # `uorca explore <results_dir>` should open straight onto those results.
    open_on_explore = bool(os.environ.get(DEFAULT_RESULTS_ENV))

    # Create page objects and store in session state for switch_page access. Each page module
    # is imported only when that page runs, inside its own guard.
    pages_pkg = "uorca.gui.pages"
    page_home = st.Page(_guarded_page("Home", f"{pages_pkg}.home"), title="Home", icon="🏠",
                        default=not open_on_explore, url_path="home")
    page_setup = st.Page(_guarded_page("Project Setup", f"{pages_pkg}.project_setup"),
                         title="Project Setup", icon="⚙️", url_path="project-setup")
    page_identify = st.Page(_guarded_page("Identify", f"{pages_pkg}.identify"),
                            title="Identify", icon="🔍", url_path="identify")
    page_run = st.Page(_guarded_page("Run", f"{pages_pkg}.run"), title="Run", icon="🚀",
                       url_path="run")
    page_explore = st.Page(_guarded_page("Explore", f"{pages_pkg}.explore"), title="Explore",
                           icon="📊", default=open_on_explore, url_path="explore")

    st.session_state["_pages"] = {
        "home": page_home,
        "project-setup": page_setup,
        "identify": page_identify,
        "run": page_run,
        "explore": page_explore,
    }

    pages = {
        "UORCA": [page_home, page_setup],
        "Workflow": [page_identify, page_run, page_explore],
    }

    pg = st.navigation(pages)
    _import_default_results_dir()
    _render_project_selector()
    pg.run()


def _import_default_results_dir():
    """`uorca explore <path>`: attach <path> to a project named after it and select it.

    Runs once per browser session. The project is created on first use; later sessions find
    the run already recorded and import nothing.
    """
    import os

    path = os.environ.get(DEFAULT_RESULTS_ENV, "").strip()
    if not path:
        return
    if not st.session_state.get("_default_results_done"):
        st.session_state["_default_results_done"] = True
        from uorca.gui.components.import_results_panel import FLASH_KEY, activate_project
        from uorca.gui.project.importer import ImportResultsError, ensure_imported
        from uorca.gui.project.manager import ProjectManager

        mgr = ProjectManager()
        try:
            project, entries = ensure_imported(mgr, path)
        except ImportResultsError as exc:
            st.session_state["_default_results_error"] = f"Could not open {path}: {exc}"
        else:
            activate_project(mgr, project)
            if entries:
                st.session_state[FLASH_KEY] = (
                    f"Imported `{path}` into new run `{entries[-1].id}` of project "
                    f"**{project.name}**."
                )
            st.switch_page(st.session_state["_pages"]["explore"])
    error = st.session_state.get("_default_results_error")
    if error:
        st.sidebar.error(error)


def _render_project_selector():
    from uorca.gui.project.manager import ProjectManager
    mgr = ProjectManager()
    projects = mgr.list_projects()
    if not projects:
        st.sidebar.caption("No projects yet")
        return
    project_names = [p.name for p in projects]
    current = st.session_state.get("active_project_name")
    index = project_names.index(current) if current in project_names else 0
    selected = st.sidebar.selectbox("Active Project", project_names, index=index, key="project_selector")
    if selected != st.session_state.get("active_project_name"):
        slug = mgr.get_project_slug(selected)
        st.session_state["active_project"] = mgr.load_project(slug)
        st.session_state["active_project_name"] = selected
        st.session_state["active_project_slug"] = slug


if __name__ == "__main__":
    main()
