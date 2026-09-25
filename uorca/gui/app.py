import streamlit as st

st.set_page_config(
    page_title="UORCA Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    from uorca.gui.pages import home, project_setup, identify, run, explore

    # Create page objects and store in session state for switch_page access
    page_home = st.Page(home.page, title="Home", icon="🏠", default=True, url_path="home")
    page_setup = st.Page(project_setup.page, title="Project Setup", icon="⚙️", url_path="project-setup")
    page_identify = st.Page(identify.page, title="Identify", icon="🔍", url_path="identify")
    page_run = st.Page(run.page, title="Run", icon="🚀", url_path="run")
    page_explore = st.Page(explore.page, title="Explore", icon="📊", url_path="explore")

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
    _render_project_selector()
    pg.run()


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
