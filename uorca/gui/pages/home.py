"""Home page — project hub dashboard for UORCA Explorer."""

from __future__ import annotations

import streamlit as st

from uorca.gui.project.manager import ProjectManager
from uorca.gui.project.models import Project


def _set_active_project(mgr: ProjectManager, project: Project) -> None:
    """Set a project as active in session state."""
    slug = mgr.get_project_slug(project.name)
    st.session_state["active_project"] = project
    st.session_state["active_project_name"] = project.name
    st.session_state["active_project_slug"] = slug


def _render_no_projects() -> None:
    """Welcome screen shown when no projects exist yet."""
    st.markdown(
        """
        ## Welcome to UORCA Explorer

        UORCA helps you identify, download, and analyse RNA-seq datasets from
        the Gene Expression Omnibus entirely through a graphical interface —
        no command-line required.

        **Get started by creating your first project.**
        """
    )
    if st.button("Create New Project", type="primary", icon="➕"):
        st.switch_page(st.session_state["_pages"]["project-setup"])


def _render_project_card(mgr: ProjectManager, project: Project) -> None:
    """Render a single project card inside a bordered container."""
    with st.container(border=True):
        col_info, col_btn = st.columns([5, 1])
        with col_info:
            st.markdown(f"### {project.name}")
            if project.description:
                st.caption(project.description)

            id_runs = project.get_runs_by_type("identification")
            pipe_runs = project.get_runs_by_type("pipeline")
            updated = (project.updated or project.created or "")[:19]

            st.markdown(
                f"Identification runs: **{len(id_runs)}** &nbsp;|&nbsp; "
                f"Pipeline runs: **{len(pipe_runs)}** &nbsp;|&nbsp; "
                f"Last updated: `{updated}`"
            )

        with col_btn:
            st.write("")  # vertical alignment spacer
            slug = mgr.get_project_slug(project.name)
            if st.button("Select", key=f"select_{slug}", type="secondary"):
                _set_active_project(mgr, project)
                st.rerun()


def _render_recent_activity(projects: list[Project]) -> None:
    """Show the 10 most recent runs across all projects."""
    all_runs: list[tuple[str, object]] = []
    for project in projects:
        for run in project.runs:
            all_runs.append((project.name, run))

    if not all_runs:
        st.caption("No runs recorded yet.")
        return

    # Sort descending by timestamp
    all_runs.sort(key=lambda t: t[1].timestamp or "", reverse=True)  # type: ignore[attr-defined]
    recent = all_runs[:10]

    for project_name, run in recent:
        ts = (run.timestamp or "")[:19]  # type: ignore[attr-defined]
        run_type = run.type or "—"  # type: ignore[attr-defined]
        run_status = run.status or "—"  # type: ignore[attr-defined]
        run_id = run.id or "—"  # type: ignore[attr-defined]
        st.markdown(
            f"- **{project_name}** &nbsp; `{run_id}` &nbsp; "
            f"`{run_type}` &nbsp; `{run_status}` &nbsp; {ts}"
        )


def page() -> None:
    st.title("UORCA Explorer")

    mgr = ProjectManager()
    projects = mgr.list_projects()

    if not projects:
        _render_no_projects()
        return

    # Header row: title + new project button
    hdr_col, btn_col = st.columns([6, 1])
    with btn_col:
        if st.button("New Project", icon="➕", type="secondary"):
            st.switch_page(st.session_state["_pages"]["project-setup"])

    st.subheader("Projects")

    active_name = st.session_state.get("active_project_name")

    for project in projects:
        # Highlight active project
        if project.name == active_name:
            st.markdown(
                f"<span style='font-size:0.8em; color:#4CAF50;'>&#9654; active</span>",
                unsafe_allow_html=True,
            )
        _render_project_card(mgr, project)

    st.divider()
    st.subheader("Recent Activity")
    _render_recent_activity(projects)
