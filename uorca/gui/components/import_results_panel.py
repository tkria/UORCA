"""The "Import existing results" panel: attach results produced outside the GUI to a project."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from uorca.gui.components.folder_browser import render_folder_browser
from uorca.gui.project.importer import (
    LAYOUT_LABELS,
    ImportCandidate,
    ImportResultsError,
    default_project_name,
    import_into_named_project,
    inspect_results,
)
from uorca.gui.project.manager import ProjectManager
from uorca.gui.project.models import Project

FLASH_KEY = "_import_results_flash"


def activate_project(mgr: ProjectManager, project: Project) -> None:
    """Make ``project`` the active project for this session."""
    st.session_state["active_project"] = project
    st.session_state["active_project_name"] = project.name
    st.session_state["active_project_slug"] = mgr.get_project_slug(project.name)


def show_flash() -> None:
    """Show (once) the message left by a successful import.

    Always emits one container, message or not, so elements below keep their position across
    reruns; otherwise the rerun that drops the message re-creates them (an open expander
    collapses).
    """
    slot = st.container()
    msg = st.session_state.pop(FLASH_KEY, None)
    if msg:
        slot.success(msg)


def _render_preview(candidate: ImportCandidate) -> None:
    st.markdown(f"**Layout:** {LAYOUT_LABELS[candidate.layout]}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Datasets found", len(candidate.datasets))
    m2.metric("Complete", len(candidate.complete),
              help="analysis_info.json says analysis_success: true with unique_groups set, and RNAseqAnalysis/CPM.csv exists")
    m3.metric("Partial", len(candidate.partial),
              help="DEG tables were written but the analysis did not finish; not shown in the Explorer")
    m4.metric("Failed", len(candidate.failed),
              help="No DEG tables (or submitted with no dataset folder)")
    st.markdown(f"**Folder the Explorer will open:** `{candidate.analysis_dir}`")
    if candidate.research_question:
        st.markdown(f"**Research question:** {candidate.research_question}")
    else:
        st.caption("No research_question.json found; the summary tab will have no research question.")
    if candidate.identify_dir:
        st.markdown(f"**Identification results:** `{candidate.identify_dir}` (imported as a linked identification run)")
    for warning in candidate.warnings:
        st.warning(warning)
    with st.expander("Datasets", expanded=False):
        st.dataframe(
            pd.DataFrame({
                "Dataset": candidate.datasets,
                "Status": [candidate.dataset_status(a) for a in candidate.datasets],
            }),
            hide_index=True,
            use_container_width=True,
        )


def render_import_results(key: str, *, intro: Optional[str] = None) -> None:
    """Render the import panel. ``key`` keeps widget keys unique per page."""
    if intro:
        st.markdown(intro)
    path = st.text_input(
        "Results folder",
        key=f"{key}_path",
        placeholder="/path/to/results  (or a run folder holding analysis/)",
        help="A folder produced by `uorca run local|slurm` or copied from an HPC: "
             "<folder>/GSE*/RNAseqAnalysis, a run folder holding analysis/ (and identify/), "
             "or one dataset folder holding RNAseqAnalysis/. Nothing is copied; the project "
             "records the folder's location.",
    )
    render_folder_browser(f"{key}_path")
    if not path.strip():
        return

    try:
        candidate = inspect_results(path)
    except ImportResultsError as exc:
        st.error(str(exc))
        return

    st.markdown("##### Preview")
    _render_preview(candidate)

    if not candidate.importable:
        st.error(
            "No complete dataset in this folder, so there is nothing to explore. "
            "Only datasets with analysis_success true, unique_groups set and a CPM.csv can be opened."
        )
        return

    mgr = ProjectManager()
    active: Optional[Project] = st.session_state.get("active_project")
    options = (["Active project", "New project"] if active else ["New project"])
    labels = {"Active project": f"Active project ({active.name})" if active else "",
              "New project": "New project"}
    target = st.radio("Import into", options, format_func=lambda o: labels[o],
                      key=f"{key}_target", horizontal=True)
    new_name = ""
    if target == "New project":
        new_name = st.text_input("New project name", value=default_project_name(candidate),
                                 key=f"{key}_new_name_{candidate.source}")
    note = st.text_input("Note (optional)", key=f"{key}_note",
                         placeholder="e.g. copied from an HPC on 2026-10-05")

    if not st.button("Import results", type="primary", key=f"{key}_submit"):
        return

    try:
        if target == "New project":
            name = new_name.strip()
            if mgr.project_exists(name):
                raise ImportResultsError(
                    f"A project named '{name}' already exists. Choose another name, "
                    "or select it as the active project and import into it."
                )
            project, entries = import_into_named_project(mgr, candidate, name, create=True,
                                                         note=note.strip() or None)
        else:
            if active is None:
                raise ImportResultsError("No active project; choose 'New project'.")
            project, entries = import_into_named_project(mgr, candidate, active.name,
                                                         note=note.strip() or None)
    except ImportResultsError as exc:
        st.error(str(exc))
        return

    activate_project(mgr, project)
    st.session_state[FLASH_KEY] = (
        f"Imported {len(candidate.complete)} complete dataset(s) from `{candidate.analysis_dir}` "
        f"into project **{project.name}** (run `{entries[-1].id}`)."
    )
    st.switch_page(st.session_state["_pages"]["explore"])
