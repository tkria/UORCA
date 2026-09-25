"""Dataset Identification page — find GEO datasets for a research question."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from uorca.core import TaskManager, TaskStatus
from uorca.gui.project.manager import ProjectManager
from uorca.gui.project.models import Project, RunEntry


# ---------------------------------------------------------------------------
# Background task function
# ---------------------------------------------------------------------------


def _run_identification(
    query: str,
    output_dir: str,
    max_per_term: int,
    rounds: int,
    threshold: float,
    additional_context: str,
    library_sources: list[str],
    biology_weight: float,
    design_min: int,
    stage1_biology_threshold: int,
    progress_callback=None,
) -> str:
    """Run `uorca identify` via Popen with live progress streaming."""
    import re

    if progress_callback:
        progress_callback(0.02, "Building command…")

    cmd = [
        sys.executable, "-m", "uorca.identification.dataset_identification",
        "-q", query,
        "-o", output_dir,
        "-m", str(max_per_term),
        "-r", str(rounds),
        "-t", str(threshold),
        "--biology-weight", str(biology_weight),
        "--design-min", str(design_min),
        "--stage1-biology-threshold", str(stage1_biology_threshold),
        "-v",  # verbose — enables INFO logging to stderr
    ]

    # Map library_sources list → --library-source choice
    if "TRANSCRIPTOMIC" in library_sources and "TRANSCRIPTOMIC SINGLE CELL" in library_sources:
        cmd.extend(["--library-source", "both"])
    elif "TRANSCRIPTOMIC SINGLE CELL" in library_sources:
        cmd.extend(["--library-source", "sc"])
    else:
        cmd.extend(["--library-source", "bulk"])

    # Pass user-provided guidance into the Stage 2 scoring prompt.
    if additional_context.strip():
        cmd.extend(["--context", additional_context])

    if progress_callback:
        progress_callback(0.05, "Starting identification pipeline…")

    # Stream output line-by-line for live progress
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout for unified parsing
        text=True,
        bufsize=1,  # line-buffered
    )

    last_lines: list[str] = []  # keep last N lines for error reporting

    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue

        # Keep tail for error reporting
        last_lines.append(line)
        if len(last_lines) > 50:
            last_lines.pop(0)

        # Parse progress from log lines
        if progress_callback:
            _parse_progress_line(line, progress_callback)

    proc.wait()

    if proc.returncode != 0:
        error_tail = "\n".join(last_lines[-20:])
        raise RuntimeError(f"Identification failed (exit code {proc.returncode}):\n{error_tail}")

    if progress_callback:
        progress_callback(1.0, "Complete")

    return output_dir


def _parse_progress_line(line: str, progress_callback) -> None:
    """Parse a log/progress line and update the progress callback."""
    import re

    # Match tqdm-style progress: "Fetching GEO summaries:  45%|..."
    tqdm_match = re.search(r"(\d+)%\|", line)

    # Match INFO stage announcements
    if "Stage 1: Lightweight scoring" in line:
        progress_callback(0.40, "Stage 1: Scoring all valid datasets…")
    elif "Stage 1 complete" in line:
        # Extract stats: "783 scored, selecting top 157"
        m = re.search(r"(\d+) scored.*?top (\d+)", line)
        if m:
            progress_callback(0.55, f"Stage 1 done: {m.group(1)} scored → top {m.group(2)} for Stage 2")
        else:
            progress_callback(0.55, "Stage 1 complete")
    elif "Stage 2: Enriching" in line:
        m = re.search(r"Enriching (\d+) candidates", line)
        msg = f"Stage 2: Enriching {m.group(1)} candidates with metadata…" if m else "Stage 2: Enriching…"
        progress_callback(0.60, msg)
    elif "Stage 2: Full scoring" in line:
        progress_callback(0.75, "Stage 2: Full scoring with enriched data…")
    elif "Scoring complete" in line:
        progress_callback(0.90, "Scoring complete — writing results…")
    elif "DATASET IDENTIFICATION RESULTS" in line:
        progress_callback(0.95, "Writing final results…")
    elif "Fetching GEO summaries" in line and tqdm_match:
        pct = int(tqdm_match.group(1))
        progress_callback(0.05 + pct * 0.15 / 100, f"Fetching GEO summaries… {pct}%")
    elif "Fetching SRA data" in line and tqdm_match:
        pct = int(tqdm_match.group(1))
        progress_callback(0.20 + pct * 0.15 / 100, f"Validating SRA metadata… {pct}%")
    elif "Assessing dataset relevance" in line and tqdm_match:
        pct = int(tqdm_match.group(1))
        progress_callback(0.75 + pct * 0.15 / 100, f"AI scoring… {pct}%")
    elif "Fetching sample metadata" in line and tqdm_match:
        pct = int(tqdm_match.group(1))
        progress_callback(0.60 + pct * 0.10 / 100, f"Fetching sample metadata… {pct}%")
    elif "search terms" in line.lower() and "INFO" in line:
        m = re.search(r"(\d+) unique search terms", line)
        if m:
            progress_callback(0.08, f"Searching GEO with {m.group(1)} terms…")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_project() -> Project | None:
    return st.session_state.get("active_project")


def _get_project_slug(project: Project) -> str:
    mgr = ProjectManager()
    return mgr.get_project_slug(project.name)


def _save_project(project: Project) -> None:
    mgr = ProjectManager()
    slug = mgr.get_project_slug(project.name)
    mgr.save_project(slug, project)
    # Keep session state in sync
    st.session_state["active_project"] = project


def _task_manager() -> TaskManager:
    return TaskManager()


def _sync_run_status(project: Project, task_id: str, status: TaskStatus) -> None:
    """Sync TaskManager status back to the project YAML RunEntry.

    Called on every render so that terminal states (completed, failed,
    cancelled) are persisted even if the user navigates away.
    """
    status_map = {
        TaskStatus.COMPLETED: "completed",
        TaskStatus.FAILED: "failed",
        TaskStatus.CANCELLED: "cancelled",
        TaskStatus.RUNNING: "running",
        TaskStatus.PENDING: "pending",
    }
    new_status = status_map.get(status)
    if not new_status:
        return

    for run in project.runs:
        if run.id == task_id and run.status != new_status:
            run.status = new_status
            _save_project(project)
            break


def _load_results(output_dir: str) -> pd.DataFrame | None:
    """Load scored results CSV from an output directory, or None if absent."""
    csv_path = Path(output_dir) / "Dataset_identification_result.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None


def _scored_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a positive RelevanceScore."""
    if "RelevanceScore" not in df.columns:
        return pd.DataFrame()
    return df[df["RelevanceScore"].notna() & (df["RelevanceScore"] > 0)].copy()


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------


def _render_left_panel(project: Project, task_manager: TaskManager) -> None:
    """Render the input form in the left column."""

    st.info(f"**Active Project:** {project.name}")

    with st.form("identify_form"):
        query = st.text_area(
            "Research Question",
            value=project.default_research_question,
            height=120,
            placeholder="e.g. What genes are differentially expressed in heart failure?",
            help="UORCA uses AI to extract search terms from this question.",
        )

        with st.expander("Advanced Options", expanded=False):
            additional_context = st.text_area(
                "Additional Context",
                height=80,
                placeholder="e.g. Exclude mouse studies. Include only human adult tissue.",
                help="Optional inclusions/exclusions to refine the search.",
            )

            st.markdown("**Library Source**")
            bulk_checked = st.checkbox("Bulk RNA-seq", value=True)
            sc_checked = st.checkbox("Single-cell RNA-seq", value=False)

            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                max_per_term = st.number_input(
                    "Max per search term",
                    min_value=10,
                    max_value=2000,
                    value=500,
                    step=50,
                    help="Maximum number of GEO entries fetched per search term.",
                )
                rounds = st.number_input(
                    "Scoring rounds",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Number of AI scoring rounds per dataset.",
                )
            with adv_col2:
                threshold = st.number_input(
                    "Relevance threshold",
                    min_value=0.0,
                    max_value=10.0,
                    value=7.0,
                    step=0.5,
                    help="Minimum relevance score to include a dataset.",
                )

            st.markdown("**Scoring**")
            sc = project.scoring_config
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                biology_weight = st.number_input(
                    "Biology weight",
                    min_value=0.0, max_value=1.0, value=float(sc.biology_weight),
                    step=0.05,
                    help="Weight on BiologyScore. Design weight = 1 - this.",
                )
                design_min = st.number_input(
                    "Design minimum",
                    min_value=0, max_value=10, value=int(sc.design_min), step=1,
                    help="If >0, RelevanceScore is capped when DesignScore < this. 0 = no cap.",
                )
            with score_col2:
                st.text_input(
                    "Design weight (auto)",
                    value=f"{1.0 - biology_weight:.2f}",
                    disabled=True,
                    help="Auto-derived = 1 - biology weight",
                )
                stage1_biology_threshold = st.number_input(
                    "Stage 1 biology gate",
                    min_value=0, max_value=10, value=int(sc.stage1_biology_threshold), step=1,
                    help="Datasets with BiologyScore >= this graduate to Stage 2.",
                )

        submitted = st.form_submit_button(
            "Run Identification", type="primary", use_container_width=True
        )

    if submitted:
        if not query.strip():
            st.error("Please enter a research question.")
            return

        # Build library sources list
        library_sources: list[str] = []
        if bulk_checked:
            library_sources.append("TRANSCRIPTOMIC")
        if sc_checked:
            library_sources.append("TRANSCRIPTOMIC SINGLE CELL")
        if not library_sources:
            st.error("Select at least one library source.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"identify_{timestamp}"
        output_dir = str(
            Path("scratch") / f"{timestamp}_{project.name.replace(' ', '_')}"
        )

        params: dict[str, Any] = {
            "query": query.strip(),
            "output_dir": output_dir,
            "max_per_term": int(max_per_term),
            "rounds": int(rounds),
            "threshold": float(threshold),
            "additional_context": additional_context,
            "library_sources": library_sources,
            "biology_weight": float(biology_weight),
            "design_min": int(design_min),
            "stage1_biology_threshold": int(stage1_biology_threshold),
        }

        # Persist scoring config to the project YAML
        from uorca.gui.project.models import ScoringConfig
        project.scoring_config = ScoringConfig(
            biology_weight=float(biology_weight),
            design_min=int(design_min),
            stage1_biology_threshold=int(stage1_biology_threshold),
        )

        task_manager.submit_task(
            task_id=run_id,
            task_type="identification",
            task_func=_run_identification,
            parameters=params,
        )

        st.session_state["identify_task_id"] = run_id

        # Record run in project YAML
        entry = RunEntry(
            id=run_id,
            type="identification",
            status="running",
            research_question=query.strip(),
            output_path=output_dir,
            timestamp=datetime.now().isoformat(),
            params=params,
        )
        project.runs.append(entry)
        project.updated = datetime.now().isoformat()
        _save_project(project)

        st.success(f"Identification task **{run_id}** submitted.")
        st.rerun()


def _render_status_panel(project: Project, task_manager: TaskManager) -> None:
    """Render progress/status in the right column (compact)."""

    task_id: str | None = st.session_state.get("identify_task_id")

    if not task_id:
        id_runs = project.get_runs_by_type("identification")
        if not id_runs:
            st.info("No identification runs yet.")
        else:
            latest = id_runs[-1]
            st.caption(f"Latest run: **{latest.id}** ({latest.status})")
        return

    status_info = task_manager.get_task_status(task_id)

    if status_info is None:
        st.caption(f"Task `{task_id}` not found in task manager.")
        return

    current_status: TaskStatus = status_info["status"]
    _sync_run_status(project, task_id, current_status)

    if current_status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        st.subheader("Identification in Progress")
        progress = status_info.get("progress", 0.0)
        message = status_info.get("progress_message", "Processing…")
        st.progress(float(progress), text=message)

        col_refresh, col_cancel = st.columns([1, 1])
        with col_refresh:
            if st.button("Refresh", key="refresh_btn"):
                st.rerun()
        with col_cancel:
            if st.button("Cancel", key="cancel_btn", type="secondary"):
                task_manager.cancel_task(task_id)
                st.warning("Cancellation requested.")
                st.rerun()

    elif current_status == TaskStatus.FAILED:
        st.error("Identification failed.")
        error = status_info.get("error", "Unknown error")
        with st.expander("Error details"):
            st.code(error)
        if st.button("Dismiss", key="dismiss_failed"):
            st.session_state.pop("identify_task_id", None)
            st.rerun()

    elif current_status == TaskStatus.CANCELLED:
        st.warning("Task was cancelled.")
        if st.button("Dismiss", key="dismiss_cancelled"):
            st.session_state.pop("identify_task_id", None)
            st.rerun()

    elif current_status == TaskStatus.COMPLETED:
        st.success("Identification completed.")


def _render_run_selector(project: Project) -> RunEntry | None:
    """Render the "View run" selector and return the chosen identification run.

    This used to live inside ``_render_results_table``, which has three early ``return``
    statements. Anything drawn after that function — the theme map panel — would then
    not know which run was selected, and would vanish whenever the table returned early
    (spec 9.1). So the selector sits here, in ``page()``, and the selected run is passed
    to both consumers.

    Returns:
        The selected run, or ``None`` when the project has no identification runs yet.
    """
    id_runs = project.get_runs_by_type("identification")
    if not id_runs:
        return None

    newest_first = id_runs[::-1]
    run_labels = [f"{r.id} — {r.status}" for r in newest_first]
    selected_label = st.selectbox("View run", run_labels, key="selected_run_label")
    return newest_first[run_labels.index(selected_label)]


def _render_results_table(project: Project, selected_run: RunEntry) -> None:
    """Render the full-width results table below the input/status panels.

    ``selected_run`` comes from :func:`_render_run_selector` in ``page()`` rather than
    from a selector owned here, so that the theme map panel below this table sees the
    same run even when this function returns early (spec 9.1).
    """

    output_dir = selected_run.output_path
    df_all = _load_results(output_dir) if output_dir else None

    if df_all is None or df_all.empty:
        st.info(
            f"No results file found for run `{selected_run.id}`. "
            "The run may still be in progress, or the output directory is unavailable."
        )
        return

    df_scored = _scored_df(df_all)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total found", len(df_all))
    with m2:
        valid_count = int((df_all["Valid"] == "Yes").sum()) if "Valid" in df_all.columns else len(df_all)
        st.metric("Valid datasets", valid_count)
    with m3:
        above_thresh = int((df_all["RelevanceScore"] >= 7.0).sum()) if "RelevanceScore" in df_all.columns else 0
        st.metric("Above threshold (7.0)", above_thresh)

    if df_scored.empty:
        st.warning("No datasets with a positive relevance score found.")
        return

    # Build display table in requested column order
    accession_col = "GEO_Accession" if "GEO_Accession" in df_scored.columns else "Accession" if "Accession" in df_scored.columns else df_scored.columns[0]

    display_cols = {}
    display_cols["Accession"] = df_scored[accession_col].astype(str)
    if "Title" in df_scored.columns:
        display_cols["Title"] = df_scored["Title"].astype(str)
    if "Summary" in df_scored.columns:
        display_cols["Description"] = df_scored["Summary"].astype(str)
    if "DatasetSizeGB" in df_scored.columns:
        display_cols["Size (GB)"] = pd.to_numeric(df_scored["DatasetSizeGB"], errors="coerce").round(1)
    if "RelevanceScore" in df_scored.columns:
        display_cols["Score"] = pd.to_numeric(df_scored["RelevanceScore"], errors="coerce").round(1)
    if "Justification" in df_scored.columns:
        display_cols["Justification"] = df_scored["Justification"].astype(str)

    # GEO link
    geo_url_col = "GEO_URL" if "GEO_URL" in df_scored.columns else None
    if geo_url_col:
        display_cols["GEO"] = df_scored[geo_url_col].astype(str)
    else:
        display_cols["GEO"] = df_scored[accession_col].apply(
            lambda a: f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={a}"
        )

    df_display = pd.DataFrame(display_cols)

    # Sort by score descending
    if "Score" in df_display.columns:
        df_display = df_display.sort_values("Score", ascending=False)

    st.caption(f"**{len(df_scored)} scored datasets** — click rows to select, click column headers to sort")

    # Show interactive dataframe with row selection
    event = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"results_table_{selected_run.id}",
        column_config={
            "GEO": st.column_config.LinkColumn("GEO", display_text="View"),
            "Score": st.column_config.NumberColumn("Score", format="%.1f"),
            "Size (GB)": st.column_config.NumberColumn("Size (GB)", format="%.1f"),
        },
    )

    # Read selected rows
    selected_rows = event.selection.rows if event and event.selection else []
    selected_accessions = {str(df_display.iloc[i]["Accession"]) for i in selected_rows}

    # Send to Run button
    n_sel = len(selected_accessions)
    send_label = f"Send {n_sel} Selected to Run →" if n_sel else "Send Selected to Run →"

    if st.button(send_label, type="primary", disabled=(n_sel == 0), key="send_to_run"):
        dataset_list = list(selected_accessions)
        st.session_state["pending_datasets"] = dataset_list
        st.session_state["pending_from_run"] = selected_run.id

        # Persist selection to the identification run's datasets field
        # so it survives browser refresh
        project = _get_project()
        if project:
            for run in project.runs:
                if run.id == selected_run.id:
                    run.datasets = dataset_list
                    run.note = f"Curated: {len(dataset_list)} datasets selected for pipeline"
                    break
            _save_project(project)

        st.switch_page(st.session_state["_pages"]["run"])


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def page() -> None:
    st.title("Identify Datasets")

    project = _get_project()
    if not project:
        st.warning("No active project. Please select or create a project first.")
        if st.button("Go to Project Setup"):
            st.switch_page(st.session_state["_pages"]["project-setup"])
        return

    task_manager = _task_manager()

    # Reconnect to a running task if session state was lost (e.g. browser refresh)
    if "identify_task_id" not in st.session_state:
        for run in reversed(project.runs):
            if run.type == "identification" and run.status == "running":
                # Check if TaskManager still knows about this task
                status_info = task_manager.get_task_status(run.id)
                if status_info and status_info["status"] in (TaskStatus.RUNNING, TaskStatus.PENDING):
                    st.session_state["identify_task_id"] = run.id
                    break
                else:
                    # Task is gone from TaskManager — mark as failed in project
                    run.status = "failed"
                    run.note = "Process interrupted"
                    _save_project(project)

    tab_search, tab_my = st.tabs(["🔍 Search GEO", "📁 My Datasets"])

    with tab_search:
        col_form, col_status = st.columns([1, 1])
        with col_form:
            _render_left_panel(project, task_manager)
        with col_status:
            _render_status_panel(project, task_manager)
        st.divider()
        selected_run = _render_run_selector(project)
        if selected_run is not None:
            _render_results_table(project, selected_run)
            st.divider()
            # Spec 9.2: the map sits below the results table, full width, so the table
            # and the map read as one view of one run. Imported here rather than at
            # module scope, like every other component on this page, so that importing
            # the page does not pull the whole components package into app startup.
            from uorca.gui.components.theme_map_panel import render_theme_map_panel

            render_theme_map_panel(selected_run)

    with tab_my:
        from uorca.gui.components.custom_datasets import render_my_datasets
        render_my_datasets(project)
