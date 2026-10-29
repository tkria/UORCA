import streamlit as st

# Set by ``uorca explore <results_dir>`` (uorca/explore.py). uorca/gui/app.py imports that
# folder into a project and opens the app on this page.
RESULTS_DIR_ENV = "UORCA_DEFAULT_RESULTS_DIR"


_IMPORT_INTRO = (
    "Have results from `uorca run local|slurm`, a site script, or copied from an HPC? "
    "Point at the folder to attach it to a project and explore it here."
)


def page():
    from uorca.gui.components.import_results_panel import render_import_results, show_flash

    st.title("📊 Explore Results")
    show_flash()

    project = st.session_state.get("active_project")
    if not project:
        st.warning("No active project. Create one in Project Setup, or import existing results below.")
        st.subheader("Import existing results")
        render_import_results("explore_import", intro=_IMPORT_INTRO)
        return

    pipeline_runs = project.get_runs_by_type("pipeline")
    completed_runs = [r for r in pipeline_runs if r.status == "completed" and r.output_path]

    if not completed_runs:
        st.info(f"No completed pipeline runs in project **{project.name}** yet.")
        st.subheader("Import existing results")
        render_import_results("explore_import", intro=_IMPORT_INTRO)
        return

    # Stay open while a path is entered, so the preview and Import button are not hidden.
    with st.expander("Import existing results",
                     expanded=bool(st.session_state.get("explore_import_more_path"))):
        render_import_results("explore_import_more", intro=_IMPORT_INTRO)

    # Run selector (newest first)
    run_labels = []
    for r in reversed(completed_runs):
        label = f"{r.timestamp[:10] if r.timestamp else '?'}"
        if r.datasets:
            label += f" ({len(r.datasets)} datasets)"
        if r.summary:
            succeeded = r.summary.get("succeeded", "?")
            label += f" — {succeeded} with DEGs"
        if r.params.get("imported"):
            label += " — imported"
        if r.note:
            label += f" [{r.note}]"
        run_labels.append(label)

    selected_idx = st.selectbox("Select pipeline run", range(len(run_labels)), format_func=lambda i: run_labels[i])
    selected_run = list(reversed(completed_runs))[selected_idx]
    from uorca.gui.project.paths import resolve_local_path

    results_path = resolve_local_path(selected_run.output_path)
    if not results_path.is_dir():
        st.error(
            f"Results folder for this run not found: `{results_path}` "
            f"(recorded as `{selected_run.output_path}`)."
        )
        return
    results_dir = str(results_path)

    if selected_run.params.get("imported"):
        st.caption(f"Imported results: `{results_dir}`")

    _render_explorer(results_dir)


def _render_explorer(results_dir: str):
    from uorca.gui.components.helpers import _validate_results_dir, get_integrator, add_custom_css
    from uorca.gui.components.sidebar_controls import render_sidebar_controls
    from uorca.gui.components.helpers.tab_guard import render_tab_safely

    add_custom_css()
    valid_dir, validation_error = _validate_results_dir(results_dir)
    if not valid_dir:
        st.error(f"Invalid results directory: {validation_error}")
        return

    ri, error = get_integrator(results_dir)
    if error or not ri or not ri.cpm_data:
        st.warning("No data found in this results directory.")
        return

    st.session_state["results_integrator"] = ri
    st.session_state["results_dir"] = results_dir
    sidebar_params = render_sidebar_controls(ri, results_dir)
    selected_datasets = sidebar_params.get("selected_datasets", [])

    tab_summary, tab_ai, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "UORCA Summary", "AI Assistant", "Explore DEG Heatmap",
        "Plot Gene Expression", "View Dataset Analyses", "View Dataset Info", "View Contrast Info",
    ])

    # Each tab imports its own module inside its own block, so a tab that fails to import or
    # render shows its error (with traceback) there, and the other tabs still work.
    components = "uorca.gui.components"
    with tab_summary:
        render_tab_safely("UORCA Summary", f"{components}.uorca_summary_tab",
                          "render_uorca_summary_tab", ri=ri, results_dir=results_dir)
    with tab_ai:
        render_tab_safely("AI Assistant", f"{components}.ai_assistant_tab",
                          "render_ai_assistant_tab", ri=ri, results_dir=results_dir,
                          selected_datasets=selected_datasets)
    with tab1:
        render_tab_safely("Explore DEG Heatmap", f"{components}.heatmap_tab",
                          "render_heatmap_tab", ri=ri, selected_datasets=selected_datasets)
    with tab2:
        render_tab_safely("Plot Gene Expression", f"{components}.expression_plots_tab",
                          "render_expression_plots_tab", ri=ri,
                          selected_datasets=selected_datasets)
    with tab3:
        render_tab_safely("View Dataset Analyses", f"{components}.analysis_plots_tab",
                          "render_analysis_plots_tab", ri=ri, results_dir=results_dir)
    with tab4:
        render_tab_safely("View Dataset Info", f"{components}.datasets_info_tab",
                          "render_datasets_info_tab", ri=ri)
    with tab5:
        render_tab_safely("View Contrast Info", f"{components}.contrasts_info_tab",
                          "render_contrasts_info_tab", ri=ri, pvalue_thresh=0.05, lfc_thresh=1.0)
