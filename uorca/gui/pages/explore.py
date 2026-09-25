import streamlit as st


def page():
    st.title("📊 Explore Results")
    project = st.session_state.get("active_project")
    if not project:
        st.warning("Please select a project first.")
        return

    pipeline_runs = project.get_runs_by_type("pipeline")
    completed_runs = [r for r in pipeline_runs if r.status == "completed" and r.output_path]

    if not completed_runs:
        st.info(f"No completed pipeline runs in project **{project.name}** yet.")
        return

    # Run selector (newest first)
    run_labels = []
    for r in reversed(completed_runs):
        label = f"{r.timestamp[:10] if r.timestamp else '?'}"
        if r.datasets:
            label += f" ({len(r.datasets)} datasets)"
        if r.summary:
            succeeded = r.summary.get("succeeded", "?")
            label += f" — {succeeded} with DEGs"
        if r.note:
            label += f" [{r.note}]"
        run_labels.append(label)

    selected_idx = st.selectbox("Select pipeline run", range(len(run_labels)), format_func=lambda i: run_labels[i])
    selected_run = list(reversed(completed_runs))[selected_idx]
    results_dir = selected_run.output_path

    # Resolve relative paths against the UORCA working directory
    from pathlib import Path
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        # Try relative to current working directory
        resolved = Path.cwd() / results_path
        if resolved.exists():
            results_dir = str(resolved)
        else:
            # Try relative to the UORCA package directory
            pkg_dir = Path(__file__).parent.parent.parent.parent
            resolved = pkg_dir / results_path
            if resolved.exists():
                results_dir = str(resolved)

    _render_explorer(results_dir)


def _render_explorer(results_dir: str):
    from uorca.gui.components.helpers import _validate_results_dir, get_integrator, add_custom_css
    from uorca.gui.components.sidebar_controls import render_sidebar_controls
    from uorca.gui.components.heatmap_tab import render_heatmap_tab
    from uorca.gui.components.expression_plots_tab import render_expression_plots_tab
    from uorca.gui.components.analysis_plots_tab import render_analysis_plots_tab
    from uorca.gui.components.datasets_info_tab import render_datasets_info_tab
    from uorca.gui.components.contrasts_info_tab import render_contrasts_info_tab
    from uorca.gui.components.ai_assistant_tab import render_ai_assistant_tab
    from uorca.gui.components.uorca_summary_tab import render_uorca_summary_tab

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

    with tab_summary:
        render_uorca_summary_tab(ri=ri, results_dir=results_dir)
    with tab_ai:
        render_ai_assistant_tab(ri=ri, results_dir=results_dir, selected_datasets=selected_datasets)
    with tab1:
        render_heatmap_tab(ri=ri, selected_datasets=selected_datasets)
    with tab2:
        render_expression_plots_tab(ri=ri, selected_datasets=selected_datasets)
    with tab3:
        render_analysis_plots_tab(ri=ri, results_dir=results_dir)
    with tab4:
        render_datasets_info_tab(ri=ri)
    with tab5:
        render_contrasts_info_tab(ri=ri, pvalue_thresh=0.05, lfc_thresh=1.0)
