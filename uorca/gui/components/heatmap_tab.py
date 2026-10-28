"""
Heatmap Tab for UORCA Explorer.

This tab displays interactive heatmaps showing log2 fold changes for selected genes across contrasts.
Features combined form for contrast and gene selection with proper select/clear all functionality.
"""

import logging
import re
import streamlit as st
import pandas as pd
import traceback
from typing import List, Tuple, Optional, Dict, Any
import io
import zipfile
import json
from datetime import datetime
import textwrap

from .helpers import (
    check_ai_generating,
    setup_fragment_decorator,
    cached_figure_creation,
    cached_identify_important_genes,
    log_streamlit_tab,
    log_streamlit_function,
    log_streamlit_event,
    log_streamlit_user_action,
    group_contrasts_by_organism,
    filter_genes_by_organism,
    get_organism_display_name,
    get_valid_contrasts_with_data,
    plotly_fig_to_pdf_bytes,
    generate_plot_filename,
    render_scrollable_plotly,
)
from uorca.gui.results_integration import ResultsIntegrator, ADJUSTED_P_LABEL, ADJUSTED_P_HELP

# Import core modules
from uorca.core import validation, script_generation
from uorca.core.gene_selection import get_available_genes_for_contrasts
from uorca.gui.ortholog_mapper import (
    expand_genes_all_vs_all,
    get_ortholog_summary,
    get_taxid_from_organism,
    create_ortholog_mapping_table,
    create_hierarchical_gene_list,
    ortholog_file_error,
)

# Set up fragment decorator
setup_fragment_decorator()

logger = logging.getLogger(__name__)


@log_streamlit_tab("Heatmap")
def render_heatmap_tab(ri: ResultsIntegrator, selected_datasets: List[str], **kwargs):
    """
    Render the heatmap tab with combined contrast and gene selection form.

    Args:
        ri: ResultsIntegrator instance
        selected_datasets: List of selected dataset IDs from sidebar
        **kwargs: Additional arguments (maintained for compatibility)
    """
    st.header("Explore DEG Heatmap")
    st.markdown("Construct a heatmap of selected genes and contrasts. The generated plots can be downloaded. The data and script to produce the heatmap can also be downloaded for further customisation.")

    # Get selected datasets from sidebar
    if not selected_datasets:
        selected_datasets = st.session_state.get('selected_datasets_from_sidebar', [])

    if not selected_datasets:
        st.markdown("To get started, select datasets from the sidebar on the left.")
        return

    # Initialize session state for selected contrasts
    if 'selected_contrasts_heatmap' not in st.session_state:
        st.session_state['selected_contrasts_heatmap'] = set()

    # Check if datasets have changed and auto-select all contrasts by default
    current_datasets = set(selected_datasets)
    if 'previous_datasets_heatmap' not in st.session_state:
        st.session_state['previous_datasets_heatmap'] = set()

    if current_datasets != st.session_state['previous_datasets_heatmap']:
        # Datasets changed - select all contrasts by default
        contrast_data = _create_contrast_table_data_filtered(ri, selected_datasets)
        all_contrasts = set()
        for item in contrast_data:
            contrast_tuple = (item['Accession'], item['Contrast'])
            all_contrasts.add(contrast_tuple)
        st.session_state['selected_contrasts_heatmap'] = all_contrasts
        st.session_state['previous_datasets_heatmap'] = current_datasets

    # Quick selection buttons (outside form)
    st.subheader("Select Contrasts")
    contrast_data = _create_contrast_table_data_filtered(ri, selected_datasets)

    if contrast_data:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All Contrasts", key="select_all_contrasts_heatmap"):
                # Select all contrasts
                all_contrasts = set()
                for item in contrast_data:
                    contrast_tuple = (item['Accession'], item['Contrast'])
                    all_contrasts.add(contrast_tuple)
                st.session_state['selected_contrasts_heatmap'] = all_contrasts
                log_streamlit_user_action(f"Selected all {len(all_contrasts)} contrasts for heatmap")
                st.rerun()
        with col2:
            if st.button("Clear All Contrasts", key="clear_all_contrasts_heatmap"):
                # Clear all selections
                st.session_state['selected_contrasts_heatmap'] = set()
                log_streamlit_user_action("Cleared all contrast selections for heatmap")
                st.rerun()

    # Render combined form
    form_results = _render_combined_heatmap_form(ri, selected_datasets, contrast_data)

    heatmap_rendered = False

    if form_results:
        selected_contrasts = form_results['contrasts']
        gene_params = form_results['gene_params']

        # Convert accession-based contrasts to analysis_id-based contrasts
        analysis_contrasts = []
        if selected_contrasts:
            for accession, contrast_id in selected_contrasts:
                # Find the analysis_id for this accession
                analysis_id = None
                for aid in ri.analysis_info:
                    if ri.analysis_info[aid].get('accession') == accession:
                        analysis_id = aid
                        break

                if analysis_id:
                    analysis_contrasts.append((analysis_id, contrast_id))

        # Auto-select genes if we have contrasts and gene parameters
        gene_sel = []
        if analysis_contrasts and gene_params:
            gene_sel = _auto_select_genes(ri, ri.results_dir, analysis_contrasts, gene_params)

        # Display current selections summary (removed metrics and gene selection info)
        if analysis_contrasts and gene_sel:
            pass  # Metrics and info messages removed

            # Group contrasts by organism and render heatmaps
            organism_groups = group_contrasts_by_organism(ri, analysis_contrasts)

            if len(organism_groups) == 1:
                # Single organism - no sub-tabs needed
                organism = list(organism_groups.keys())[0]
                organism_contrasts = organism_groups[organism]
                organism_genes = filter_genes_by_organism(ri, gene_sel, organism, organism_contrasts)

                # Frequent DEG helper: species-specific copy dropdown
                species_genes_for_cache = []
                if gene_params.get('gene_selection_method') == 'Frequent DEGs':
                    try:
                        species_genes = cached_identify_important_genes(
                            results_dir=ri.results_dir,
                            selected_contrasts=organism_contrasts,
                            top_frequent=gene_params.get('gene_count', 50),
                            p_value_threshold=gene_params.get('pvalue_thresh', 0.05),
                            lfc_threshold=gene_params.get('lfc_thresh', 1.0)
                        ) or []
                        species_genes_for_cache = species_genes
                        with st.expander(f"Copy Frequent DEGs for {get_organism_display_name(organism)} ({len(species_genes)})", expanded=False):
                            st.code("\n".join(species_genes) if species_genes else "", language=None)
                    except Exception as e:
                        logger.error(f"Could not list frequent DEGs for {organism}: {e}", exc_info=True)
                        st.warning(f"Could not list frequent DEGs for {get_organism_display_name(organism)}: {e}")

                # Info panel: appears once plots are generated (placed above heatmap)
                with st.expander("What this heatmap shows", expanded=False):
                    st.markdown(
                        f"- Values are log2 fold change (log2FC) computed within each dataset's contrast.\n"
                        f"- Only significant values are shown: {ADJUSTED_P_LABEL} < {gene_params.get('pvalue_thresh', 0.05)} and |logFC| > {gene_params.get('lfc_thresh', 1.0)}.\n"
                        f"  Non-significant entries are set to 0 and hidden by filtering."
                    )

                _draw_heatmap(
                    ri,
                    organism_genes,
                    organism_contrasts,
                    gene_params.get('pvalue_thresh', 0.05),
                    gene_params.get('lfc_thresh', 1.0),
                    use_dynamic_filtering=True,
                    hide_empty_rows_cols=True,
                    font_size=12,
                    show_grid_lines=True,
                    grid_opacity=0.3,
                    cluster_genes=gene_params.get('cluster_genes', True)
                )
                heatmap_rendered = True
                # Persist full render specification for seamless re-display
                st.session_state['heatmap_last_spec'] = {
                    'tabs': False,
                    'sections': [
                        {
                            'organism': organism,
                            'organism_display': get_organism_display_name(organism),
                            'genes': organism_genes,
                            'contrasts': organism_contrasts,
                            'frequent_genes': species_genes_for_cache,
                        }
                    ],
                    'params': {
                        'pvalue_thresh': gene_params.get('pvalue_thresh', 0.05),
                        'lfc_thresh': gene_params.get('lfc_thresh', 1.0),
                        'use_dynamic_filtering': True,
                        'hide_empty_rows_cols': True,
                        'font_size': 12,
                        'show_grid_lines': True,
                        'grid_opacity': 0.3,
                        'cluster_genes': gene_params.get('cluster_genes', True),
                    },
                }
            else:
                # Multiple organisms - create sub-tabs
                organism_names = list(organism_groups.keys())
                tab_names = [get_organism_display_name(org) for org in organism_names]
                tabs = st.tabs(tab_names)

                frequent_by_org: Dict[str, List[str]] = {}
                for i, organism in enumerate(organism_names):
                    with tabs[i]:
                        organism_contrasts = organism_groups[organism]
                        organism_genes = filter_genes_by_organism(ri, gene_sel, organism, organism_contrasts)

                        # Frequent DEG helper: species-specific copy UI in each tab
                        if gene_params.get('gene_selection_method') == 'Frequent DEGs':
                            try:
                                species_genes = cached_identify_important_genes(
                                    results_dir=ri.results_dir,
                                    selected_contrasts=organism_contrasts,
                                    top_frequent=gene_params.get('gene_count', 50),
                                    p_value_threshold=gene_params.get('pvalue_thresh', 0.05),
                                    lfc_threshold=gene_params.get('lfc_thresh', 1.0)
                                ) or []
                                frequent_by_org[organism] = species_genes
                                with st.expander(f"Copy Frequent DEGs for {get_organism_display_name(organism)} ({len(species_genes)})", expanded=False):
                                    st.code("\n".join(species_genes) if species_genes else "", language=None)
                            except Exception:
                                pass

                        if organism_genes:
                            # Info panel within each organism tab
                            with st.expander("What this heatmap shows", expanded=False):
                                st.markdown(
                                    f"- Values are log2 fold change (log2FC) computed within each dataset's contrast.\n"
                                    f"- Only significant values are shown: {ADJUSTED_P_LABEL} < {gene_params.get('pvalue_thresh', 0.05)} and |logFC| > {gene_params.get('lfc_thresh', 1.0)}.\n"
                                    f"  Non-significant entries are set to 0 and hidden by filtering."
                                )
                            _draw_heatmap(
                                ri,
                                organism_genes,
                                organism_contrasts,
                                gene_params.get('pvalue_thresh', 0.05),
                                gene_params.get('lfc_thresh', 1.0),
                                use_dynamic_filtering=True,
                                hide_empty_rows_cols=True,
                                font_size=12,
                                show_grid_lines=True,
                                grid_opacity=0.3,
                                cluster_genes=gene_params.get('cluster_genes', True)
                            )
                            heatmap_rendered = True
                        else:
                            st.warning(f"No genes found for {get_organism_display_name(organism)} with current parameters.")
                            st.info("Try adjusting the significance thresholds in the gene selection form or selecting more datasets for this species.")

                # Persist full render specification for seamless re-display (multi-organism)
                if heatmap_rendered:
                    sections = []
                    for organism in organism_names:
                        organism_contrasts = organism_groups[organism]
                        organism_genes = filter_genes_by_organism(ri, gene_sel, organism, organism_contrasts)
                        if organism_genes:
                            sections.append({
                                'organism': organism,
                                'organism_display': get_organism_display_name(organism),
                                'genes': organism_genes,
                                'contrasts': organism_contrasts,
                                'frequent_genes': frequent_by_org.get(organism, []),
                            })
                    st.session_state['heatmap_last_spec'] = {
                        'tabs': True,
                        'tab_names': [get_organism_display_name(org) for org in organism_names],
                        'sections': sections,
                        'params': {
                            'pvalue_thresh': gene_params.get('pvalue_thresh', 0.05),
                            'lfc_thresh': gene_params.get('lfc_thresh', 1.0),
                            'use_dynamic_filtering': True,
                            'hide_empty_rows_cols': True,
                            'font_size': 12,
                            'show_grid_lines': True,
                            'grid_opacity': 0.3,
                            'cluster_genes': gene_params.get('cluster_genes', True),
                        },
                    }

        elif analysis_contrasts and not gene_sel:
            st.warning("No genes selected. Please configure gene selection parameters above.")
        elif not analysis_contrasts:
            st.info("Please select contrasts using the form above.")

    # If no new heatmap was rendered in this run, re-render the last one identically
    if not heatmap_rendered:
        last_spec = st.session_state.get('heatmap_last_spec')
        if last_spec:
            params = last_spec.get('params', {})
            if last_spec.get('tabs'):
                tab_names = last_spec.get('tab_names') or [s.get('organism_display') for s in last_spec.get('sections', [])]
                tabs = st.tabs(tab_names)
                for i, section in enumerate(last_spec.get('sections', [])):
                    with tabs[i]:
                        # If frequent genes were cached for this section, show the copy expander
                        freq = section.get('frequent_genes') or []
                        if freq:
                            org_disp = section.get('organism_display') or 'Species'
                            with st.expander(f"Copy Frequent DEGs for {org_disp} ({len(freq)})", expanded=False):
                                st.code("\n".join(freq), language=None)
                        # Info panel on cached re-render
                        with st.expander("What does this plot mean?", expanded=False):
                            st.markdown(
                                f"- Values are log2 fold change (log2FC) computed within each dataset's contrast.\n"
                                f"- Only significant values are shown: {ADJUSTED_P_LABEL} < {params.get('pvalue_thresh', 0.05)} and |logFC| > {params.get('lfc_thresh', 1.0)}.\n"
                                f"  Non-significant entries are set to 0 and hidden by filtering."
                            )
                        _draw_heatmap(
                            ri,
                            section.get('genes', []),
                            section.get('contrasts', []),
                            params.get('pvalue_thresh', 0.05),
                            params.get('lfc_thresh', 1.0),
                            use_dynamic_filtering=params.get('use_dynamic_filtering', True),
                            hide_empty_rows_cols=params.get('hide_empty_rows_cols', True),
                            font_size=params.get('font_size', 12),
                            show_grid_lines=params.get('show_grid_lines', True),
                            grid_opacity=params.get('grid_opacity', 0.3),
                            cluster_genes=params.get('cluster_genes', True),
                        )
            else:
                # Single organism re-render
                section = (last_spec.get('sections') or [{}])[0]
                # If frequent genes were cached, show the copy expander
                freq = section.get('frequent_genes') or []
                if freq:
                    org_disp = section.get('organism_display') or 'Species'
                    with st.expander(f"Copy Frequent DEGs for {org_disp} ({len(freq)})", expanded=False):
                        st.code("\n".join(freq), language=None)
                # Info panel on cached re-render (single organism)
                with st.expander("What does this plot mean?", expanded=False):
                    st.markdown(
                        f"- Values are log2 fold change (log2FC) computed within each dataset's contrast.\n"
                        f"- Only significant values are shown: {ADJUSTED_P_LABEL} < {params.get('pvalue_thresh', 0.05)} and |logFC| > {params.get('lfc_thresh', 1.0)}.\n"
                        f"  Non-significant entries are set to 0 and hidden by filtering."
                    )
                _draw_heatmap(
                    ri,
                    section.get('genes', []),
                    section.get('contrasts', []),
                    params.get('pvalue_thresh', 0.05),
                    params.get('lfc_thresh', 1.0),
                    use_dynamic_filtering=params.get('use_dynamic_filtering', True),
                    hide_empty_rows_cols=params.get('hide_empty_rows_cols', True),
                    font_size=params.get('font_size', 12),
                    show_grid_lines=params.get('show_grid_lines', True),
                    grid_opacity=params.get('grid_opacity', 0.3),
                    cluster_genes=params.get('cluster_genes', True),
                )


@log_streamlit_function
def _render_combined_heatmap_form(ri: ResultsIntegrator, selected_datasets: List[str], contrast_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Render the combined form for contrast and gene selection."""

    # 1) Contrast selection (kept at the top)
    with st.form("heatmap_contrasts_form"):
        st.subheader("Analysis Configuration")
        st.markdown("**Contrasts**")

        if contrast_data:
            df = pd.DataFrame(contrast_data)
            # Pre-select based on session state
            df['Select'] = df.apply(
                lambda row: (row['Accession'], row['Contrast']) in st.session_state['selected_contrasts_heatmap'],
                axis=1
            )

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "",
                        help="Check to include this contrast in the heatmap",
                        default=False
                    ),
                    "Accession": st.column_config.TextColumn(
                        "Dataset",
                        help="GEO accession number",
                        width=100
                    ),
                    "Contrast": st.column_config.TextColumn(
                        "Contrast",
                        help="Contrast identifier",
                        width=200
                    ),
                    "Description": st.column_config.TextColumn(
                        "Description",
                        help="Contrast description",
                        width=400
                    )
                },
                key="heatmap_combined_contrast_table"
            )
        else:
            edited_df = pd.DataFrame()
            st.info("No contrasts available for selected datasets")

        contrasts_submitted = st.form_submit_button("Apply Contrast Selection", type="primary")
        if contrasts_submitted:
            # Get selected contrasts and store in session state, then rerun
            selected_contrasts = []
            if not edited_df.empty:
                selected_rows = edited_df[edited_df["Select"] == True]
                selected_contrasts = [
                    (row["Accession"], row["Contrast"])
                    for _, row in selected_rows.iterrows()
                ]
            st.session_state['selected_contrasts_heatmap'] = set(selected_contrasts)
            log_streamlit_user_action(f"Contrast selection updated: {len(selected_contrasts)} contrasts selected for heatmap")
            st.rerun()

    # 2) Gene selection method (below contrasts; outside form to trigger rerun)
    st.markdown("---")
    st.markdown("**Gene Selection**")
    gene_selection_method = st.radio(
        "Choose gene selection method",
        options=["Frequent DEGs", "Custom"],
        key="heatmap_combined_gene_method",
        help="**Frequent DEGs**: Automatically finds genes that are consistently differentially expressed across multiple contrasts. **Custom**: Use your own gene list for targeted analysis.",
        horizontal=True,
    )

    # 3) Gene/threshold inputs and final submit
    with st.form("heatmap_gene_form"):
        # Conditional input display based on currently selected method
        if gene_selection_method == "Frequent DEGs":
            gene_count_input = st.text_input(
                "Number of genes to display",
                value="50",
                help="Number of top frequently differentially expressed genes to include",
                key="heatmap_combined_gene_count"
            )
            custom_genes_list = []
        else:
            custom_genes_input = st.text_area(
                "Custom Gene List",
                height=150,
                placeholder="Enter one gene per line, e.g.:\nTP53\nEGFR\nMYC\nBRCA1",
                help="Enter gene symbols, one per line. Genes will be filtered by significance thresholds.",
                key="heatmap_combined_custom_genes"
            )
            custom_genes_list = []
            if custom_genes_input.strip():
                custom_genes_list = [gene.strip() for gene in custom_genes_input.strip().split('\n') if gene.strip()]
            gene_count_input = "50"  # not used in custom mode
            keep_gene_order = st.checkbox(
                "Keep gene order (disable y-axis clustering)",
                value=False,
                help="If enabled, the heatmap will preserve the order of your custom gene list and will not cluster genes on the y-axis.",
                key="heatmap_keep_gene_order_custom"
            )

        # Ortholog expansion option (only for Custom genes)
        if gene_selection_method == "Custom":
            # The orthologue table is downloaded separately; say so loudly if it is missing
            ortholog_error = ortholog_file_error()
            if ortholog_error:
                st.error(ortholog_error)
            check_orthologs = st.checkbox(
                "Check for orthologues",
                value=False,
                disabled=ortholog_error is not None,
                help="Expand the input gene list to include orthologues in other species present in the selected datasets. Ortholog identification uses GeneOrthology.",
                key="heatmap_check_orthologs"
            )
        else:
            check_orthologs = False

        # Significance Thresholds
        st.markdown("**Significance Thresholds**")
        col1, col2 = st.columns(2)
        with col1:
            lfc_thresh = st.text_input(
                "Log2FC Threshold",
                value="1.0",
                help="Absolute log2 fold change threshold for significance",
                key="heatmap_combined_lfc_threshold"
            )
        with col2:
            pvalue_thresh = st.text_input(
                f"{ADJUSTED_P_LABEL} threshold",
                value="0.05",
                help=f"Genes count as significant when their {ADJUSTED_P_HELP} is below this value",
                key="heatmap_combined_pvalue_threshold"
            )

        submitted = st.form_submit_button("Generate Heatmap", type="primary")

        if submitted:
            validation_error = False

            # Validate thresholds using core module
            lfc_val, pval_val, thresh_error = validation.validate_threshold_values(lfc_thresh, pvalue_thresh)
            if thresh_error:
                st.error("Please enter valid numeric values for thresholds")
                validation_error = True

            # Validate gene count for Frequent DEGs using core module
            gene_count = 50
            if gene_selection_method == "Frequent DEGs":
                gene_count, count_error = validation.validate_gene_count(gene_count_input)
                if count_error:
                    st.error("Please enter a valid positive number for gene count")
                    validation_error = True

            # Validate custom gene list using core module
            if gene_selection_method == "Custom" and not validation.validate_custom_gene_list(custom_genes_list):
                st.error("Please enter at least one gene for custom selection")
                validation_error = True

            # Handle ortholog expansion if enabled
            original_genes = custom_genes_list.copy() if custom_genes_list else []
            ortholog_mapping = None
            if gene_selection_method == "Custom" and check_orthologs and custom_genes_list and not validation_error:
                # Get unique organisms from selected contrasts
                target_organisms = set()
                for accession, contrast_id in st.session_state.get('selected_contrasts_heatmap', set()):
                    for aid in ri.analysis_info:
                        if ri.analysis_info[aid].get('accession') == accession:
                            organism = ri.analysis_info[aid].get('organism', 'Unknown')
                            if organism != 'Unknown':
                                target_organisms.add(organism)
                            break

                # If multiple organisms, try to expand genes
                if len(target_organisms) > 1:
                    with st.spinner('Searching for orthologues across species...'):
                        # Use all-vs-all approach
                        expanded_genes, ortholog_mapping = expand_genes_all_vs_all(
                            custom_genes_list,
                            list(target_organisms),
                            return_mapping=True
                        )

                        if len(expanded_genes) > len(custom_genes_list):
                            st.success(f"✓ Expanded {len(custom_genes_list)} genes to {len(expanded_genes)} genes (+{len(expanded_genes) - len(custom_genes_list)} orthologues)")

                            # Show ortholog mapping table
                            with st.expander("Ortholog mapping table", expanded=False):
                                if ortholog_mapping:
                                    # Create the mapping table
                                    mapping_df = create_ortholog_mapping_table(
                                        original_genes,
                                        ortholog_mapping,
                                        list(target_organisms)
                                    )
                                    st.dataframe(mapping_df, use_container_width=True)

                            # Show copyable gene list
                            with st.expander("Copy full gene list", expanded=False):
                                if ortholog_mapping:
                                    hierarchical_list = create_hierarchical_gene_list(
                                        original_genes,
                                        ortholog_mapping,
                                        expanded_genes
                                    )
                                    st.code(hierarchical_list, language=None)

                            custom_genes_list = expanded_genes
                        else:
                            st.info("No additional orthologues found for the input genes")
                else:
                    if check_orthologs:
                        st.info("Ortholog expansion requires multiple species in selected datasets")

            # Validate selected contrasts using core module
            selected_contrasts = list(st.session_state.get('selected_contrasts_heatmap', set()))
            if not validation.validate_contrasts_selected(selected_contrasts):
                st.error("Please select at least one contrast")
                validation_error = True

            if not validation_error:
                # Additional gene/contrast validation for Custom mode
                if gene_selection_method == "Custom":
                    with st.expander("Gene and contrast validation", expanded=False):
                        st.write(f"**Total genes entered:** {len(custom_genes_list)}")
                        st.write(f"**Total contrasts selected:** {len(selected_contrasts)}")

                        st.write("")

                        available_genes = set()
                        significant_genes = set()
                        contrasts_with_data = []
                        contrasts_without_data = []
                        contrasts_with_no_significant_custom_genes = []

                        for accession, contrast_id in selected_contrasts:
                            analysis_id = None
                            for aid in ri.analysis_info:
                                if ri.analysis_info[aid].get('accession') == accession:
                                    analysis_id = aid
                                    break

                            if analysis_id and analysis_id in ri.deg_data and contrast_id in ri.deg_data[analysis_id]:
                                deg_df = ri.deg_data[analysis_id][contrast_id]
                                if 'Gene' in deg_df.columns:
                                    all_contrast_genes = set(deg_df['Gene'].tolist())
                                    available_genes.update(all_contrast_genes)
                                    contrasts_with_data.append(f"{accession}:{contrast_id}")

                                    contrast_has_significant_custom_genes = False
                                    if 'adj.P.Val' in deg_df.columns and 'logFC' in deg_df.columns:
                                        significant_contrast_genes = deg_df[
                                            (deg_df['adj.P.Val'] < pval_val) &
                                            (abs(deg_df['logFC']) > lfc_val)
                                        ]['Gene'].tolist()
                                        significant_genes.update(significant_contrast_genes)

                                        custom_genes_in_contrast = [gene for gene in custom_genes_list if gene in significant_contrast_genes]
                                        if custom_genes_in_contrast:
                                            contrast_has_significant_custom_genes = True

                                    if not contrast_has_significant_custom_genes:
                                        contrasts_with_no_significant_custom_genes.append(f"{accession}:{contrast_id}")
                            else:
                                contrasts_without_data.append(f"{accession}:{contrast_id}")

                        found_genes = [gene for gene in custom_genes_list if gene in available_genes]
                        missing_genes = [gene for gene in custom_genes_list if gene not in available_genes]
                        found_but_filtered = [gene for gene in found_genes if gene not in significant_genes]
                        will_display = [gene for gene in found_genes if gene in significant_genes]

                        st.write("**Gene Analysis:**")

                        if will_display:
                            st.write(f"**{len(will_display)} genes will be displayed** in the heatmap")

                        if missing_genes:
                            st.write(f"\n**{len(missing_genes)} genes not found in selected contrasts:**")
                            if len(missing_genes) <= 10:
                                st.write(", ".join(missing_genes))
                            else:
                                st.write(f"{', '.join(missing_genes[:10])}, ... (+{len(missing_genes)-10} more)")

                        if found_but_filtered:
                            st.write(f"\n**{len(found_but_filtered)} genes found but will be filtered out** (never significant in selected contrasts):")
                            if len(found_but_filtered) <= 10:
                                st.write(", ".join(found_but_filtered))
                            else:
                                st.write(f"{', '.join(found_but_filtered[:10])}, ... (+{len(found_but_filtered)-10} more)")

                        st.write(f"\n**Contrast Analysis:**")
                        st.write(f"**{len(contrasts_with_data)} contrasts have data available**")

                        if contrasts_without_data:
                            st.write(f"\n**{len(contrasts_without_data)} contrasts missing data:**")
                            for contrast in contrasts_without_data:
                                st.write(f"• {contrast}")

                        if contrasts_with_no_significant_custom_genes:
                            st.write(f"\n**{len(contrasts_with_no_significant_custom_genes)} contrasts have no significant custom genes:**")
                            for contrast in contrasts_with_no_significant_custom_genes:
                                st.write(f"• {contrast}")

                        if not will_display:
                            st.write(f"\n**ERROR:** No genes will be displayed! All custom genes are either missing or never significant.")
                            return None

                if gene_selection_method == "Frequent DEGs":
                    log_streamlit_user_action(f"Heatmap analysis: {len(selected_contrasts)} contrasts, LFC={lfc_val}, P={pval_val}, genes={gene_count}, method=Frequent DEGs")
                else:
                    log_streamlit_user_action(f"Heatmap analysis: {len(selected_contrasts)} contrasts, LFC={lfc_val}, P={pval_val}, method=Custom, custom_genes={len(custom_genes_list)}")

                return {
                    'contrasts': selected_contrasts,
                    'gene_params': {
                        'lfc_thresh': lfc_val,
                        'pvalue_thresh': pval_val,
                        'gene_count': gene_count,
                        'gene_selection_method': gene_selection_method,
                        'custom_genes': custom_genes_list,
                        'original_genes': original_genes if check_orthologs else None,
                        'ortholog_mapping': ortholog_mapping if check_orthologs else None,
                        'cluster_genes': False if (gene_selection_method == "Custom" and st.session_state.get("heatmap_keep_gene_order_custom", False)) else True
                    }
                }

    return None


@log_streamlit_function
def _create_contrast_table_data_filtered(ri: ResultsIntegrator, selected_datasets: List[str]) -> List[Dict[str, Any]]:
    """Create data for the contrast selection table using standardized validation logic."""
    # Use the centralized validation function
    valid_contrasts = get_valid_contrasts_with_data(ri, selected_datasets)

    # Convert to format expected by heatmap table
    contrast_data = []
    for contrast in valid_contrasts:
        contrast_data.append({
            "Select": False,
            "Accession": contrast["accession"],
            "Contrast": contrast["contrast_name"],
            "Description": contrast["description"]
        })
    # Sort by GEO accession numeric portion, then by contrast name
    def _acc_num(acc: str) -> int:
        m = re.search(r"(\d+)", str(acc) or "")
        return int(m.group(1)) if m else float('inf')

    contrast_data.sort(key=lambda x: (_acc_num(x["Accession"]), x["Accession"], x["Contrast"]))
    return contrast_data


@log_streamlit_function
def _auto_select_genes(
    ri: ResultsIntegrator,
    results_dir: str,
    selected_contrasts: List[Tuple[str, str]],
    gene_params: Dict[str, Any]
) -> List[str]:
    """Select genes based on gene parameters and selected contrasts (analysis_id format)."""
    if not selected_contrasts:
        return []

    gene_selection_method = gene_params.get('gene_selection_method', 'Frequent DEGs')

    try:
        # Group contrasts by organism (selected_contrasts should already be in analysis_id format)
        organism_groups = group_contrasts_by_organism(ri, selected_contrasts)

        if gene_selection_method == 'Custom':
            # Custom gene selection
            custom_genes = gene_params.get('custom_genes', [])
            if not custom_genes:
                return []

            # Collect all available genes using core module
            all_contrasts = [c for contrasts in organism_groups.values() for c in contrasts]
            available_genes = get_available_genes_for_contrasts(ri.deg_data, all_contrasts)

            # Filter custom genes to only those available
            available_custom_genes = [gene for gene in custom_genes if gene in available_genes]

            log_streamlit_event(f"Custom gene selection: {len(available_custom_genes)} of {len(custom_genes)} genes available")
            return available_custom_genes

        else:
            # Frequent DEGs selection
            if len(organism_groups) == 1:
                # Single organism
                organism = list(organism_groups.keys())[0]
                organism_contrasts = organism_groups[organism]

                top_genes = cached_identify_important_genes(
                    results_dir=results_dir,
                    selected_contrasts=organism_contrasts,
                    top_frequent=gene_params.get('gene_count', 50),
                    p_value_threshold=gene_params['pvalue_thresh'],
                    lfc_threshold=gene_params['lfc_thresh']
                )

                log_streamlit_event(f"Auto-selected {len(top_genes)} frequent DEGs for {organism}")
                return top_genes

            else:
                # Multiple organisms - combine genes from each
                all_selected_genes = []
                gene_count_per_organism = gene_params.get('gene_count', 50)

                for organism, organism_contrasts in organism_groups.items():
                    organism_genes = cached_identify_important_genes(
                        results_dir=results_dir,
                        selected_contrasts=organism_contrasts,
                        top_frequent=gene_count_per_organism,
                        p_value_threshold=gene_params['pvalue_thresh'],
                        lfc_threshold=gene_params['lfc_thresh']
                    )
                    all_selected_genes.extend(organism_genes)

                log_streamlit_event(f"Auto-selected {len(all_selected_genes)} frequent DEGs across {len(organism_groups)} organisms")
                return all_selected_genes

    except Exception as e:
        logger.error(f"Error in gene selection: {e}")
        st.error(f"Error selecting genes: {str(e)}")
        return []


@st.fragment
@log_streamlit_function
def _draw_heatmap(
    ri: ResultsIntegrator,
    gene_selection: List[str],
    contrast_pairs: List[Tuple[str, str]],
    p_thresh: Optional[float],
    lfc_thresh_val: Optional[float],
    use_dynamic_filtering: bool,
    hide_empty_rows_cols: bool,
    font_size: int,
    show_grid_lines: bool,
    grid_opacity: float,
    cluster_genes: bool = True
):
    """
    Create and display the heatmap using fragment isolation.
    """
    # Skip execution if AI is currently generating
    if check_ai_generating():
        return

    with st.spinner("Generating heatmap..."):
        try:
            # Use cached figure creation for better performance
            p_threshold = p_thresh if use_dynamic_filtering else None
            lfc_threshold = lfc_thresh_val if use_dynamic_filtering else None

            fig = cached_figure_creation(
                "create_lfc_heatmap",
                ri.results_dir,
                gene_selection,
                contrast_pairs,
                None,  # output_file
                p_threshold,
                lfc_threshold,
                hide_empty_rows_cols,
                font_size,
                show_grid_lines,
                grid_opacity,
                cluster_genes=cluster_genes,
            )

            if fig:
                log_streamlit_event("Heatmap generated successfully")
                # Keep each contrast column readable: scroll instead of squeezing
                render_scrollable_plotly(fig, key="lfc_heatmap_scroll")
                _render_column_key(fig)
                # Persist the last generated figure (legacy fallback not used for UI identity)
                st.session_state['heatmap_last_fig'] = fig

                # Download buttons section
                st.markdown("---")
                st.subheader("Download Options")

                # Prepare reproducible download package with caching for speed
                try:
                    last_df = getattr(ri, 'last_heatmap_df', None)
                    last_ctx = getattr(ri, 'last_heatmap_context', None)

                    # Create two columns for download buttons
                    col1, col2 = st.columns(2)

                    # Build a stable cache key from last_ctx
                    cache_key = None
                    if last_ctx is not None:
                        try:
                            # Use deterministic JSON of context as key
                            cache_key = json.dumps(last_ctx, sort_keys=True, default=str)
                        except Exception:
                            cache_key = str(last_ctx)

                    # Ensure a session cache exists
                    if 'heatmap_dl_cache' not in st.session_state:
                        st.session_state['heatmap_dl_cache'] = {}
                    dl_cache = st.session_state['heatmap_dl_cache']

                    # PDF download button in first column (cached and prepared immediately, using dynamic figure size)
                    with col1:
                        pdf_bytes = None
                        if cache_key and cache_key in dl_cache and dl_cache[cache_key].get('pdf_bytes'):
                            pdf_bytes = dl_cache[cache_key]['pdf_bytes']
                        else:
                            # Use figure's own dimensions for parity with on-screen rendering
                            fig_width = int(fig.layout.width) if fig.layout.width else None
                            fig_height = int(fig.layout.height) if fig.layout.height else None
                            pdf_bytes = plotly_fig_to_pdf_bytes(fig, width=fig_width, height=fig_height, scale=2)
                            if cache_key and pdf_bytes:
                                dl_cache.setdefault(cache_key, {})['pdf_bytes'] = pdf_bytes
                        if pdf_bytes:
                            st.download_button(
                                label="Download as PDF",
                                data=pdf_bytes,
                                file_name=generate_plot_filename("heatmap", "pdf"),
                                mime="application/pdf",
                                help="Download the heatmap as a high-resolution PDF suitable for publications"
                            )

                    # Reproducible package button in second column (cached)
                    with col2:
                        if last_df is not None and last_ctx is not None:
                            zip_bytes = None
                            if cache_key and cache_key in dl_cache and dl_cache[cache_key].get('zip_bytes'):
                                zip_bytes = dl_cache[cache_key]['zip_bytes']
                            else:
                                # Build in-memory ZIP package once
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                                    # heatmap data CSV
                                    csv_bytes = last_df.reset_index().to_csv(index=False).encode('utf-8')
                                    zf.writestr('heatmap_data.csv', csv_bytes)

                                    # metadata JSON - include x-axis default labels mapping
                                    try:
                                        clustered = last_ctx.get('clustered_contrasts') or []
                                        contrast_labels = last_ctx.get('contrast_labels') or []
                                        simplified = last_ctx.get('simplified_labels') or []
                                        x_axis_labels = {c: s for c, s in zip(contrast_labels, simplified) if c in clustered}
                                    except Exception:
                                        x_axis_labels = {}

                                    fig_meta = dict(fig.layout.meta or {})
                                    meta = {
                                        'generated_at': datetime.utcnow().isoformat() + 'Z',
                                        'p_value_threshold': last_ctx.get('p_value_threshold'),
                                        'lfc_threshold': last_ctx.get('lfc_threshold'),
                                        'genes': last_ctx.get('genes'),
                                        'contrast_labels': last_ctx.get('contrast_labels'),
                                        'clustered_genes': last_ctx.get('clustered_genes'),
                                        'clustered_contrasts': last_ctx.get('clustered_contrasts'),
                                        'x_axis_labels': x_axis_labels,
                                        # Accession shown once above each block of columns
                                        'dataset_blocks': fig_meta.get('dataset_blocks', []),
                                        # Default to PDF for publication-quality output
                                        'output_format': 'pdf',
                                        'dpi': 300,
                                        # Plotly layout defaults mirror the app for parity
                                        'font_size': 12,
                                        'show_grid_lines': True,
                                        'grid_opacity': 0.3
                                    }
                                    zf.writestr('metadata.json', json.dumps(meta, indent=2))
                                    zf.writestr('column_key.csv', column_key_csv(fig))

                                    repro_script = script_generation.build_repro_script()
                                    zf.writestr('reproduce_heatmap.py', repro_script)

                                    readme_text = script_generation.build_readme_text()
                                    zf.writestr('README.txt', readme_text)

                                zip_buffer.seek(0)
                                zip_bytes = zip_buffer.getvalue()
                                if cache_key and zip_bytes:
                                    dl_cache.setdefault(cache_key, {})['zip_bytes'] = zip_bytes

                            file_name = f"uorca_heatmap_repro_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.zip"
                            st.download_button(
                                label="Reproduce and edit heatmap",
                                data=zip_bytes,
                                file_name=file_name,
                                mime='application/zip',
                                help="Download a folder to reproduce this heatmap. Use this to make edits to the heatmap, for example to labels, colours, or margins."
                            )
                except Exception as e:
                    logger.debug(f"Could not prepare reproducible package: {e}")

                # Filtering information is now included in the form validation
            else:
                log_streamlit_event("Failed to generate heatmap")
                st.error("Could not generate heatmap. Please check your selections and try adjusting the parameters above.")

        except Exception as e:
            logger.error(f"Error generating heatmap: {str(e)}", exc_info=True)
            st.error(f"Error generating heatmap: {str(e)}")
            _display_heatmap_error_details(e)


COLUMN_KEY_FIELDS = ["#", "Label", "Dataset", "Contrast", "Description"]


def column_key_csv(fig) -> str:
    """CSV of the heatmap's numbered column key (label, dataset, full name, description)."""
    key = (fig.layout.meta or {}).get("column_key", [])
    return pd.DataFrame(key, columns=pd.Index(COLUMN_KEY_FIELDS)).to_csv(index=False)


def _render_column_key(fig) -> None:
    """Numbered key under the heatmap: each column's short label, dataset,
    full contrast name and description, with a CSV download."""
    key = (fig.layout.meta or {}).get("column_key", [])
    if not key:
        st.error("The heatmap has no column key; the column labels cannot be explained.")
        return
    st.markdown("**Column key** (left to right)")
    st.dataframe(
        pd.DataFrame(key, columns=pd.Index(COLUMN_KEY_FIELDS)),
        hide_index=True,
        use_container_width=True,
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Label": st.column_config.TextColumn("Label on chart"),
            "Contrast": st.column_config.TextColumn("Full contrast name"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
        key="heatmap_column_key_table",
    )
    st.download_button(
        "Download column key (CSV)",
        data=column_key_csv(fig),
        file_name=generate_plot_filename("heatmap_column_key", "csv"),
        mime="text/csv",
        key="heatmap_column_key_download",
    )


@log_streamlit_function
def _display_filtered_elements_info(ri: ResultsIntegrator):
    """Display information about filtered genes and contrasts."""
    try:
        filtered_info = ri.get_last_heatmap_filtered_info()
        filtered_genes = filtered_info.get('genes', [])
        filtered_contrasts = filtered_info.get('contrasts', [])

        if filtered_genes or filtered_contrasts:
            with st.expander("Hidden Elements (No Significant Values)", expanded=False):
                if filtered_genes:
                    st.subheader("Hidden Genes")
                    st.write(f"**{len(filtered_genes)} genes** hidden because they have no significant values in any selected contrasts:")
                    genes_text = ", ".join(filtered_genes)
                    st.text_area("Genes:", value=genes_text, height=100, disabled=True, key="heatmap_filtered_genes_display")

                if filtered_contrasts:
                    st.subheader("Hidden Contrasts")
                    st.write(f"**{len(filtered_contrasts)} contrasts** hidden because they have no significant values for any selected genes:")
                    for contrast in filtered_contrasts:
                        st.write(f"• {contrast}")

    except Exception as e:
        logger.warning(f"Could not display filtered elements info: {e}")


@log_streamlit_function
def _display_heatmap_error_details(error: Exception):
    """Display detailed error information in an expandable section."""
    with st.expander("Heatmap Error Details", expanded=False):
        st.code(traceback.format_exc())


