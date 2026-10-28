"""
Analysis Plots Tab for UORCA Explorer.

This tab displays quality control and differential expression plots from individual datasets.
"""

import os
import logging
import pandas as pd
import streamlit as st
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

from .helpers import (
    check_ai_generating,
    setup_fragment_decorator,
    log_streamlit_function,
    log_streamlit_event,
    get_organism_display_name,
    log_streamlit_tab,
    is_analysis_successful,
    get_valid_contrasts_from_analysis_info,
    validate_contrast_has_deg_data,
    create_dataset_download_package,
    generate_plot_filename
)
from uorca.gui.results_integration import ResultsIntegrator, ADJUSTED_P_LABEL, ADJUSTED_P_HELP, RAW_P_LABEL
# R script bundling is now integrated into create_dataset_download_package via components['r_script']

# Import single analysis plotting functions
try:
    from uorca.gui.single_analysis_plots import (
        create_pca_plot,
        create_volcano_plot,
        create_ma_plot,
        create_deg_heatmap,
        load_sample_groups,
    )
    SINGLE_PLOTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Single analysis plots not available: {e}")
    SINGLE_PLOTS_AVAILABLE = False

# Set up fragment decorator
setup_fragment_decorator()

logger = logging.getLogger(__name__)


@log_streamlit_tab("View Dataset Analyses")
def render_analysis_plots_tab(ri: ResultsIntegrator, results_dir: str):
    """
    Render the analysis plots tab.

    Args:
        ri: ResultsIntegrator instance
        results_dir: Path to the results directory
    """
    st.header("View Dataset Analyses")
    st.markdown("Explore QC plots, analysis results, and metadata for individual datasets.")



    # Main analysis plots interface
    _render_analysis_plots_interface(ri, results_dir)





@st.fragment
@log_streamlit_function
def _render_analysis_plots_interface(ri: ResultsIntegrator, results_dir: str):
    """Render the main analysis plots interface using fragment isolation."""
    # Skip execution if AI is currently generating
    if check_ai_generating():
        return

    # Create formatted dataset options with GSExxxx - Title format
    dataset_options = []
    dataset_mapping = {}  # Map display name to analysis_id

    if ri:
        # Get datasets with successful analyses only (using standardized validation)
        for analysis_id in ri.analysis_info.keys():
            if is_analysis_successful(ri, analysis_id):
                info = ri.analysis_info[analysis_id]
                accession = info.get("accession", analysis_id)

                # Get title from dataset info if available
                title = ""
                if hasattr(ri, "dataset_info") and analysis_id in getattr(ri, "dataset_info", {}):
                    title = ri.dataset_info[analysis_id].get("title", "")
                    # Remove "Title:" prefix if present
                    if isinstance(title, str) and title.startswith("Title:"):
                        title = title[6:].strip()

                # Format as GSExxxx - Title or just GSExxxx if no title
                display_name = f"{accession} - {title}" if title else accession

                dataset_options.append(display_name)
                dataset_mapping[display_name] = analysis_id

        # Sort alphabetically
        dataset_options.sort()

    # If there are no valid datasets, inform the user and exit early
    if not dataset_options:
        log_streamlit_event("No valid datasets available for analysis plots")
        st.info("No valid datasets available. Ensure analyses completed successfully.")
        return

    selected_dataset_display = st.selectbox(
        "Select a dataset to view analysis plots:",
        dataset_options,
        key="analysis_dataset_select"
    )
    selected_dataset = dataset_mapping.get(selected_dataset_display, selected_dataset_display) if selected_dataset_display else None

    if not selected_dataset:
        log_streamlit_event("No dataset selected for analysis plots")
        st.info("Please select a dataset to view RNA-seq analysis plots.")
        return

    # Add download options in expandable panel
    if selected_dataset and ri:
        dataset_accession = ri.analysis_info.get(selected_dataset, {}).get("accession", selected_dataset)
        
        with st.expander("Download Options", expanded=False):
            # Initialize session state for checkboxes if not exists
            if f"download_components_{selected_dataset}" not in st.session_state:
                st.session_state[f"download_components_{selected_dataset}"] = {
                    'metadata': True,
                    'abundance': True,
                    'qc_plots': True,
                    'deg_analysis': True,
                    'include_pdfs': False  # Default to false for speed
                }
            
            # Create checkboxes for component selection
            st.markdown("**Select components to include:**")
            
            col1, col2 = st.columns(2)
            with col1:
                include_metadata = st.checkbox(
                    "Metadata files",
                    value=st.session_state[f"download_components_{selected_dataset}"].get('metadata', True),
                    key=f"checkbox_metadata_{selected_dataset}",
                    help="Include contrasts.csv, sample information, and analysis configuration"
                )
                include_qc = st.checkbox(
                    "QC plots",
                    value=st.session_state[f"download_components_{selected_dataset}"].get('qc_plots', True),
                    key=f"checkbox_qc_{selected_dataset}",
                    help="Include PCA, MDS, normalization, and other quality control plots"
                )
            
            with col2:
                include_abundance = st.checkbox(
                    "Abundance data (Kallisto)",
                    value=st.session_state[f"download_components_{selected_dataset}"].get('abundance', True),
                    key=f"checkbox_abundance_{selected_dataset}",
                    help="Include transcript-level quantification results for each sample"
                )
                include_deg = st.checkbox(
                    "DEG analysis",
                    value=st.session_state[f"download_components_{selected_dataset}"].get('deg_analysis', True),
                    key=f"checkbox_deg_{selected_dataset}",
                    help="Include differential expression results, plots, and CPM data for all contrasts"
                )
            
            # R script option
            include_rscript = st.checkbox(
                "R script",
                value=st.session_state[f"download_components_{selected_dataset}"].get('r_script', False),
                key=f"checkbox_rscript_{selected_dataset}",
                help=(
                    "Include a single self-contained RNAseq.R that reproduces the analysis. "
                    "If selected, sample metadata and abundance files are always included regardless of other selections."
                )
            )
            
            # PDF generation option
            st.markdown("**Additional options:**")
            include_pdfs = st.checkbox(
                "Generate PDF versions of DEG plots (slower)",
                value=st.session_state[f"download_components_{selected_dataset}"].get('include_pdfs', False),
                key=f"checkbox_pdfs_{selected_dataset}",
                help="Generate high-resolution PDF versions of volcano plots, MA plots, and heatmaps for each contrast. This significantly increases processing time but provides publication-quality vector graphics."
            )
            
            # Update session state
            components = {
                'metadata': include_metadata,
                'abundance': include_abundance,
                'qc_plots': include_qc,
                'deg_analysis': include_deg,
                'r_script': include_rscript,
                'include_pdfs': include_pdfs
            }
            st.session_state[f"download_components_{selected_dataset}"] = components
            
            # Download section with clear two-step process
            st.markdown("---")
            st.markdown("**Download Process:**")
            
            # Check if package is ready
            if f"package_ready_{selected_dataset}" not in st.session_state:
                # Step 1: Prepare package button
                st.info("**Step 1**: Click below to prepare your customized package. Once ready, a download button will appear.")
                
                if st.button("Prepare Download Package", key=f"download_btn_{selected_dataset}", type="primary"):
                    # Check if at least one component is selected
                    if not any([include_metadata, include_abundance, include_qc, include_deg, include_rscript]):
                        st.error("Please select at least one component to download.")
                    else:
                        spinner_text = "Preparing your download package..."
                        if include_pdfs and include_deg:
                            num_contrasts = len(ri.deg_data.get(selected_dataset, {}))
                            if num_contrasts > 0:
                                spinner_text = f"Preparing package with {num_contrasts} contrast(s)... Generating PDF versions of plots, this may take a moment."
                        
                        # If R script selected, enforce metadata + abundance inclusion
                        effective_components = components.copy()
                        if effective_components.get('r_script'):
                            effective_components['metadata'] = True
                            effective_components['abundance'] = True
                        with st.spinner(spinner_text):
                            package_bytes = create_dataset_download_package(
                                ri=ri,
                                results_dir=results_dir,
                                dataset_id=selected_dataset,
                                dataset_accession=dataset_accession,
                                components=effective_components
                            )
                        
                        if package_bytes:
                            # Store package in session state
                            st.session_state[f"package_ready_{selected_dataset}"] = {
                                'bytes': package_bytes,
                                'filename': f"{dataset_accession}_analysis_package.zip"
                            }
                            st.rerun()
                        else:
                            st.warning("Could not create download package. Some files may be missing.")
            else:
                # Step 2: Download ready
                package_data = st.session_state[f"package_ready_{selected_dataset}"]
                st.success("**Step 2**: Your package is ready! Click the button below to download.")
                
                col1, col2, col3 = st.columns([2, 3, 2])
                with col1:
                    st.download_button(
                        label="Download Package",
                        data=package_data['bytes'],
                        file_name=package_data['filename'],
                        mime="application/zip",
                        key=f"download_final_{selected_dataset}",
                        use_container_width=True
                    )
                with col2:
                    if st.button("Prepare New Package", key=f"reset_download_{selected_dataset}"):
                        # Clear the package data to allow reconfiguration
                        del st.session_state[f"package_ready_{selected_dataset}"]
                        st.rerun()

            # Note: R reproduction is integrated into the package via the 'R script' option above
    
    st.markdown("---")  # Add separator after download section

    # Load sample groups for the selected dataset
    groups = _load_sample_groups(ri, selected_dataset)

    # Handle the plot display for the selected dataset
    base_path = os.path.join(ri.results_dir, selected_dataset, "RNAseqAnalysis")

    # Check if the path exists
    if not os.path.exists(base_path):
        log_streamlit_event(f"RNAseqAnalysis folder not found for {selected_dataset}")
        st.error(f"Could not find RNAseqAnalysis folder for {selected_dataset}")
        return

    # Display QC plots
    _render_qc_plots(ri, selected_dataset, base_path, groups)

    # Display differential expression plots
    _render_de_plots(ri, selected_dataset, base_path, groups)

    # Display metadata section
    _render_metadata_section(ri, selected_dataset)


@log_streamlit_function
def _load_sample_groups(ri: ResultsIntegrator, selected_dataset: str) -> Optional[Dict]:
    """Load sample groups for the selected dataset."""
    groups = None
    if selected_dataset in ri.cpm_data:
        try:
            cpm_df = ri.cpm_data[selected_dataset]
            analysis_info = None
            if hasattr(ri, "analysis_info"):
                analysis_info = ri.analysis_info.get(selected_dataset)

            if SINGLE_PLOTS_AVAILABLE:
                groups = load_sample_groups(ri.results_dir, selected_dataset, cpm_df, analysis_info)
        except Exception as e:
            logger.warning(f"Could not load sample groups for {selected_dataset}: {e}")
            groups = None
    return groups


@log_streamlit_function
def _render_qc_plots(ri: ResultsIntegrator, selected_dataset: str, base_path: str, groups: Optional[Dict]):
    """Render quality control plots section."""
    st.subheader("Quality Control and Normalisation")

    # Define QC plot files and descriptions
    qc_plot_files = {
        "MDS Plot": {
            "file": os.path.join(base_path, "MDS.png"),
            "description": "**Multi-dimensional scaling plot** showing sample relationships based on their gene expression profiles. This is analogous to a PCA plot.\n\n- **What to look for**: Samples from the same experimental group should cluster together. Outliers may indicate technical issues.\n\n- **Technical details**: Distances on the plot represent leading log2-fold changes between samples (the average of the largest absolute log2-fold changes)."
        },
        "Filtering Density": {
            "file": os.path.join(base_path, "filtering_density.png"),
            "description": "**Density plot** showing distribution of gene expression levels (log-CPM) before and after filtering low-expression genes.\n\n- **What to look for**: After filtering (red curve), the left peak of very low-expressed genes should be reduced or eliminated.\n\n- **Technical details**: Low-expression genes are filtered out because they provide little statistical power for differential expression analysis and can increase the multiple testing burden."
        },
        "Normalisation Boxplots": {
            "file": os.path.join(base_path, "normalization_boxplots.png"),
            "description": "**Boxplots** showing expression distributions before (left) and after (right) normalisation.\n\n- **What to look for**: After normalisation, all samples should have similar median expression and comparable distributions.\n\n- **Technical details**: Normalisation adjusts for technical differences in sequencing depth and composition between libraries to make samples comparable."
        },
        "Mean-Variance Relationship (voom)": {
            "file": os.path.join(base_path, "voom_mean_variance.png"),
            "description": "**Voom transformation plot** showing how the variance of gene expression depends on the mean expression level.\n\n- **What to look for**: A smooth decreasing trend where variance is higher for low-expressed genes and stabilises for highly-expressed genes.\n\n- **Technical details**: The voom method transforms count data to log-CPM values and estimates the mean-variance relationship to assign appropriate weights for linear modelling."
        },
        "SA Plot": {
            "file": os.path.join(base_path, "sa_plot.png"),
            "description": "**Sigma vs Average plot** showing the standard deviation of normalised expression values against their average expression.\n\n- **What to look for**: A smooth trend without unusual patterns or outliers.\n\n- **Technical details**: This diagnostic plot from limma helps visualise how gene-wise variances change with expression level after fitting the linear model."
        }
    }

    # Display QC plots in tabs
    plot_tabs = st.tabs(["Quality Overview", "Expression Filtering", "Normalisation", "Variance Modeling"])

    # Tab 1: Quality Overview (interactive PCA plot replacing MDS)
    with plot_tabs[0]:
        _render_quality_overview_tab(ri, selected_dataset, qc_plot_files, groups)

    # Tab 2: Expression Filtering
    with plot_tabs[1]:
        _render_filtering_tab(qc_plot_files)

    # Tab 3: Normalisation
    with plot_tabs[2]:
        _render_normalisation_tab(qc_plot_files)

    # Tab 4: Variance Modeling
    with plot_tabs[3]:
        _render_variance_modeling_tab(qc_plot_files)


@log_streamlit_function
def _render_quality_overview_tab(ri: ResultsIntegrator, selected_dataset: str, qc_plot_files: Dict, groups: Optional[Dict]):
    """Render the quality overview tab with PCA plot."""
    pca_fig = None
    if ri and selected_dataset in ri.cpm_data and SINGLE_PLOTS_AVAILABLE:
        try:
            cpm_df = ri.cpm_data[selected_dataset]
            pca_fig = create_pca_plot(cpm_df, groups)
        except Exception:
            pca_fig = None

    if pca_fig is not None:
        st.plotly_chart(pca_fig, use_container_width=True)
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(qc_plot_files["MDS Plot"]["description"])
    elif os.path.exists(qc_plot_files["MDS Plot"]["file"]):
        st.image(qc_plot_files["MDS Plot"]["file"])
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(qc_plot_files["MDS Plot"]["description"])
    else:
        st.info("MDS plot not available for this dataset.")


@log_streamlit_function
def _render_filtering_tab(qc_plot_files: Dict):
    """Render the expression filtering tab."""
    if os.path.exists(qc_plot_files["Filtering Density"]["file"]):
        st.image(qc_plot_files["Filtering Density"]["file"])
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(qc_plot_files["Filtering Density"]["description"])
    else:
        st.info("Filtering density plot not available for this dataset.")


@log_streamlit_function
def _render_normalisation_tab(qc_plot_files: Dict):
    """Render the normalisation tab."""
    if os.path.exists(qc_plot_files["Normalisation Boxplots"]["file"]):
        st.image(qc_plot_files["Normalisation Boxplots"]["file"])
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(qc_plot_files["Normalisation Boxplots"]["description"])
    else:
        st.info("Normalisation boxplots not available for this dataset.")


@log_streamlit_function
def _render_variance_modeling_tab(qc_plot_files: Dict):
    """Render the variance modeling tab."""
    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(qc_plot_files["Mean-Variance Relationship (voom)"]["file"]):
            st.subheader("Voom Transformation")
            st.image(qc_plot_files["Mean-Variance Relationship (voom)"]["file"])
            with st.expander("What does this plot mean?"):
                st.markdown(qc_plot_files["Mean-Variance Relationship (voom)"]["description"])

    with col2:
        if os.path.exists(qc_plot_files["SA Plot"]["file"]):
            st.subheader("Variance Trend")
            st.image(qc_plot_files["SA Plot"]["file"])
            with st.expander("What does this plot mean?"):
                st.markdown(qc_plot_files["SA Plot"]["description"])


@log_streamlit_function
def _render_de_plots(ri: ResultsIntegrator, selected_dataset: str, base_path: str, groups: Optional[Dict]):
    """Render differential expression plots section."""
    st.markdown("---")
    st.subheader("Differential Expression Results")

    # Get valid contrasts from analysis_info.json (no directory fallback)
    analysis_id = selected_dataset

    # Check if analysis was successful
    if not is_analysis_successful(ri, analysis_id):
        log_streamlit_event(f"Analysis not successful for {selected_dataset}")
        st.info("This analysis was not completed successfully.")
        return

    # Get contrasts from analysis_info
    contrasts = get_valid_contrasts_from_analysis_info(ri, analysis_id)

    # Filter to only contrasts that have DEG data
    valid_contrasts = []
    for contrast in contrasts:
        contrast_name = contrast["name"]
        if validate_contrast_has_deg_data(ri, analysis_id, contrast_name):
            valid_contrasts.append(contrast_name)

    if not valid_contrasts:
        log_streamlit_event(f"No valid contrasts with DEG data found for {selected_dataset}")
        st.info("No contrast-specific plots found for this dataset.")
        return

    # Create contrast selection interface
    selected_contrast = _render_contrast_selection(ri, valid_contrasts, selected_dataset)

    if selected_contrast:
        contrast_path = os.path.join(base_path, selected_contrast)

        # Show contrast description
        _display_contrast_description(ri, selected_contrast, selected_dataset)

        # Display DE plots in tabs
        _render_de_plot_tabs(ri, selected_dataset, selected_contrast, contrast_path, groups)

        # Display DEG table
        _render_deg_table(selected_dataset, selected_contrast, contrast_path)


@log_streamlit_function
def _render_contrast_selection(ri: ResultsIntegrator, contrast_dirs: List[str],
                               analysis_id: str) -> Optional[str]:
    """Render contrast selection interface and return selected contrast."""
    # Find contrast descriptions for this dataset's contrasts
    contrast_descriptions = {}
    for c_id in contrast_dirs:
        entry = ri.get_contrast_entry(analysis_id, c_id)
        if entry and isinstance(entry.get('description'), str):
            contrast_descriptions[c_id] = entry['description']

    # Create a list of contrast options with descriptions
    contrast_options = []
    for c_id in contrast_dirs:
        label = f"{c_id}"
        if c_id in contrast_descriptions and contrast_descriptions[c_id]:
            # Truncate description if too long
            desc = contrast_descriptions[c_id]
            if len(desc) > 80:
                desc = desc[:77] + "..."
            label += f" - {desc}"
        contrast_options.append(label)

    # Create mapping from display string to actual contrast ID
    contrast_display_to_id = {opt: contrast_dirs[i] for i, opt in enumerate(contrast_options)}

    selected_contrast_display = st.selectbox("Select a contrast:", contrast_options, key="analysis_contrast_select")
    return contrast_display_to_id.get(selected_contrast_display)


@log_streamlit_function
def _display_contrast_description(ri: ResultsIntegrator, selected_contrast: str, analysis_id: str):
    """Display the full contrast description if available."""
    entry = ri.get_contrast_entry(analysis_id, selected_contrast)
    if entry and isinstance(entry.get('description'), str) and entry['description']:
        st.markdown(f"**Description:** {entry['description']}")


@log_streamlit_function
def _render_de_plot_tabs(ri: ResultsIntegrator, selected_dataset: str, selected_contrast: str,
                        contrast_path: str, groups: Optional[Dict]):
    """Render differential expression plot tabs."""
    # Define plot information with descriptions
    contrast_plot_files = {
        "Heatmap of Top DEGs": {
            "file": os.path.join(contrast_path, "heatmap_top50.png"),
            "description": "**Hierarchical clustering heatmap** showing the top 50 differentially expressed genes for this contrast.\n\n- **What to look for**: Distinct expression patterns that separate the experimental groups. Groups of co-regulated genes (clusters) may represent functional modules.\n\n- **Technical details**: Genes are clustered based on their expression similarity. Color intensity represents expression level (red = high, blue = low)."
        },
        "MA Plot": {
            "file": os.path.join(contrast_path, "ma_plot.png"),
            "description": "**MA plot** showing the relationship between average expression level (x-axis) and log2 fold change (y-axis).\n\n- **What to look for**: Symmetrical distribution around y=0 with significant genes (red points) distributed across expression levels. Asymmetry might indicate normalisation issues.\n\n- **Technical details**: Each point represents a gene. M (y-axis) is the log-ratio of expression (log2 fold change), and A (x-axis) is the average expression. Red points are statistically significant DEGs."
        },
        "Volcano Plot": {
            "file": os.path.join(contrast_path, "volcano_plot.png"),
            "description": f"**Volcano plot** showing the relationship between statistical significance (-log10 adjusted p-value (BH) on y-axis) and biological significance (log2 fold change on x-axis).\n\n- **What to look for**: Genes in the upper left and right corners are both statistically significant and have large fold changes, making them the most interesting candidates.\n\n- **Technical details**: Each point represents a gene. Red and blue points have {ADJUSTED_P_LABEL} below the threshold and |log2FC| above it.",
            # The static PNG written by RNAseq.R plots the raw p-value on its y-axis
            "static_description": "**Volcano plot** (static image from the R pipeline) showing statistical significance (-log10 raw p-value on y-axis) against log2 fold change (x-axis).\n\n- **What to look for**: Genes in the upper left and right corners have small raw p-values and large fold changes.\n\n- **Technical details**: Each point represents a gene. The y-axis is the raw p-value. The 10 labelled genes have the smallest raw p-values; this image does not mark significance by adjusted p-value (BH)."
        }
    }

    # Create tabs for the contrast-specific plots
    de_plot_tabs = st.tabs(["Volcano Plot", "MA Plot", "Heatmap"])

    # Tab 1: Volcano Plot
    with de_plot_tabs[0]:
        _render_volcano_plot_tab(ri, selected_dataset, selected_contrast, contrast_plot_files)

    # Tab 2: MA Plot
    with de_plot_tabs[1]:
        _render_ma_plot_tab(ri, selected_dataset, selected_contrast, contrast_plot_files)

    # Tab 3: Heatmap
    with de_plot_tabs[2]:
        _render_heatmap_plot_tab(ri, selected_dataset, selected_contrast, contrast_path, contrast_plot_files, groups)


@log_streamlit_function
def _render_volcano_plot_tab(ri: ResultsIntegrator, selected_dataset: str, selected_contrast: str,
                           contrast_plot_files: Dict):
    """Render volcano plot tab."""
    volcano_fig = None
    if (SINGLE_PLOTS_AVAILABLE and ri and
        selected_dataset in ri.deg_data and
        selected_contrast in ri.deg_data[selected_dataset]):
        try:
            deg_df = ri.deg_data[selected_dataset][selected_contrast]
            volcano_fig = create_volcano_plot(deg_df)
        except Exception as e:
            logger.error(f"Interactive volcano plot failed for {selected_dataset}/{selected_contrast}: {e}", exc_info=True)
            st.warning(f"Could not create the interactive volcano plot: {e}")
            volcano_fig = None

    if volcano_fig is not None:
        st.plotly_chart(volcano_fig, use_container_width=True)
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(contrast_plot_files["Volcano Plot"]["description"])
    elif os.path.exists(contrast_plot_files["Volcano Plot"]["file"]):
        st.image(contrast_plot_files["Volcano Plot"]["file"])
        st.caption("Static image: the y-axis is -log10(raw p-value).")
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(contrast_plot_files["Volcano Plot"]["static_description"])
    else:
        st.info("Volcano plot not available for this contrast.")


@log_streamlit_function
def _render_ma_plot_tab(ri: ResultsIntegrator, selected_dataset: str, selected_contrast: str,
                       contrast_plot_files: Dict):
    """Render MA plot tab."""
    ma_fig = None
    if (SINGLE_PLOTS_AVAILABLE and ri and
        selected_dataset in ri.deg_data and
        selected_contrast in ri.deg_data[selected_dataset]):
        try:
            deg_df = ri.deg_data[selected_dataset][selected_contrast]
            ma_fig = create_ma_plot(deg_df)
        except Exception as e:
            logger.error(f"Interactive MA plot failed for {selected_dataset}/{selected_contrast}: {e}", exc_info=True)
            st.warning(f"Could not create the interactive MA plot: {e}")
            ma_fig = None

    if ma_fig is not None:
        st.plotly_chart(ma_fig, use_container_width=True)
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(contrast_plot_files["MA Plot"]["description"])
    elif os.path.exists(contrast_plot_files["MA Plot"]["file"]):
        st.image(contrast_plot_files["MA Plot"]["file"])
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(contrast_plot_files["MA Plot"]["description"])
    else:
        st.info("MA plot not available for this contrast.")


@log_streamlit_function
def _render_heatmap_plot_tab(ri: ResultsIntegrator, selected_dataset: str, selected_contrast: str,
                           contrast_path: str, contrast_plot_files: Dict, groups: Optional[Dict]):
    """Render heatmap plot tab."""
    heatmap_fig = None
    if (SINGLE_PLOTS_AVAILABLE and ri and
        selected_dataset in ri.deg_data and
        selected_contrast in ri.deg_data[selected_dataset] and
        selected_dataset in ri.cpm_data):
        try:
            deg_df = ri.deg_data[selected_dataset][selected_contrast]
            cpm_df = ri.cpm_data[selected_dataset]
            heatmap_fig = create_deg_heatmap(cpm_df, deg_df, groups)
        except Exception as e:
            st.warning(f"Error creating interactive DEG heatmap: {e}")
            heatmap_fig = None

    if heatmap_fig is not None:
        st.plotly_chart(heatmap_fig, use_container_width=True)
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(contrast_plot_files["Heatmap of Top DEGs"]["description"])
    elif os.path.exists(contrast_plot_files["Heatmap of Top DEGs"]["file"]):
        st.warning("Interactive heatmap unavailable; attempting to load static image as fallback.")
        st.image(contrast_plot_files["Heatmap of Top DEGs"]["file"])
        with st.expander("What does this plot mean?", expanded=True):
            st.markdown(contrast_plot_files["Heatmap of Top DEGs"]["description"])
    else:
        st.info("Heatmap not available for this contrast.")


@log_streamlit_function
def _render_deg_table(selected_dataset: str, selected_contrast: str, contrast_path: str):
    """Render the DEG table section."""
    deg_file = os.path.join(contrast_path, "DEG.csv")
    if not os.path.exists(deg_file):
        return

    st.markdown("---")
    with st.expander("View Differentially Expressed Genes Table", expanded=False):
        try:
            deg_df = pd.read_csv(deg_file)

            # Create a copy for display, preserving original numeric types
            display_df = deg_df.copy()

            # Format p-values for display while keeping them numeric for sorting
            column_config = {}
            if 'adj.P.Val' in display_df.columns:
                try:
                    display_df['adj.P.Val'] = pd.to_numeric(display_df['adj.P.Val'])
                    column_config['adj.P.Val'] = st.column_config.NumberColumn(
                        ADJUSTED_P_LABEL,
                        format="%.2e",
                        help=ADJUSTED_P_HELP
                    )
                except:
                    st.warning("Could not convert adj.P.Val to numeric format for sorting.")

            if 'P.Value' in display_df.columns:
                try:
                    display_df['P.Value'] = pd.to_numeric(display_df['P.Value'])
                    column_config['P.Value'] = st.column_config.NumberColumn(
                        RAW_P_LABEL,
                        format="%.2e",
                        help="Raw (unadjusted) p-value (limma column P.Value)"
                    )
                except:
                    st.warning("Could not convert the raw p-value column (P.Value) to numeric format for sorting.")

            if 'logFC' in display_df.columns:
                column_config['logFC'] = st.column_config.NumberColumn(
                    "logFC",
                    format="%.2f",
                    help="Log2 fold change"
                )

            if 'Gene' in display_df.columns:
                column_config['Gene'] = st.column_config.TextColumn(
                    "Gene",
                    width="auto",
                    help="Gene symbol or identifier"
                )

            if 'Description' in display_df.columns:
                column_config['Description'] = st.column_config.TextColumn(
                    "Description",
                    width="medium",
                    help="Contrast description"
                )

            # Show the dataframe with proper formatting
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config=column_config
            )


        except Exception as e:
            st.error(f"Error loading DEG file: {str(e)}")


@log_streamlit_function
def _render_metadata_section(ri: ResultsIntegrator, selected_dataset: str):
    """Render the metadata section showing the original dataset metadata."""
    st.markdown("---")

    # Get accession for the selected dataset
    accession = selected_dataset
    if selected_dataset in ri.analysis_info:
        accession = ri.analysis_info[selected_dataset].get('accession', selected_dataset)

    # Look for metadata file
    metadata_file = os.path.join(ri.results_dir, selected_dataset, "metadata", f"{accession}_metadata.csv")

    if not os.path.exists(metadata_file):
        return

    with st.expander("View Original Dataset Metadata", expanded=False):
        try:
            metadata_df = pd.read_csv(metadata_file)

            st.subheader(f"Original Metadata for {accession}")
            st.markdown(f"*Source file: {accession}_metadata.csv*")

            # Display the metadata table
            st.dataframe(
                metadata_df,
                use_container_width=True
            )


        except Exception as e:
            st.error(f"Error loading metadata file: {str(e)}")
