"""Integration tests for cross-module workflows."""
import pytest
import pandas as pd
from uorca.core import organism_utils, data_formatters, gene_selection, validation


@pytest.mark.integration
def test_organism_to_gene_filtering_workflow(sample_deg_data_dict, sample_analysis_info):
    """Test complete workflow: group by organism -> filter genes."""
    # Step 1: Group datasets by organism
    selected_datasets = ['GSE123456', 'GSE789012']
    organism_groups = organism_utils.group_datasets_by_organism(sample_analysis_info, selected_datasets)

    assert len(organism_groups) == 2
    assert 'Homo sapiens' in organism_groups

    # Step 2: Get contrasts for one organism
    contrasts = [('GSE123456', 'Treatment_vs_Control')]

    # Step 3: Get available genes for those contrasts
    available_genes = gene_selection.get_available_genes_for_contrasts(
        sample_deg_data_dict,
        contrasts
    )

    assert len(available_genes) > 0

    # Step 4: Filter genes by organism
    test_genes = list(available_genes)[:3]
    filtered = organism_utils.filter_genes_by_organism(
        deg_data=sample_deg_data_dict,
        cpm_data={},
        analysis_info=sample_analysis_info,
        genes=test_genes,
        organism='Homo sapiens',
        selected_contrasts=contrasts
    )

    assert len(filtered) <= len(test_genes)
    assert all(g in available_genes for g in filtered)


@pytest.mark.integration
def test_deg_identification_to_validation_workflow(sample_deg_data_dict):
    """Test workflow: identify DEGs -> validate custom genes."""
    # Step 1: Identify frequent DEGs
    contrasts = [('GSE123456', 'Treatment_vs_Control'), ('GSE789012', 'KO_vs_WT')]
    top_genes = gene_selection.identify_frequent_degs(
        deg_data=sample_deg_data_dict,
        contrasts=contrasts,
        top_n=10,
        p_thresh=0.05,
        lfc_thresh=1.0
    )

    assert len(top_genes) > 0

    # Step 2: Get all available genes
    available = gene_selection.get_available_genes_for_contrasts(
        sample_deg_data_dict,
        contrasts
    )

    # Step 3: Validate that identified genes are in available set
    found, missing = gene_selection.validate_custom_genes(top_genes, available)

    assert len(found) == len(top_genes)
    assert len(missing) == 0


@pytest.mark.integration
def test_data_formatting_with_organism_grouping(sample_analysis_info, sample_deg_data_dict):
    """Test workflow: format data tables with organism awareness."""
    # Step 1: Create dataset info table
    dataset_table = data_formatters.create_dataset_info_table(
        analysis_info=sample_analysis_info,
        dataset_info={},
        deg_data=sample_deg_data_dict
    )

    assert len(dataset_table) == 2
    assert 'Organism' in dataset_table.columns

    # Step 2: Group by organisms
    selected_datasets = dataset_table['Accession'].tolist()
    organism_groups = organism_utils.group_datasets_by_organism(
        sample_analysis_info,
        selected_datasets
    )

    # Step 3: Verify organism counts match
    unique_organisms = dataset_table['Organism'].nunique()
    assert len(organism_groups) == unique_organisms


@pytest.mark.integration
def test_validation_chain(sample_deg_data_dict):
    """Test complete validation chain for heatmap parameters."""
    # Validate thresholds
    lfc_val, pval_val, thresh_error = validation.validate_threshold_values("1.5", "0.05")
    assert not thresh_error
    assert lfc_val == 1.5

    # Validate gene count
    gene_count, count_error = validation.validate_gene_count("50")
    assert not count_error
    assert gene_count == 50

    # Get available genes using validated parameters
    contrasts = [('GSE123456', 'Treatment_vs_Control')]
    top_genes = gene_selection.identify_frequent_degs(
        deg_data=sample_deg_data_dict,
        contrasts=contrasts,
        top_n=gene_count,
        p_thresh=pval_val,
        lfc_thresh=lfc_val
    )

    # Validate custom gene list
    assert validation.validate_custom_gene_list(top_genes)

    # Validate contrasts
    assert validation.validate_contrasts_selected(contrasts)


@pytest.mark.integration
def test_multi_organism_gene_selection(sample_deg_data_dict, sample_analysis_info):
    """Test gene selection across multiple organisms."""
    # Step 1: Group contrasts by organism
    contrasts = [
        ('GSE123456', 'Treatment_vs_Control'),
        ('GSE789012', 'KO_vs_WT')
    ]
    organism_groups = organism_utils.group_contrasts_by_organism(
        sample_analysis_info,
        contrasts
    )

    assert len(organism_groups) == 2

    # Step 2: Select genes for each organism
    all_selected_genes = []
    for organism, org_contrasts in organism_groups.items():
        genes = gene_selection.identify_frequent_degs(
            deg_data=sample_deg_data_dict,
            contrasts=org_contrasts,
            top_n=5,
            p_thresh=0.05,
            lfc_thresh=1.0
        )
        all_selected_genes.extend(genes)

    # Step 3: Verify we got genes from multiple organisms
    assert len(all_selected_genes) > 0


@pytest.mark.integration
def test_contrast_table_with_sorting(sample_analysis_info, sample_deg_data_dict):
    """Test creating and sorting contrast tables."""
    # Step 1: Create contrast table
    contrast_table = data_formatters.create_contrast_table_data(
        analysis_info=sample_analysis_info,
        deg_data=sample_deg_data_dict,
        contrast_info={},
        selected_datasets=['GSE123456', 'GSE789012']
    )

    assert len(contrast_table) > 0

    # Step 2: Convert to DataFrame and sort
    df = pd.DataFrame(contrast_table)
    sorted_df = data_formatters.sort_by_geo_accession(df)

    # Verify sorting maintained all rows
    assert len(sorted_df) == len(df)

    # Step 3: Create contrast info table with thresholds
    contrast_info = data_formatters.create_contrast_info_table(
        analysis_info=sample_analysis_info,
        deg_data=sample_deg_data_dict,
        pvalue_thresh=0.05,
        lfc_thresh=1.0
    )

    assert len(contrast_info) == len(sorted_df)


@pytest.mark.integration
def test_full_heatmap_preparation_workflow(sample_deg_data_dict, sample_analysis_info):
    """Test complete workflow for preparing heatmap data."""
    # Step 1: Validate user inputs
    lfc_val, pval_val, _ = validation.validate_threshold_values("1.5", "0.05")
    gene_count, _ = validation.validate_gene_count("10")

    # Step 2: Select contrasts
    contrasts = [('GSE123456', 'Treatment_vs_Control')]
    assert validation.validate_contrasts_selected(contrasts)

    # Step 3: Group by organism
    organism_groups = organism_utils.group_contrasts_by_organism(
        sample_analysis_info,
        contrasts
    )

    # Step 4: Select genes per organism
    final_genes = []
    for organism, org_contrasts in organism_groups.items():
        # Get top DEGs
        top_genes = gene_selection.identify_frequent_degs(
            deg_data=sample_deg_data_dict,
            contrasts=org_contrasts,
            top_n=gene_count,
            p_thresh=pval_val,
            lfc_thresh=lfc_val
        )

        # Filter by organism
        filtered_genes = organism_utils.filter_genes_by_organism(
            deg_data=sample_deg_data_dict,
            cpm_data={},
            analysis_info=sample_analysis_info,
            genes=top_genes,
            organism=organism,
            selected_contrasts=org_contrasts
        )

        final_genes.extend(filtered_genes)

    # Step 5: Final validation
    assert validation.validate_custom_gene_list(final_genes)
    assert len(final_genes) <= gene_count