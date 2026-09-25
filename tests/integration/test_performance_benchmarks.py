"""Performance benchmarks for core modules."""
import pytest
import pandas as pd
import time
from uorca.core import organism_utils, data_formatters, gene_selection, validation


@pytest.fixture
def large_deg_data():
    """Create larger DEG dataset for performance testing."""
    genes = [f'GENE{i}' for i in range(1000)]
    return pd.DataFrame({
        'Gene': genes,
        'logFC': [2.5 if i % 2 == 0 else -1.8 for i in range(1000)],
        'adj.P.Val': [0.001 if i % 3 == 0 else 0.15 for i in range(1000)],
        'AveExpr': [10.5 + i * 0.1 for i in range(1000)]
    })


@pytest.fixture
def large_deg_data_dict(large_deg_data):
    """Large nested DEG data structure."""
    return {
        f'GSE{i}': {
            f'Contrast{j}': large_deg_data.copy()
            for j in range(5)
        }
        for i in range(10)
    }


@pytest.fixture
def large_analysis_info():
    """Large analysis info structure."""
    return {
        f'GSE{i}': {
            'accession': f'GSE{i}',
            'organism': 'Homo sapiens' if i % 2 == 0 else 'Mus musculus',
            'analysis_success': True,
            'unique_groups': ['Control', 'Treatment'],
            'contrasts': [
                {
                    'name': f'Contrast{j}',
                    'expression': 'Treatment - Control',
                    'description': 'Test contrast'
                }
                for j in range(5)
            ]
        }
        for i in range(10)
    }


@pytest.mark.slow
@pytest.mark.integration
def test_identify_frequent_degs_performance(large_deg_data_dict, benchmark_timer):
    """Benchmark identify_frequent_degs with large dataset."""
    contrasts = [(f'GSE{i}', f'Contrast{j}') for i in range(10) for j in range(5)]

    with benchmark_timer() as times:
        result = gene_selection.identify_frequent_degs(
            deg_data=large_deg_data_dict,
            contrasts=contrasts,
            top_n=50,
            p_thresh=0.05,
            lfc_thresh=1.0
        )

    assert len(result) <= 50
    assert times['elapsed'] < 1.0  # Should complete in under 1 second
    print(f"\n✓ identify_frequent_degs: {times['elapsed']:.3f}s")


@pytest.mark.slow
@pytest.mark.integration
def test_group_datasets_by_organism_performance(large_analysis_info, benchmark_timer):
    """Benchmark organism grouping with many datasets."""
    selected_datasets = [f'GSE{i}' for i in range(10)]

    with benchmark_timer() as times:
        result = organism_utils.group_datasets_by_organism(
            large_analysis_info,
            selected_datasets
        )

    assert len(result) == 2  # Two organisms
    assert times['elapsed'] < 0.1  # Should be very fast
    print(f"\n✓ group_datasets_by_organism: {times['elapsed']:.3f}s")


@pytest.mark.slow
@pytest.mark.integration
def test_create_dataset_info_table_performance(large_analysis_info, large_deg_data_dict, benchmark_timer):
    """Benchmark dataset info table creation."""
    with benchmark_timer() as times:
        result = data_formatters.create_dataset_info_table(
            analysis_info=large_analysis_info,
            dataset_info={},
            deg_data=large_deg_data_dict
        )

    assert len(result) == 10
    assert times['elapsed'] < 0.5  # Should be reasonably fast
    print(f"\n✓ create_dataset_info_table: {times['elapsed']:.3f}s")


@pytest.mark.slow
@pytest.mark.integration
def test_filter_genes_by_organism_performance(large_deg_data_dict, large_analysis_info, benchmark_timer):
    """Benchmark gene filtering with large gene list."""
    genes = [f'GENE{i}' for i in range(500)]
    contrasts = [(f'GSE{i}', f'Contrast0') for i in range(5)]

    with benchmark_timer() as times:
        result = organism_utils.filter_genes_by_organism(
            deg_data=large_deg_data_dict,
            cpm_data={},
            analysis_info=large_analysis_info,
            genes=genes,
            organism='Homo sapiens',
            selected_contrasts=contrasts
        )

    assert times['elapsed'] < 0.5
    print(f"\n✓ filter_genes_by_organism: {times['elapsed']:.3f}s")


@pytest.mark.slow
@pytest.mark.integration
def test_get_available_genes_performance(large_deg_data_dict, benchmark_timer):
    """Benchmark getting available genes from many contrasts."""
    contrasts = [(f'GSE{i}', f'Contrast{j}') for i in range(10) for j in range(5)]

    with benchmark_timer() as times:
        result = gene_selection.get_available_genes_for_contrasts(
            large_deg_data_dict,
            contrasts
        )

    assert len(result) > 0
    assert times['elapsed'] < 0.5
    print(f"\n✓ get_available_genes_for_contrasts: {times['elapsed']:.3f}s")


@pytest.mark.integration
def test_validation_performance(benchmark_timer):
    """Benchmark validation functions (should be near-instant)."""
    with benchmark_timer() as times:
        for _ in range(1000):
            validation.validate_threshold_values("1.5", "0.05")
            validation.validate_gene_count("50")
            validation.validate_custom_gene_list(['GENE1', 'GENE2'])
            validation.validate_contrasts_selected([('GSE1', 'C1')])

    assert times['elapsed'] < 0.1  # 1000 iterations should be very fast
    print(f"\n✓ validation functions (1000 calls): {times['elapsed']:.3f}s")


@pytest.mark.slow
@pytest.mark.integration
def test_sort_large_dataframe_performance(benchmark_timer):
    """Benchmark sorting large GEO dataframe."""
    # Create large dataframe
    df = pd.DataFrame({
        'Accession': [f'GSE{i}' for i in range(1000, 1, -1)],
        'Data': list(range(999))
    })

    with benchmark_timer() as times:
        result = data_formatters.sort_by_geo_accession(df)

    assert len(result) == 999
    assert result['Accession'].iloc[0] == 'GSE2'
    assert times['elapsed'] < 0.1
    print(f"\n✓ sort_by_geo_accession (999 rows): {times['elapsed']:.3f}s")


@pytest.mark.slow
@pytest.mark.integration
def test_end_to_end_workflow_performance(large_deg_data_dict, large_analysis_info, benchmark_timer):
    """Benchmark complete heatmap preparation workflow."""
    with benchmark_timer() as times:
        # Validation
        lfc_val, pval_val, _ = validation.validate_threshold_values("1.5", "0.05")
        gene_count, _ = validation.validate_gene_count("50")

        # Select contrasts
        contrasts = [(f'GSE{i}', 'Contrast0') for i in range(5)]
        validation.validate_contrasts_selected(contrasts)

        # Group by organism
        organism_groups = organism_utils.group_contrasts_by_organism(
            large_analysis_info,
            contrasts
        )

        # Select genes
        all_genes = []
        for organism, org_contrasts in organism_groups.items():
            top_genes = gene_selection.identify_frequent_degs(
                deg_data=large_deg_data_dict,
                contrasts=org_contrasts,
                top_n=gene_count,
                p_thresh=pval_val,
                lfc_thresh=lfc_val
            )
            all_genes.extend(top_genes)

    assert len(all_genes) > 0
    assert times['elapsed'] < 2.0  # Full workflow should be reasonably fast
    print(f"\n✓ End-to-end workflow: {times['elapsed']:.3f}s")