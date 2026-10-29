"""Tests for gene_selection module."""
import pytest
import pandas as pd
from uorca.core.pvalues import MissingAdjustedPValueError
from uorca.core.gene_selection import (
    identify_frequent_degs,
    get_all_genes_from_cpm,
    get_available_genes_for_contrasts,
    validate_custom_genes
)


def test_identify_frequent_degs(sample_deg_data_dict):
    """Test identifying frequently DE genes."""
    contrasts = [
        ('GSE123456', 'Treatment_vs_Control'),
        ('GSE789012', 'KO_vs_WT')
    ]

    result = identify_frequent_degs(
        deg_data=sample_deg_data_dict,
        contrasts=contrasts,
        top_n=3,
        p_thresh=0.05,
        lfc_thresh=1.0
    )

    assert len(result) <= 3
    assert all(isinstance(gene, str) for gene in result)


def test_identify_frequent_degs_with_high_threshold(sample_deg_data_dict):
    """Test with high threshold returns fewer genes."""
    contrasts = [('GSE123456', 'Treatment_vs_Control')]

    # With high threshold
    result_strict = identify_frequent_degs(
        deg_data=sample_deg_data_dict,
        contrasts=contrasts,
        top_n=10,
        p_thresh=0.001,
        lfc_thresh=2.0
    )

    # With lenient threshold
    result_lenient = identify_frequent_degs(
        deg_data=sample_deg_data_dict,
        contrasts=contrasts,
        top_n=10,
        p_thresh=0.05,
        lfc_thresh=1.0
    )

    # Lenient should find more or equal genes
    assert len(result_lenient) >= len(result_strict)


def test_identify_frequent_degs_empty():
    """Test with no contrasts."""
    result = identify_frequent_degs(
        deg_data={},
        contrasts=[],
        top_n=10,
        p_thresh=0.05,
        lfc_thresh=1.0
    )

    assert result == []


def test_identify_frequent_degs_missing_columns(sample_analysis_info):
    """DEG data without adj.P.Val raises rather than being silently skipped."""
    bad_deg_data = {
        'GSE123456': {
            'Treatment_vs_Control': pd.DataFrame({
                'Gene': ['GENE1', 'GENE2'],
                'BadColumn': [1, 2]
            })
        }
    }

    # A table without adj.P.Val fails loudly instead of being skipped
    with pytest.raises(MissingAdjustedPValueError, match="GSE123456/Treatment_vs_Control"):
        identify_frequent_degs(
            deg_data=bad_deg_data,
            contrasts=[('GSE123456', 'Treatment_vs_Control')],
            top_n=10,
            p_thresh=0.05,
            lfc_thresh=1.0
        )


def test_get_all_genes_from_cpm(sample_cpm_data):
    """Test extracting all genes from CPM data."""
    cpm_dict = {'dataset1': sample_cpm_data}
    result = get_all_genes_from_cpm(cpm_dict)

    assert len(result) == 3
    assert result == ['GENE1', 'GENE2', 'GENE3']  # Should be sorted


def test_get_all_genes_from_cpm_multiple_datasets():
    """Test with multiple datasets."""
    cpm_dict = {
        'dataset1': pd.DataFrame({'Gene': ['GENE1', 'GENE2']}),
        'dataset2': pd.DataFrame({'Gene': ['GENE2', 'GENE3']})
    }

    result = get_all_genes_from_cpm(cpm_dict)

    # Should have unique genes, sorted
    assert len(result) == 3
    assert result == ['GENE1', 'GENE2', 'GENE3']


def test_get_all_genes_from_cpm_empty():
    """Test with no CPM data."""
    result = get_all_genes_from_cpm({})
    assert result == []


def test_get_available_genes_for_contrasts(sample_deg_data_dict):
    """Test getting available genes for contrasts."""
    contrasts = [('GSE123456', 'Treatment_vs_Control')]
    result = get_available_genes_for_contrasts(sample_deg_data_dict, contrasts)

    assert len(result) == 5
    assert 'GENE1' in result
    assert 'GENE5' in result


def test_get_available_genes_for_contrasts_multiple(sample_deg_data_dict):
    """Test with multiple contrasts."""
    contrasts = [
        ('GSE123456', 'Treatment_vs_Control'),
        ('GSE789012', 'KO_vs_WT')
    ]
    result = get_available_genes_for_contrasts(sample_deg_data_dict, contrasts)

    # Should have genes from both (but same genes since fixtures are copies)
    assert len(result) == 5


def test_get_available_genes_for_contrasts_empty():
    """Test with no contrasts."""
    result = get_available_genes_for_contrasts({}, [])
    assert result == set()


def test_validate_custom_genes():
    """Test custom gene validation."""
    custom = ['GENE1', 'GENE2', 'MISSING1', 'MISSING2']
    available = {'GENE1', 'GENE2', 'GENE3'}

    found, missing = validate_custom_genes(custom, available)

    assert found == ['GENE1', 'GENE2']
    assert missing == ['MISSING1', 'MISSING2']


def test_validate_custom_genes_all_found():
    """Test when all genes are found."""
    custom = ['GENE1', 'GENE2']
    available = {'GENE1', 'GENE2', 'GENE3'}

    found, missing = validate_custom_genes(custom, available)

    assert found == ['GENE1', 'GENE2']
    assert missing == []


def test_validate_custom_genes_none_found():
    """Test when no genes are found."""
    custom = ['MISSING1', 'MISSING2']
    available = {'GENE1', 'GENE2'}

    found, missing = validate_custom_genes(custom, available)

    assert found == []
    assert missing == ['MISSING1', 'MISSING2']


def test_validate_custom_genes_empty():
    """Test with empty gene list."""
    found, missing = validate_custom_genes([], {'GENE1'})
    assert found == []
    assert missing == []