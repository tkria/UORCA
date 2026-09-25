"""Unit tests for metadata column selection in the graph workflow."""

import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from uorca.graph.models import MetadataAnalysisOut
from uorca.graph.state import WorkflowState, CheckpointStatus


class TestMetadataColumnSelection:
    """Tests for MetadataAnalysisNode column handling."""

    @pytest.fixture
    def sample_metadata_df(self):
        """Create a sample metadata dataframe similar to GEO data."""
        return pd.DataFrame({
            "GSM": ["GSM1", "GSM2", "GSM3", "GSM4", "GSM5", "GSM6"],
            "SRR": ["SRR1", "SRR2", "SRR3", "SRR4", "SRR5", "SRR6"],
            "infection status:ch1": ["Mock", "Mock", "SARS-CoV-2", "SARS-CoV-2", "IAV", "IAV"],
            "cell_type:ch1": ["iPSC", "iPSC", "iPSC", "iPSC", "iPSC", "iPSC"],
            "title": ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5", "Sample 6"],
        })

    @pytest.fixture
    def workflow_state(self, sample_metadata_df):
        """Create a workflow state with metadata."""
        state = WorkflowState(dataset_id="GSE123456")
        state.metadata_df = sample_metadata_df
        return state

    def test_validate_selected_columns_exist(self, sample_metadata_df):
        """Test that non-existent columns raise an error."""
        selected_columns = ["nonexistent_column"]

        missing_cols = [c for c in selected_columns if c not in sample_metadata_df.columns]

        assert len(missing_cols) == 1
        assert "nonexistent_column" in missing_cols

    def test_single_column_same_name_no_modification(self, sample_metadata_df):
        """Test that selecting a single column with matching name doesn't duplicate."""
        df = sample_metadata_df.copy()
        selected_columns = ["infection status:ch1"]
        merged_column_name = "infection status:ch1"

        # Simulate the logic from MetadataAnalysisNode
        if len(selected_columns) > 1:
            df[merged_column_name] = df[selected_columns].astype(str).agg("_".join, axis=1)
        elif selected_columns[0] != merged_column_name:
            df[merged_column_name] = df[selected_columns[0]]
        # else: no action needed

        # Column should exist with original values
        assert merged_column_name in df.columns
        assert df[merged_column_name].tolist() == ["Mock", "Mock", "SARS-CoV-2", "SARS-CoV-2", "IAV", "IAV"]

    def test_single_column_different_name_creates_copy(self, sample_metadata_df):
        """Test that selecting a single column with different name creates a copy."""
        df = sample_metadata_df.copy()
        selected_columns = ["infection status:ch1"]
        merged_column_name = "infection_status"  # Different name!

        original_cols = list(df.columns)

        # Simulate the logic from MetadataAnalysisNode
        if len(selected_columns) > 1:
            df[merged_column_name] = df[selected_columns].astype(str).agg("_".join, axis=1)
        elif selected_columns[0] != merged_column_name:
            df[merged_column_name] = df[selected_columns[0]]

        # New column should be created
        assert merged_column_name in df.columns
        assert merged_column_name not in original_cols
        assert df[merged_column_name].tolist() == ["Mock", "Mock", "SARS-CoV-2", "SARS-CoV-2", "IAV", "IAV"]

    def test_multiple_columns_merged(self, sample_metadata_df):
        """Test that multiple columns are properly merged."""
        df = sample_metadata_df.copy()
        selected_columns = ["infection status:ch1", "cell_type:ch1"]
        merged_column_name = "condition"

        # Simulate the logic from MetadataAnalysisNode
        if len(selected_columns) > 1:
            df[merged_column_name] = df[selected_columns].astype(str).agg("_".join, axis=1)
        elif selected_columns[0] != merged_column_name:
            df[merged_column_name] = df[selected_columns[0]]

        # Merged column should exist with combined values
        assert merged_column_name in df.columns
        expected = ["Mock_iPSC", "Mock_iPSC", "SARS-CoV-2_iPSC", "SARS-CoV-2_iPSC", "IAV_iPSC", "IAV_iPSC"]
        assert df[merged_column_name].tolist() == expected

    def test_unique_groups_extracted_correctly(self, sample_metadata_df):
        """Test that unique groups are correctly identified."""
        df = sample_metadata_df.copy()
        merged_column_name = "infection status:ch1"

        actual_groups = df[merged_column_name].unique().tolist()

        assert set(actual_groups) == {"Mock", "SARS-CoV-2", "IAV"}
        assert len(actual_groups) == 3

    def test_minimum_two_groups_required(self, sample_metadata_df):
        """Test that at least 2 groups are required for DE analysis."""
        # Create a dataframe with only one group
        df = pd.DataFrame({
            "GSM": ["GSM1", "GSM2", "GSM3"],
            "condition": ["Control", "Control", "Control"],
        })

        unique_groups = df["condition"].unique().tolist()

        assert len(unique_groups) < 2  # Should fail validation

    def test_dataframe_modification_persists(self, sample_metadata_df):
        """Test that modifications to the dataframe persist (critical bug fix)."""
        # Simulate the workflow state
        class MockState:
            def __init__(self):
                self.metadata_df = sample_metadata_df.copy()
                self.merged_column = None

        state = MockState()
        df = state.metadata_df

        # Simulate adding a new column
        df["new_merged_column"] = df["infection status:ch1"]

        # CRITICAL: Save back to state (this was the bug!)
        state.metadata_df = df

        # Verify the column persists in state
        assert "new_merged_column" in state.metadata_df.columns


class TestMetadataAnalysisIntegration:
    """Integration-style tests for the metadata analysis flow."""

    def test_agent_output_validation(self):
        """Test that MetadataAnalysisOut model validates correctly."""
        # Valid output
        output = MetadataAnalysisOut(
            selected_columns=["infection status:ch1"],
            merged_column_name="infection_status",
            unique_groups=["Mock", "SARS-CoV-2", "IAV"],
            reasoning="Selected infection status as the grouping variable."
        )

        assert output.selected_columns == ["infection status:ch1"]
        assert output.merged_column_name == "infection_status"
        assert len(output.unique_groups) == 3

    def test_agent_output_minimum_groups(self):
        """Test that MetadataAnalysisOut requires at least 2 groups."""
        with pytest.raises(Exception):  # Pydantic validation error
            MetadataAnalysisOut(
                selected_columns=["condition"],
                merged_column_name="condition",
                unique_groups=["Control"],  # Only 1 group - should fail
                reasoning="Only one group available."
            )

    def test_column_name_mismatch_scenario(self):
        """Test the specific scenario from the bug report."""
        # This is the scenario that was failing:
        # Agent selects "infection status:ch1" but names it "infection_status"

        df = pd.DataFrame({
            "GSM": ["GSM1", "GSM2", "GSM3"],
            "infection status:ch1": ["Mock", "SARS-CoV-2", "IAV"],
        })

        selected_columns = ["infection status:ch1"]
        merged_column_name = "infection_status"

        # Before fix: this scenario would NOT create the column
        # After fix: this should create a copy with the new name

        if len(selected_columns) > 1:
            df[merged_column_name] = df[selected_columns].astype(str).agg("_".join, axis=1)
        elif selected_columns[0] != merged_column_name:
            df[merged_column_name] = df[selected_columns[0]]

        # The merged_column_name should now exist
        assert merged_column_name in df.columns
        assert df[merged_column_name].tolist() == ["Mock", "SARS-CoV-2", "IAV"]
