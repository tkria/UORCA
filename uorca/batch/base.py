"""
Base Batch Processor Interface

This module defines the abstract base class for all batch processing systems.
All concrete batch processors (SLURM, local, etc.) must inherit from this class.
"""

import json
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class BatchProcessor(ABC):
    """
    Abstract base class for batch processing systems.

    This class defines the common interface that all batch processors must implement.
    """

    def __init__(self):
        """Initialize the batch processor."""
        self.name = self.__class__.__name__.replace('BatchProcessor', '').lower()

    @abstractmethod
    def submit_datasets(self, input_path: str, output_dir: str, **kwargs) -> int:
        """
        Submit batch jobs for processing datasets.

        Args:
            input_path: Either CSV file with dataset information or directory containing CSV and metadata
            output_dir: Output directory for results
            **kwargs: Additional system-specific parameters

        Returns:
            Number of jobs submitted

        Raises:
            FileNotFoundError: If input doesn't exist or CSV not found
            ValueError: If invalid input format
        """
        pass

    @abstractmethod
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a specific job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary containing job status information
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was successfully cancelled, False otherwise
        """
        pass

    @abstractmethod
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get information about available computational resources.

        Returns:
            Dictionary containing resource information (CPUs, memory, etc.)
        """
        pass

    def resolve_input_and_extract_metadata(self, input_path: str) -> tuple[str, str]:
        """
        Resolve input path to CSV file and extract research question if available.

        Args:
            input_path: Either a CSV file path or directory containing CSV and metadata

        Returns:
            Tuple of (csv_file_path, research_question)

        Raises:
            FileNotFoundError: If input doesn't exist or CSV not found
            ValueError: If invalid input format
        """
        input_path_obj = Path(input_path)

        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        research_question = ""

        if input_path_obj.is_file():
            # Direct CSV file
            if not input_path_obj.suffix.lower() == '.csv':
                raise ValueError(f"File must be a CSV: {input_path}")
            csv_file = str(input_path_obj)
        elif input_path_obj.is_dir():
            # Directory containing CSV and metadata

            # Look for identification metadata JSON
            metadata_file = input_path_obj / "identification_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        research_question = metadata.get("input_query", "")
                        print(f"Found research question from metadata: {research_question}")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not read research question from metadata: {e}")

            # Look for CSV file in directory
            csv_files = list(input_path_obj.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
            elif len(csv_files) == 1:
                csv_file = str(csv_files[0])
                print(f"Found CSV file: {csv_files[0].name}")
            else:
                # Look for the expected CSV file from dataset identification
                expected_csv = input_path_obj / "selected_datasets.csv"
                if expected_csv.exists():
                    csv_file = str(expected_csv)
                    print(f"Found expected CSV file: selected_datasets.csv")
                else:
                    csv_names = [f.name for f in csv_files]
                    raise ValueError(f"Expected 'selected_datasets.csv' not found in directory. Found CSV files: {csv_names}. Please ensure you're using the output from 'uorca identify' or specify the CSV file directly.")
        else:
            raise ValueError(f"Input must be a CSV file or directory: {input_path}")

        return csv_file, research_question

    def validate_csv_file(self, input_path: str) -> tuple[pd.DataFrame, str]:
        """
        Validate input and prepare dataset information.

        Args:
            input_path: Either CSV file path or directory containing CSV and metadata

        Returns:
            Tuple of (validated DataFrame, research_question)

        Raises:
            FileNotFoundError: If input doesn't exist or CSV not found
            ValueError: If required columns are missing
        """
        csv_file, research_question = self.resolve_input_and_extract_metadata(input_path)

        csv_path = Path(csv_file)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        # Check required columns
        required_columns = ['Accession']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Handle dataset size columns (prefer DatasetSizeGB, fallback to DatasetSizeBytes)
        if 'DatasetSizeGB' in df.columns:
            # Size already in GB, apply 2x safety factor
            df['SafeStorageGB'] = df['DatasetSizeGB'] * 2.0
            print("Using recorded dataset size × 2 as approximation for required storage space")
        elif 'DatasetSizeBytes' in df.columns:
            # Convert from bytes to GB and apply 2x safety factor
            df['SafeStorageGB'] = (df['DatasetSizeBytes'] / (1024**3)) * 2.0
            print("Using recorded dataset size × 2 as approximation for required storage space")
        else:
            # No size information available
            df['SafeStorageGB'] = 0
            print("Warning: Neither DatasetSizeGB nor DatasetSizeBytes column found, using default value of 0")

        return df, research_question

    def save_research_question(self, output_dir: str, research_question: str) -> None:
        """
        Save research question to output directory for later use.

        Args:
            output_dir: Output directory path
            research_question: Research question to save
        """
        if research_question.strip():
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            question_file = output_path / "research_question.json"
            question_data = {
                "research_question": research_question.strip(),
                "source": "dataset_identification_metadata"
            }

            try:
                with open(question_file, 'w', encoding='utf-8') as f:
                    json.dump(question_data, f, indent=2)
                print(f"Saved research question to: {question_file}")
            except Exception as e:
                print(f"Warning: Could not save research question: {e}")

    def setup_job_tracking(self, output_dir: str) -> Path:
        """
        Set up job status tracking directory.

        Args:
            output_dir: Base output directory

        Returns:
            Path to job status directory
        """
        status_dir = Path(output_dir) / "job_status"
        status_dir.mkdir(parents=True, exist_ok=True)
        return status_dir

    def create_job_status_file(self, status_dir: Path, accession: str,
                              job_id: str, **extra_info) -> Path:
        """
        Create a job status tracking file.

        Args:
            status_dir: Directory for status files
            accession: Dataset accession
            job_id: Job identifier
            **extra_info: Additional information to store

        Returns:
            Path to created status file
        """
        status_file = status_dir / f"{accession}_status.json"

        status_info = {
            'accession': accession,
            'job_id': job_id,
            'state': 'submitted',
            'submitted_time': datetime.now().isoformat(),
            'system': self.name,
            **extra_info
        }

        with open(status_file, 'w') as f:
            json.dump(status_info, f, indent=2)

        return status_file

    def update_job_status(self, status_file: Path, **updates):
        """
        Update job status information.

        Args:
            status_file: Path to status file
            **updates: Fields to update
        """
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
        else:
            status = {}

        status.update(updates)
        status['last_updated'] = datetime.now().isoformat()

        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def get_running_jobs_storage(self, output_dir: str) -> float:
        """
        Calculate total storage being used by currently running jobs.

        Args:
            output_dir: Output directory containing job status

        Returns:
            Total storage in GB
        """
        total_storage = 0.0
        status_dir = Path(output_dir) / "job_status"

        if not status_dir.exists():
            return total_storage

        for status_file in status_dir.glob("*_status.json"):
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)

                if status.get('state') in ['submitted', 'running']:
                    total_storage += status.get('storage_gb', 0)
            except (json.JSONDecodeError, FileNotFoundError):
                continue

        return total_storage

    def display_scheduling_order(self, df: pd.DataFrame, max_storage_gb: float):
        """
        Display the dataset scheduling order and storage analysis.

        Args:
            df: DataFrame with dataset information
            max_storage_gb: Maximum storage limit
        """
        df_sorted = df.sort_values('SafeStorageGB', ascending=False)

        print(f"\n{'='*60}")
        print(f"DATASET SCHEDULING ORDER (Largest First)")
        print(f"{'='*60}")
        print(f"Batch system: {self.name.upper()}")
        print(f"Maximum storage limit: {max_storage_gb:.2f} GB")
        print(f"Total datasets: {len(df_sorted)}")

        total_storage = df_sorted['SafeStorageGB'].sum()
        print(f"Total storage needed: {total_storage:.2f} GB")

        # Show largest 10 datasets
        print(f"\nLargest 10 datasets (will be prioritized first):")
        for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['Accession']:12s} - {row['SafeStorageGB']:6.2f} GB")

        if len(df_sorted) > 10:
            print(f"  ... and {len(df_sorted)-10} more datasets")

        # Show smallest 5 datasets
        if len(df_sorted) > 5:
            print(f"\nSmallest 5 datasets (will be processed last):")
            for i, (_, row) in enumerate(df_sorted.tail(5).iterrows()):
                pos = len(df_sorted) - 4 + i
                print(f"  {pos:2d}. {row['Accession']:12s} - {row['SafeStorageGB']:6.2f} GB")

        # Storage analysis
        datasets_that_fit = df_sorted[df_sorted['SafeStorageGB'] <= max_storage_gb]
        datasets_too_large = df_sorted[df_sorted['SafeStorageGB'] > max_storage_gb]

        print(f"\nStorage Analysis:")
        print(f"  Datasets that fit individually: {len(datasets_that_fit)}/{len(df_sorted)}")
        if len(datasets_too_large) > 0:
            print(f"  Datasets too large for storage limit:")
            for _, row in datasets_too_large.iterrows():
                print(f"    {row['Accession']:12s} - {row['SafeStorageGB']:6.2f} GB (exceeds {max_storage_gb:.2f} GB limit)")

        print(f"{'='*60}\n")

    def check_environment_requirements(self) -> List[str]:
        """
        Check if all required environment variables and tools are available.

        Note: Environment variable validation is now handled at the CLI level.

        Returns:
            List of missing requirements (empty if all satisfied)
        """
        # Environment variables are now validated in CLI before reaching this point
        return []

    @property
    @abstractmethod
    def default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for this batch system.

        Returns:
            Dictionary of default parameters
        """
        pass

    def __str__(self) -> str:
        """String representation of the batch processor."""
        return f"{self.name.capitalize()} Batch Processor"

    def __repr__(self) -> str:
        """Developer representation of the batch processor."""
        return f"<{self.__class__.__name__}()>"
