"""Shared mutable state for the RNA-seq workflow graph."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import pandas as pd


class CheckpointStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowState:
    """Per-run mutable state shared across all graph nodes."""

    # === User-provided inputs ===
    dataset_id: str  # GSE accession (e.g., "GSE12345") or user dataset ID
    research_question: Optional[str] = None  # Optional context for AI
    data_source: Literal["geo", "user"] = "geo"  # Track data source mode

    # === User data config (for retry support) ===
    user_fastq_dir: Optional[str] = None  # Original FASTQ directory from config
    user_metadata_path: Optional[str] = None  # Original metadata path from config
    user_organism: Optional[str] = None  # Organism from config (may be None)
    user_description: Optional[str] = None  # Description from config

    # === Extracted/derived data ===
    metadata_df: Optional[pd.DataFrame] = None
    metadata_path: Optional[str] = None
    dataset_information: Optional[str] = None  # Title, summary from GEO
    organism: Optional[str] = None
    fastq_dir: Optional[str] = None

    # === Analysis configuration ===
    merged_column: Optional[str] = None  # Grouping variable name
    unique_groups: Optional[List[str]] = None  # Unique values
    contrasts: Optional[List[Dict[str, str]]] = None
    contrast_path: Optional[str] = None

    # === Execution outputs ===
    abundance_files: Optional[List[str]] = None
    kallisto_index_used: Optional[str] = None
    tx2gene_file_used: Optional[str] = None
    sample_mapping_path: Optional[str] = None
    fastq_sample_mapping: Optional[Dict[str, str]] = None  # fastq_prefix -> sample_id
    deg_results_path: Optional[str] = None

    # === Checkpoint tracking (graph-level) ===
    checkpoints: Dict[str, CheckpointStatus] = field(
        default_factory=lambda: {
            "entry": CheckpointStatus.NOT_STARTED,
            "extraction": CheckpointStatus.NOT_STARTED,
            "metadata_analysis": CheckpointStatus.NOT_STARTED,
            "contrast_design": CheckpointStatus.NOT_STARTED,
            "kallisto_quant": CheckpointStatus.NOT_STARTED,
            "edger_prep": CheckpointStatus.NOT_STARTED,
            "de_analysis": CheckpointStatus.NOT_STARTED,
        }
    )
    last_successful_checkpoint: Optional[str] = None

    # === Reflection/retry tracking ===
    analysis_history: List[Dict[str, Any]] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    reflection_iterations: int = 0

    # === Result ===
    analysis_success: Optional[bool] = None
    analysis_diagnostics: Optional[str] = None

    def update_checkpoint(self, name: str, status: CheckpointStatus) -> None:
        """Update checkpoint and track last successful."""
        self.checkpoints[name] = status
        if status == CheckpointStatus.COMPLETED:
            self.last_successful_checkpoint = name
