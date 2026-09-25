"""Pydantic output models for AI agents. All agents return typed models."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MetadataAnalysisOut(BaseModel):
    """AI output for metadata column selection."""

    selected_columns: List[str] = Field(
        ..., description="Columns selected for differential expression grouping"
    )
    merged_column_name: str = Field(
        ..., description="Name for the merged analysis column"
    )
    unique_groups: List[str] = Field(
        ..., min_length=2, description="Unique experimental groups (minimum 2 for DE)"
    )
    reasoning: str = Field(
        ..., description="Why these columns were selected for analysis"
    )


class ContrastOut(BaseModel):
    """Single contrast definition for differential expression."""

    name: str = Field(..., description="Contrast identifier (e.g., 'treatment_vs_control')")
    expression: str = Field(..., description="R-style contrast expression")
    description: str = Field(..., description="Biological interpretation")
    justification: str = Field(..., description="Scientific rationale for this contrast")


class ContrastsOut(BaseModel):
    """AI output for contrast design."""

    contrasts: List[ContrastOut] = Field(..., min_length=1)
    analysis_summary: str = Field(..., description="Summary of the analysis design")


class KallistoIndexOut(BaseModel):
    """AI output for Kallisto index selection."""

    selected_index: str = Field(..., description="Path to selected Kallisto index")
    tx2gene_file: str = Field(..., description="Path to corresponding tx2gene file")
    organism_match: str = Field(..., description="Detected organism match")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation of why this index was selected")


class ReflectionOut(BaseModel):
    """AI output for failure diagnosis and retry routing."""

    root_cause: str = Field(..., description="Identified root cause of failure")
    failed_checkpoint: str = Field(..., description="Which checkpoint failed")
    retry_strategy: Literal["retry_same", "retry_from_checkpoint", "abort"] = Field(
        ..., description="Recommended action"
    )
    retry_from: Optional[str] = Field(
        None, description="Checkpoint to retry from (if retry_from_checkpoint)"
    )
    parameter_changes: Dict[str, Any] = Field(
        default_factory=dict, description="Suggested parameter modifications for retry"
    )
    reasoning: str = Field(..., description="Detailed explanation of diagnosis")


# === User Data Models ===


class FASTQPair(BaseModel):
    """A pair of FASTQ files (R1 and R2) representing one sample."""

    sample_prefix: str = Field(
        ..., description="Sample identifier extracted from filenames"
    )
    r1_file: str = Field(..., description="Filename of read 1 FASTQ file")
    r2_file: str = Field(..., description="Filename of read 2 FASTQ file")
    reasoning: str = Field(
        ..., description="Why these files were paired together"
    )


class FASTQPairDiscoveryResult(BaseModel):
    """Output from FASTQ pair discovery agent."""

    pairs: List[FASTQPair] = Field(
        ..., description="List of discovered FASTQ pairs"
    )
    naming_pattern: str = Field(
        ..., description="Description of the naming pattern detected"
    )
    unpaired_files: List[str] = Field(
        default_factory=list, description="Files that could not be paired"
    )
    reasoning: str = Field(
        ..., description="Explanation of how pairs were determined"
    )


class FASTQSampleMapping(BaseModel):
    """Single FASTQ-to-metadata mapping."""

    fastq_prefix: str = Field(
        ..., description="FASTQ file prefix (e.g., 'sample1' from sample1_R1.fastq.gz)"
    )
    metadata_id: str = Field(
        ..., description="Value from the identified sample ID column in metadata"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this mapping"
    )


class FASTQMetadataLinkResult(BaseModel):
    """Output from FASTQ-metadata linker agent."""

    # Matching result
    can_match: bool = Field(
        ..., description="Whether FASTQ files can be matched to metadata rows"
    )

    # Column identification
    sample_id_column: Optional[str] = Field(
        None, description="Column in metadata that contains sample identifiers"
    )

    # Mappings (only populated if can_match=True)
    mappings: List[FASTQSampleMapping] = Field(
        default_factory=list, description="FASTQ to metadata mappings"
    )

    # Organism inference (if not provided in config)
    inferred_organism: Optional[str] = Field(
        None, description="Organism inferred from metadata"
    )
    organism_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in organism inference"
    )

    # Clarification request (only populated if can_match=False)
    clarification_needed: Optional[str] = Field(
        None, description="What clarification is needed from user"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Possible solutions for user"
    )

    # Reasoning
    reasoning: str = Field(..., description="Explanation of matching logic")

    # Warnings (non-fatal issues)
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal warnings about the data"
    )


class KallistoSampleMapping(BaseModel):
    """Single Kallisto output to metadata mapping."""

    kallisto_sample: str = Field(
        ..., description="Kallisto output directory name"
    )
    metadata_row_id: str = Field(
        ..., description="Identifier to match in metadata (e.g., file_name value)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this mapping"
    )


class KallistoSampleLinkerResult(BaseModel):
    """Output from Kallisto sample linker agent (fallback for edgeR prep)."""

    can_match: bool = Field(
        ..., description="Whether Kallisto samples can be matched to metadata"
    )

    metadata_column: str = Field(
        ..., description="Column in metadata to use for matching"
    )

    mappings: List[KallistoSampleMapping] = Field(
        default_factory=list, description="Kallisto sample to metadata mappings"
    )

    pattern_detected: str = Field(
        ..., description="Description of naming pattern detected in Kallisto outputs"
    )

    transformation_rule: str = Field(
        ..., description="Rule for transforming Kallisto names to metadata IDs"
    )

    reasoning: str = Field(
        ..., description="Explanation of matching logic used"
    )

    unmatched_samples: List[str] = Field(
        default_factory=list, description="Samples that could not be matched"
    )
